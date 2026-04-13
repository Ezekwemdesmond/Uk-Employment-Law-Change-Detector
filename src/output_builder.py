"""
Structured JSON output builder for material legislation changes.

Consumes scored predictions from Layer 2, filters to material changes only
(predicted_label == 1), and constructs a structured record per change with:
  - domain
  - effective_date
  - penalty_threshold
  - affected_business_types
  - obligation_type
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.domain_classifier import classify_domain, classify_domain_from_text
from src.penalty_extractor import extract_penalty

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

# ---------------------------------------------------------------------------
# Obligation-type inference
# ---------------------------------------------------------------------------

# change_type values from scorer / data_builder
_OBLIGATION_MAP: dict[str, str] = {
    "added_section": "new_duty",
    "new_obligation": "new_duty",
    "removed_section": "removed_duty",
    "removed_duty": "removed_duty",
    "penalty_change": "changed_penalty",
    "scope_extension": "extended_scope",
    "new_definition": "new_duty",
    "new_exemption": "new_duty",
    "modified_section": "changed_duty",
    "cross_reference": "changed_duty",
    "renumbering": "changed_duty",
}

_PENALTY_TEXT_SIGNALS = re.compile(
    r"\b(?:penalt|fine|compensation|award|sanction|maximum|up to £|not exceeding)\b",
    re.IGNORECASE,
)

_SCOPE_TEXT_SIGNALS = re.compile(
    r"\b(?:extended?|now (?:appl|cover|include)|scope|gig economy|"
    r"agency worker|platform worker|zero.hours|from day one)\b",
    re.IGNORECASE,
)

VALID_OBLIGATION_TYPES = frozenset(
    {"new_duty", "changed_duty", "removed_duty", "changed_penalty", "extended_scope"}
)

# ---------------------------------------------------------------------------
# Effective date extraction
# ---------------------------------------------------------------------------

_DATE_CONTEXT_PATTERN = re.compile(
    r"(?:from|with effect from|commencing|in force (?:from|on|at)|"
    r"effective|taking effect|applies from|operative from)\s+"
    r"(\d{1,2}\s+\w+\s+\d{4}|\d{4}-\d{2}-\d{2}|\w+\s+\d{4})",
    re.IGNORECASE,
)

_PLAIN_DATE_PATTERN = re.compile(
    r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4})\b",
    re.IGNORECASE,
)

_ISO_DATE_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def extract_effective_date(text: str) -> str | None:
    """
    Extract the earliest effective date from legislation change text.

    Searches for contextual phrases like "with effect from 1 April 2024"
    before falling back to any standalone date.

    Args:
        text: Preprocessed change text.

    Returns:
        Date string as it appears in the text, or ``None``.
    """
    # Prefer dates appearing after an "effective from" phrase
    for match in _DATE_CONTEXT_PATTERN.finditer(text):
        return match.group(1).strip()

    # Then any standalone named-month date
    match = _PLAIN_DATE_PATTERN.search(text)
    if match:
        return match.group(1).strip()

    # Finally ISO-format dates
    match = _ISO_DATE_PATTERN.search(text)
    if match:
        return match.group(1).strip()

    return None


# ---------------------------------------------------------------------------
# Affected business types extraction
# ---------------------------------------------------------------------------

_BUSINESS_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"small\s+businesses?|SME|fewer\s+than\s+\d+\s+employees?", re.IGNORECASE), "SMEs"),
    (re.compile(r"micro\s+businesses?|fewer\s+than\s+[59]\s+employees?", re.IGNORECASE), "micro businesses"),
    (re.compile(r"large\s+employers?|more\s+than\s+250\s+employees?", re.IGNORECASE), "large employers"),
    (re.compile(r"public\s+sector|public\s+authority|government\s+body|NHS|council", re.IGNORECASE), "public sector employers"),
    (re.compile(r"charit(?:y|ies)|voluntary\s+sector|not.for.profit", re.IGNORECASE), "charities and voluntary organisations"),
    (re.compile(r"agency|staffing|recruitment\s+agenc", re.IGNORECASE), "employment agencies"),
    (re.compile(r"gig\s+economy|platform\s+(?:worker|compan)", re.IGNORECASE), "gig economy platforms"),
    (re.compile(r"family\s+business|close\s+relative", re.IGNORECASE), "family businesses"),
]


def extract_affected_business_types(text: str) -> list[str]:
    """
    Identify affected business types from change text.

    Applies keyword patterns; defaults to ``["all employers"]`` when no
    specific business type is mentioned.

    Args:
        text: Preprocessed change text.

    Returns:
        Non-empty list of business-type strings.
    """
    found = []
    for pattern, label in _BUSINESS_PATTERNS:
        if pattern.search(text) and label not in found:
            found.append(label)

    if not found:
        return ["all employers"]
    return found


# ---------------------------------------------------------------------------
# Obligation type inference
# ---------------------------------------------------------------------------

def infer_obligation_type(change_type: str, text: str) -> str:
    """
    Infer the obligation type from the change type code and text content.

    Text signals can override the default mapping — e.g. a ``modified_section``
    that mentions penalties becomes ``changed_penalty``.

    Args:
        change_type: Change-type string from the scored prediction.
        text: Preprocessed change text.

    Returns:
        One of: ``"new_duty"``, ``"changed_duty"``, ``"removed_duty"``,
        ``"changed_penalty"``, ``"extended_scope"``.
    """
    # Start from the explicit mapping
    base = _OBLIGATION_MAP.get(change_type, "changed_duty")

    # Text-signal overrides (only on non-removal types)
    if base != "removed_duty":
        if _PENALTY_TEXT_SIGNALS.search(text):
            return "changed_penalty"
        if _SCOPE_TEXT_SIGNALS.search(text):
            return "extended_scope"

    return base


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_material_record(prediction: dict[str, Any], act_name: str) -> dict[str, Any]:
    """
    Build a single structured output record from a material prediction.

    Args:
        prediction: A scored prediction dict (from ``Scorer.predict_diff``).
        act_name: Canonical act name for domain lookup.

    Returns:
        Structured record with the five required fields plus metadata.
    """
    text = prediction.get("text", "")
    change_type = prediction.get("change_type", "modified_section")

    # Domain: prefer act-name lookup, fall back to text inference
    domain = classify_domain(act_name)
    if domain == "employment":
        # May be more specific based on text
        text_domain = classify_domain_from_text(text)
        if text_domain != "employment":
            domain = text_domain

    return {
        "section_id": prediction.get("section_id", "unknown"),
        "domain": domain,
        "effective_date": extract_effective_date(text),
        "penalty_threshold": extract_penalty(text),
        "affected_business_types": extract_affected_business_types(text),
        "obligation_type": infer_obligation_type(change_type, text),
        # Metadata for traceability
        "confidence": prediction.get("confidence"),
        "needs_review": prediction.get("needs_review", False),
        "change_type": change_type,
        "text_excerpt": text[:200] + "…" if len(text) > 200 else text,
    }


def build_output(
    scored_result: dict[str, Any],
    min_confidence: float = 0.0,
) -> dict[str, Any]:
    """
    Build the full structured JSON output for all material changes in a scored diff.

    Filters predictions to those where ``predicted_label == 1`` (material).
    Optionally further filters by minimum confidence.

    Args:
        scored_result: Return value of ``Scorer.predict_diff()``.
        min_confidence: Skip predictions below this confidence (default 0.0 = keep all material).

    Returns:
        Dict with keys:
          - ``act_name``
          - ``processed_at`` (ISO-8601 timestamp)
          - ``total_material_changes``
          - ``changes`` (list of structured records)
    """
    act_name = scored_result.get("act_name", "unknown")
    predictions = scored_result.get("predictions", [])

    material = [
        p for p in predictions
        if p.get("predicted_label") == 1
        and p.get("confidence", 0.0) >= min_confidence
    ]

    changes = [build_material_record(p, act_name) for p in material]

    logger.info(
        "Built output for '%s': %d material changes from %d total predictions",
        act_name, len(changes), len(predictions),
    )

    return {
        "act_name": act_name,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "total_material_changes": len(changes),
        "changes": changes,
    }


def save_output(
    output: dict[str, Any],
    act_name: str,
    output_dir: Path | None = None,
) -> Path:
    """
    Persist the structured output to a JSON file.

    Filename format: ``<act_name>_output_<YYYYMMDDTHHMMSSz>.json``

    Args:
        output: Dict produced by :func:`build_output`.
        act_name: Used in the filename.
        output_dir: Override directory; defaults to :data:`OUTPUTS_DIR`.

    Returns:
        :class:`pathlib.Path` of the written file.
    """
    target_dir = output_dir if output_dir is not None else OUTPUTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{act_name}_output_{timestamp}.json"
    filepath = target_dir / filename

    filepath.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved output to %s", filepath)
    return filepath
