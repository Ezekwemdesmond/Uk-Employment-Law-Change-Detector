"""
Penalty extractor for UK employment legislation change text.

Uses regex patterns to identify monetary penalty thresholds.
Returns None when no penalty amount is found.

Recognises formats:
  - £500
  - £1,500
  - £10,000 per week / per day / per employee
  - unlimited fine / unlimited penalty / unlimited compensation
"""

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Matches "unlimited" qualifiers before fine/penalty/compensation/award
_UNLIMITED_PATTERN = re.compile(
    r"\bunlimited\s+(?:fine|penalty|compensation|award|damages)\b",
    re.IGNORECASE,
)

# Matches a pound-sterling amount with optional thousands separators and pence
# e.g. £500, £1,500, £10,000.00
_AMOUNT_PATTERN = re.compile(
    r"£([\d]{1,3}(?:,\d{3})*(?:\.\d{2})?)",
    re.IGNORECASE,
)

# Qualifiers that may follow a monetary amount
_QUALIFIER_PATTERN = re.compile(
    r"£[\d,]+(?:\.\d{2})?"
    r"(?:\s*(?:per\s+(?:week|day|month|year|employee|worker|breach|contravention)"
    r"|a\s+(?:day|week|month)"
    r"|each\s+(?:day|week|month|employee|worker)))?",
    re.IGNORECASE,
)

# Common penalty-signal phrases that indicate the figure IS a threshold
_PENALTY_CONTEXT_PATTERN = re.compile(
    r"(?:fine|penalty|penalt(?:y|ies)|award|compensation|sanction|"
    r"liability|maximum|up\s+to|not\s+exceeding|civil\s+penalty|"
    r"daily\s+rate|basic\s+award|compensatory\s+award)",
    re.IGNORECASE,
)


def _parse_amount(raw: str) -> int:
    """
    Convert a raw pound-string (digits, commas, and optional pence) to an integer.

    Strips commas and truncates any decimal portion before conversion,
    so both ``'10,000'`` and ``'1,000.00'`` are handled correctly.

    Args:
        raw: String like ``'10,000'``, ``'500'``, or ``'9.50'``.

    Returns:
        Integer value (pence truncated, not rounded).
    """
    cleaned = raw.replace(",", "")
    return int(float(cleaned))


def extract_penalty(text: str) -> str | None:
    """
    Extract the most significant penalty threshold from legislation change text.

    Priority order:
      1. "unlimited" — returned as the string ``"unlimited"``.
      2. Largest pound-sterling amount appearing near a penalty-signal word.
      3. Largest pound-sterling amount in the text regardless of context.
      4. ``None`` — if no amount is found.

    Args:
        text: Preprocessed legislation change text.

    Returns:
        Penalty threshold as a formatted string (e.g. ``"£10,000"``,
        ``"£500 per day"``, ``"unlimited"``), or ``None``.
    """
    if not text:
        return None

    # 1. Unlimited penalty
    if _UNLIMITED_PATTERN.search(text):
        logger.debug("Unlimited penalty detected")
        return "unlimited"

    # 2. Find all full match spans (with qualifier) and their numeric values
    qualified_matches: list[tuple[int, str]] = []
    for match in _QUALIFIER_PATTERN.finditer(text):
        raw_match = match.group(0)
        amount_match = _AMOUNT_PATTERN.match(raw_match)
        if amount_match:
            value = _parse_amount(amount_match.group(1))
            qualified_matches.append((value, raw_match.strip()))

    if not qualified_matches:
        return None

    # 3. If a penalty-context word is present, prefer the highest contextual amount
    if _PENALTY_CONTEXT_PATTERN.search(text):
        # Return the text's highest amount as the threshold
        qualified_matches.sort(key=lambda x: x[0], reverse=True)
        logger.debug("Penalty threshold found: %s", qualified_matches[0][1])
        return qualified_matches[0][1]

    # 4. No penalty context — only return if exactly one amount exists (likely it IS the threshold)
    if len(qualified_matches) == 1:
        return qualified_matches[0][1]

    return None


def extract_all_amounts(text: str) -> list[str]:
    """
    Return all monetary amounts found in the text, regardless of context.

    Useful for debugging and inspection.

    Args:
        text: Preprocessed legislation change text.

    Returns:
        List of matched amount strings in order of appearance.
    """
    return [m.group(0).strip() for m in _QUALIFIER_PATTERN.finditer(text)]
