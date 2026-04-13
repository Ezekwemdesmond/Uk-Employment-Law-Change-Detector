"""
FastAPI application for the UK Employment Law Change Detector.

Exposes the full pipeline — scrape → diff → preprocess → classify → output —
via a single POST /analyse endpoint.  A GET /health endpoint is also provided.

Pipeline flow
-------------
1. Parse the legislation URL to identify the act.
2. Fetch the current XML from legislation.gov.uk.
3. Locate the most recent cached XML for that act in data/raw/.
   - If no baseline exists: save the current XML and return an empty result
     with a "baseline_established" status.
   - If a baseline exists: proceed to diff.
4. Compute the XML diff with diff_engine.
5. Preprocess the diff with preprocessor.
6. Score with the trained BERT classifier (scorer).
7. Build and return structured JSON output (output_builder).
"""

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, field_validator

from src.diff_engine import compute_diff
from src.domain_classifier import url_to_act_name
from src.output_builder import build_output
from src.preprocessor import preprocess_diff
from src.scraper import build_session, fetch_legislation_xml, save_xml

logger = logging.getLogger(__name__)

DATA_RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
MODELS_DIR = Path(__file__).parent.parent / "models"

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class AnalyseRequest(BaseModel):
    """Request body for the /analyse endpoint."""

    url: str
    min_confidence: float = 0.0

    @field_validator("url")
    @classmethod
    def url_must_be_legislation_gov_uk(cls, v: str) -> str:
        """Validate that the URL points to legislation.gov.uk."""
        if "legislation.gov.uk" not in v:
            raise ValueError("URL must be a legislation.gov.uk address")
        return v.rstrip("/")

    @field_validator("min_confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        return v


class HealthResponse(BaseModel):
    """Response schema for /health."""

    status: str
    service: str
    model_available: bool


class AnalyseResponse(BaseModel):
    """Response schema for /analyse."""

    act_name: str
    pipeline_status: str  # "changes_found" | "baseline_established" | "no_changes"
    processed_at: str
    total_material_changes: int
    changes: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_latest_cached_xml(act_name: str, raw_dir: Path) -> Path | None:
    """
    Return the most recently saved XML file for *act_name*, or None.

    Args:
        act_name: Canonical act identifier.
        raw_dir: Directory to search (data/raw).

    Returns:
        Path to the newest matching file, or None.
    """
    matches = sorted(raw_dir.glob(f"{act_name}_*.xml"), reverse=True)
    return matches[0] if matches else None


def _model_is_available() -> bool:
    """Return True if the HuggingFace model cache has been downloaded locally."""
    hub_dir = MODELS_DIR / "hub"
    return hub_dir.exists() and any(hub_dir.iterdir())


def _run_scorer(preprocessed: dict[str, Any]) -> dict[str, Any]:
    """
    Load the trained scorer and run inference on a preprocessed diff.

    Separated into its own function so it can be patched in tests.

    Args:
        preprocessed: Output of preprocessor.preprocess_diff().

    Returns:
        Scored result dict (see scorer.Scorer.predict_diff).

    Raises:
        RuntimeError: If no trained model is available.
    """
    if not _model_is_available():
        raise RuntimeError(
            "HuggingFace model cache not found. "
            "The model will be downloaded automatically on the first /analyse request."
        )

    # Import lazily to keep startup fast when model is not needed
    from src.scorer import Scorer  # noqa: PLC0415

    scorer = Scorer()
    return scorer.predict_diff(preprocessed)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="UK Employment Law Change Detector",
    description=(
        "Monitors legislation.gov.uk for material changes to UK employment law "
        "and surfaces structured compliance alerts for SMEs."
    ),
    version="0.3.0",
)


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health() -> HealthResponse:
    """
    Health check endpoint.

    Returns service status and whether a trained model is available.
    """
    return HealthResponse(
        status="ok",
        service="uk-employment-law-change-detector",
        model_available=_model_is_available(),
    )


@app.post(
    "/analyse",
    response_model=AnalyseResponse,
    status_code=status.HTTP_200_OK,
    tags=["Pipeline"],
)
def analyse(request: AnalyseRequest) -> AnalyseResponse:
    """
    Run the full legislation monitoring pipeline for a given URL.

    **Steps:**

    1. Identify the act from the URL.
    2. Fetch the current XML from legislation.gov.uk.
    3. Compare against the cached baseline in ``data/raw/``.
    4. Diff → preprocess → classify → return structured changes.

    If no baseline exists the current XML is saved and an empty result
    with ``pipeline_status="baseline_established"`` is returned.

    **Request body:**
    ```json
    { "url": "https://www.legislation.gov.uk/ukpga/1996/18", "min_confidence": 0.0 }
    ```
    """
    url = request.url

    # 1. Identify act
    act_name = url_to_act_name(url)
    if act_name is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"URL '{url}' does not match any monitored legislation. "
                "Supported acts: Employment Rights Act 1996, National Minimum Wage Act 1998, "
                "Equality Act 2010, Working Time Regulations 1998."
            ),
        )

    # 2. Fetch current XML
    try:
        session = build_session()
        # Extract path component: everything after legislation.gov.uk/
        path_start = url.find("legislation.gov.uk/") + len("legislation.gov.uk/")
        act_path = url[path_start:].lstrip("/")
        current_xml = fetch_legislation_xml(session, act_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to fetch XML for %s: %s", act_name, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to fetch legislation XML: {exc}",
        ) from exc

    # 3. Look for baseline
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    baseline_path = _find_latest_cached_xml(act_name, DATA_RAW_DIR)

    if baseline_path is None:
        # No baseline — save current as baseline and return empty
        save_xml(act_name, current_xml)
        logger.info("Baseline established for %s", act_name)
        return AnalyseResponse(
            act_name=act_name,
            pipeline_status="baseline_established",
            processed_at=__import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
            total_material_changes=0,
            changes=[],
        )

    # 4. Diff
    diff = compute_diff(baseline_path, current_xml, act_name=act_name)

    # Save current version as new baseline
    save_xml(act_name, current_xml)

    total_sections_changed = sum(diff["summary"].values())
    if total_sections_changed == 0:
        return AnalyseResponse(
            act_name=act_name,
            pipeline_status="no_changes",
            processed_at=__import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
            total_material_changes=0,
            changes=[],
        )

    # 5. Preprocess
    preprocessed = preprocess_diff(diff)

    # 6. Score (raises RuntimeError if model not trained)
    try:
        scored = _run_scorer(preprocessed)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    # 7. Build output
    output = build_output(scored, min_confidence=request.min_confidence)

    status_str = "changes_found" if output["total_material_changes"] > 0 else "no_changes"

    return AnalyseResponse(
        act_name=act_name,
        pipeline_status=status_str,
        processed_at=output["processed_at"],
        total_material_changes=output["total_material_changes"],
        changes=output["changes"],
    )
