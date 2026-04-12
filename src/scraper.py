"""
Legislation scraper for the legislation.gov.uk REST API.

Fetches XML documents for target UK employment acts and saves them
to data/raw with UTC timestamps in the filename.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

BASE_URL = "https://www.legislation.gov.uk"
DATA_RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

# Maps short act identifiers to their legislation.gov.uk path components.
ACTS: dict[str, str] = {
    "employment_rights_act_1996": "ukpga/1996/18",
    "national_minimum_wage_act_1998": "ukpga/1998/39",
    "equality_act_2010": "ukpga/2010/15",
    "working_time_regulations_1998": "uksi/1998/1833",
}

# Polite delay between requests (seconds) to avoid hammering the public API.
RATE_LIMIT_DELAY: float = 1.5


def build_session() -> requests.Session:
    """
    Build a requests Session with automatic retry logic.

    Retries up to 3 times with exponential back-off on server errors and
    rate-limit (429) responses.

    Returns:
        Configured requests.Session instance.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "Accept": "application/xml",
            "User-Agent": "complyAI-legislation-monitor/1.0 (portfolio project; contact via github)",
        }
    )
    return session


def fetch_legislation_xml(session: requests.Session, act_path: str) -> str:
    """
    Fetch the XML representation of a piece of legislation from legislation.gov.uk.

    Appends ``/data.xml`` to the act path to request the canonical XML feed.
    Handles a 429 response by honouring the ``Retry-After`` header before
    making one additional attempt.

    Args:
        session: Configured requests Session (from :func:`build_session`).
        act_path: Legislation path component, e.g. ``'ukpga/1996/18'``.

    Returns:
        Raw XML string.

    Raises:
        requests.HTTPError: If the request fails after retries.
        requests.ConnectionError: On network-level failures.
        requests.Timeout: If the server does not respond within 30 s.
    """
    url = f"{BASE_URL}/{act_path}/data.xml"
    logger.info("Fetching %s", url)

    response = session.get(url, timeout=30)

    if response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 60))
        logger.warning("Rate limited by server. Waiting %d s before retry.", retry_after)
        time.sleep(retry_after)
        response = session.get(url, timeout=30)

    response.raise_for_status()
    return response.text


def save_xml(act_name: str, xml_content: str, output_dir: Path | None = None) -> Path:
    """
    Persist raw XML to the data/raw directory with a UTC timestamp.

    Filename format: ``<act_name>_<YYYYMMDDTHHMMSSz>.xml``

    Args:
        act_name: Short snake_case identifier for the act.
        xml_content: Raw XML string to save.
        output_dir: Optional override for the output directory.
                    Defaults to :data:`DATA_RAW_DIR`.

    Returns:
        :class:`pathlib.Path` of the saved file.
    """
    target_dir = output_dir if output_dir is not None else DATA_RAW_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{act_name}_{timestamp}.xml"
    filepath = target_dir / filename
    filepath.write_text(xml_content, encoding="utf-8")

    size_kb = len(xml_content.encode("utf-8")) / 1024
    logger.info("Saved %s (%.1f KB)", filepath, size_kb)
    return filepath


def scrape_all(
    acts: dict[str, str] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Scrape all configured acts and save their raw XML.

    Iterates through *acts*, fetching each document and saving it to
    *output_dir*.  A :data:`RATE_LIMIT_DELAY` pause is observed between
    requests regardless of success or failure.

    Args:
        acts: Optional override mapping ``act_name -> act_path``.
              Defaults to :data:`ACTS`.
        output_dir: Optional output directory override.

    Returns:
        Dict mapping ``act_name -> Path`` for every successfully saved file.
        Acts that fail are logged and omitted from the result.
    """
    if acts is None:
        acts = ACTS

    session = build_session()
    results: dict[str, Path] = {}

    for act_name, act_path in acts.items():
        try:
            xml_content = fetch_legislation_xml(session, act_path)
            saved_path = save_xml(act_name, xml_content, output_dir)
            results[act_name] = saved_path
        except requests.RequestException as exc:
            logger.error("Failed to fetch '%s': %s", act_name, exc)

        time.sleep(RATE_LIMIT_DELAY)

    logger.info("Scrape complete. %d/%d acts saved.", len(results), len(acts))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    saved = scrape_all()
    for name, path in saved.items():
        print(f"  {name}: {path}")
