"""
Domain classifier for UK employment legislation.

Maps act names and legislation.gov.uk URLs to one of four compliance domains:
  - "employment"   — Employment Rights Act 1996
  - "wage"         — National Minimum Wage Act 1998
  - "equality"     — Equality Act 2010
  - "working_time" — Working Time Regulations 1998
"""

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical act-name → domain mapping
# ---------------------------------------------------------------------------

ACT_TO_DOMAIN: dict[str, str] = {
    "employment_rights_act_1996": "employment",
    "national_minimum_wage_act_1998": "wage",
    "equality_act_2010": "equality",
    "working_time_regulations_1998": "working_time",
}

# legislation.gov.uk path fragments that uniquely identify each act
_URL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"ukpga/1996/18"), "employment"),
    (re.compile(r"ukpga/1998/39"), "wage"),
    (re.compile(r"ukpga/2010/15"), "equality"),
    (re.compile(r"uksi/1998/1833"), "working_time"),
]

# Keywords in free text that suggest a domain
_TEXT_KEYWORDS: list[tuple[list[str], str]] = [
    (["national minimum wage", "minimum wage", "nmwa", "nmw"], "wage"),
    (["equality act", "discrimination", "protected characteristic", "reasonable adjustment", "eqa"], "equality"),
    (["working time", "rest break", "working hours", "night worker", "wtr"], "working_time"),
    (["employment rights", "unfair dismissal", "redundancy", "notice period", "era"], "employment"),
]

FALLBACK_DOMAIN = "employment"


def classify_domain(act_name: str) -> str:
    """
    Map a canonical act name to its compliance domain.

    Args:
        act_name: Snake-case act identifier (e.g. ``'equality_act_2010'``).

    Returns:
        Domain string, falling back to ``'employment'`` for unknown acts.
    """
    domain = ACT_TO_DOMAIN.get(act_name.lower().strip())
    if domain is None:
        logger.warning("Unknown act name '%s', defaulting to '%s'", act_name, FALLBACK_DOMAIN)
        return FALLBACK_DOMAIN
    return domain


def classify_domain_from_url(url: str) -> str:
    """
    Infer a compliance domain from a legislation.gov.uk URL.

    Matches against known path patterns such as ``ukpga/1996/18``.

    Args:
        url: Full or partial legislation.gov.uk URL.

    Returns:
        Domain string, falling back to ``'employment'`` if no pattern matches.
    """
    for pattern, domain in _URL_PATTERNS:
        if pattern.search(url):
            return domain
    logger.warning("Could not infer domain from URL '%s', defaulting to '%s'", url, FALLBACK_DOMAIN)
    return FALLBACK_DOMAIN


def classify_domain_from_text(text: str) -> str:
    """
    Infer a compliance domain from free text using keyword matching.

    The first keyword group to match wins; matching is case-insensitive.

    Args:
        text: Preprocessed legislation change text.

    Returns:
        Domain string, falling back to ``'employment'`` if no keywords match.
    """
    lower = text.lower()
    for keywords, domain in _TEXT_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return domain
    return FALLBACK_DOMAIN


def url_to_act_name(url: str) -> str | None:
    """
    Reverse-map a legislation.gov.uk URL to a canonical act name.

    Args:
        url: Full or partial legislation.gov.uk URL.

    Returns:
        Canonical act name string, or ``None`` if unrecognised.
    """
    _URL_TO_ACT = {
        r"ukpga/1996/18": "employment_rights_act_1996",
        r"ukpga/1998/39": "national_minimum_wage_act_1998",
        r"ukpga/2010/15": "equality_act_2010",
        r"uksi/1998/1833": "working_time_regulations_1998",
    }
    for fragment, act_name in _URL_TO_ACT.items():
        if fragment in url:
            return act_name
    return None
