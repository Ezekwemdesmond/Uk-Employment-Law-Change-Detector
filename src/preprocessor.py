"""
Text preprocessor for UK legislation diff text.

Cleans raw or XML-tagged legislation text to produce normalised plain text
that is ready for BERT tokenisation.  The pipeline is:

  1. Strip XML/HTML tags (BeautifulSoup + lxml parser)
  2. Remove CLML editorial markers (e.g. [F1], [E+W])
  3. Expand common UK legal abbreviations
  4. Normalise whitespace
  5. Optionally truncate to BERT's effective token limit
"""

import logging
import re
from typing import Any

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Legal abbreviation expansion map
# Each key is a regex pattern; each value is a substitution string.
# ---------------------------------------------------------------------------
LEGAL_ABBREVIATIONS: dict[str, str] = {
    # Section references — "s. 12", "ss. 3–5"
    r"\bss?\.\s*(\d+)": r"Section \1",
    # Article references
    r"\bArt\.\s*(\d+)": r"Article \1",
    # Schedule references
    r"\bSch\.\s*(\d+)": r"Schedule \1",
    # Paragraph references
    r"\bPara\.\s*(\d+)": r"Paragraph \1",
    # Subsection references — "subs. (3)"
    r"\bsubss?\.\s*\((\w+)\)": r"Subsection (\1)",
    # Regulations / Regulation
    r"\bRegs\b": "Regulations",
    r"\bReg\b": "Regulation",
    # Statutory Instrument
    r"\bSI\b": "Statutory Instrument",
    # Latin / shorthand
    r"\bIbid\b": "in the same place",
    r"\betc\.": "and so on",
    r"\bviz\.": "namely",
    r"\be\.g\.": "for example",
    r"\bi\.e\.": "that is",
    # UK-specific
    r"\bHMRC\b": "His Majesty's Revenue and Customs",
    r"\bEAT\b": "Employment Appeal Tribunal",
    r"\bEqA\b": "Equality Act",
    r"\bERA\b": "Employment Rights Act",
    r"\bNMWA\b": "National Minimum Wage Act",
    r"\bWTR\b": "Working Time Regulations",
}

# Conservative character budget to stay safely under BERT's 512-token limit.
# Average English word ≈ 5 chars; BERT subword pieces average ~4 chars each.
BERT_MAX_CHARS: int = 1_800


def strip_xml_tags(text: str) -> str:
    """
    Remove all XML/HTML markup and return plain text.

    Uses BeautifulSoup with the ``lxml`` parser for robust handling of
    malformed or deeply nested CLML markup.

    Args:
        text: Raw string, potentially containing XML or HTML tags.

    Returns:
        Plain text with all tags removed.  Tag boundaries are replaced
        with a space so adjoining words are not accidentally merged.
    """
    soup = BeautifulSoup(text, "lxml")
    return soup.get_text(separator=" ")


def remove_citation_markers(text: str) -> str:
    """
    Remove CLML editorial and commencement markers that add noise for NLP.

    Examples of markers stripped:
      - ``[F1]``, ``[F12]``  — textual amendment markers
      - ``[E+W]``, ``[E+W+S+NI]`` — extent markers
      - ``[X1]`` — cross-heading markers
      - ``[S. 1 in force ...]`` — commencement notes

    Args:
        text: Input text.

    Returns:
        Text with CLML markers removed.
    """
    # Editorial markers: [F1], [E+W+S], [X1], etc.
    text = re.sub(r"\[[A-Z][A-Z0-9+\s]*\]", "", text)
    # Inline commencement notes: [S. 3 in force at 1.1.2000 ...]
    text = re.sub(r"\[S\.\s+\d+[^\]]{0,120}\]", "", text)
    return text


def expand_abbreviations(text: str) -> str:
    """
    Expand common UK legal abbreviations for improved tokenisation.

    Applies each pattern in :data:`LEGAL_ABBREVIATIONS` in order.

    Args:
        text: Input text.

    Returns:
        Text with recognised abbreviations replaced by their full forms.
    """
    for pattern, replacement in LEGAL_ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)
    return text


def normalise_whitespace(text: str) -> str:
    """
    Collapse any sequence of whitespace characters to a single space.

    Covers spaces, tabs, carriage returns, newlines, form-feeds, and
    vertical tabs.

    Args:
        text: Input text.

    Returns:
        Text with normalised whitespace, stripped of leading/trailing space.
    """
    text = re.sub(r"[\r\n\t\f\v]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def truncate_for_bert(text: str, max_chars: int = BERT_MAX_CHARS) -> str:
    """
    Truncate *text* to fit within BERT's effective token limit.

    Where possible, truncation occurs at the last sentence boundary
    (``". "``) that falls within the first 50 %–100 % of the budget,
    to avoid cutting mid-sentence.

    Args:
        text: Preprocessed plain text.
        max_chars: Maximum character count.  Defaults to :data:`BERT_MAX_CHARS`.

    Returns:
        Truncated text (unchanged if already within budget).
    """
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    last_period = truncated.rfind(". ")
    if last_period >= max_chars // 2:
        return truncated[: last_period + 1]
    return truncated


def preprocess_text(raw_text: str, truncate: bool = True) -> str:
    """
    Run the full preprocessing pipeline on a single text block.

    Pipeline stages (in order):

    1. :func:`strip_xml_tags`
    2. :func:`remove_citation_markers`
    3. :func:`expand_abbreviations`
    4. :func:`normalise_whitespace`
    5. :func:`truncate_for_bert` (if *truncate* is ``True``)

    Args:
        raw_text: Raw text, possibly containing XML/HTML markup.
        truncate: Whether to truncate to :data:`BERT_MAX_CHARS`.
                  Set to ``False`` when the full text is needed (e.g. for
                  display or manual review).

    Returns:
        Clean, tokenisation-ready plain text string.
    """
    text = strip_xml_tags(raw_text)
    text = remove_citation_markers(text)
    text = expand_abbreviations(text)
    text = normalise_whitespace(text)
    if truncate:
        text = truncate_for_bert(text)
    return text


def preprocess_diff(diff: dict[str, Any], truncate: bool = True) -> dict[str, Any]:
    """
    Apply :func:`preprocess_text` to every text field in a diff object.

    For ``modified`` sections an extra ``combined`` field is added that
    concatenates old and new text (useful as a single classifier input).

    Args:
        diff: Structured diff dict produced by ``diff_engine.compute_diff()``.
        truncate: Passed through to :func:`preprocess_text`.

    Returns:
        New dict with the same structure but all text fields preprocessed.
        Metadata fields (``act_name``, ``timestamp``, ``summary``) are
        preserved unchanged.
    """
    processed: dict[str, Any] = {
        "act_name": diff.get("act_name", ""),
        "timestamp": diff.get("timestamp", ""),
        "summary": diff.get("summary", {}),
        "added": {
            k: preprocess_text(v, truncate)
            for k, v in diff.get("added", {}).items()
        },
        "removed": {
            k: preprocess_text(v, truncate)
            for k, v in diff.get("removed", {}).items()
        },
        "modified": {
            k: {
                "old": preprocess_text(v["old"], truncate),
                "new": preprocess_text(v["new"], truncate),
                # [SEP] is inserted after preprocessing so it isn't stripped
                # by the citation-marker cleaner.
                "combined": preprocess_text(v["old"], truncate)
                + " [SEP] "
                + preprocess_text(v["new"], truncate),
            }
            for k, v in diff.get("modified", {}).items()
        },
    }

    logger.info(
        "Preprocessed diff for '%s': %d added, %d removed, %d modified",
        processed["act_name"],
        len(processed["added"]),
        len(processed["removed"]),
        len(processed["modified"]),
    )
    return processed


if __name__ == "__main__":
    _sample = (
        "<Section><Number>1</Number>"
        "<P>[F1] The employer shall <em>not</em> dismiss the employee "
        "without Regs. 5 notice, i.e. written notice. [E+W+S+NI]</P></Section>"
    )
    print(preprocess_text(_sample))
