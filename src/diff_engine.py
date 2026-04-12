"""
XML diff engine for UK legislation documents.

Compares two versions of a CLML (Crown Legislation Markup Language) XML
document, identifies sections that were added, removed, or modified, and
writes a structured diff to data/diffs as JSON.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lxml import etree

logger = logging.getLogger(__name__)

DATA_DIFFS_DIR = Path(__file__).parent.parent / "data" / "diffs"

# Tags treated as discrete "sections" for diffing purposes.
# Covers both the older CLML P1group style and the newer Section/Article style.
SECTION_TAGS: frozenset[str] = frozenset(
    {
        "Section",
        "P1group",
        "Part",
        "Chapter",
        "Schedule",
        "Article",
        "Regulation",
        "Rule",
    }
)


def parse_xml(xml_source: str | Path | bytes) -> etree._Element:
    """
    Parse an XML document from a string, bytes, or file path.

    Args:
        xml_source: One of:
            - A raw XML string.
            - A ``bytes`` object containing XML.
            - A :class:`pathlib.Path` pointing to an XML file.

    Returns:
        Root :class:`lxml.etree._Element` of the parsed document.

    Raises:
        lxml.etree.XMLSyntaxError: If the source is not valid XML.
        FileNotFoundError: If a Path is given but the file does not exist.
    """
    if isinstance(xml_source, Path):
        return etree.parse(str(xml_source)).getroot()
    if isinstance(xml_source, str):
        xml_source = xml_source.encode("utf-8")
    return etree.fromstring(xml_source)


def _local_tag(element: etree._Element) -> str:
    """Return the local (non-namespaced) tag name of *element*."""
    tag = element.tag
    if tag and "{" in tag:
        return tag.split("}")[1]
    return tag


def _element_id(element: etree._Element) -> str | None:
    """
    Derive a stable section identifier from element attributes or child text.

    Checks (in order):
      1. ``id`` attribute
      2. ``IdURI`` attribute
      3. ``DocumentURI`` attribute
      4. Text of a direct ``<Number>`` child element

    Args:
        element: An lxml element representing a legislation section.

    Returns:
        Identifier string, or ``None`` if none could be found.
    """
    for attr in ("id", "IdURI", "DocumentURI"):
        val = element.get(attr)
        if val:
            return val.strip()
    for child in element:
        if _local_tag(child) == "Number" and child.text:
            return child.text.strip()
    return None


def _element_text(element: etree._Element) -> str:
    """
    Concatenate all descendant text nodes of *element* into a single string.

    Args:
        element: Any lxml element.

    Returns:
        Whitespace-normalised text content.
    """
    return " ".join(element.itertext()).strip()


def extract_sections(root: etree._Element) -> dict[str, str]:
    """
    Walk the element tree and collect section-level text blocks.

    Only elements whose local tag is in :data:`SECTION_TAGS` and that have a
    derivable identifier (via :func:`_element_id`) are included.

    Args:
        root: Root element of a parsed legislation XML document.

    Returns:
        Dict mapping ``section_id -> full_text_content``.
    """
    sections: dict[str, str] = {}

    for element in root.iter():
        local = _local_tag(element)
        if local in SECTION_TAGS:
            section_id = _element_id(element)
            if section_id:
                text = _element_text(element)
                if text:
                    # Prefer the most granular (deepest) match for duplicate ids.
                    sections[section_id] = text

    return sections


def compute_diff(
    old_xml: str | Path | bytes,
    new_xml: str | Path | bytes,
    act_name: str = "unknown",
) -> dict[str, Any]:
    """
    Compare two versions of a legislation XML document.

    Parses both documents, extracts sections via :func:`extract_sections`,
    then classifies each section as added, removed, or modified.

    Args:
        old_xml: Older document version — string, bytes, or Path.
        new_xml: Newer document version — string, bytes, or Path.
        act_name: Human-readable legislation identifier used in the output.

    Returns:
        Structured diff dict with the following keys:

        - ``act_name`` (str)
        - ``timestamp`` (ISO-8601 UTC string)
        - ``summary`` (dict with ``added_count``, ``removed_count``, ``modified_count``)
        - ``added`` (dict mapping section_id -> new text)
        - ``removed`` (dict mapping section_id -> old text)
        - ``modified`` (dict mapping section_id -> ``{"old": ..., "new": ...}``)
    """
    old_root = parse_xml(old_xml)
    new_root = parse_xml(new_xml)

    old_sections = extract_sections(old_root)
    new_sections = extract_sections(new_root)

    old_keys = set(old_sections)
    new_keys = set(new_sections)

    added = {k: new_sections[k] for k in new_keys - old_keys}
    removed = {k: old_sections[k] for k in old_keys - new_keys}
    modified = {
        k: {"old": old_sections[k], "new": new_sections[k]}
        for k in old_keys & new_keys
        if old_sections[k] != new_sections[k]
    }

    diff: dict[str, Any] = {
        "act_name": act_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "added_count": len(added),
            "removed_count": len(removed),
            "modified_count": len(modified),
        },
        "added": added,
        "removed": removed,
        "modified": modified,
    }

    logger.info(
        "Diff for '%s': +%d added, -%d removed, ~%d modified",
        act_name,
        len(added),
        len(removed),
        len(modified),
    )
    return diff


def save_diff(
    diff: dict[str, Any],
    act_name: str,
    output_dir: Path | None = None,
) -> Path:
    """
    Serialise a diff dict to a timestamped JSON file.

    Filename format: ``<act_name>_<YYYYMMDDTHHMMSSz>.json``

    Args:
        diff: Structured diff produced by :func:`compute_diff`.
        act_name: Used in the output filename.
        output_dir: Override directory; defaults to :data:`DATA_DIFFS_DIR`.

    Returns:
        :class:`pathlib.Path` of the written JSON file.
    """
    target_dir = output_dir if output_dir is not None else DATA_DIFFS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{act_name}_{timestamp}.json"
    filepath = target_dir / filename
    filepath.write_text(json.dumps(diff, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved diff to %s", filepath)
    return filepath


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if len(sys.argv) != 4:
        print("Usage: diff_engine.py <old_xml_path> <new_xml_path> <act_name>")
        sys.exit(1)

    result = compute_diff(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
    out = save_diff(result, sys.argv[3])
    print(f"Diff saved to: {out}")
    print(json.dumps(result["summary"], indent=2))
