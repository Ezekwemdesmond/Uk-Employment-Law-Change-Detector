"""
Tests for src/diff_engine.py
"""

import json
from pathlib import Path

import pytest
from lxml import etree

from src.diff_engine import (
    SECTION_TAGS,
    compute_diff,
    extract_sections,
    parse_xml,
    save_diff,
)

# ---------------------------------------------------------------------------
# Minimal CLML-like XML fixtures
# ---------------------------------------------------------------------------

SIMPLE_XML = """\
<Legislation>
  <Body>
    <Section id="s1">
      <Number>1</Number>
      <P>An employer shall give notice.</P>
    </Section>
    <Section id="s2">
      <Number>2</Number>
      <P>The notice period is one week.</P>
    </Section>
  </Body>
</Legislation>
"""

MODIFIED_XML = """\
<Legislation>
  <Body>
    <Section id="s1">
      <Number>1</Number>
      <P>An employer shall give written notice.</P>
    </Section>
    <Section id="s3">
      <Number>3</Number>
      <P>A new section about redundancy.</P>
    </Section>
  </Body>
</Legislation>
"""

IDENTICAL_XML = SIMPLE_XML  # same as SIMPLE_XML for no-diff tests


# ---------------------------------------------------------------------------
# parse_xml
# ---------------------------------------------------------------------------

class TestParseXml:
    def test_parse_string(self):
        root = parse_xml("<Root><Child/></Root>")
        assert root.tag == "Root"

    def test_parse_bytes(self):
        root = parse_xml(b"<Root/>")
        assert root.tag == "Root"

    def test_parse_path(self, tmp_path):
        xml_file = tmp_path / "test.xml"
        xml_file.write_text("<Root><Item id='1'/></Root>", encoding="utf-8")
        root = parse_xml(xml_file)
        assert root.tag == "Root"

    def test_invalid_xml_raises(self):
        with pytest.raises(etree.XMLSyntaxError):
            parse_xml("<Unclosed>")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(OSError):
            parse_xml(tmp_path / "nonexistent.xml")


# ---------------------------------------------------------------------------
# extract_sections
# ---------------------------------------------------------------------------

class TestExtractSections:
    def test_extracts_two_sections(self):
        root = parse_xml(SIMPLE_XML)
        sections = extract_sections(root)
        assert len(sections) == 2

    def test_section_keys_are_ids(self):
        root = parse_xml(SIMPLE_XML)
        sections = extract_sections(root)
        assert "s1" in sections
        assert "s2" in sections

    def test_section_text_content(self):
        root = parse_xml(SIMPLE_XML)
        sections = extract_sections(root)
        assert "notice" in sections["s1"]

    def test_empty_document_returns_empty_dict(self):
        root = parse_xml("<Legislation/>")
        sections = extract_sections(root)
        assert sections == {}

    def test_namespaced_section_extracted(self):
        """Sections under a CLML namespace should still be found."""
        ns_xml = """\
<leg:Legislation xmlns:leg="http://www.legislation.gov.uk/namespaces/legislation">
  <leg:Body>
    <leg:Section id="s1">
      <leg:Number>1</leg:Number>
      <leg:P>Namespaced section content.</leg:P>
    </leg:Section>
  </leg:Body>
</leg:Legislation>
"""
        root = parse_xml(ns_xml)
        sections = extract_sections(root)
        assert "s1" in sections
        assert "Namespaced section content" in sections["s1"]

    def test_section_without_id_skipped(self):
        xml = """\
<Legislation>
  <Section>
    <P>No id attribute and no Number child.</P>
  </Section>
</Legislation>
"""
        root = parse_xml(xml)
        sections = extract_sections(root)
        assert sections == {}

    def test_all_section_tag_types_recognised(self):
        """All tags in SECTION_TAGS should be extracted when present."""
        xml_parts = [
            f"<{tag} id='{tag.lower()}1'><Number>1</Number><P>Content.</P></{tag}>"
            for tag in SECTION_TAGS
        ]
        xml = "<Legislation>" + "".join(xml_parts) + "</Legislation>"
        root = parse_xml(xml)
        sections = extract_sections(root)
        assert len(sections) == len(SECTION_TAGS)


# ---------------------------------------------------------------------------
# compute_diff
# ---------------------------------------------------------------------------

class TestComputeDiff:
    def test_identical_docs_produce_empty_diff(self):
        diff = compute_diff(IDENTICAL_XML, IDENTICAL_XML, "test_act")
        assert diff["summary"]["added_count"] == 0
        assert diff["summary"]["removed_count"] == 0
        assert diff["summary"]["modified_count"] == 0

    def test_added_section_detected(self):
        # MODIFIED_XML has s3 which SIMPLE_XML does not
        diff = compute_diff(SIMPLE_XML, MODIFIED_XML, "test_act")
        assert "s3" in diff["added"]

    def test_removed_section_detected(self):
        # SIMPLE_XML has s2 which MODIFIED_XML does not
        diff = compute_diff(SIMPLE_XML, MODIFIED_XML, "test_act")
        assert "s2" in diff["removed"]

    def test_modified_section_detected(self):
        # s1 text changed between SIMPLE_XML and MODIFIED_XML
        diff = compute_diff(SIMPLE_XML, MODIFIED_XML, "test_act")
        assert "s1" in diff["modified"]
        assert "old" in diff["modified"]["s1"]
        assert "new" in diff["modified"]["s1"]

    def test_modified_old_new_differ(self):
        diff = compute_diff(SIMPLE_XML, MODIFIED_XML, "test_act")
        s1 = diff["modified"]["s1"]
        assert s1["old"] != s1["new"]

    def test_act_name_in_output(self):
        diff = compute_diff(SIMPLE_XML, MODIFIED_XML, "employment_rights_act_1996")
        assert diff["act_name"] == "employment_rights_act_1996"

    def test_timestamp_present(self):
        diff = compute_diff(SIMPLE_XML, MODIFIED_XML, "test_act")
        ts = diff["timestamp"]
        # Accept both "+00:00" (Python 3.11+) and legacy "Z" suffixes
        assert ts.endswith("+00:00") or ts.endswith("Z")

    def test_summary_counts_accurate(self):
        diff = compute_diff(SIMPLE_XML, MODIFIED_XML, "test_act")
        assert diff["summary"]["added_count"] == len(diff["added"])
        assert diff["summary"]["removed_count"] == len(diff["removed"])
        assert diff["summary"]["modified_count"] == len(diff["modified"])

    def test_accepts_path_inputs(self, tmp_path):
        old_file = tmp_path / "old.xml"
        new_file = tmp_path / "new.xml"
        old_file.write_text(SIMPLE_XML, encoding="utf-8")
        new_file.write_text(MODIFIED_XML, encoding="utf-8")
        diff = compute_diff(old_file, new_file, "test_act")
        assert "summary" in diff


# ---------------------------------------------------------------------------
# save_diff
# ---------------------------------------------------------------------------

class TestSaveDiff:
    def _make_diff(self) -> dict:
        return compute_diff(SIMPLE_XML, MODIFIED_XML, "test_act")

    def test_creates_json_file(self, tmp_path):
        diff = self._make_diff()
        path = save_diff(diff, "test_act", output_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".json"

    def test_filename_starts_with_act_name(self, tmp_path):
        diff = self._make_diff()
        path = save_diff(diff, "employment_rights_act_1996", output_dir=tmp_path)
        assert path.name.startswith("employment_rights_act_1996_")

    def test_json_is_valid_and_contains_summary(self, tmp_path):
        diff = self._make_diff()
        path = save_diff(diff, "test_act", output_dir=tmp_path)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert "summary" in loaded
        assert "added_count" in loaded["summary"]

    def test_creates_output_dir_if_missing(self, tmp_path):
        nested = tmp_path / "new_diffs_dir"
        diff = self._make_diff()
        save_diff(diff, "test_act", output_dir=nested)
        assert nested.exists()
