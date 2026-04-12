"""
Tests for src/preprocessor.py
"""

import pytest

from src.preprocessor import (
    BERT_MAX_CHARS,
    expand_abbreviations,
    normalise_whitespace,
    preprocess_diff,
    preprocess_text,
    remove_citation_markers,
    strip_xml_tags,
    truncate_for_bert,
)


# ---------------------------------------------------------------------------
# strip_xml_tags
# ---------------------------------------------------------------------------

class TestStripXmlTags:
    def test_removes_simple_tags(self):
        result = strip_xml_tags("<P>Hello world</P>")
        assert "<P>" not in result
        assert "Hello world" in result

    def test_handles_nested_tags(self):
        result = strip_xml_tags("<Section><Number>1</Number><P>Text here.</P></Section>")
        assert "Text here." in result
        assert "<" not in result

    def test_plain_text_unchanged(self):
        text = "Plain text with no markup."
        result = strip_xml_tags(text)
        assert "Plain text" in result

    def test_empty_string(self):
        result = strip_xml_tags("")
        assert result == "" or result.strip() == ""

    def test_tag_boundary_whitespace(self):
        """Adjacent tags should not produce merged words."""
        result = strip_xml_tags("<A>Hello</A><B>World</B>")
        assert "Hello" in result
        assert "World" in result
        # They should not be merged into "HelloWorld"
        assert "HelloWorld" not in result


# ---------------------------------------------------------------------------
# remove_citation_markers
# ---------------------------------------------------------------------------

class TestRemoveCitationMarkers:
    def test_removes_f_markers(self):
        result = remove_citation_markers("The employer [F1] shall give notice.")
        assert "[F1]" not in result
        assert "employer" in result

    def test_removes_extent_markers(self):
        result = remove_citation_markers("Section 1 [E+W+S+NI] applies.")
        assert "[E+W+S+NI]" not in result

    def test_removes_x_markers(self):
        result = remove_citation_markers("[X1] Cross-heading here.")
        assert "[X1]" not in result

    def test_removes_commencement_notes(self):
        text = "Provision [S. 3 in force at 1.1.2000 by SI 2000/1] is active."
        result = remove_citation_markers(text)
        assert "[S." not in result
        assert "Provision" in result

    def test_no_false_positives_on_normal_brackets(self):
        text = "Subsection (3) applies to employers."
        result = remove_citation_markers(text)
        assert "(3)" in result


# ---------------------------------------------------------------------------
# expand_abbreviations
# ---------------------------------------------------------------------------

class TestExpandAbbreviations:
    def test_expands_reg(self):
        result = expand_abbreviations("Under Reg 5 of the instrument.")
        assert "Regulation" in result

    def test_expands_regs(self):
        result = expand_abbreviations("The Regs apply here.")
        assert "Regulations" in result

    def test_expands_section_reference(self):
        result = expand_abbreviations("See s. 12 of the Act.")
        assert "Section 12" in result

    def test_expands_ie(self):
        result = expand_abbreviations("The notice, i.e. written notice.")
        assert "that is" in result

    def test_expands_eg(self):
        result = expand_abbreviations("Benefits, e.g. holiday pay.")
        assert "for example" in result

    def test_no_expansion_for_unmatched_text(self):
        text = "No abbreviations in this sentence."
        result = expand_abbreviations(text)
        assert result == text


# ---------------------------------------------------------------------------
# normalise_whitespace
# ---------------------------------------------------------------------------

class TestNormaliseWhitespace:
    def test_collapses_multiple_spaces(self):
        result = normalise_whitespace("Hello   world")
        assert result == "Hello world"

    def test_replaces_newlines_with_space(self):
        result = normalise_whitespace("Hello\nworld")
        assert result == "Hello world"

    def test_replaces_tabs(self):
        result = normalise_whitespace("Hello\tworld")
        assert result == "Hello world"

    def test_strips_leading_trailing(self):
        result = normalise_whitespace("  Hello world  ")
        assert result == "Hello world"

    def test_mixed_whitespace(self):
        result = normalise_whitespace("  Hello\n\n\tworld  ")
        assert result == "Hello world"


# ---------------------------------------------------------------------------
# truncate_for_bert
# ---------------------------------------------------------------------------

class TestTruncateForBert:
    def test_short_text_unchanged(self):
        text = "Short text."
        result = truncate_for_bert(text, max_chars=100)
        assert result == text

    def test_long_text_truncated(self):
        text = "word " * 500  # definitely exceeds BERT_MAX_CHARS
        result = truncate_for_bert(text)
        assert len(result) <= BERT_MAX_CHARS

    def test_truncates_at_sentence_boundary(self):
        # Build a text where a ". " boundary sits well past the 50% mark of
        # the budget so the sentence-boundary branch is taken.
        max_chars = 100
        # Pad before the sentence so the period lands at ~80% of max_chars
        prefix = "x" * 78
        text = prefix + ". Trailing words that push us over the limit aaaaaaa"
        result = truncate_for_bert(text, max_chars=max_chars)
        # Should end at the sentence boundary (the period at position 78)
        assert result.endswith(".")

    def test_custom_max_chars(self):
        text = "a" * 200
        result = truncate_for_bert(text, max_chars=100)
        assert len(result) <= 100


# ---------------------------------------------------------------------------
# preprocess_text (integration pipeline)
# ---------------------------------------------------------------------------

class TestPreprocessText:
    def test_removes_xml_tags(self):
        result = preprocess_text("<Section><P>Hello.</P></Section>")
        assert "<" not in result

    def test_removes_citation_markers(self):
        result = preprocess_text("[F1] The employer shall comply.")
        assert "[F1]" not in result

    def test_expands_abbreviations(self):
        result = preprocess_text("Under Reg 5 of the instrument.")
        assert "Regulation" in result

    def test_normalises_whitespace(self):
        result = preprocess_text("<P>Hello\n\n\tworld</P>")
        assert "  " not in result

    def test_truncate_false_returns_full_text(self):
        long_text = "word " * 500
        result = preprocess_text(long_text, truncate=False)
        assert len(result) > BERT_MAX_CHARS

    def test_full_pipeline_realistic_input(self):
        xml_fragment = (
            "<Section id='s1'><Number>1</Number>"
            "<P>[F1] The employer shall <em>not</em> dismiss the employee "
            "without Regs. 5 notice, i.e. written notice. [E+W+S+NI]</P>"
            "</Section>"
        )
        result = preprocess_text(xml_fragment)
        assert "<" not in result
        assert "[F1]" not in result
        assert "[E+W+S+NI]" not in result
        assert "that is" in result
        assert "Regulation" in result


# ---------------------------------------------------------------------------
# preprocess_diff
# ---------------------------------------------------------------------------

class TestPreprocessDiff:
    def _make_diff(self) -> dict:
        return {
            "act_name": "test_act",
            "timestamp": "2024-01-01T00:00:00Z",
            "summary": {"added_count": 1, "removed_count": 1, "modified_count": 1},
            "added": {"s3": "<P>[F1] New section text.</P>"},
            "removed": {"s2": "<P>Old removed text.</P>"},
            "modified": {
                "s1": {
                    "old": "<P>Old s1 text with Regs. 5.</P>",
                    "new": "<P>[F2] New s1 text.</P>",
                }
            },
        }

    def test_preserves_metadata(self):
        diff = self._make_diff()
        result = preprocess_diff(diff)
        assert result["act_name"] == "test_act"
        assert result["timestamp"] == "2024-01-01T00:00:00Z"
        assert result["summary"]["added_count"] == 1

    def test_added_text_cleaned(self):
        diff = self._make_diff()
        result = preprocess_diff(diff)
        assert "<P>" not in result["added"]["s3"]
        assert "[F1]" not in result["added"]["s3"]

    def test_removed_text_cleaned(self):
        diff = self._make_diff()
        result = preprocess_diff(diff)
        assert "<P>" not in result["removed"]["s2"]

    def test_modified_has_old_new_combined(self):
        diff = self._make_diff()
        result = preprocess_diff(diff)
        s1 = result["modified"]["s1"]
        assert "old" in s1
        assert "new" in s1
        assert "combined" in s1

    def test_combined_contains_sep_token(self):
        diff = self._make_diff()
        result = preprocess_diff(diff)
        assert "[SEP]" in result["modified"]["s1"]["combined"]

    def test_empty_diff_handled_gracefully(self):
        empty = {
            "act_name": "empty_act",
            "timestamp": "2024-01-01T00:00:00Z",
            "summary": {},
            "added": {},
            "removed": {},
            "modified": {},
        }
        result = preprocess_diff(empty)
        assert result["added"] == {}
        assert result["removed"] == {}
        assert result["modified"] == {}
