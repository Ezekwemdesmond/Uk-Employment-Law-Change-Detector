"""
Tests for src/output_builder.py
"""

import json
from pathlib import Path

import pytest

from src.output_builder import (
    VALID_OBLIGATION_TYPES,
    build_material_record,
    build_output,
    extract_affected_business_types,
    extract_effective_date,
    infer_obligation_type,
    save_output,
)


# ---------------------------------------------------------------------------
# extract_effective_date
# ---------------------------------------------------------------------------

class TestExtractEffectiveDate:
    def test_with_effect_from_phrase(self):
        text = "The provision takes effect with effect from 1 April 2024."
        result = extract_effective_date(text)
        assert result is not None
        assert "April 2024" in result

    def test_from_phrase(self):
        text = "From 1 January 2025, all employers must comply."
        result = extract_effective_date(text)
        assert result is not None
        assert "2025" in result

    def test_iso_date(self):
        text = "Regulations come into force on 2024-04-01."
        result = extract_effective_date(text)
        assert result is not None
        assert "2024" in result

    def test_plain_named_date(self):
        text = "The Act applies from 15 March 2023."
        result = extract_effective_date(text)
        assert result is not None
        assert "March 2023" in result

    def test_no_date_returns_none(self):
        result = extract_effective_date("The employer must provide written notice.")
        assert result is None

    def test_empty_string_returns_none(self):
        assert extract_effective_date("") is None

    def test_commencing_phrase(self):
        text = "Commencing 1 October 2024, the new rules apply."
        result = extract_effective_date(text)
        assert result is not None
        assert "2024" in result


# ---------------------------------------------------------------------------
# extract_affected_business_types
# ---------------------------------------------------------------------------

class TestExtractAffectedBusinessTypes:
    def test_returns_list(self):
        result = extract_affected_business_types("All employers must comply.")
        assert isinstance(result, list)

    def test_default_is_all_employers(self):
        result = extract_affected_business_types("The employer shall give notice.")
        assert result == ["all employers"]

    def test_sme_keyword(self):
        result = extract_affected_business_types("Small businesses are exempt.")
        # label returned is "SMEs" — compare lowercased on both sides
        assert any("sme" in r.lower() or "small" in r.lower() for r in result)

    def test_fewer_than_employees(self):
        result = extract_affected_business_types("Employers with fewer than 50 employees qualify.")
        assert len(result) > 0

    def test_large_employer_keyword(self):
        result = extract_affected_business_types("Large employers with more than 250 employees.")
        assert any("large" in r.lower() for r in result)

    def test_public_sector_keyword(self):
        result = extract_affected_business_types("Public authority employers must comply.")
        assert any("public" in r.lower() for r in result)

    def test_charity_keyword(self):
        result = extract_affected_business_types("Charities and voluntary sector bodies are affected.")
        assert any("charit" in r.lower() or "voluntar" in r.lower() for r in result)

    def test_gig_economy_keyword(self):
        result = extract_affected_business_types("Gig economy platforms are now covered.")
        assert any("gig" in r.lower() for r in result)

    def test_no_duplicates(self):
        # Multiple SME keywords should not produce duplicate entries
        result = extract_affected_business_types("SME or small business with fewer than 50 employees.")
        assert len(result) == len(set(result))

    def test_non_empty_always(self):
        result = extract_affected_business_types("")
        assert len(result) > 0


# ---------------------------------------------------------------------------
# infer_obligation_type
# ---------------------------------------------------------------------------

class TestInferObligationType:
    def test_added_section_is_new_duty(self):
        assert infer_obligation_type("added_section", "New duty imposed.") == "new_duty"

    def test_removed_section_is_removed_duty(self):
        assert infer_obligation_type("removed_section", "Old section removed.") == "removed_duty"

    def test_modified_section_default_is_changed_duty(self):
        result = infer_obligation_type("modified_section", "The text is slightly amended.")
        assert result == "changed_duty"

    def test_penalty_text_triggers_changed_penalty(self):
        text = "The maximum fine has been increased."
        result = infer_obligation_type("modified_section", text)
        assert result == "changed_penalty"

    def test_scope_text_triggers_extended_scope(self):
        text = "Protections are now extended to agency workers from day one."
        result = infer_obligation_type("modified_section", text)
        assert result == "extended_scope"

    def test_removed_duty_not_overridden_by_penalty_text(self):
        # Even if text mentions penalty, removed_duty stays removed_duty
        text = "The fine of £5,000 no longer applies."
        result = infer_obligation_type("removed_section", text)
        assert result == "removed_duty"

    def test_unknown_change_type_defaults_to_changed_duty(self):
        result = infer_obligation_type("some_unknown_type", "Generic text.")
        assert result == "changed_duty"

    def test_all_returned_types_are_valid(self):
        cases = [
            ("added_section", "New duty."),
            ("removed_section", "Removed."),
            ("modified_section", "Changed text."),
            ("modified_section", "The maximum fine increases."),
            ("modified_section", "Scope extended to gig economy workers."),
        ]
        for change_type, text in cases:
            result = infer_obligation_type(change_type, text)
            assert result in VALID_OBLIGATION_TYPES, f"{result} not in VALID_OBLIGATION_TYPES"


# ---------------------------------------------------------------------------
# build_material_record
# ---------------------------------------------------------------------------

class TestBuildMaterialRecord:
    def _prediction(self, **kwargs):
        defaults = {
            "text": "Employers must now provide written notice within 14 days.",
            "change_type": "added_section",
            "section_id": "s1",
            "predicted_label": 1,
            "confidence": 0.92,
            "needs_review": False,
        }
        return {**defaults, **kwargs}

    def test_returns_dict(self):
        record = build_material_record(self._prediction(), "employment_rights_act_1996")
        assert isinstance(record, dict)

    def test_has_all_five_required_fields(self):
        record = build_material_record(self._prediction(), "employment_rights_act_1996")
        assert "domain" in record
        assert "effective_date" in record
        assert "penalty_threshold" in record
        assert "affected_business_types" in record
        assert "obligation_type" in record

    def test_domain_correct_for_era(self):
        record = build_material_record(self._prediction(), "employment_rights_act_1996")
        assert record["domain"] == "employment"

    def test_domain_correct_for_equality(self):
        record = build_material_record(self._prediction(), "equality_act_2010")
        assert record["domain"] == "equality"

    def test_domain_correct_for_wage(self):
        record = build_material_record(self._prediction(), "national_minimum_wage_act_1998")
        assert record["domain"] == "wage"

    def test_domain_correct_for_working_time(self):
        record = build_material_record(self._prediction(), "working_time_regulations_1998")
        assert record["domain"] == "working_time"

    def test_obligation_type_valid(self):
        record = build_material_record(self._prediction(), "employment_rights_act_1996")
        assert record["obligation_type"] in VALID_OBLIGATION_TYPES

    def test_affected_business_types_is_list(self):
        record = build_material_record(self._prediction(), "employment_rights_act_1996")
        assert isinstance(record["affected_business_types"], list)

    def test_penalty_extracted_when_present(self):
        pred = self._prediction(
            text="The maximum fine has been increased from £5,000 to £20,000."
        )
        record = build_material_record(pred, "employment_rights_act_1996")
        assert record["penalty_threshold"] is not None
        assert "£20,000" in record["penalty_threshold"]

    def test_penalty_none_when_absent(self):
        pred = self._prediction(text="Employers must now give written notice.")
        record = build_material_record(pred, "employment_rights_act_1996")
        assert record["penalty_threshold"] is None

    def test_text_excerpt_truncated_at_200(self):
        long_text = "x" * 300
        pred = self._prediction(text=long_text)
        record = build_material_record(pred, "employment_rights_act_1996")
        assert len(record["text_excerpt"]) <= 204  # 200 + "…"

    def test_short_text_not_truncated(self):
        pred = self._prediction(text="Short text.")
        record = build_material_record(pred, "employment_rights_act_1996")
        assert "…" not in record["text_excerpt"]


# ---------------------------------------------------------------------------
# build_output
# ---------------------------------------------------------------------------

class TestBuildOutput:
    def _scored_result(self, predictions=None):
        if predictions is None:
            predictions = [
                {
                    "text": "New obligation for all employers.",
                    "change_type": "added_section",
                    "section_id": "s1",
                    "predicted_label": 1,
                    "confidence": 0.95,
                    "needs_review": False,
                    "act_name": "employment_rights_act_1996",
                },
                {
                    "text": "Cross-reference updated.",
                    "change_type": "cross_reference",
                    "section_id": "s2",
                    "predicted_label": 0,
                    "confidence": 0.88,
                    "needs_review": False,
                    "act_name": "employment_rights_act_1996",
                },
            ]
        return {
            "act_name": "employment_rights_act_1996",
            "timestamp": "2024-01-01T00:00:00Z",
            "predictions": predictions,
            "summary": {"total": len(predictions)},
        }

    def test_returns_dict(self):
        result = build_output(self._scored_result())
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = build_output(self._scored_result())
        assert "act_name" in result
        assert "processed_at" in result
        assert "total_material_changes" in result
        assert "changes" in result

    def test_filters_to_material_only(self):
        result = build_output(self._scored_result())
        # Only 1 of the 2 predictions is material (label=1)
        assert result["total_material_changes"] == 1

    def test_changes_is_list(self):
        result = build_output(self._scored_result())
        assert isinstance(result["changes"], list)

    def test_all_minor_predictions_excluded(self):
        minor_preds = [
            {
                "text": "Cross-reference fix.",
                "change_type": "cross_reference",
                "section_id": "s1",
                "predicted_label": 0,
                "confidence": 0.9,
                "needs_review": False,
            }
        ]
        result = build_output(self._scored_result(predictions=minor_preds))
        assert result["total_material_changes"] == 0
        assert result["changes"] == []

    def test_min_confidence_filter(self):
        preds = [
            {
                "text": "New obligation.",
                "change_type": "added_section",
                "section_id": "s1",
                "predicted_label": 1,
                "confidence": 0.60,
                "needs_review": True,
            },
            {
                "text": "Penalty increased.",
                "change_type": "modified_section",
                "section_id": "s2",
                "predicted_label": 1,
                "confidence": 0.95,
                "needs_review": False,
            },
        ]
        result = build_output(self._scored_result(predictions=preds), min_confidence=0.75)
        assert result["total_material_changes"] == 1  # Only the 0.95 confidence one

    def test_act_name_preserved(self):
        result = build_output(self._scored_result())
        assert result["act_name"] == "employment_rights_act_1996"

    def test_processed_at_is_iso_timestamp(self):
        result = build_output(self._scored_result())
        assert "T" in result["processed_at"]


# ---------------------------------------------------------------------------
# save_output
# ---------------------------------------------------------------------------

class TestSaveOutput:
    def _make_output(self):
        return build_output({
            "act_name": "equality_act_2010",
            "predictions": [
                {
                    "text": "New protected characteristic added.",
                    "change_type": "new_definition",
                    "section_id": "s1",
                    "predicted_label": 1,
                    "confidence": 0.97,
                    "needs_review": False,
                }
            ],
        })

    def test_creates_json_file(self, tmp_path):
        output = self._make_output()
        path = save_output(output, "equality_act_2010", output_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".json"

    def test_filename_contains_act_name(self, tmp_path):
        output = self._make_output()
        path = save_output(output, "equality_act_2010", output_dir=tmp_path)
        assert "equality_act_2010" in path.name

    def test_json_is_valid_and_contains_changes(self, tmp_path):
        output = self._make_output()
        path = save_output(output, "equality_act_2010", output_dir=tmp_path)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert "changes" in loaded

    def test_creates_output_dir_if_missing(self, tmp_path):
        nested = tmp_path / "new_outputs_dir"
        output = self._make_output()
        save_output(output, "test_act", output_dir=nested)
        assert nested.exists()
