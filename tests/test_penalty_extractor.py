"""
Tests for src/penalty_extractor.py
"""

import pytest

from src.penalty_extractor import extract_all_amounts, extract_penalty


# ---------------------------------------------------------------------------
# extract_penalty — unlimited
# ---------------------------------------------------------------------------

class TestExtractPenaltyUnlimited:
    def test_unlimited_fine(self):
        assert extract_penalty("The court may impose an unlimited fine.") == "unlimited"

    def test_unlimited_compensation(self):
        assert extract_penalty("Unlimited compensation may be awarded.") == "unlimited"

    def test_unlimited_penalty(self):
        assert extract_penalty("An unlimited penalty applies.") == "unlimited"

    def test_unlimited_award(self):
        assert extract_penalty("An unlimited award is available.") == "unlimited"

    def test_unlimited_damages(self):
        assert extract_penalty("Unlimited damages may be sought.") == "unlimited"

    def test_unlimited_case_insensitive(self):
        assert extract_penalty("UNLIMITED FINE may be imposed.") == "unlimited"

    def test_unlimited_takes_priority_over_amount(self):
        text = "A fine of £5,000 or unlimited penalty applies."
        assert extract_penalty(text) == "unlimited"


# ---------------------------------------------------------------------------
# extract_penalty — monetary amounts
# ---------------------------------------------------------------------------

class TestExtractPenaltyMonetary:
    def test_simple_amount(self):
        result = extract_penalty("The maximum fine is £500.")
        assert result is not None
        assert "£500" in result

    def test_thousands_separator(self):
        result = extract_penalty("A penalty of up to £10,000 may be imposed.")
        assert result is not None
        assert "£10,000" in result

    def test_larger_amount(self):
        result = extract_penalty("Maximum award of £20,000 in the employment tribunal.")
        assert result is not None
        assert "£20,000" in result

    def test_amount_with_per_day_qualifier(self):
        result = extract_penalty("A daily fine of £500 per day applies.")
        assert result is not None
        assert "£500" in result

    def test_amount_with_per_week_qualifier(self):
        result = extract_penalty("Compensation of £1,500 per week.")
        assert result is not None
        assert "£1,500" in result

    def test_amount_with_per_employee_qualifier(self):
        result = extract_penalty("A fine of £200 per employee.")
        assert result is not None
        assert "£200" in result

    def test_no_penalty_context_single_amount(self):
        # Single amount with no penalty context still returned (likely the threshold)
        result = extract_penalty("The rate is £9.50.")
        assert result is not None
        assert "£9" in result

    def test_no_penalty_context_multiple_amounts_returns_none(self):
        # Multiple amounts with no penalty signal — too ambiguous, return None
        result = extract_penalty("The rate is £9.50 and the cap is £12.00.")
        assert result is None

    def test_no_amount_returns_none(self):
        result = extract_penalty("The employer must provide written notice.")
        assert result is None

    def test_empty_string_returns_none(self):
        result = extract_penalty("")
        assert result is None

    def test_none_input_returns_none(self):
        result = extract_penalty(None)
        assert result is None

    def test_highest_amount_selected_with_context(self):
        text = "The basic award is £5,000 but the maximum compensatory award is £20,000."
        result = extract_penalty(text)
        assert result is not None
        assert "£20,000" in result

    def test_up_to_phrase_triggers_context(self):
        result = extract_penalty("Employers face a fine of up to £50,000.")
        assert result is not None
        assert "£50,000" in result


# ---------------------------------------------------------------------------
# extract_penalty — edge cases
# ---------------------------------------------------------------------------

class TestExtractPenaltyEdgeCases:
    def test_whitespace_only_returns_none(self):
        assert extract_penalty("   ") is None

    def test_pence_amounts_parsed(self):
        result = extract_penalty("Maximum award of £1,000.00.")
        assert result is not None

    def test_cross_reference_number_not_mistaken_for_penalty(self):
        # A plain section number should not match the amount pattern
        result = extract_penalty("See section 18 of the Act.")
        assert result is None


# ---------------------------------------------------------------------------
# extract_all_amounts
# ---------------------------------------------------------------------------

class TestExtractAllAmounts:
    def test_returns_list(self):
        result = extract_all_amounts("Fine of £500 and award of £1,000.")
        assert isinstance(result, list)

    def test_finds_multiple_amounts(self):
        result = extract_all_amounts("Fine of £500 and award of £1,000.")
        assert len(result) == 2

    def test_finds_single_amount(self):
        result = extract_all_amounts("Maximum penalty of £5,000.")
        assert len(result) == 1
        assert "£5,000" in result[0]

    def test_empty_text_returns_empty_list(self):
        assert extract_all_amounts("") == []

    def test_no_amounts_returns_empty_list(self):
        assert extract_all_amounts("The employer must give notice.") == []
