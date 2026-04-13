"""
Tests for src/domain_classifier.py
"""

import pytest

from src.domain_classifier import (
    ACT_TO_DOMAIN,
    FALLBACK_DOMAIN,
    classify_domain,
    classify_domain_from_text,
    classify_domain_from_url,
    url_to_act_name,
)


# ---------------------------------------------------------------------------
# classify_domain
# ---------------------------------------------------------------------------

class TestClassifyDomain:
    def test_employment_rights_act(self):
        assert classify_domain("employment_rights_act_1996") == "employment"

    def test_national_minimum_wage_act(self):
        assert classify_domain("national_minimum_wage_act_1998") == "wage"

    def test_equality_act(self):
        assert classify_domain("equality_act_2010") == "equality"

    def test_working_time_regulations(self):
        assert classify_domain("working_time_regulations_1998") == "working_time"

    def test_unknown_act_returns_fallback(self):
        assert classify_domain("unknown_act_2099") == FALLBACK_DOMAIN

    def test_case_insensitive(self):
        assert classify_domain("EMPLOYMENT_RIGHTS_ACT_1996") == "employment"

    def test_leading_trailing_whitespace_stripped(self):
        assert classify_domain("  equality_act_2010  ") == "equality"

    def test_all_four_acts_covered(self):
        expected = {"employment", "wage", "equality", "working_time"}
        actual = set(ACT_TO_DOMAIN.values())
        assert actual == expected


# ---------------------------------------------------------------------------
# classify_domain_from_url
# ---------------------------------------------------------------------------

class TestClassifyDomainFromUrl:
    def test_employment_rights_url(self):
        url = "https://www.legislation.gov.uk/ukpga/1996/18/data.xml"
        assert classify_domain_from_url(url) == "employment"

    def test_minimum_wage_url(self):
        url = "https://www.legislation.gov.uk/ukpga/1998/39/data.xml"
        assert classify_domain_from_url(url) == "wage"

    def test_equality_act_url(self):
        url = "https://www.legislation.gov.uk/ukpga/2010/15/data.xml"
        assert classify_domain_from_url(url) == "equality"

    def test_working_time_url(self):
        url = "https://www.legislation.gov.uk/uksi/1998/1833/data.xml"
        assert classify_domain_from_url(url) == "working_time"

    def test_unknown_url_returns_fallback(self):
        url = "https://www.legislation.gov.uk/ukpga/2099/99"
        assert classify_domain_from_url(url) == FALLBACK_DOMAIN

    def test_partial_url_still_matched(self):
        assert classify_domain_from_url("ukpga/1996/18") == "employment"


# ---------------------------------------------------------------------------
# classify_domain_from_text
# ---------------------------------------------------------------------------

class TestClassifyDomainFromText:
    def test_minimum_wage_keyword(self):
        assert classify_domain_from_text("The national minimum wage rate increases.") == "wage"

    def test_equality_keyword(self):
        assert classify_domain_from_text("Protected characteristic discrimination rules apply.") == "equality"

    def test_working_time_keyword(self):
        assert classify_domain_from_text("The employer must provide adequate rest breaks.") == "working_time"

    def test_employment_keyword(self):
        assert classify_domain_from_text("Unfair dismissal claims may now be brought sooner.") == "employment"

    def test_no_keyword_returns_fallback(self):
        assert classify_domain_from_text("Lorem ipsum dolor sit amet.") == FALLBACK_DOMAIN

    def test_case_insensitive_matching(self):
        assert classify_domain_from_text("NATIONAL MINIMUM WAGE applies here.") == "wage"

    def test_equality_act_abbreviation(self):
        assert classify_domain_from_text("Under the EqA, employers must make adjustments.") == "equality"

    def test_wtr_abbreviation(self):
        assert classify_domain_from_text("WTR provisions extend to agency workers.") == "working_time"


# ---------------------------------------------------------------------------
# url_to_act_name
# ---------------------------------------------------------------------------

class TestUrlToActName:
    def test_employment_rights_url(self):
        assert url_to_act_name("https://www.legislation.gov.uk/ukpga/1996/18") == "employment_rights_act_1996"

    def test_minimum_wage_url(self):
        assert url_to_act_name("https://www.legislation.gov.uk/ukpga/1998/39") == "national_minimum_wage_act_1998"

    def test_equality_act_url(self):
        assert url_to_act_name("https://www.legislation.gov.uk/ukpga/2010/15") == "equality_act_2010"

    def test_working_time_url(self):
        assert url_to_act_name("https://www.legislation.gov.uk/uksi/1998/1833") == "working_time_regulations_1998"

    def test_unknown_url_returns_none(self):
        assert url_to_act_name("https://www.legislation.gov.uk/ukpga/2099/99") is None

    def test_returns_string_not_none_for_known(self):
        result = url_to_act_name("https://www.legislation.gov.uk/ukpga/2010/15")
        assert isinstance(result, str)
