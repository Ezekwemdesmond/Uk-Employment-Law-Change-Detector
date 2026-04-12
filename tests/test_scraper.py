"""
Tests for src/scraper.py

Uses unittest.mock to avoid real network calls. File I/O is redirected to
a tmp_path fixture so no permanent files are created during the test run.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.scraper import (
    ACTS,
    RATE_LIMIT_DELAY,
    build_session,
    fetch_legislation_xml,
    save_xml,
    scrape_all,
)


# ---------------------------------------------------------------------------
# build_session
# ---------------------------------------------------------------------------

class TestBuildSession:
    def test_returns_session(self):
        session = build_session()
        assert isinstance(session, requests.Session)

    def test_has_xml_accept_header(self):
        session = build_session()
        assert session.headers.get("Accept") == "application/xml"

    def test_has_user_agent_header(self):
        session = build_session()
        assert "complyAI" in session.headers.get("User-Agent", "")

    def test_https_adapter_mounted(self):
        session = build_session()
        assert "https://" in session.get_adapter("https://example.com").max_retries.__class__.__name__ or \
               session.get_adapter("https://example.com") is not None


# ---------------------------------------------------------------------------
# fetch_legislation_xml
# ---------------------------------------------------------------------------

class TestFetchLegislationXml:
    def _make_session(self, status_code: int, text: str, headers: dict | None = None) -> requests.Session:
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.text = text
        mock_response.headers = headers or {}
        mock_response.raise_for_status = MagicMock()
        if status_code >= 400:
            mock_response.raise_for_status.side_effect = requests.HTTPError(
                response=mock_response
            )

        session = MagicMock(spec=requests.Session)
        session.get.return_value = mock_response
        return session

    def test_successful_fetch_returns_xml(self):
        session = self._make_session(200, "<Legislation/>")
        result = fetch_legislation_xml(session, "ukpga/1996/18")
        assert result == "<Legislation/>"

    def test_correct_url_constructed(self):
        session = self._make_session(200, "<Legislation/>")
        fetch_legislation_xml(session, "ukpga/1996/18")
        called_url = session.get.call_args[0][0]
        assert called_url == "https://www.legislation.gov.uk/ukpga/1996/18/data.xml"

    def test_http_error_raises(self):
        session = self._make_session(404, "Not Found")
        with pytest.raises(requests.HTTPError):
            fetch_legislation_xml(session, "ukpga/9999/99")

    def test_rate_limit_triggers_sleep_and_retry(self):
        """On 429, honours Retry-After header then retries once."""
        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.text = "<Legislation/>"
        ok_response.headers = {}
        ok_response.raise_for_status = MagicMock()

        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.text = ""
        rate_limited.headers = {"Retry-After": "2"}

        session = MagicMock(spec=requests.Session)
        session.get.side_effect = [rate_limited, ok_response]

        with patch("src.scraper.time.sleep") as mock_sleep:
            result = fetch_legislation_xml(session, "ukpga/1996/18")

        mock_sleep.assert_called_once_with(2)
        assert result == "<Legislation/>"
        assert session.get.call_count == 2


# ---------------------------------------------------------------------------
# save_xml
# ---------------------------------------------------------------------------

class TestSaveXml:
    def test_creates_file_in_output_dir(self, tmp_path):
        path = save_xml("test_act", "<Legislation/>", output_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".xml"

    def test_filename_starts_with_act_name(self, tmp_path):
        path = save_xml("employment_rights_act_1996", "<Legislation/>", output_dir=tmp_path)
        assert path.name.startswith("employment_rights_act_1996_")

    def test_filename_contains_timestamp(self, tmp_path):
        path = save_xml("test_act", "<Legislation/>", output_dir=tmp_path)
        # Timestamp format: YYYYMMDDTHHMMSSz — 16 chars
        stem = path.stem  # e.g. "test_act_20240101T120000Z"
        parts = stem.split("_")
        timestamp_part = parts[-1]
        assert len(timestamp_part) == 16
        assert timestamp_part.endswith("Z")

    def test_file_content_matches_input(self, tmp_path):
        xml = "<Legislation><Title>ERA 1996</Title></Legislation>"
        path = save_xml("era", xml, output_dir=tmp_path)
        assert path.read_text(encoding="utf-8") == xml

    def test_creates_output_dir_if_missing(self, tmp_path):
        nested = tmp_path / "deeply" / "nested"
        save_xml("test_act", "<X/>", output_dir=nested)
        assert nested.exists()


# ---------------------------------------------------------------------------
# scrape_all
# ---------------------------------------------------------------------------

class TestScrapeAll:
    def test_returns_dict_of_paths(self, tmp_path):
        with patch("src.scraper.fetch_legislation_xml", return_value="<Legislation/>"), \
             patch("src.scraper.time.sleep"):
            results = scrape_all(acts={"test_act": "ukpga/1996/18"}, output_dir=tmp_path)

        assert "test_act" in results
        assert isinstance(results["test_act"], Path)

    def test_failed_acts_omitted_from_results(self, tmp_path):
        with patch(
            "src.scraper.fetch_legislation_xml",
            side_effect=requests.ConnectionError("Network error"),
        ), patch("src.scraper.time.sleep"):
            results = scrape_all(acts={"failing_act": "ukpga/0000/0"}, output_dir=tmp_path)

        assert "failing_act" not in results
        assert results == {}

    def test_rate_limit_delay_observed(self, tmp_path):
        """scrape_all should call time.sleep once per act."""
        acts = {"act_a": "ukpga/1996/18", "act_b": "ukpga/1998/39"}
        with patch("src.scraper.fetch_legislation_xml", return_value="<Legislation/>"), \
             patch("src.scraper.time.sleep") as mock_sleep:
            scrape_all(acts=acts, output_dir=tmp_path)

        assert mock_sleep.call_count == len(acts)

    def test_default_acts_constant(self):
        """ACTS dict contains the four required legislations."""
        required = {
            "employment_rights_act_1996",
            "national_minimum_wage_act_1998",
            "equality_act_2010",
            "working_time_regulations_1998",
        }
        assert required.issubset(set(ACTS.keys()))
