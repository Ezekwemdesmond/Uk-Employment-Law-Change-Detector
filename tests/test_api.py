"""
Tests for src/api.py

Uses FastAPI's TestClient (via httpx) for endpoint testing.
All external I/O (network, disk, model) is mocked.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

if FASTAPI_AVAILABLE:
    from src.api import app, _model_is_available

pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE,
    reason="fastapi[testclient] not installed"
)


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status_ok(self, client):
        response = client.get("/health")
        assert response.json()["status"] == "ok"

    def test_health_service_name(self, client):
        response = client.get("/health")
        assert response.json()["service"] == "complyAI"

    def test_health_has_model_available_field(self, client):
        response = client.get("/health")
        assert "model_available" in response.json()

    def test_health_model_available_is_bool(self, client):
        response = client.get("/health")
        assert isinstance(response.json()["model_available"], bool)

    def test_health_model_available_false_without_model(self, client, tmp_path):
        with patch("src.api.MODELS_DIR", tmp_path / "nonexistent"):
            response = client.get("/health")
        assert response.json()["model_available"] is False


# ---------------------------------------------------------------------------
# /analyse — validation
# ---------------------------------------------------------------------------

class TestAnalyseValidation:
    def test_missing_url_returns_422(self, client):
        response = client.post("/analyse", json={})
        assert response.status_code == 422

    def test_non_legislation_url_returns_422(self, client):
        response = client.post("/analyse", json={"url": "https://www.google.com"})
        assert response.status_code == 422

    def test_invalid_confidence_too_high_returns_422(self, client):
        response = client.post("/analyse", json={
            "url": "https://www.legislation.gov.uk/ukpga/1996/18",
            "min_confidence": 1.5,
        })
        assert response.status_code == 422

    def test_invalid_confidence_negative_returns_422(self, client):
        response = client.post("/analyse", json={
            "url": "https://www.legislation.gov.uk/ukpga/1996/18",
            "min_confidence": -0.1,
        })
        assert response.status_code == 422

    def test_unknown_act_url_returns_422(self, client):
        with patch("src.api.fetch_legislation_xml"):
            response = client.post("/analyse", json={
                "url": "https://www.legislation.gov.uk/ukpga/2099/99"
            })
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# /analyse — baseline establishment
# ---------------------------------------------------------------------------

class TestAnalyseBaseline:
    def test_first_call_establishes_baseline(self, client, tmp_path):
        """When no baseline exists, save and return baseline_established."""
        mock_xml = "<Legislation><Body><Section id='s1'><Number>1</Number><P>Text.</P></Section></Body></Legislation>"

        with patch("src.api.DATA_RAW_DIR", tmp_path), \
             patch("src.api.fetch_legislation_xml", return_value=mock_xml), \
             patch("src.api.save_xml") as mock_save:

            response = client.post("/analyse", json={
                "url": "https://www.legislation.gov.uk/ukpga/1996/18"
            })

        assert response.status_code == 200
        data = response.json()
        assert data["pipeline_status"] == "baseline_established"
        assert data["total_material_changes"] == 0
        assert data["changes"] == []

    def test_baseline_act_name_correct(self, client, tmp_path):
        mock_xml = "<Legislation/>"

        with patch("src.api.DATA_RAW_DIR", tmp_path), \
             patch("src.api.fetch_legislation_xml", return_value=mock_xml), \
             patch("src.api.save_xml"):

            response = client.post("/analyse", json={
                "url": "https://www.legislation.gov.uk/ukpga/1996/18"
            })

        assert response.json()["act_name"] == "employment_rights_act_1996"


# ---------------------------------------------------------------------------
# /analyse — network errors
# ---------------------------------------------------------------------------

class TestAnalyseNetworkErrors:
    def test_network_error_returns_502(self, client, tmp_path):
        import requests as req

        with patch("src.api.DATA_RAW_DIR", tmp_path), \
             patch("src.api.fetch_legislation_xml", side_effect=req.ConnectionError("timeout")):

            response = client.post("/analyse", json={
                "url": "https://www.legislation.gov.uk/ukpga/1996/18"
            })

        assert response.status_code == 502

    def test_502_detail_mentions_xml(self, client, tmp_path):
        import requests as req

        with patch("src.api.DATA_RAW_DIR", tmp_path), \
             patch("src.api.fetch_legislation_xml", side_effect=req.ConnectionError("timeout")):

            response = client.post("/analyse", json={
                "url": "https://www.legislation.gov.uk/ukpga/1996/18"
            })

        assert "XML" in response.json()["detail"] or "legislation" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# /analyse — no changes
# ---------------------------------------------------------------------------

class TestAnalyseNoChanges:
    def _setup_identical_xml(self, tmp_path):
        """Write a baseline XML file so a diff can be computed."""
        xml = (
            "<Legislation><Body>"
            "<Section id='s1'><Number>1</Number><P>Unchanged text.</P></Section>"
            "</Body></Legislation>"
        )
        baseline = tmp_path / "employment_rights_act_1996_20240101T000000Z.xml"
        baseline.write_text(xml, encoding="utf-8")
        return xml

    def test_identical_xml_returns_no_changes(self, client, tmp_path):
        xml = self._setup_identical_xml(tmp_path)

        with patch("src.api.DATA_RAW_DIR", tmp_path), \
             patch("src.api.fetch_legislation_xml", return_value=xml), \
             patch("src.api.save_xml"):

            response = client.post("/analyse", json={
                "url": "https://www.legislation.gov.uk/ukpga/1996/18"
            })

        assert response.status_code == 200
        data = response.json()
        assert data["pipeline_status"] == "no_changes"
        assert data["total_material_changes"] == 0


# ---------------------------------------------------------------------------
# /analyse — full pipeline (model unavailable → 503)
# ---------------------------------------------------------------------------

class TestAnalyseModelUnavailable:
    def _setup_baseline(self, tmp_path, old_xml, new_xml=None):
        baseline = tmp_path / "employment_rights_act_1996_20240101T000000Z.xml"
        baseline.write_text(old_xml, encoding="utf-8")
        return new_xml or old_xml.replace("Old text.", "New obligation text.")

    def test_no_model_returns_503(self, client, tmp_path):
        old_xml = (
            "<Legislation><Body>"
            "<Section id='s1'><Number>1</Number><P>Old text.</P></Section>"
            "</Body></Legislation>"
        )
        new_xml = (
            "<Legislation><Body>"
            "<Section id='s1'><Number>1</Number><P>New obligation text.</P></Section>"
            "<Section id='s2'><Number>2</Number><P>Brand new section.</P></Section>"
            "</Body></Legislation>"
        )
        self._setup_baseline(tmp_path, old_xml, new_xml)

        with patch("src.api.DATA_RAW_DIR", tmp_path), \
             patch("src.api.fetch_legislation_xml", return_value=new_xml), \
             patch("src.api.save_xml"), \
             patch("src.api._model_is_available", return_value=False):

            response = client.post("/analyse", json={
                "url": "https://www.legislation.gov.uk/ukpga/1996/18"
            })

        assert response.status_code == 503


# ---------------------------------------------------------------------------
# /analyse — full pipeline (model available)
# ---------------------------------------------------------------------------

class TestAnalyseFullPipeline:
    def test_material_changes_returned(self, client, tmp_path):
        old_xml = (
            "<Legislation><Body>"
            "<Section id='s1'><Number>1</Number><P>Old notice text.</P></Section>"
            "</Body></Legislation>"
        )
        new_xml = (
            "<Legislation><Body>"
            "<Section id='s1'><Number>1</Number><P>New obligation for employers.</P></Section>"
            "<Section id='s2'><Number>2</Number><P>Penalty of up to £10,000.</P></Section>"
            "</Body></Legislation>"
        )
        baseline = tmp_path / "employment_rights_act_1996_20240101T000000Z.xml"
        baseline.write_text(old_xml, encoding="utf-8")

        mock_scored = {
            "act_name": "employment_rights_act_1996",
            "timestamp": "2024-01-01T00:00:00Z",
            "predictions": [
                {
                    "text": "New obligation for employers.",
                    "change_type": "modified_section",
                    "section_id": "s1",
                    "predicted_label": 1,
                    "confidence": 0.93,
                    "needs_review": False,
                },
                {
                    "text": "Penalty of up to £10,000.",
                    "change_type": "added_section",
                    "section_id": "s2",
                    "predicted_label": 1,
                    "confidence": 0.88,
                    "needs_review": False,
                },
            ],
            "summary": {"total": 2},
        }

        with patch("src.api.DATA_RAW_DIR", tmp_path), \
             patch("src.api.fetch_legislation_xml", return_value=new_xml), \
             patch("src.api.save_xml"), \
             patch("src.api._model_is_available", return_value=True), \
             patch("src.api._run_scorer", return_value=mock_scored):

            response = client.post("/analyse", json={
                "url": "https://www.legislation.gov.uk/ukpga/1996/18"
            })

        assert response.status_code == 200
        data = response.json()
        assert data["pipeline_status"] == "changes_found"
        assert data["total_material_changes"] == 2
        assert len(data["changes"]) == 2

    def test_each_change_has_five_fields(self, client, tmp_path):
        old_xml = "<Legislation><Body><Section id='s1'><Number>1</Number><P>Old.</P></Section></Body></Legislation>"
        new_xml = "<Legislation><Body><Section id='s1'><Number>1</Number><P>New obligation.</P></Section></Body></Legislation>"
        baseline = tmp_path / "employment_rights_act_1996_20240101T000000Z.xml"
        baseline.write_text(old_xml, encoding="utf-8")

        mock_scored = {
            "act_name": "employment_rights_act_1996",
            "predictions": [{
                "text": "New obligation for all employers.",
                "change_type": "modified_section",
                "section_id": "s1",
                "predicted_label": 1,
                "confidence": 0.91,
                "needs_review": False,
            }],
        }

        with patch("src.api.DATA_RAW_DIR", tmp_path), \
             patch("src.api.fetch_legislation_xml", return_value=new_xml), \
             patch("src.api.save_xml"), \
             patch("src.api._model_is_available", return_value=True), \
             patch("src.api._run_scorer", return_value=mock_scored):

            response = client.post("/analyse", json={
                "url": "https://www.legislation.gov.uk/ukpga/1996/18"
            })

        change = response.json()["changes"][0]
        for field in ("domain", "effective_date", "penalty_threshold",
                      "affected_business_types", "obligation_type"):
            assert field in change, f"Missing field: {field}"
