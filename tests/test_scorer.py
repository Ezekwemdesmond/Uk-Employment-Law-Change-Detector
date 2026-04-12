"""
Tests for src/scorer.py
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.scorer import (
    CONFIDENCE_THRESHOLD,
    LABEL_NAMES,
    Scorer,
    save_predictions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model():
    """Create a mock BERT model that returns predictable logits."""
    model = MagicMock()
    model.eval = MagicMock()
    model.to = MagicMock(return_value=model)
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.ones(1, 256, dtype=torch.long),
        "attention_mask": torch.ones(1, 256, dtype=torch.long),
    }
    return tokenizer


@pytest.fixture
def scorer_with_mocks(mock_model, mock_tokenizer):
    """Create a Scorer with mocked model and tokenizer."""
    with patch("src.scorer.load_model") as mock_load:
        mock_load.return_value = (mock_model, mock_tokenizer)
        scorer = Scorer(model_path="fake/path")
    return scorer, mock_model, mock_tokenizer


# ---------------------------------------------------------------------------
# Scorer initialization
# ---------------------------------------------------------------------------

class TestScorerInit:
    def test_uses_provided_model_and_tokenizer(self, mock_model, mock_tokenizer):
        scorer = Scorer(model=mock_model, tokenizer=mock_tokenizer)
        assert scorer.model is mock_model
        assert scorer.tokenizer is mock_tokenizer

    def test_default_confidence_threshold(self, mock_model, mock_tokenizer):
        scorer = Scorer(model=mock_model, tokenizer=mock_tokenizer)
        assert scorer.confidence_threshold == CONFIDENCE_THRESHOLD

    def test_custom_confidence_threshold(self, mock_model, mock_tokenizer):
        scorer = Scorer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            confidence_threshold=0.9,
        )
        assert scorer.confidence_threshold == 0.9

    def test_default_max_length(self, mock_model, mock_tokenizer):
        scorer = Scorer(model=mock_model, tokenizer=mock_tokenizer)
        assert scorer.max_length == 256

    def test_model_set_to_eval_mode(self, mock_model, mock_tokenizer):
        Scorer(model=mock_model, tokenizer=mock_tokenizer)
        mock_model.eval.assert_called_once()


# ---------------------------------------------------------------------------
# predict_single
# ---------------------------------------------------------------------------

class TestPredictSingle:
    def test_returns_dict(self, scorer_with_mocks):
        scorer, model, _ = scorer_with_mocks
        # Mock model output: strong material prediction
        model.return_value = MagicMock(logits=torch.tensor([[0.1, 0.9]]))

        result = scorer.predict_single("Test text.")
        assert isinstance(result, dict)

    def test_has_required_keys(self, scorer_with_mocks):
        scorer, model, _ = scorer_with_mocks
        model.return_value = MagicMock(logits=torch.tensor([[0.1, 0.9]]))

        result = scorer.predict_single("Test text.")
        assert "predicted_label" in result
        assert "predicted_class" in result
        assert "confidence" in result
        assert "probability_material" in result
        assert "needs_review" in result

    def test_high_confidence_material_prediction(self, scorer_with_mocks):
        scorer, model, _ = scorer_with_mocks
        # Logits that result in ~90% probability for material
        model.return_value = MagicMock(logits=torch.tensor([[-2.0, 2.0]]))

        result = scorer.predict_single("Penalty increased.")
        assert result["predicted_label"] == 1
        assert result["predicted_class"] == "material"
        assert result["confidence"] > 0.75
        assert not result["needs_review"]

    def test_high_confidence_minor_prediction(self, scorer_with_mocks):
        scorer, model, _ = scorer_with_mocks
        # Logits that result in ~90% probability for minor
        model.return_value = MagicMock(logits=torch.tensor([[2.0, -2.0]]))

        result = scorer.predict_single("Cross-reference update.")
        assert result["predicted_label"] == 0
        assert result["predicted_class"] == "minor"
        assert result["confidence"] > 0.75
        assert not result["needs_review"]

    def test_low_confidence_flags_for_review(self, scorer_with_mocks):
        scorer, model, _ = scorer_with_mocks
        # Logits that result in ~55% probability (low confidence)
        model.return_value = MagicMock(logits=torch.tensor([[0.0, 0.2]]))

        result = scorer.predict_single("Ambiguous text.")
        assert result["needs_review"] is True
        assert result["confidence"] < 0.75

    def test_probability_material_in_range(self, scorer_with_mocks):
        scorer, model, _ = scorer_with_mocks
        model.return_value = MagicMock(logits=torch.tensor([[0.5, 0.5]]))

        result = scorer.predict_single("Test.")
        assert 0.0 <= result["probability_material"] <= 1.0

    def test_confidence_rounded_to_4_decimals(self, scorer_with_mocks):
        scorer, model, _ = scorer_with_mocks
        model.return_value = MagicMock(logits=torch.tensor([[0.123456, 0.654321]]))

        result = scorer.predict_single("Test.")
        # Check that confidence has at most 4 decimal places
        conf_str = str(result["confidence"])
        if "." in conf_str:
            decimals = len(conf_str.split(".")[1])
            assert decimals <= 4


# ---------------------------------------------------------------------------
# predict_batch
# ---------------------------------------------------------------------------

class TestPredictBatch:
    def test_returns_list(self, scorer_with_mocks):
        scorer, model, tokenizer = scorer_with_mocks
        # Mock batch output
        model.return_value = MagicMock(logits=torch.tensor([[-2.0, 2.0], [2.0, -2.0]]))
        tokenizer.return_value = {
            "input_ids": torch.ones(2, 256, dtype=torch.long),
            "attention_mask": torch.ones(2, 256, dtype=torch.long),
        }

        results = scorer.predict_batch(["Text 1.", "Text 2."])
        assert isinstance(results, list)

    def test_correct_length(self, scorer_with_mocks):
        scorer, model, tokenizer = scorer_with_mocks
        model.return_value = MagicMock(logits=torch.tensor([[-2.0, 2.0], [2.0, -2.0], [0.0, 0.0]]))
        tokenizer.return_value = {
            "input_ids": torch.ones(3, 256, dtype=torch.long),
            "attention_mask": torch.ones(3, 256, dtype=torch.long),
        }

        results = scorer.predict_batch(["Text 1.", "Text 2.", "Text 3."])
        assert len(results) == 3

    def test_each_result_has_required_keys(self, scorer_with_mocks):
        scorer, model, tokenizer = scorer_with_mocks
        model.return_value = MagicMock(logits=torch.tensor([[-2.0, 2.0]]))
        tokenizer.return_value = {
            "input_ids": torch.ones(1, 256, dtype=torch.long),
            "attention_mask": torch.ones(1, 256, dtype=torch.long),
        }

        results = scorer.predict_batch(["Text."])
        for result in results:
            assert "predicted_label" in result
            assert "confidence" in result


# ---------------------------------------------------------------------------
# predict_examples
# ---------------------------------------------------------------------------

class TestPredictExamples:
    def test_merges_prediction_into_example(self, scorer_with_mocks):
        scorer, model, tokenizer = scorer_with_mocks
        model.return_value = MagicMock(logits=torch.tensor([[-2.0, 2.0]]))
        tokenizer.return_value = {
            "input_ids": torch.ones(1, 256, dtype=torch.long),
            "attention_mask": torch.ones(1, 256, dtype=torch.long),
        }

        examples = [{"text": "Test.", "change_type": "new_obligation", "act_name": "test_act"}]
        results = scorer.predict_examples(examples)

        assert len(results) == 1
        assert results[0]["text"] == "Test."
        assert results[0]["change_type"] == "new_obligation"
        assert results[0]["act_name"] == "test_act"
        assert "predicted_label" in results[0]

    def test_preserves_original_fields(self, scorer_with_mocks):
        scorer, model, tokenizer = scorer_with_mocks
        model.return_value = MagicMock(logits=torch.tensor([[-2.0, 2.0]]))
        tokenizer.return_value = {
            "input_ids": torch.ones(1, 256, dtype=torch.long),
            "attention_mask": torch.ones(1, 256, dtype=torch.long),
        }

        examples = [{"text": "Test.", "custom_field": "custom_value"}]
        results = scorer.predict_examples(examples)

        assert results[0]["custom_field"] == "custom_value"


# ---------------------------------------------------------------------------
# predict_diff
# ---------------------------------------------------------------------------

class TestPredictDiff:
    def test_returns_dict_with_summary(self, scorer_with_mocks):
        scorer, model, tokenizer = scorer_with_mocks
        model.return_value = MagicMock(logits=torch.tensor([[-2.0, 2.0]]))
        tokenizer.return_value = {
            "input_ids": torch.ones(1, 256, dtype=torch.long),
            "attention_mask": torch.ones(1, 256, dtype=torch.long),
        }

        diff = {
            "act_name": "test_act",
            "added": {"s1": "New section."},
            "removed": {},
            "modified": {},
        }

        result = scorer.predict_diff(diff)
        assert isinstance(result, dict)
        assert "summary" in result
        assert "predictions" in result

    def test_summary_has_counts(self, scorer_with_mocks):
        scorer, model, tokenizer = scorer_with_mocks
        # Return different predictions for batch
        model.return_value = MagicMock(logits=torch.tensor([[-2.0, 2.0], [2.0, -2.0]]))
        tokenizer.return_value = {
            "input_ids": torch.ones(2, 256, dtype=torch.long),
            "attention_mask": torch.ones(2, 256, dtype=torch.long),
        }

        diff = {
            "act_name": "test_act",
            "added": {"s1": "New obligation."},
            "removed": {"s2": "Old section."},
            "modified": {},
        }

        result = scorer.predict_diff(diff)
        summary = result["summary"]

        assert "total" in summary
        assert "material_changes" in summary
        assert "minor_changes" in summary
        assert "needs_review" in summary

    def test_preserves_act_name(self, scorer_with_mocks):
        scorer, model, tokenizer = scorer_with_mocks
        model.return_value = MagicMock(logits=torch.tensor([[-2.0, 2.0]]))
        tokenizer.return_value = {
            "input_ids": torch.ones(1, 256, dtype=torch.long),
            "attention_mask": torch.ones(1, 256, dtype=torch.long),
        }

        diff = {
            "act_name": "equality_act_2010",
            "added": {"s1": "Test."},
            "removed": {},
            "modified": {},
        }

        result = scorer.predict_diff(diff)
        assert result["act_name"] == "equality_act_2010"


# ---------------------------------------------------------------------------
# save_predictions
# ---------------------------------------------------------------------------

class TestSavePredictions:
    def test_creates_json_file(self, tmp_path):
        predictions = {"test": "data"}
        path = save_predictions(predictions, filename="test.json", output_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".json"

    def test_creates_output_dir(self, tmp_path):
        nested = tmp_path / "nested" / "dir"
        predictions = {"test": "data"}
        save_predictions(predictions, filename="test.json", output_dir=nested)
        assert nested.exists()

    def test_default_filename_has_timestamp(self, tmp_path):
        predictions = {"test": "data"}
        path = save_predictions(predictions, output_dir=tmp_path)
        assert path.name.startswith("predictions_")
        assert path.name.endswith(".json")

    def test_json_is_valid(self, tmp_path):
        predictions = {
            "timestamp": "2024-01-01T00:00:00Z",
            "predictions": [{"label": 1, "confidence": 0.95}],
        }
        path = save_predictions(predictions, filename="test.json", output_dir=tmp_path)

        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["timestamp"] == "2024-01-01T00:00:00Z"

    def test_handles_list_input(self, tmp_path):
        predictions = [{"label": 1}, {"label": 0}]
        path = save_predictions(predictions, filename="test.json", output_dir=tmp_path)

        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert isinstance(loaded, list)
        assert len(loaded) == 2


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_confidence_threshold_in_valid_range(self):
        assert 0.0 < CONFIDENCE_THRESHOLD < 1.0

    def test_label_names_complete(self):
        assert 0 in LABEL_NAMES
        assert 1 in LABEL_NAMES
        assert LABEL_NAMES[0] == "minor"
        assert LABEL_NAMES[1] == "material"
