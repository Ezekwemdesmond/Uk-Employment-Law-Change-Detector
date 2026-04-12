"""
Tests for src/classifier.py

Note: Full training tests are skipped by default as they require
significant compute time. Use pytest -m "not slow" to skip them,
or run the full suite to include training tests.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.classifier import (
    DEFAULT_CONFIG,
    LegislationDataset,
    compute_metrics,
    split_dataset,
)


# ---------------------------------------------------------------------------
# LegislationDataset
# ---------------------------------------------------------------------------

class TestLegislationDataset:
    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.ones(1, 256, dtype=torch.long),
            "attention_mask": torch.ones(1, 256, dtype=torch.long),
        }
        return tokenizer

    @pytest.fixture
    def sample_examples(self):
        return [
            {"text": "New obligation text.", "label": 1},
            {"text": "Cross-reference update.", "label": 0},
            {"text": "Penalty increased.", "label": 1},
        ]

    def test_len_returns_correct_count(self, mock_tokenizer, sample_examples):
        dataset = LegislationDataset(sample_examples, mock_tokenizer)
        assert len(dataset) == 3

    def test_getitem_returns_dict(self, mock_tokenizer, sample_examples):
        dataset = LegislationDataset(sample_examples, mock_tokenizer)
        item = dataset[0]
        assert isinstance(item, dict)

    def test_getitem_has_required_keys(self, mock_tokenizer, sample_examples):
        dataset = LegislationDataset(sample_examples, mock_tokenizer)
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_labels_are_tensors(self, mock_tokenizer, sample_examples):
        dataset = LegislationDataset(sample_examples, mock_tokenizer)
        item = dataset[0]
        assert isinstance(item["labels"], torch.Tensor)

    def test_labels_match_input(self, mock_tokenizer, sample_examples):
        dataset = LegislationDataset(sample_examples, mock_tokenizer)
        assert dataset[0]["labels"].item() == 1
        assert dataset[1]["labels"].item() == 0

    def test_tokenizer_called_with_text(self, mock_tokenizer, sample_examples):
        dataset = LegislationDataset(sample_examples, mock_tokenizer, max_length=128)
        _ = dataset[0]
        mock_tokenizer.assert_called_once()
        call_args = mock_tokenizer.call_args
        assert call_args[0][0] == "New obligation text."
        assert call_args[1]["max_length"] == 128


# ---------------------------------------------------------------------------
# split_dataset
# ---------------------------------------------------------------------------

class TestSplitDataset:
    @pytest.fixture
    def balanced_examples(self):
        return [{"text": f"text {i}", "label": i % 2} for i in range(100)]

    def test_returns_three_lists(self, balanced_examples):
        train, val, test = split_dataset(balanced_examples)
        assert isinstance(train, list)
        assert isinstance(val, list)
        assert isinstance(test, list)

    def test_default_split_ratios(self, balanced_examples):
        train, val, test = split_dataset(balanced_examples)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_custom_split_ratios(self, balanced_examples):
        train, val, test = split_dataset(
            balanced_examples,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )
        assert len(train) == 60
        assert len(val) == 20
        assert len(test) == 20

    def test_no_overlap_between_splits(self, balanced_examples):
        train, val, test = split_dataset(balanced_examples)

        train_texts = {ex["text"] for ex in train}
        val_texts = {ex["text"] for ex in val}
        test_texts = {ex["text"] for ex in test}

        assert len(train_texts & val_texts) == 0
        assert len(train_texts & test_texts) == 0
        assert len(val_texts & test_texts) == 0

    def test_all_examples_preserved(self, balanced_examples):
        train, val, test = split_dataset(balanced_examples)
        total = len(train) + len(val) + len(test)
        assert total == len(balanced_examples)

    def test_stratified_by_label(self, balanced_examples):
        train, val, test = split_dataset(balanced_examples)

        # Each split should have roughly balanced labels
        train_labels = [ex["label"] for ex in train]
        assert 35 <= train_labels.count(1) <= 45  # ~40 of 80

        val_labels = [ex["label"] for ex in val]
        assert 3 <= val_labels.count(1) <= 7  # ~5 of 10

    def test_seed_produces_reproducible_splits(self, balanced_examples):
        train1, val1, test1 = split_dataset(balanced_examples, seed=42)
        train2, val2, test2 = split_dataset(balanced_examples, seed=42)

        assert [ex["text"] for ex in train1] == [ex["text"] for ex in train2]
        assert [ex["text"] for ex in val1] == [ex["text"] for ex in val2]
        assert [ex["text"] for ex in test1] == [ex["text"] for ex in test2]

    def test_different_seeds_different_splits(self, balanced_examples):
        train1, _, _ = split_dataset(balanced_examples, seed=42)
        train2, _, _ = split_dataset(balanced_examples, seed=123)
        assert [ex["text"] for ex in train1] != [ex["text"] for ex in train2]

    def test_invalid_ratios_raise_assertion(self, balanced_examples):
        with pytest.raises(AssertionError):
            split_dataset(balanced_examples, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_returns_dict(self):
        metrics = compute_metrics([0, 1, 1, 0], [0, 1, 1, 0])
        assert isinstance(metrics, dict)

    def test_has_required_keys(self):
        metrics = compute_metrics([0, 1, 1, 0], [0, 1, 1, 0])
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    def test_perfect_predictions(self):
        labels = [0, 1, 1, 0, 1, 0]
        predictions = [0, 1, 1, 0, 1, 0]
        metrics = compute_metrics(labels, predictions)
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_all_wrong_predictions(self):
        labels = [0, 0, 0, 0]
        predictions = [1, 1, 1, 1]
        metrics = compute_metrics(labels, predictions)
        assert metrics["accuracy"] == 0.0

    def test_accuracy_calculation(self):
        labels = [0, 1, 1, 0, 1, 0]
        predictions = [0, 1, 0, 0, 1, 1]  # 4 correct out of 6
        metrics = compute_metrics(labels, predictions)
        assert abs(metrics["accuracy"] - 4 / 6) < 0.001

    def test_precision_with_no_positive_predictions(self):
        labels = [1, 1, 1]
        predictions = [0, 0, 0]
        metrics = compute_metrics(labels, predictions)
        assert metrics["precision"] == 0.0  # zero_division=0

    def test_recall_with_no_actual_positives(self):
        labels = [0, 0, 0]
        predictions = [1, 1, 1]
        metrics = compute_metrics(labels, predictions)
        assert metrics["recall"] == 0.0  # zero_division=0


# ---------------------------------------------------------------------------
# DEFAULT_CONFIG
# ---------------------------------------------------------------------------

class TestDefaultConfig:
    def test_has_model_name(self):
        assert "model_name" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["model_name"] == "bert-base-uncased"

    def test_has_batch_size(self):
        assert "batch_size" in DEFAULT_CONFIG
        assert isinstance(DEFAULT_CONFIG["batch_size"], int)

    def test_has_learning_rate(self):
        assert "learning_rate" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["learning_rate"] > 0

    def test_has_num_epochs(self):
        assert "num_epochs" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["num_epochs"] > 0

    def test_has_max_length(self):
        assert "max_length" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["max_length"] <= 512  # BERT max


# ---------------------------------------------------------------------------
# Integration test markers
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestTrainingIntegration:
    """
    These tests require a GPU or significant CPU time.
    Skip with: pytest -m "not slow"
    """

    def test_training_placeholder(self):
        # Placeholder — actual training tests would go here
        # but are skipped by default due to compute requirements
        pytest.skip("Skipping slow training test")
