"""
Tests for src/data_builder.py
"""

import csv
from pathlib import Path

import pytest

from src.data_builder import (
    ACT_NAMES,
    MATERIAL_TEMPLATES,
    MINOR_TEMPLATES,
    _fill_template,
    build_training_dataset,
    examples_from_diff,
    generate_examples,
    load_dataset,
    save_dataset,
)


# ---------------------------------------------------------------------------
# _fill_template
# ---------------------------------------------------------------------------

class TestFillTemplate:
    def test_replaces_days_placeholder(self):
        result = _fill_template("Notice period is {days} days.")
        assert "{days}" not in result
        assert "days" in result

    def test_replaces_amount_placeholder(self):
        result = _fill_template("Fine up to {amount}.")
        assert "{amount}" not in result
        assert "£" in result

    def test_replaces_multiple_placeholders(self):
        result = _fill_template("{old_amount} increased to {new_amount}")
        assert "{old_amount}" not in result
        assert "{new_amount}" not in result

    def test_no_placeholders_unchanged(self):
        text = "Plain text with no placeholders."
        result = _fill_template(text)
        assert result == text

    def test_numeric_placeholders_are_digits(self):
        result = _fill_template("Section {num} applies.")
        # Extract the number
        parts = result.split()
        section_idx = parts.index("Section")
        num_str = parts[section_idx + 1]
        assert num_str.isdigit()


# ---------------------------------------------------------------------------
# generate_examples
# ---------------------------------------------------------------------------

class TestGenerateExamples:
    def test_returns_list(self):
        examples = generate_examples(n_per_class=10)
        assert isinstance(examples, list)

    def test_correct_total_count(self):
        examples = generate_examples(n_per_class=50)
        assert len(examples) == 100  # 50 per class

    def test_balanced_classes(self):
        examples = generate_examples(n_per_class=100)
        material = sum(1 for ex in examples if ex["label"] == 1)
        minor = sum(1 for ex in examples if ex["label"] == 0)
        assert material == 100
        assert minor == 100

    def test_examples_have_required_keys(self):
        examples = generate_examples(n_per_class=10)
        for ex in examples:
            assert "text" in ex
            assert "label" in ex
            assert "change_type" in ex
            assert "act_name" in ex

    def test_labels_are_binary(self):
        examples = generate_examples(n_per_class=50)
        for ex in examples:
            assert ex["label"] in (0, 1)

    def test_act_names_from_valid_set(self):
        examples = generate_examples(n_per_class=50)
        for ex in examples:
            assert ex["act_name"] in ACT_NAMES

    def test_material_change_types_valid(self):
        examples = generate_examples(n_per_class=50)
        material_types = set(MATERIAL_TEMPLATES.keys())
        for ex in examples:
            if ex["label"] == 1:
                assert ex["change_type"] in material_types

    def test_minor_change_types_valid(self):
        examples = generate_examples(n_per_class=50)
        minor_types = set(MINOR_TEMPLATES.keys())
        for ex in examples:
            if ex["label"] == 0:
                assert ex["change_type"] in minor_types

    def test_seed_produces_reproducible_output(self):
        examples1 = generate_examples(n_per_class=10, seed=123)
        examples2 = generate_examples(n_per_class=10, seed=123)
        assert examples1 == examples2

    def test_different_seeds_produce_different_output(self):
        examples1 = generate_examples(n_per_class=10, seed=123)
        examples2 = generate_examples(n_per_class=10, seed=456)
        assert examples1 != examples2

    def test_text_is_non_empty(self):
        examples = generate_examples(n_per_class=50)
        for ex in examples:
            assert len(ex["text"]) > 0


# ---------------------------------------------------------------------------
# examples_from_diff
# ---------------------------------------------------------------------------

class TestExamplesFromDiff:
    def _make_diff(self) -> dict:
        return {
            "act_name": "test_act",
            "timestamp": "2024-01-01T00:00:00Z",
            "summary": {},
            "added": {"s1": "New section about penalties."},
            "removed": {"s2": "Old section removed."},
            "modified": {
                "s3": {
                    "old": "Original text.",
                    "new": "Modified text.",
                    "combined": "Original text. [SEP] Modified text.",
                }
            },
        }

    def test_returns_list(self):
        diff = self._make_diff()
        examples = examples_from_diff(diff)
        assert isinstance(examples, list)

    def test_extracts_added_sections(self):
        diff = self._make_diff()
        examples = examples_from_diff(diff)
        added = [ex for ex in examples if ex["change_type"] == "added_section"]
        assert len(added) == 1
        assert "penalties" in added[0]["text"]

    def test_extracts_removed_sections(self):
        diff = self._make_diff()
        examples = examples_from_diff(diff)
        removed = [ex for ex in examples if ex["change_type"] == "removed_section"]
        assert len(removed) == 1

    def test_extracts_modified_sections(self):
        diff = self._make_diff()
        examples = examples_from_diff(diff)
        modified = [ex for ex in examples if ex["change_type"] == "modified_section"]
        assert len(modified) == 1
        assert "[SEP]" in modified[0]["text"]

    def test_includes_act_name(self):
        diff = self._make_diff()
        examples = examples_from_diff(diff)
        for ex in examples:
            assert ex["act_name"] == "test_act"

    def test_includes_section_id(self):
        diff = self._make_diff()
        examples = examples_from_diff(diff)
        ids = {ex["section_id"] for ex in examples}
        assert ids == {"s1", "s2", "s3"}

    def test_default_label_applied(self):
        diff = self._make_diff()
        examples = examples_from_diff(diff, default_label=1)
        for ex in examples:
            assert ex["label"] == 1

    def test_no_label_when_default_none(self):
        diff = self._make_diff()
        examples = examples_from_diff(diff, default_label=None)
        for ex in examples:
            assert "label" not in ex

    def test_empty_diff_returns_empty_list(self):
        diff = {
            "act_name": "empty",
            "added": {},
            "removed": {},
            "modified": {},
        }
        examples = examples_from_diff(diff)
        assert examples == []


# ---------------------------------------------------------------------------
# save_dataset / load_dataset
# ---------------------------------------------------------------------------

class TestSaveLoadDataset:
    def test_save_creates_file(self, tmp_path):
        examples = generate_examples(n_per_class=5)
        path = save_dataset(examples, filename="test.csv", output_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".csv"

    def test_save_creates_output_dir(self, tmp_path):
        nested = tmp_path / "nested" / "dir"
        examples = generate_examples(n_per_class=5)
        save_dataset(examples, filename="test.csv", output_dir=nested)
        assert nested.exists()

    def test_default_filename_has_timestamp(self, tmp_path):
        examples = generate_examples(n_per_class=5)
        path = save_dataset(examples, output_dir=tmp_path)
        assert path.name.startswith("labelled_dataset_")
        assert path.name.endswith(".csv")

    def test_csv_has_correct_columns(self, tmp_path):
        examples = generate_examples(n_per_class=5)
        path = save_dataset(examples, filename="test.csv", output_dir=tmp_path)

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)

        assert "text" in headers
        assert "label" in headers
        assert "change_type" in headers
        assert "act_name" in headers

    def test_load_returns_correct_count(self, tmp_path):
        examples = generate_examples(n_per_class=10)
        path = save_dataset(examples, filename="test.csv", output_dir=tmp_path)
        loaded = load_dataset(path)
        assert len(loaded) == 20

    def test_load_preserves_text(self, tmp_path):
        examples = [{"text": "Test text.", "label": 1, "change_type": "test", "act_name": "test"}]
        path = save_dataset(examples, filename="test.csv", output_dir=tmp_path)
        loaded = load_dataset(path)
        assert loaded[0]["text"] == "Test text."

    def test_load_converts_label_to_int(self, tmp_path):
        examples = [{"text": "Test.", "label": 1, "change_type": "test", "act_name": "test"}]
        path = save_dataset(examples, filename="test.csv", output_dir=tmp_path)
        loaded = load_dataset(path)
        assert isinstance(loaded[0]["label"], int)
        assert loaded[0]["label"] == 1

    def test_roundtrip_preserves_data(self, tmp_path):
        original = generate_examples(n_per_class=20, seed=42)
        path = save_dataset(original, filename="test.csv", output_dir=tmp_path)
        loaded = load_dataset(path)

        # Compare key fields (order may differ due to CSV serialization)
        for orig, load in zip(original, loaded):
            assert orig["text"] == load["text"]
            assert orig["label"] == load["label"]
            assert orig["change_type"] == load["change_type"]


# ---------------------------------------------------------------------------
# build_training_dataset
# ---------------------------------------------------------------------------

class TestBuildTrainingDataset:
    def test_creates_csv_file(self, tmp_path):
        path = build_training_dataset(n_per_class=10, output_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".csv"

    def test_correct_row_count(self, tmp_path):
        path = build_training_dataset(n_per_class=50, output_dir=tmp_path)
        loaded = load_dataset(path)
        assert len(loaded) == 100

    def test_seed_produces_reproducible_file(self, tmp_path):
        path1 = build_training_dataset(n_per_class=10, seed=42, output_dir=tmp_path)
        path2_dir = tmp_path / "second"
        path2 = build_training_dataset(n_per_class=10, seed=42, output_dir=path2_dir)

        data1 = load_dataset(path1)
        data2 = load_dataset(path2)

        for d1, d2 in zip(data1, data2):
            assert d1["text"] == d2["text"]
            assert d1["label"] == d2["label"]
