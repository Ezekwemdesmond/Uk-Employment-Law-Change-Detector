"""
Confidence scorer for material-change classification.

Runs inference on new preprocessed diffs, outputs probability scores
between 0 and 1, and flags predictions below 0.75 confidence for
human review.

Results are saved to /outputs/predictions.json.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer

from src.classifier import load_model
from src.data_builder import examples_from_diff

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

# Confidence threshold for human review flagging
CONFIDENCE_THRESHOLD = 0.75

# Label names for output
LABEL_NAMES = {0: "minor", 1: "material"}


class Scorer:
    """
    Inference scorer for material-change classification.

    Wraps a trained BERT model and provides probability-scored predictions
    with confidence flagging.
    """

    def __init__(
        self,
        model: BertForSequenceClassification | None = None,
        tokenizer: BertTokenizer | None = None,
        model_path: Path | str | None = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        max_length: int = 256,
    ):
        """
        Initialize the scorer.

        Args:
            model: Pre-loaded BERT model. If None, loads from model_path.
            tokenizer: Pre-loaded tokenizer. If None, loads from model_path.
            model_path: Path to saved model directory.
            confidence_threshold: Threshold below which to flag for review.
            max_length: Maximum sequence length for tokenization.
        """
        if model is None or tokenizer is None:
            model, tokenizer = load_model(model_path)

        self.model = model
        self.tokenizer = tokenizer
        self.confidence_threshold = confidence_threshold
        self.max_length = max_length

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "Scorer initialized on %s with confidence threshold %.2f",
            self.device, confidence_threshold,
        )

    def predict_single(self, text: str) -> dict[str, Any]:
        """
        Run inference on a single text.

        Args:
            text: The text to classify.

        Returns:
            Dict with keys:
              - predicted_label: int (0 or 1)
              - predicted_class: str ("minor" or "material")
              - confidence: float (probability of predicted class)
              - probability_material: float (probability of class 1)
              - needs_review: bool (True if confidence < threshold)
        """
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            probabilities = F.softmax(outputs.logits, dim=1)

        probs = probabilities.squeeze().cpu().tolist()
        prob_minor, prob_material = probs[0], probs[1]

        predicted_label = 1 if prob_material > 0.5 else 0
        confidence = prob_material if predicted_label == 1 else prob_minor

        return {
            "predicted_label": predicted_label,
            "predicted_class": LABEL_NAMES[predicted_label],
            "confidence": round(confidence, 4),
            "probability_material": round(prob_material, 4),
            "needs_review": confidence < self.confidence_threshold,
        }

    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = 16,
    ) -> list[dict[str, Any]]:
        """
        Run inference on a batch of texts.

        Args:
            texts: List of texts to classify.
            batch_size: Batch size for inference.

        Returns:
            List of prediction dicts (same structure as predict_single).
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            encoding = self.tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                probabilities = F.softmax(outputs.logits, dim=1)

            for j, probs in enumerate(probabilities.cpu().tolist()):
                prob_minor, prob_material = probs[0], probs[1]
                predicted_label = 1 if prob_material > 0.5 else 0
                confidence = prob_material if predicted_label == 1 else prob_minor

                results.append({
                    "predicted_label": predicted_label,
                    "predicted_class": LABEL_NAMES[predicted_label],
                    "confidence": round(confidence, 4),
                    "probability_material": round(prob_material, 4),
                    "needs_review": confidence < self.confidence_threshold,
                })

        return results

    def predict_examples(
        self,
        examples: list[dict[str, Any]],
        batch_size: int = 16,
    ) -> list[dict[str, Any]]:
        """
        Run inference on a list of example dicts.

        Merges prediction results into the example dicts.

        Args:
            examples: List of dicts with at least a 'text' key.
            batch_size: Batch size for inference.

        Returns:
            List of examples with prediction fields added.
        """
        texts = [ex["text"] for ex in examples]
        predictions = self.predict_batch(texts, batch_size)

        results = []
        for example, pred in zip(examples, predictions):
            merged = {**example, **pred}
            results.append(merged)

        return results

    def predict_diff(
        self,
        diff: dict[str, Any],
        batch_size: int = 16,
    ) -> dict[str, Any]:
        """
        Run inference on a preprocessed diff object.

        Args:
            diff: Preprocessed diff from preprocessor.preprocess_diff().
            batch_size: Batch size for inference.

        Returns:
            Dict with:
              - act_name: str
              - timestamp: str
              - predictions: list of prediction dicts per section
              - summary: counts of material, minor, needs_review
        """
        examples = examples_from_diff(diff)
        predictions = self.predict_examples(examples, batch_size)

        # Compute summary
        material_count = sum(1 for p in predictions if p["predicted_label"] == 1)
        minor_count = sum(1 for p in predictions if p["predicted_label"] == 0)
        review_count = sum(1 for p in predictions if p["needs_review"])

        return {
            "act_name": diff.get("act_name", "unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "predictions": predictions,
            "summary": {
                "total": len(predictions),
                "material_changes": material_count,
                "minor_changes": minor_count,
                "needs_review": review_count,
            },
        }


def save_predictions(
    predictions: dict[str, Any] | list[dict[str, Any]],
    filename: str | None = None,
    output_dir: Path | None = None,
) -> Path:
    """
    Save prediction results to a JSON file.

    Args:
        predictions: Prediction dict or list of prediction dicts.
        filename: Output filename. Defaults to timestamped name.
        output_dir: Directory to save to. Defaults to /outputs.

    Returns:
        Path to the saved JSON file.
    """
    target_dir = output_dir if output_dir is not None else OUTPUTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"predictions_{timestamp}.json"

    filepath = target_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    logger.info("Saved predictions to %s", filepath)
    return filepath


def score_diff(
    diff: dict[str, Any],
    model_path: Path | str | None = None,
    save: bool = True,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Convenience function to score a preprocessed diff.

    Args:
        diff: Preprocessed diff from preprocessor.preprocess_diff().
        model_path: Path to trained model. Defaults to models/best_model.
        save: Whether to save predictions to JSON.
        output_dir: Override output directory.

    Returns:
        Prediction results dict.
    """
    scorer = Scorer(model_path=model_path)
    results = scorer.predict_diff(diff)

    if save:
        save_predictions(results, output_dir=output_dir)

    logger.info(
        "Scored %d sections: %d material, %d minor, %d need review",
        results["summary"]["total"],
        results["summary"]["material_changes"],
        results["summary"]["minor_changes"],
        results["summary"]["needs_review"],
    )

    return results


def score_texts(
    texts: list[str],
    model_path: Path | str | None = None,
    save: bool = True,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Convenience function to score a list of raw texts.

    Args:
        texts: List of text strings to classify.
        model_path: Path to trained model. Defaults to models/best_model.
        save: Whether to save predictions to JSON.
        output_dir: Override output directory.

    Returns:
        List of prediction dicts.
    """
    scorer = Scorer(model_path=model_path)
    results = scorer.predict_batch(texts)

    if save:
        output = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "predictions": [
                {"text": text, **pred}
                for text, pred in zip(texts, results)
            ],
        }
        save_predictions(output, output_dir=output_dir)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Demo with sample texts
    sample_texts = [
        "The maximum penalty for non-compliance has been increased from £5,000 to £20,000.",
        "In section 12, for 'section 5' substitute 'section 6'.",
        "The employer must now provide written particulars within 14 days.",
        "The semicolon at the end of paragraph (a) is replaced with a full stop.",
    ]

    results = score_texts(sample_texts, save=False)

    for text, pred in zip(sample_texts, results):
        status = "REVIEW" if pred["needs_review"] else "OK"
        print(f"[{status}] {pred['predicted_class']} ({pred['confidence']:.2%}): {text[:60]}...")
