"""
BERT fine-tuner for material-change classification.

Loads bert-base-uncased from HuggingFace and fine-tunes it as a binary
classifier on the labelled dataset. Uses 80/10/10 train/validation/test
split and tracks accuracy, precision, recall, and F1.

Saves best model checkpoint to /models and logs training progress to
/outputs/training_log.json.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from src.data_builder import load_dataset

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

# Default hyperparameters
DEFAULT_CONFIG = {
    "model_name": "bert-base-uncased",
    "max_length": 256,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "seed": 42,
}


class LegislationDataset(Dataset):
    """
    PyTorch Dataset for legislation change classification.

    Tokenizes text examples using BERT tokenizer and returns tensors
    suitable for BertForSequenceClassification.
    """

    def __init__(
        self,
        examples: list[dict[str, Any]],
        tokenizer: BertTokenizer,
        max_length: int = 256,
    ):
        """
        Initialize the dataset.

        Args:
            examples: List of dicts with 'text' and 'label' keys.
            tokenizer: BERT tokenizer instance.
            max_length: Maximum sequence length for tokenization.
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.examples[idx]
        text = example["text"]
        label = example.get("label", 0)

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def split_dataset(
    examples: list[dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Split examples into train/validation/test sets.

    Args:
        examples: Full list of labelled examples.
        train_ratio: Fraction for training (default 0.8).
        val_ratio: Fraction for validation (default 0.1).
        test_ratio: Fraction for test (default 0.1).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_examples, val_examples, test_examples).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    # First split: train vs (val + test)
    train_examples, temp_examples = train_test_split(
        examples,
        train_size=train_ratio,
        random_state=seed,
        stratify=[ex["label"] for ex in examples],
    )

    # Second split: val vs test from the remaining
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_examples, test_examples = train_test_split(
        temp_examples,
        train_size=relative_val,
        random_state=seed,
        stratify=[ex["label"] for ex in temp_examples],
    )

    logger.info(
        "Split dataset: %d train, %d val, %d test",
        len(train_examples), len(val_examples), len(test_examples),
    )
    return train_examples, val_examples, test_examples


def compute_metrics(
    labels: list[int],
    predictions: list[int],
) -> dict[str, float]:
    """
    Compute classification metrics.

    Args:
        labels: Ground truth labels.
        predictions: Model predictions.

    Returns:
        Dict with accuracy, precision, recall, and f1 scores.
    """
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def evaluate(
    model: BertForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], float]:
    """
    Evaluate model on a dataset.

    Args:
        model: The BERT model.
        dataloader: DataLoader for the evaluation set.
        device: torch device (cuda/cpu).

    Returns:
        Tuple of (metrics_dict, average_loss).
    """
    model.eval()
    all_labels = []
    all_predictions = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_labels, all_predictions)
    metrics["loss"] = avg_loss

    return metrics, avg_loss


def train(
    dataset_path: Path | str,
    config: dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Fine-tune BERT on the labelled dataset.

    Args:
        dataset_path: Path to the labelled CSV file.
        config: Hyperparameter dict. Uses DEFAULT_CONFIG if not provided.
        output_dir: Directory to save model and logs. Defaults to /models.

    Returns:
        Dict containing training history and final metrics.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    model_dir = output_dir if output_dir is not None else MODELS_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Set seed for reproducibility
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load data
    examples = load_dataset(dataset_path)
    train_examples, val_examples, test_examples = split_dataset(
        examples, seed=cfg["seed"]
    )

    # Initialize tokenizer and model
    logger.info("Loading model: %s", cfg["model_name"])
    tokenizer = BertTokenizer.from_pretrained(cfg["model_name"])
    model = BertForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=2,
    )
    model.to(device)

    # Create datasets and dataloaders
    train_dataset = LegislationDataset(train_examples, tokenizer, cfg["max_length"])
    val_dataset = LegislationDataset(val_examples, tokenizer, cfg["max_length"])
    test_dataset = LegislationDataset(test_examples, tokenizer, cfg["max_length"])

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"])

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    total_steps = len(train_loader) * cfg["num_epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Training loop
    history = {
        "config": cfg,
        "epochs": [],
        "best_val_f1": 0.0,
        "best_epoch": 0,
    }

    best_val_f1 = 0.0

    for epoch in range(cfg["num_epochs"]):
        model.train()
        total_train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if (batch_idx + 1) % 10 == 0:
                logger.debug(
                    "Epoch %d, Batch %d/%d, Loss: %.4f",
                    epoch + 1, batch_idx + 1, len(train_loader), loss.item(),
                )

        # Evaluate on validation set
        val_metrics, val_loss = evaluate(model, val_loader, device)
        avg_train_loss = total_train_loss / len(train_loader)

        epoch_stats = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        }
        history["epochs"].append(epoch_stats)

        logger.info(
            "Epoch %d/%d — Train Loss: %.4f, Val Loss: %.4f, Val F1: %.4f",
            epoch + 1, cfg["num_epochs"], avg_train_loss, val_loss, val_metrics["f1"],
        )

        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            history["best_val_f1"] = best_val_f1
            history["best_epoch"] = epoch + 1

            checkpoint_path = model_dir / "best_model"
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            logger.info("Saved best model to %s (F1: %.4f)", checkpoint_path, best_val_f1)

    # Final evaluation on test set
    test_metrics, test_loss = evaluate(model, test_loader, device)
    history["test_metrics"] = {
        "loss": test_loss,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
    }

    logger.info(
        "Test Results — Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f",
        test_metrics["accuracy"],
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["f1"],
    )

    # Save training log
    history["completed_at"] = datetime.now(timezone.utc).isoformat()
    log_path = OUTPUTS_DIR / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info("Saved training log to %s", log_path)

    return history


HF_MODEL_NAME = "bert-base-uncased"
HF_CACHE_DIR = MODELS_DIR / "hub"


def load_model(
    model_path: Path | str | None = None,
) -> tuple[BertForSequenceClassification, BertTokenizer]:
    """
    Load bert-base-uncased from HuggingFace, caching weights locally.

    On first call the weights are downloaded from HuggingFace Hub and stored
    in ``models/hub/``.  Subsequent calls are served entirely from that local
    cache with no network access required.

    Args:
        model_path: Unused.  Kept for call-site compatibility; the model is
            always sourced from HuggingFace (``bert-base-uncased``).

    Returns:
        Tuple of (model, tokenizer).
    """
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(HF_MODEL_NAME, cache_dir=HF_CACHE_DIR)
    model = BertForSequenceClassification.from_pretrained(
        HF_MODEL_NAME,
        num_labels=2,
        cache_dir=HF_CACHE_DIR,
    )

    logger.info("Loaded %s (cache: %s)", HF_MODEL_NAME, HF_CACHE_DIR)
    return model, tokenizer


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if len(sys.argv) < 2:
        print("Usage: classifier.py <dataset_path>")
        sys.exit(1)

    result = train(Path(sys.argv[1]))
    print(f"\nTraining complete. Best validation F1: {result['best_val_f1']:.4f}")
    print(f"Test F1: {result['test_metrics']['f1']:.4f}")
