"""
Evaluation script for trained multi-class achievement standard classifier.

This script loads a trained model and evaluates it on test data.
Uses the same data preparation and evaluation functions as training script.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.train_multiclass_classifier import (
    AchievementClassifier,
    AchievementDataset,
    prepare_data,
)
from src.utils.common import detect_encoding


def load_model(
    model_dir: str,
    device: torch.device,
    num_classes: int = None,
    base_model: str = None,
    dropout: float = None,
    pooling: str = None,
):
    """
    Load trained model from directory.

    Args:
        model_dir: Path to model directory (should contain model.pt and config.json)
        device: Torch device
        num_classes: Number of classes (if not in config)
        base_model: Base model name (if not in config)
        dropout: Dropout value (if not in config)
        pooling: Pooling method (if not in config)

    Returns:
        model, tokenizer, config, mappings
    """
    model_dir = Path(model_dir)

    # Load config
    config_path = model_dir / "config.json"
    if not config_path.exists():
        # Try parent directory
        config_path = model_dir.parent / "config.json"

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        print("Warning: config.json not found. Using provided/default values.")
        config = {}

    # Load mappings (required)
    mappings_path = model_dir.parent / "label_mappings.json"
    if not mappings_path.exists():
        mappings_path = model_dir / "label_mappings.json"

    if not mappings_path.exists():
        raise FileNotFoundError(
            f"label_mappings.json not found in {model_dir} or {model_dir.parent}"
        )

    with open(mappings_path, "r", encoding="utf-8") as f:
        mappings = json.load(f)

    # Get config values (prefer config file, then provided args, then defaults)
    base_model = base_model or config.get("base_model", "klue/roberta-large")
    num_classes = (
        num_classes or config.get("num_classes") or len(mappings.get("code_to_idx", {}))
    )
    dropout = dropout if dropout is not None else config.get("dropout", 0.1)
    pooling = pooling or config.get("pooling", "cls")
    max_length = config.get("max_length", 256)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load model
    model = AchievementClassifier(
        model_name=base_model,
        num_classes=num_classes,
        dropout=dropout,
        pooling=pooling,
    )

    # Load model weights
    model_path = model_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    config["max_length"] = max_length
    config["num_classes"] = num_classes
    config["base_model"] = base_model
    config["dropout"] = dropout
    config["pooling"] = pooling

    return model, tokenizer, config, mappings


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    idx_to_code: Dict[int, str],
) -> Dict:
    """Comprehensive evaluation"""

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]

            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)

    # Top-k accuracies
    # Only calculate top-k accuracy if k <= number of classes to avoid errors
    top_k_accs = {}
    num_classes = len(idx_to_code)
    for k in [1, 3, 5, 10, 20]:
        if k <= num_classes:
            top_k_acc = top_k_accuracy_score(
                all_labels, all_probs, k=k, labels=list(range(num_classes))
            )
            top_k_accs[f"top_{k}_acc"] = top_k_acc

    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    # Macro F1 for balanced view
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Per-class metrics
    per_class_metrics = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        per_class_metrics
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_weighted": f1,
        "f1_macro": macro_f1,
        **top_k_accs,
        "per_class_precision": per_class_precision.tolist(),
        "per_class_recall": per_class_recall.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "per_class_support": per_class_support.tolist(),
    }

    return metrics, all_preds, all_labels, all_probs


def evaluate_classifier(
    model_dir: str,
    input_csv: str,
    encoding: str = None,
    test_size: float = None,
    max_samples_per_class: int = None,
    max_length: int = None,
    batch_size: int = 32,
    output_dir: str = None,
    base_model: str = None,
    dropout: float = None,
    pooling: str = None,
    save_predictions: bool = False,
):
    """
    Evaluate trained multi-class classifier on test data.

    Args:
        model_dir: Path to trained model directory
        input_csv: Path to input CSV file
        encoding: CSV encoding (auto-detect if None)
        test_size: Test split ratio (if None, use entire dataset as test)
        max_samples_per_class: Maximum samples per class (for testing)
        max_length: Max sequence length (from config if None)
        batch_size: Batch size for evaluation
        output_dir: Directory to save results
        base_model: Base model name (from config if None)
        dropout: Dropout value (from config if None)
        pooling: Pooling method (from config if None)
        save_predictions: Whether to save individual predictions
    """

    # Setup
    if output_dir is None:
        output_dir = PROJECT_ROOT / "output" / "classification" / "eval"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("MULTI-CLASS ACHIEVEMENT STANDARD CLASSIFIER - EVALUATION")
    print("=" * 80)

    # Load model
    print(f"\nLoading model from: {model_dir}")
    model, tokenizer, config, mappings = load_model(
        model_dir, device, base_model=base_model, dropout=dropout, pooling=pooling
    )

    print(f"   Model: {config['base_model']}")
    print(f"   Number of classes: {config['num_classes']}")
    print(f"   Max length: {config['max_length']}")
    print(f"   Dropout: {config['dropout']}")
    print(f"   Pooling: {config['pooling']}")
    print(f"   Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Use max_length from config if not provided
    if max_length is None:
        max_length = config["max_length"]

    # Prepare data
    print(f"\nLoading data from: {input_csv}")
    texts, labels, code_to_idx, idx_to_code, code_to_content = prepare_data(
        input_csv, encoding, max_samples_per_class
    )

    num_classes = len(code_to_idx)
    print(f"   Total samples: {len(texts)}")
    print(f"   Number of classes: {num_classes}")

    # Split data if test_size is provided
    if test_size is not None and test_size > 0:
        print(f"\nSplitting data (test_size={test_size})...")
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        print(f"   Train samples: {len(train_texts)}")
        print(f"   Test samples: {len(test_texts)}")
    else:
        print("\nUsing entire dataset for evaluation...")
        test_texts = texts
        test_labels = labels

    # Create dataset and dataloader
    test_dataset = AchievementDataset(test_texts, test_labels, tokenizer, max_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATING MODEL")
    print("=" * 80)

    metrics, all_preds, all_labels, all_probs = evaluate_model(
        model, test_loader, device, idx_to_code
    )

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\n   Main Metrics:")
    print(f"   Accuracy:        {metrics['accuracy']:.4f}")
    print(f"   F1 (weighted):   {metrics['f1_weighted']:.4f}")
    print(f"   F1 (macro):      {metrics['f1_macro']:.4f}")
    print(f"   Precision:       {metrics['precision']:.4f}")
    print(f"   Recall:          {metrics['recall']:.4f}")

    print(f"\n   Top-K Accuracies:")
    for k in [1, 3, 5, 10, 20]:
        key = f"top_{k}_acc"
        if key in metrics:
            print(f"   Top-{k:2d}:  {metrics[key]:.4f}")

    # Classification report
    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)
    target_names = [idx_to_code[i] for i in range(num_classes)]
    report = classification_report(
        all_labels, all_preds, target_names=target_names, zero_division=0
    )
    print(report)

    # Save results
    results = {
        "model_dir": str(model_dir),
        "input_csv": str(input_csv),
        "num_samples": len(test_texts),
        "num_classes": num_classes,
        "metrics": metrics,
    }

    results_path = output_dir / "eval_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n   Results saved to: {results_path}")

    # Save predictions if requested
    if save_predictions:
        predictions_df = pd.DataFrame(
            {
                "text": test_texts,
                "true_label_idx": all_labels,
                "pred_label_idx": all_preds,
                "true_code": [idx_to_code[int(idx)] for idx in all_labels],
                "pred_code": [idx_to_code[int(idx)] for idx in all_preds],
                "correct": all_labels == all_preds,
            }
        )

        # Add top-k predictions
        for k in [1, 3, 5, 10]:
            if k <= num_classes:
                top_k_indices = np.argsort(all_probs, axis=1)[:, -k:][:, ::-1]
                predictions_df[f"top{k}_codes"] = [
                    ",".join([idx_to_code[int(idx)] for idx in row])
                    for row in top_k_indices
                ]
                predictions_df[f"top{k}_probs"] = [
                    ",".join([f"{all_probs[i][idx]:.4f}" for idx in row])
                    for i, row in enumerate(top_k_indices)
                ]

        predictions_path = output_dir / "predictions.csv"
        predictions_df.to_csv(predictions_path, index=False, encoding="utf-8")
        print(f"   Predictions saved to: {predictions_path}")

    print("=" * 80)

    return metrics


def evaluate_and_select_best_checkpoint(
    output_dir: str,
    input_csv: str,
    checkpoint_epochs: List[int],
    encoding: str = None,
    test_size: float = None,
    max_samples_per_class: int = None,
    max_length: int = None,
    batch_size: int = 32,
    base_model: str = None,
    dropout: float = None,
    pooling: str = None,
    metric_for_selection: str = "f1_weighted",  # f1_weighted, f1_macro, accuracy
):
    """
    Evaluate multiple checkpoints and select the best one, then save it as best_model.

    Args:
        output_dir: Path to output directory containing checkpoints
        input_csv: Path to input CSV file for evaluation
        checkpoint_epochs: List of epoch numbers to evaluate (e.g., [8, 9, 10])
        encoding: CSV encoding (auto-detect if None)
        test_size: Test split ratio (if None, use entire dataset as test)
        max_samples_per_class: Maximum samples per class (for testing)
        max_length: Max sequence length (from config if None)
        batch_size: Batch size for evaluation
        base_model: Base model name (from config if None)
        dropout: Dropout value (from config if None)
        pooling: Pooling method (from config if None)
        metric_for_selection: Metric to use for selecting best model
                             ('f1_weighted', 'f1_macro', 'accuracy')

    Returns:
        Dictionary with best checkpoint info and all evaluation results
    """
    output_dir = Path(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("EVALUATING MULTIPLE CHECKPOINTS AND SELECTING BEST MODEL")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Checkpoints to evaluate: {checkpoint_epochs}")
    print(f"Metric for selection: {metric_for_selection}")

    # Prepare data once (will be reused for all checkpoints)
    print(f"\nLoading data from: {input_csv}")
    texts, labels, code_to_idx, idx_to_code, code_to_content = prepare_data(
        input_csv, encoding, max_samples_per_class
    )

    num_classes = len(code_to_idx)
    print(f"   Total samples: {len(texts)}")
    print(f"   Number of classes: {num_classes}")

    # Split data if test_size is provided
    if test_size is not None and test_size > 0:
        print(f"\nSplitting data (test_size={test_size})...")
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        print(f"   Train samples: {len(train_texts)}")
        print(f"   Test samples: {len(test_texts)}")
    else:
        print("\nUsing entire dataset for evaluation...")
        test_texts = texts
        test_labels = labels

    # Load config and mappings from output_dir (should be same for all checkpoints)
    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            base_config = json.load(f)
    else:
        base_config = {}

    mappings_path = output_dir / "label_mappings.json"
    if not mappings_path.exists():
        raise FileNotFoundError(f"label_mappings.json not found in {output_dir}")

    with open(mappings_path, "r", encoding="utf-8") as f:
        mappings = json.load(f)

    # Get config values
    base_model = base_model or base_config.get("base_model", "klue/roberta-large")
    dropout = dropout if dropout is not None else base_config.get("dropout", 0.1)
    pooling = pooling or base_config.get("pooling", "cls")
    if max_length is None:
        max_length = base_config.get("max_length", 256)

    # Load tokenizer (should be same for all checkpoints)
    # Try to load from first checkpoint, or from output_dir
    tokenizer = None
    for epoch in checkpoint_epochs:
        checkpoint_dir = output_dir / f"checkpoint_epoch_{epoch}"
        if checkpoint_dir.exists():
            try:
                tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
                print(f"\nLoaded tokenizer from checkpoint_epoch_{epoch}")
                break
            except:
                continue

    if tokenizer is None:
        # Try to load from best_model if exists
        best_model_dir = output_dir / "best_model"
        if best_model_dir.exists():
            try:
                tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
                print("\nLoaded tokenizer from best_model")
            except:
                pass

    if tokenizer is None:
        # Load from base model
        print(f"\nLoading tokenizer from base model: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Create dataset and dataloader (will be reused)
    test_dataset = AchievementDataset(test_texts, test_labels, tokenizer, max_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Evaluate each checkpoint
    all_results = []
    best_metric_value = -1.0
    best_checkpoint_epoch = None
    best_metrics = None

    print("\n" + "=" * 80)
    print("EVALUATING CHECKPOINTS")
    print("=" * 80)

    for epoch in checkpoint_epochs:
        checkpoint_dir = output_dir / f"checkpoint_epoch_{epoch}"
        model_path = checkpoint_dir / "model.pt"

        if not model_path.exists():
            print(f"\n⚠️  Checkpoint {epoch}: model.pt not found, skipping...")
            continue

        print(f"\n{'='*80}")
        print(f"Evaluating checkpoint_epoch_{epoch}")
        print(f"{'='*80}")

        # Load model
        model = AchievementClassifier(
            model_name=base_model,
            num_classes=num_classes,
            dropout=dropout,
            pooling=pooling,
        )

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Evaluate
        metrics, _, _, _ = evaluate_model(model, test_loader, device, idx_to_code)

        # Get metric value for selection
        metric_value = metrics.get(metric_for_selection, 0.0)

        print(f"\n   Results for checkpoint_epoch_{epoch}:")
        print(f"   Accuracy:        {metrics['accuracy']:.4f}")
        print(f"   F1 (weighted):   {metrics['f1_weighted']:.4f}")
        print(f"   F1 (macro):      {metrics['f1_macro']:.4f}")
        print(f"   {metric_for_selection}: {metric_value:.4f}")

        all_results.append(
            {
                "epoch": epoch,
                "checkpoint_dir": str(checkpoint_dir),
                "metrics": metrics,
                "metric_for_selection": metric_for_selection,
                "metric_value": metric_value,
            }
        )

        # Check if this is the best so far
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_checkpoint_epoch = epoch
            best_metrics = metrics
            print(f"   ⭐ NEW BEST! ({metric_for_selection}: {metric_value:.4f})")

    if best_checkpoint_epoch is None:
        raise ValueError("No valid checkpoints found to evaluate!")

    # Save best model
    print("\n" + "=" * 80)
    print("SELECTING AND SAVING BEST MODEL")
    print("=" * 80)
    print(f"\nBest checkpoint: checkpoint_epoch_{best_checkpoint_epoch}")
    print(f"Best {metric_for_selection}: {best_metric_value:.4f}")
    print(f"\nBest metrics:")
    print(f"   Accuracy:        {best_metrics['accuracy']:.4f}")
    print(f"   F1 (weighted):   {best_metrics['f1_weighted']:.4f}")
    print(f"   F1 (macro):      {best_metrics['f1_macro']:.4f}")

    best_checkpoint_dir = output_dir / f"checkpoint_epoch_{best_checkpoint_epoch}"
    best_model_dir = output_dir / "best_model"
    best_model_dir.mkdir(exist_ok=True)

    # Copy model.pt
    shutil.copy2(best_checkpoint_dir / "model.pt", best_model_dir / "model.pt")
    print(f"\n✓ Copied model.pt to {best_model_dir}")

    # Copy tokenizer (if exists in checkpoint)
    if (best_checkpoint_dir / "tokenizer_config.json").exists():
        # Copy all tokenizer files
        for tokenizer_file in best_checkpoint_dir.glob("tokenizer*"):
            if tokenizer_file.is_file():
                shutil.copy2(tokenizer_file, best_model_dir / tokenizer_file.name)
        print(f"✓ Copied tokenizer files to {best_model_dir}")
    else:
        # Save tokenizer from current tokenizer object
        tokenizer.save_pretrained(best_model_dir)
        print(f"✓ Saved tokenizer to {best_model_dir}")

    # Update config.json with best metrics
    config = {
        **base_config,
        "base_model": base_model,
        "num_classes": num_classes,
        "max_length": max_length,
        "dropout": dropout,
        "pooling": pooling,
        "best_f1": best_metrics["f1_weighted"],
        "best_accuracy": best_metrics["accuracy"],
        "best_checkpoint_epoch": best_checkpoint_epoch,
        "metric_for_selection": metric_for_selection,
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Save evaluation summary
    summary = {
        "best_checkpoint_epoch": best_checkpoint_epoch,
        "best_metric": metric_for_selection,
        "best_metric_value": best_metric_value,
        "best_metrics": best_metrics,
        "all_results": all_results,
    }

    summary_path = output_dir / "checkpoint_evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved evaluation summary to {summary_path}")
    print(f"✓ Updated config.json with best metrics")
    print(f"\n{'='*80}")
    print("BEST MODEL SAVED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"\nBest model location: {best_model_dir}")
    print(f"Best checkpoint: checkpoint_epoch_{best_checkpoint_epoch}")
    print(f"Best {metric_for_selection}: {best_metric_value:.4f}")
    print("=" * 80)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Multi-Class Classifier for Achievement Standard Prediction"
    )

    # Data arguments
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Path to trained model directory (required for single model evaluation)",
    )
    parser.add_argument("--input_csv", type=str, required=True, help="Input CSV file")
    parser.add_argument("--encoding", type=str, default=None, help="CSV encoding")
    parser.add_argument(
        "--test_size",
        type=float,
        default=None,
        help="Test split ratio (if None, use entire dataset as test)",
    )
    parser.add_argument("--max_samples_per_class", type=int, default=None)
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Max sequence length (from config if None)",
    )

    # Model arguments (optional, will use config if not provided)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--pooling", type=str, default=None, choices=["cls", "mean"])

    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save individual predictions to CSV",
    )

    # Multiple checkpoint evaluation
    parser.add_argument(
        "--checkpoint_epochs",
        type=int,
        nargs="+",
        default=None,
        help="List of checkpoint epochs to evaluate and select best (e.g., --checkpoint_epochs 8 9 10). "
        "If provided, will evaluate all checkpoints and save the best one as best_model.",
    )
    parser.add_argument(
        "--metric_for_selection",
        type=str,
        default="f1_weighted",
        choices=["f1_weighted", "f1_macro", "accuracy"],
        help="Metric to use for selecting best checkpoint",
    )

    args = parser.parse_args()

    # If checkpoint_epochs is provided, evaluate multiple checkpoints and select best
    if args.checkpoint_epochs:
        if args.model_dir:
            # Use model_dir as output_dir if output_dir not specified
            output_dir = args.output_dir or args.model_dir
        else:
            output_dir = args.output_dir
            if not output_dir:
                raise ValueError(
                    "Either --model_dir or --output_dir must be provided when using --checkpoint_epochs"
                )

        evaluate_and_select_best_checkpoint(
            output_dir=output_dir,
            input_csv=args.input_csv,
            checkpoint_epochs=args.checkpoint_epochs,
            encoding=args.encoding,
            test_size=args.test_size,
            max_samples_per_class=args.max_samples_per_class,
            max_length=args.max_length,
            batch_size=args.batch_size,
            base_model=args.base_model,
            dropout=args.dropout,
            pooling=args.pooling,
            metric_for_selection=args.metric_for_selection,
        )
    else:
        # Single model evaluation
        if not args.model_dir:
            parser.error("--model_dir is required when not using --checkpoint_epochs")

        evaluate_classifier(
            model_dir=args.model_dir,
            input_csv=args.input_csv,
            encoding=args.encoding,
            test_size=args.test_size,
            max_samples_per_class=args.max_samples_per_class,
            max_length=args.max_length,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            base_model=args.base_model,
            dropout=args.dropout,
            pooling=args.pooling,
            save_predictions=args.save_predictions,
        )
