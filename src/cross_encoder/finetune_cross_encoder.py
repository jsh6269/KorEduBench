import argparse
import os
import random
import sys
from pathlib import Path

import chardet
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, InputExample
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Get project root (3 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.common import build_pairs_from_df, detect_encoding
from src.utils.random_seed import set_train_random_seed


def fine_tune_cross_encoder(
    input_csv,
    base_model="bongsoo/albert-small-kor-cross-encoder-v1",
    output_dir=None,
    encoding=None,
    test_size=0.2,
    batch_size=8,
    epochs=2,
    lr=2e-5,
    max_samples_per_row=None,
):
    if output_dir is None:
        output_dir = PROJECT_ROOT / "model" / "cross_finetuned"
    output_dir = str(output_dir)

    if not encoding:
        encoding = detect_encoding(input_csv)

    df = pd.read_csv(input_csv, encoding=encoding)
    if "content" not in df.columns or "code" not in df.columns:
        raise ValueError("CSV must contain 'code' and 'content' columns.")

    # Split by rows first
    row_train, row_test = train_test_split(df, test_size=test_size, random_state=42)
    print(f"Train rows: {len(row_train)} | Test rows: {len(row_test)}")

    # Build pairs within each split
    print("Building sentence pairs for train/test...")
    train_pairs = build_pairs_from_df(
        row_train, max_samples_per_row, neg_ratio=1.0, use_labels=True
    )
    test_pairs = build_pairs_from_df(
        row_test, max_samples_per_row, neg_ratio=1.0, use_labels=True
    )

    print(f"Train pairs: {len(train_pairs)} | Test pairs: {len(test_pairs)}")

    # Model
    model = CrossEncoder(base_model, num_labels=1)
    train_dataloader = torch.utils.data.DataLoader(
        train_pairs, shuffle=True, batch_size=batch_size
    )
    warmup_steps = int(len(train_dataloader) * epochs * 0.1)

    # Train
    print("\nFine-tuning Cross-Encoder...")
    model.fit(
        train_dataloader=train_dataloader,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        optimizer_params={"lr": lr},
        show_progress_bar=True,
    )

    # Evaluate
    print("\nEvaluating on test set...")
    test_texts = [ex.texts for ex in test_pairs]
    y_true = [ex.label for ex in test_pairs]
    y_pred = model.predict(test_texts)

    y_pred_bin = [1 if p >= 0.5 else 0 for p in y_pred]
    acc = accuracy_score(y_true, y_pred_bin)
    f1 = f1_score(y_true, y_pred_bin)
    auc = roc_auc_score(y_true, y_pred)

    print("\n=== Cross-Encoder Fine-Tuning Results ===")
    print(f"Base Model: {base_model}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)
    print(f"Saved fine-tuned model to: {output_dir}")

    return acc, f1, auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune CrossEncoder on CSV dataset (code-text mapping)."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to CSV file with code, content, and text_ columns.",
    )
    parser.add_argument(
        "--base_model", type=str, default="bongsoo/albert-small-kor-cross-encoder-v1"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: {PROJECT_ROOT}/model/cross_finetuned)",
    )
    parser.add_argument("--encoding", type=str)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_samples_per_row", type=int, default=None)
    args = parser.parse_args()

    set_train_random_seed(42)
    fine_tune_cross_encoder(
        args.input_csv,
        base_model=args.base_model,
        output_dir=args.output_dir,
        encoding=args.encoding,
        test_size=args.test_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        max_samples_per_row=args.max_samples_per_row,
    )
