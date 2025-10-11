import argparse
import os
import random
from pathlib import Path

import chardet
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, InputExample
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from utils.random_seed import set_train_random_seed

# Get project root (3 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def detect_encoding(csv_path: str) -> str:
    """Detect encoding using chardet (fallback to utf-8)."""
    try:
        with open(csv_path, "rb") as f:
            result = chardet.detect(f.read(50000))
        enc = result.get("encoding") or "utf-8"
        conf = result.get("confidence", 0)
        return enc if conf >= 0.5 else "utf-8"
    except Exception:
        return "utf-8"


def build_pairs_from_df(df, max_samples_per_row=None, neg_ratio=1.0):
    """Build positive and negative pairs within one split (train/test)."""
    sample_cols = [c for c in df.columns if c.startswith("text_")]
    pairs = []

    # Positive pairs
    pos_keys = set()
    for _, row in df.iterrows():
        content = str(row["content"]).strip()
        texts = [
            str(row[c]).strip()
            for c in sample_cols
            if pd.notna(row[c]) and str(row[c]).strip() != ""
        ]
        if max_samples_per_row:
            texts = texts[:max_samples_per_row]
        for t in texts:
            key = (t, content)
            if key not in pos_keys:
                pairs.append(InputExample(texts=[t, content], label=1.0))
                pos_keys.add(key)

    # Negative pairs (within this df only)
    contents = df["content"].astype(str).str.strip().tolist()
    all_texts = []
    for _, row in df.iterrows():
        for c in sample_cols:
            if pd.notna(row[c]) and str(row[c]).strip() != "":
                all_texts.append(str(row[c]).strip())

    num_neg = int(len(pos_keys) * neg_ratio)
    neg_keys = set()
    tries = 0
    while len(neg_keys) < num_neg and tries < num_neg * 20:
        t = random.choice(all_texts)
        neg_content = random.choice(contents)
        if (t, neg_content) in pos_keys:
            tries += 1
            continue
        key = (t, neg_content)
        if key not in neg_keys:
            pairs.append(InputExample(texts=[t, neg_content], label=0.0))
            neg_keys.add(key)
        tries += 1

    random.shuffle(pairs)
    return pairs


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
    train_pairs = build_pairs_from_df(row_train, max_samples_per_row, neg_ratio=1.0)
    test_pairs = build_pairs_from_df(row_test, max_samples_per_row, neg_ratio=1.0)

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
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: {PROJECT_ROOT}/model/cross_finetuned)")
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
