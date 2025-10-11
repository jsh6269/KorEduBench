import argparse
import os
import random
import sys
from pathlib import Path

import chardet
import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Get project root (3 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.random_seed import set_train_random_seed
from src.utils.common import build_pairs_from_df, detect_encoding


def fine_tune_dual_encoder(
    input_csv,
    base_model="klue/roberta-base",
    output_dir=None,
    encoding=None,
    test_size=0.2,
    batch_size=16,
    epochs=2,
    lr=2e-5,
    max_samples_per_row=None,
):
    if output_dir is None:
        output_dir = PROJECT_ROOT / "model" / "biencoder_finetuned"
    output_dir = str(output_dir)
    
    if not encoding:
        encoding = detect_encoding(input_csv)

    df = pd.read_csv(input_csv, encoding=encoding)
    if "content" not in df.columns or "code" not in df.columns:
        raise ValueError("CSV must contain 'code' and 'content' columns.")

    # Split train/test
    row_train, row_test = train_test_split(df, test_size=test_size, random_state=42)
    print(f"Train rows: {len(row_train)} | Test rows: {len(row_test)}")

    # Build pairs
    print("Building sentence pairs for train/test...")
    train_pairs = build_pairs_from_df(row_train, max_samples_per_row, neg_ratio=1.0, use_labels=False)
    test_pairs = build_pairs_from_df(row_test, max_samples_per_row, neg_ratio=0.5, use_labels=False)

    print(f"Train pairs: {len(train_pairs)} | Test pairs: {len(test_pairs)}")

    # Model
    model = SentenceTransformer(base_model)
    train_dataloader = DataLoader(train_pairs, shuffle=True, batch_size=batch_size)
    warmup_steps = int(len(train_dataloader) * epochs * 0.1)

    # MultipleNegativesRankingLoss (Contrastive)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Evaluator
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        test_pairs, name="biencoder-eval"
    )

    print("\nFine-tuning Bi-Encoder (Dual Encoder)...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        optimizer_params={"lr": lr},
        evaluation_steps=max(500, len(train_dataloader) // 2),
        show_progress_bar=True,
    )

    print(f"\nSaved fine-tuned model to: {output_dir}")

    # Evaluation (cosine similarity based)
    test_texts = [ex.texts[0] for ex in test_pairs]
    test_standards = [ex.texts[1] for ex in test_pairs]

    emb_texts = model.encode(test_texts, convert_to_tensor=True)
    emb_contents = model.encode(test_standards, convert_to_tensor=True)
    sims = util.cos_sim(emb_texts, emb_contents)
    diag_scores = sims.diag().cpu().numpy()
    print(f"\nMean self-similarity (diagonal mean): {np.mean(diag_scores):.4f}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune Dual (Bi) Encoder on CSV dataset (code-text mapping)."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to CSV file with code, content, and text_ columns.",
    )
    parser.add_argument("--base_model", type=str, default="klue/roberta-base")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: {PROJECT_ROOT}/model/biencoder_finetuned)")
    parser.add_argument("--encoding", type=str)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_samples_per_row", type=int, default=None)
    args = parser.parse_args()

    set_train_random_seed(42)
    fine_tune_dual_encoder(
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
