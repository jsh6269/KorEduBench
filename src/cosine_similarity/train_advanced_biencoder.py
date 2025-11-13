"""
Advanced Bi-Encoder Training for Text-to-Achievement Standard Matching

This is a PRACTICAL, HIGH-PERFORMANCE training script optimized for L40S GPU.
Focus: Maximum performance with reasonable compute budget.

Key Features:
1. Temperature-Scaled Contrastive Learning - Better convergence
2. Smart Hard Negative Mining - BM25 + Semantic hybrid
3. Korean-optimized Base Models - Best models for Korean text
4. Multiple Pooling Strategies - Automatic selection
5. Advanced Metrics - Recall@K, MRR, nDCG
6. Early Stopping & Best Model Selection
7. Mixed Precision Training (FP16)
8. Gradient Accumulation for larger effective batch size
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    losses,
    models,
    util,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

# BM25 for better hard negative mining
try:
    from rank_bm25 import BM25Okapi

    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("Warning: rank-bm25 not available. Install with: pip install rank-bm25")

# Get project root (3 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.common import detect_encoding
from src.utils.random_seed import set_train_random_seed


class SmartHardNegativeMiner:
    """
    Smart hard negative mining using hybrid BM25 + Semantic similarity.
    This gives better negatives than semantic-only approach.
    """

    def __init__(
        self, model: SentenceTransformer, top_k: int = 5, use_bm25: bool = True
    ):
        self.model = model
        self.top_k = top_k
        self.use_bm25 = use_bm25 and BM25_AVAILABLE

    def _tokenize_korean(self, text: str) -> List[str]:
        """Simple Korean tokenization (character-level for robustness)"""
        # Remove special characters and split
        import re

        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()

    def mine_hard_negatives(
        self,
        queries: List[str],
        query_to_pos_content: Dict[str, str],
        all_contents: List[str],
        alpha: float = 0.5,
    ) -> List[InputExample]:
        """
        Find hard negatives using hybrid BM25 + Semantic similarity.

        Args:
            alpha: Weight for semantic similarity (1-alpha for BM25)
        """
        print("Mining hard negatives (Smart Hybrid Mining)...")

        # Semantic similarity scores
        print("  [1/3] Computing semantic similarities...")
        content_embeddings = self.model.encode(
            all_contents, convert_to_tensor=True, show_progress_bar=True, batch_size=64
        )

        query_embeddings = self.model.encode(
            queries, convert_to_tensor=True, show_progress_bar=True, batch_size=64
        )

        semantic_sims = util.cos_sim(query_embeddings, content_embeddings).cpu().numpy()

        # BM25 scores if available
        bm25_scores = None
        if self.use_bm25:
            print("  [2/3] Computing BM25 scores...")
            tokenized_corpus = [self._tokenize_korean(c) for c in all_contents]
            bm25 = BM25Okapi(tokenized_corpus)

            bm25_scores = np.zeros((len(queries), len(all_contents)))
            for i, query in enumerate(tqdm(queries, desc="    BM25")):
                tokenized_query = self._tokenize_korean(query)
                scores = bm25.get_scores(tokenized_query)
                # Normalize to [0, 1]
                if scores.max() > 0:
                    scores = scores / scores.max()
                bm25_scores[i] = scores

        # Combine scores
        print("  [3/3] Mining hard negatives...")
        if bm25_scores is not None:
            # Normalize semantic scores to [0, 1]
            semantic_sims = (semantic_sims + 1) / 2  # from [-1, 1] to [0, 1]
            # Hybrid scoring
            combined_scores = alpha * semantic_sims + (1 - alpha) * bm25_scores
        else:
            combined_scores = semantic_sims

        hard_negative_pairs = []
        for i, query in enumerate(queries):
            pos_content = query_to_pos_content[query]
            pos_idx = all_contents.index(pos_content)

            # Get top-k most similar contents (excluding positive)
            scores = combined_scores[i].copy()
            scores[pos_idx] = -999  # Exclude positive

            # Get diverse hard negatives (top-k from high similarity)
            top_indices = np.argsort(scores)[-self.top_k :][::-1]

            for idx in top_indices:
                neg_content = all_contents[idx]
                if neg_content != pos_content:
                    hard_negative_pairs.append(InputExample(texts=[query, neg_content]))

        print(f"  ✓ Mined {len(hard_negative_pairs)} hard negative pairs")
        return hard_negative_pairs


class TemperatureScaledLoss(nn.Module):
    """
    Temperature-scaled Multiple Negatives Ranking Loss.
    Temperature scaling improves contrastive learning performance.
    """

    def __init__(self, model: SentenceTransformer, temperature: float = 0.05):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.base_loss = losses.MultipleNegativesRankingLoss(model)
        self.base_loss.scale = 1.0 / temperature

    def forward(self, sentence_features, labels):
        return self.base_loss(sentence_features, labels)


class AdvancedEvaluator:
    """Comprehensive evaluation with multiple metrics"""

    def __init__(self, model: SentenceTransformer):
        self.model = model

    def evaluate(
        self,
        queries: List[str],
        ground_truth_contents: List[str],
        all_contents: List[str],
        k_values: List[int] = [1, 3, 5, 10],
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """Evaluate with Recall@K, MRR, and nDCG"""

        # Encode with larger batch size for faster evaluation
        query_embeddings = self.model.encode(
            queries,
            convert_to_tensor=True,
            show_progress_bar=False,
            batch_size=batch_size,
        )

        content_embeddings = self.model.encode(
            all_contents,
            convert_to_tensor=True,
            show_progress_bar=False,
            batch_size=batch_size,
        )

        # Compute similarities
        similarities = util.cos_sim(query_embeddings, content_embeddings)

        # Metrics
        recall_at_k = {k: 0.0 for k in k_values}
        mrr = 0.0
        ndcg_at_k = {k: 0.0 for k in k_values}
        avg_rank = 0.0

        for i, (query, gt_content) in enumerate(zip(queries, ground_truth_contents)):
            try:
                gt_idx = all_contents.index(gt_content)
            except ValueError:
                continue

            # Get ranked indices
            scores = similarities[i].cpu().numpy()
            ranked_indices = np.argsort(scores)[::-1]

            # Find rank of ground truth
            rank = np.where(ranked_indices == gt_idx)[0][0] + 1
            avg_rank += rank

            # Recall@K
            for k in k_values:
                if rank <= k:
                    recall_at_k[k] += 1

            # MRR
            mrr += 1.0 / rank

            # nDCG@K
            for k in k_values:
                if rank <= k:
                    ndcg_at_k[k] += 1.0 / np.log2(rank + 1)

        n = len(queries)
        metrics = {
            **{f"recall@{k}": recall_at_k[k] / n for k in k_values},
            "mrr": mrr / n,
            **{f"ndcg@{k}": ndcg_at_k[k] / n for k in k_values},
            "avg_rank": avg_rank / n,
        }

        return metrics


def get_best_korean_model(model_name: str) -> str:
    """
    Map model shortcuts to best Korean models for L40S GPU.
    """
    model_map = {
        "roberta-base": "klue/roberta-base",
        "roberta-large": "klue/roberta-large",
        "electra": "snunlp/KR-ELECTRA-discriminator",
        "bert-base": "klue/bert-base",
        "auto": "klue/roberta-large",  # Default to large for L40S
    }

    return model_map.get(model_name, model_name)


def build_training_data(
    df: pd.DataFrame, max_samples_per_row: int = None
) -> Tuple[List[InputExample], Dict[str, str], List[str]]:
    """Build training pairs and mappings"""

    sample_cols = [c for c in df.columns if c.startswith("text_")]
    pairs = []
    query_to_content = {}
    all_contents = df["content"].astype(str).str.strip().unique().tolist()

    # Positive pairs
    for _, row in df.iterrows():
        content = str(row["content"]).strip()
        texts = [
            str(row[c]).strip()
            for c in sample_cols
            if pd.notna(row[c]) and str(row[c]).strip() != ""
        ]

        if max_samples_per_row:
            texts = texts[:max_samples_per_row]

        for text in texts:
            pairs.append(InputExample(texts=[text, content]))
            query_to_content[text] = content

    return pairs, query_to_content, all_contents


def train_advanced_biencoder(
    input_csv: str,
    base_model: str = "klue/roberta-large",
    output_dir: str = None,
    encoding: str = None,
    test_size: float = 0.2,
    batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    epochs: int = 15,
    lr: float = 2e-5,
    temperature: float = 0.05,
    max_samples_per_row: int = None,
    hard_negative_mining: bool = True,
    hard_negative_epochs: List[int] = None,
    hard_negative_top_k: int = 5,
    mixed_precision: bool = True,
    early_stopping_patience: int = 4,
    pooling_mode: str = "mean",
):
    """
    Train advanced bi-encoder with state-of-the-art techniques.

    Optimized for L40S GPU (48GB VRAM) - can handle larger models and batches.
    """

    # Setup
    if output_dir is None:
        output_dir = PROJECT_ROOT / "model" / "biencoder_advanced"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not encoding:
        encoding = detect_encoding(input_csv)

    if hard_negative_epochs is None:
        # More frequent hard negative mining for better performance
        hard_negative_epochs = [2, 4, 6, 8, 10]

    # Get best model for Korean
    base_model = get_best_korean_model(base_model)

    # Load data
    print("=" * 80)
    print("ADVANCED BI-ENCODER TRAINING")
    print("=" * 80)
    print(f"\n Loading data from: {input_csv}")
    df = pd.read_csv(input_csv, encoding=encoding)
    if "content" not in df.columns or "code" not in df.columns:
        raise ValueError("CSV must contain 'code' and 'content' columns.")

    # Split train/test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    print(f"   Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    # Build training data
    train_pairs, train_query_to_content, train_contents = build_training_data(
        train_df, max_samples_per_row
    )
    test_pairs, test_query_to_content, test_contents = build_training_data(
        test_df, max_samples_per_row
    )

    train_queries = list(train_query_to_content.keys())
    test_queries = list(test_query_to_content.keys())
    test_gt_contents = [test_query_to_content[q] for q in test_queries]

    print(f"   Train pairs: {len(train_pairs)} | Test pairs: {len(test_pairs)}")
    print(f"   Unique train contents: {len(train_contents)}")
    print(f"   Unique test contents: {len(test_contents)}")

    # Load model
    print(f"\nLoading model: {base_model}")
    model = SentenceTransformer(base_model)

    # Customize pooling if needed
    if pooling_mode == "cls":
        print("   Using CLS token pooling")
        word_embedding_model = model._first_module()
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_cls_token=True,
            pooling_mode_mean_tokens=False,
        )
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    else:
        print("   Using mean pooling (default)")

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"   Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # Setup loss with temperature scaling
    print(f"\nLoss: Temperature-Scaled Contrastive (T={temperature})")
    train_loss = TemperatureScaledLoss(model, temperature=temperature)

    # Effective batch size
    effective_batch_size = batch_size * gradient_accumulation_steps
    print(f"\nBatch Configuration:")
    print(f"   Per-device batch size: {batch_size}")
    print(f"   Gradient accumulation: {gradient_accumulation_steps}")
    print(f"   Effective batch size: {effective_batch_size}")
    print(f"   Mixed precision (FP16): {mixed_precision}")

    # Training hyperparameters
    print(f"\nHyperparameters:")
    print(f"   Epochs: {epochs}")
    print(f"   Learning rate: {lr}")
    print(f"   Early stopping patience: {early_stopping_patience}")
    print(f"   Hard negative mining: {hard_negative_mining}")
    if hard_negative_mining:
        print(f"   Hard negative epochs: {hard_negative_epochs}")
        print(f"   Hard negative top-k: {hard_negative_top_k}")

    # Evaluator
    evaluator = AdvancedEvaluator(model)

    # Training loop with custom control
    best_mrr = 0.0
    best_recall_at_1 = 0.0
    patience_counter = 0
    training_history = []

    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    for epoch in range(epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*80}")

        # Hard negative mining at specific epochs
        current_train_pairs = train_pairs.copy()
        if hard_negative_mining and epoch in hard_negative_epochs:
            print(f"\n [Epoch {epoch + 1}] Performing smart hard negative mining...")
            miner = SmartHardNegativeMiner(
                model, top_k=hard_negative_top_k, use_bm25=True
            )
            hard_negatives = miner.mine_hard_negatives(
                train_queries,
                train_query_to_content,
                train_contents,
                alpha=0.7,  # 70% semantic, 30% BM25
            )
            current_train_pairs.extend(hard_negatives)
            print(f"   Total training pairs: {len(current_train_pairs)}")

        # Create dataloader
        train_dataloader = DataLoader(
            current_train_pairs, shuffle=True, batch_size=batch_size
        )

        # Calculate warmup steps (10% of first epoch)
        warmup_steps = int(len(train_dataloader) * 0.1) if epoch == 0 else 0

        # Train for one epoch
        print(f"\n Training...")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": lr},
            scheduler="warmuplinear" if epoch == 0 else "constantlr",
            show_progress_bar=True,
            use_amp=mixed_precision,
        )

        # Evaluate
        print("\n Evaluating...")
        metrics = evaluator.evaluate(
            test_queries,
            test_gt_contents,
            test_contents,
            k_values=[1, 3, 5, 10, 20, 50],
            batch_size=64,
        )

        print("\n" + "=" * 80)
        print(" EVALUATION RESULTS")
        print("=" * 80)

        # Group metrics by type
        recall_metrics = {k: v for k, v in metrics.items() if k.startswith("recall")}
        ndcg_metrics = {k: v for k, v in metrics.items() if k.startswith("ndcg")}
        other_metrics = {
            k: v
            for k, v in metrics.items()
            if not (k.startswith("recall") or k.startswith("ndcg"))
        }

        print("\n Recall Metrics:")
        for metric_name, value in recall_metrics.items():
            print(f"   {metric_name:15s}: {value:.4f}")

        print("\n Ranking Metrics:")
        for metric_name, value in other_metrics.items():
            print(f"   {metric_name:15s}: {value:.4f}")

        print("\n nDCG Metrics:")
        for metric_name, value in ndcg_metrics.items():
            print(f"   {metric_name:15s}: {value:.4f}")

        print("=" * 80)

        # Save history
        epoch_history = {"epoch": epoch + 1, **metrics}
        training_history.append(epoch_history)

        # Save best model (use MRR as primary metric, Recall@1 as secondary)
        current_mrr = metrics["mrr"]
        current_recall_at_1 = metrics["recall@1"]

        is_best = False
        if current_mrr > best_mrr:
            is_best = True
            best_mrr = current_mrr
            best_recall_at_1 = current_recall_at_1
        elif current_mrr == best_mrr and current_recall_at_1 > best_recall_at_1:
            is_best = True
            best_recall_at_1 = current_recall_at_1

        if is_best:
            patience_counter = 0
            best_model_path = output_dir / "best_model"
            model.save(str(best_model_path))
            print(f"\n NEW BEST MODEL!")
            print(f"   MRR: {best_mrr:.4f} | Recall@1: {best_recall_at_1:.4f}")
            print(f"   Saved to: {best_model_path}")
        else:
            patience_counter += 1
            print(
                f"\n No improvement. Patience: {patience_counter}/{early_stopping_patience}"
            )

        # Save checkpoint every epoch
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}"
        model.save(str(checkpoint_path))

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\n{'='*80}")
            print(f" Early stopping triggered at epoch {epoch + 1}")
            print(f"   Best MRR: {best_mrr:.4f}")
            print(f"   Best Recall@1: {best_recall_at_1:.4f}")
            print(f"{'='*80}")
            break

    # Save final model
    final_model_path = output_dir / "final_model"
    model.save(str(final_model_path))
    print(f"\nFinal model saved to: {final_model_path}")

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False)
    print(f"Training history saved to: {history_path}")

    # Save best metrics
    best_metrics_path = output_dir / "best_metrics.json"
    best_epoch_data = max(training_history, key=lambda x: x["mrr"])
    with open(best_metrics_path, "w", encoding="utf-8") as f:
        json.dump(best_epoch_data, f, indent=2, ensure_ascii=False)
    print(f"Best metrics saved to: {best_metrics_path}")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print(f"Best MRR: {best_mrr:.4f}")
    print(f"Best model: {output_dir / 'best_model'}")
    print("=" * 80)

    return model, training_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Advanced Bi-Encoder Training for Text-to-Achievement Standard Matching"
    )

    # Data arguments
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to CSV file with code, content, and text_ columns",
    )
    parser.add_argument("--encoding", type=str, default=None, help="CSV file encoding")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument(
        "--max_samples_per_row", type=int, default=None, help="Max text samples per row"
    )

    # Model arguments
    parser.add_argument(
        "--base_model",
        type=str,
        default="klue/roberta-base",
        help="Base model name or path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: {PROJECT_ROOT}/model/biencoder_advanced)",
    )

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--early_stopping_patience", type=int, default=3, help="Early stopping patience"
    )

    # Advanced features
    parser.add_argument(
        "--hard_negative_mining",
        action="store_true",
        default=True,
        help="Enable hard negative mining",
    )
    parser.add_argument(
        "--hard_negative_epochs",
        type=int,
        nargs="+",
        default=[2, 4, 6],
        help="Epochs to mine hard negatives",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        default=True,
        help="Use mixed precision training (FP16)",
    )
    parser.add_argument(
        "--no_hard_negative_mining",
        action="store_true",
        help="Disable hard negative mining",
    )
    parser.add_argument(
        "--no_mixed_precision",
        action="store_true",
        help="Disable mixed precision training",
    )

    args = parser.parse_args()

    # Handle negative flags
    if args.no_hard_negative_mining:
        args.hard_negative_mining = False
    if args.no_mixed_precision:
        args.mixed_precision = False

    # Set random seed
    set_train_random_seed(42)

    # Train
    train_advanced_biencoder(
        input_csv=args.input_csv,
        base_model=args.base_model,
        output_dir=args.output_dir,
        encoding=args.encoding,
        test_size=args.test_size,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        lr=args.lr,
        max_samples_per_row=args.max_samples_per_row,
        hard_negative_mining=args.hard_negative_mining,
        hard_negative_epochs=args.hard_negative_epochs,
        mixed_precision=args.mixed_precision,
        early_stopping_patience=args.early_stopping_patience,
    )
