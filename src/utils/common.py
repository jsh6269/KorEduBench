import random

import chardet
import pandas as pd
from sentence_transformers import InputExample


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


def build_pairs_from_df(df, max_samples_per_row=None, neg_ratio=1.0, use_labels=True):
    """Build positive and negative pairs for training.

    Args:
        df: DataFrame with 'content' column and 'text_' columns
        max_samples_per_row: Maximum number of text samples to use per row
        neg_ratio: Ratio of negative to positive samples
        use_labels: If True, add labels (1.0/0.0) to InputExample for cross-encoder
                   If False, no labels for dual-encoder

    Returns:
        List of InputExample objects
    """
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
                if use_labels:
                    pairs.append(InputExample(texts=[t, content], label=1.0))
                else:
                    pairs.append(InputExample(texts=[t, content]))
                pos_keys.add(key)

    # Negative pairs
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
            if use_labels:
                pairs.append(InputExample(texts=[t, neg_content], label=0.0))
            else:
                pairs.append(InputExample(texts=[t, neg_content]))
            neg_keys.add(key)
        tries += 1

    random.shuffle(pairs)
    return pairs
