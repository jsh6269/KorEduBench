import argparse
import json
import os
import random
import re
from pathlib import Path

import chardet
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from utils.random_seed import set_predict_random_seed

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


def evaluate_cosine_similarity_baseline(
    input_csv: str,
    model_name: str,
    encoding: str | None,
    json_path: str = None,
    max_samples_per_row: int = None,
):
    if json_path is None:
        json_path = PROJECT_ROOT / "output" / "cosine_similarity" / "results.json"
    json_path = str(json_path)
    if not encoding:
        encoding = detect_encoding(input_csv)

    # Load CSV
    df = pd.read_csv(input_csv, encoding=encoding)
    required_cols = ["code", "content"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Detect parent folder name (e.g., train / valid / test)
    parent_folder = os.path.basename(os.path.dirname(os.path.abspath(input_csv)))
    folder_name = None
    if re.search(r"(train|valid|val|test)", parent_folder, re.IGNORECASE):
        folder_name = parent_folder

    sample_cols = [c for c in df.columns if c.startswith("text_")]
    if not sample_cols:
        raise ValueError("No text_ columns found for evaluation.")

    # Extract meta info
    subject = df["subject"].iloc[0] if "subject" in df.columns else "Unknown"
    num_rows = len(df)

    # Auto compute max_samples_per_row if None
    if max_samples_per_row is None:
        max_samples_per_row = int(max((df[sample_cols].notna().sum(axis=1)).max(), 0))
        print(f"Auto-detected max_samples_per_row = {max_samples_per_row}")

    # Load embedding model
    print(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)
    model.eval()

    # Encode achievement standards
    print("Encoding achievement standards...")
    contents = df["content"].astype(str).tolist()
    codes = df["code"].astype(str).tolist()
    emb_contents = model.encode(
        contents, convert_to_tensor=True, show_progress_bar=True
    )

    # Flatten sample texts and true codes
    sample_texts, true_codes = [], []
    for _, row in df.iterrows():
        code = str(row["code"])
        texts = []
        for col in sample_cols:
            text = str(row[col]).strip()
            if text and text.lower() != "nan":
                texts.append(text)

        # Apply max_samples_per_row
        if len(texts) > max_samples_per_row:
            texts = texts[:max_samples_per_row]

        for t in texts:
            sample_texts.append(t)
            true_codes.append(code)

    num_samples = len(sample_texts)
    print(f"Total evaluation samples: {num_samples}")

    # Encode all sample texts
    print("Encoding sample texts...")
    emb_samples = model.encode(
        sample_texts, convert_to_tensor=True, show_progress_bar=True
    )

    # Compute cosine similarity matrix
    print("Computing cosine similarity matrix...")
    sims = util.cos_sim(emb_samples, emb_contents)

    # Top-k accuracy
    topk_list = [1, 3, 10, 20, 40, 60]
    acc_dict = {}

    for k in topk_list:
        topk_indices = torch.topk(sims, k=k, dim=1).indices.cpu().numpy()
        correct = sum(
            t in [codes[i] for i in idxs] for t, idxs in zip(true_codes, topk_indices)
        )
        acc_dict[f"top{k}_acc"] = correct / len(true_codes)

    # Mean Reciprocal Rank (MRR)
    code_to_index = {c: i for i, c in enumerate(codes)}
    reciprocal_ranks = []
    for t, sim_row in zip(true_codes, sims):
        if t not in code_to_index:
            reciprocal_ranks.append(0)
            continue
        true_idx = code_to_index[t]
        sorted_indices = torch.argsort(sim_row, descending=True)
        rank = (sorted_indices == true_idx).nonzero(as_tuple=True)[0].item() + 1
        reciprocal_ranks.append(1 / rank)
    mrr = np.mean(reciprocal_ranks)

    # Print summary
    print("\n=== Cosine Similarity Baseline Evaluation ===")
    print(f"Model: {model_name}")
    print(f"Subject: {subject}")
    print(f"Samples evaluated: {num_samples}")

    for k in topk_list:
        print(f"Top-{k} Accuracy: {acc_dict[f'top{k}_acc']:.4f}")

    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"Max Samples per Row: {max_samples_per_row}")

    result_entry = {}

    # Add folder name if detected
    if folder_name:
        result_entry["folder"] = folder_name

    # JSON logging
    result_entry.update(
        {
            "model_name": model_name,
            "subject": subject,
            "num_standards": num_rows,
            "max_samples_per_row": int(max_samples_per_row),
            "total_samples": num_samples,
            **{k: round(float(v), 4) for k, v in acc_dict.items()},
            "mrr": round(float(mrr), 4),
        }
    )

    # Load existing JSON if exists
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except json.JSONDecodeError:
            results = []
    else:
        results = []

    # Update or append
    replaced = False
    for i, r in enumerate(results):
        if (
            r["model_name"] == result_entry["model_name"]
            and r["subject"] == result_entry["subject"]
            and r["num_standards"] == result_entry["num_standards"]
            and r["total_samples"] == result_entry["total_samples"]
            and r.get("max_samples_per_row") == result_entry["max_samples_per_row"]
        ):
            results[i] = result_entry  # 덮어쓰기
            replaced = True
            print(f"Updated existing entry for subject '{subject}' in {json_path}")
            break

    if not replaced:
        results.append(result_entry)
        print(f"Appended new entry for subject '{subject}' to {json_path}")

    # Save file
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate cosine-similarity-based curriculum mapping accuracy."
    )
    parser.add_argument(
        "--input_csv", type=str, required=True, help="Path to input CSV file."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="jhgan/ko-sroberta-multitask",
        help="SentenceTransformer model name (default: jhgan/ko-sroberta-multitask).",
    )
    parser.add_argument(
        "--encoding", type=str, help="CSV encoding (default: auto-detect)."
    )
    parser.add_argument(
        "--json_path", type=str, default=None, help="Path to JSON log file (default: {PROJECT_ROOT}/output/cosine_similarity/results.json)."
    )
    parser.add_argument(
        "--max-samples-per-row",
        type=int,
        default=None,
        help="Maximum number of text samples to evaluate per row (default: auto-detect).",
    )
    args = parser.parse_args()

    set_predict_random_seed(42)
    evaluate_cosine_similarity_baseline(
        args.input_csv,
        args.model_name,
        encoding=args.encoding,
        json_path=args.json_path,
        max_samples_per_row=args.max_samples_per_row,
    )
