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
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from tqdm import tqdm

# Get project root (3 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def evaluate_bi_cross_pipeline(
    input_csv: str,
    bi_model_name: str,
    cross_model_name: str,
    top_k: int = 20,
    encoding: str = None,
    json_path: str = None,
    max_samples_per_row: int = None,
):
    if json_path is None:
        json_path = PROJECT_ROOT / "output" / "cross_encoder" / "results_rerank.json"
    json_path = str(json_path)
    
    if not encoding:
        encoding = detect_encoding(input_csv)

    # === Load CSV ===
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

    subject = df["subject"].iloc[0] if "subject" in df.columns else "Unknown"
    num_rows = len(df)

    if max_samples_per_row is None:
        max_samples_per_row = int(max((df[sample_cols].notna().sum(axis=1)).max(), 0))
        print(f"Auto-detected max_samples_per_row = {max_samples_per_row}")

    # === Load Models ===
    print(f"\nLoading Bi-Encoder: {bi_model_name}")
    bi_model = SentenceTransformer(bi_model_name)
    bi_model.eval()

    print(f"Loading Cross-Encoder: {cross_model_name}")
    cross_model = CrossEncoder(cross_model_name)
    cross_model.model.eval()

    # === Encode Achievement Standards ===
    contents = df["content"].astype(str).tolist()
    codes = df["code"].astype(str).tolist()
    print("\nEncoding achievement standards with Bi-Encoder...")
    emb_contents = bi_model.encode(
        contents, convert_to_tensor=True, show_progress_bar=True
    )

    # === Collect Samples ===
    sample_texts, true_codes = [], []
    for _, row in df.iterrows():
        code = str(row["code"])
        texts = []
        for col in sample_cols:
            text = str(row[col]).strip()
            if text and text.lower() != "nan":
                texts.append(text)
        if len(texts) > max_samples_per_row:
            texts = texts[:max_samples_per_row]
        for t in texts:
            sample_texts.append(t)
            true_codes.append(code)

    num_samples = len(sample_texts)
    print(f"\nTotal evaluation samples: {num_samples}")

    # === Bi-Encoder Stage ===
    print("Encoding sample texts (Bi-Encoder)...")
    emb_samples = bi_model.encode(
        sample_texts, convert_to_tensor=True, show_progress_bar=True
    )
    print("Computing cosine similarities for Bi-Encoder top-k retrieval...")
    sims = util.cos_sim(emb_samples, emb_contents)
    # For each sample, get top-k candidate indices
    topk_indices = torch.topk(sims, k=top_k, dim=1).indices.cpu().numpy()

    # === Cross-Encoder Stage ===
    print(f"\nRe-ranking top-{top_k} candidates using Cross-Encoder...")
    correct_top1, correct_top3, correct_top10, correct_top20 = 0, 0, 0, 0
    reciprocal_ranks = []

    # 잘못 분류된 샘플 저장용 list
    wrong_samples = []

    for i in tqdm(range(num_samples), desc="Reranking"):
        query = sample_texts[i]
        candidates = [contents[j] for j in topk_indices[i]]
        pairs = [(query, c) for c in candidates]

        scores = cross_model.predict(pairs)
        reranked = np.argsort(scores)[::-1]
        ranked_codes = [codes[topk_indices[i][j]] for j in reranked]

        true_code = true_codes[i]
        top1_pred = ranked_codes[0]

        # Evaluate ranking metrics
        if top1_pred == true_code:
            correct_top1 += 1
        else:
            true_content = (
                contents[codes.index(true_code)] if true_code in codes else "N/A"
            )

            wrong_samples.append(
                {
                    "sample_idx": i,
                    "input_text": query,
                    "true_code": true_code,
                    "pred_code": top1_pred,
                    "true_content": true_content,
                    "pred_content": candidates[reranked[0]],
                }
            )

        if true_code in ranked_codes[:3]:
            correct_top3 += 1
        if true_code in ranked_codes[:10]:
            correct_top10 += 1
        if true_code in ranked_codes[:20]:
            correct_top20 += 1
        if true_code in ranked_codes:
            rank = ranked_codes.index(true_code) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

    # === Metrics ===
    acc_top1 = correct_top1 / num_samples
    acc_top3 = correct_top3 / num_samples
    acc_top10 = correct_top10 / num_samples
    acc_top20 = correct_top20 / num_samples
    mrr = np.mean(reciprocal_ranks)

    # === Summary ===
    print("\n=== Bi-Encoder + Cross-Encoder Reranking Evaluation ===")
    print(f"Bi-Encoder: {bi_model_name}")
    print(f"Cross-Encoder: {cross_model_name}")
    print(f"Subject: {subject}")
    print(f"Samples evaluated: {num_samples}")
    print(f"Top-1 Accuracy: {acc_top1:.4f}")
    print(f"Top-3 Accuracy: {acc_top3:.4f}")
    print(f"Top-10 Accuracy: {acc_top10:.4f}")
    print(f"Top-20 Accuracy: {acc_top20:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")

    # === JSON Logging ===
    result_entry = {}
    if folder_name:
        result_entry["folder"] = folder_name

    result_entry.update(
        {
            "bi_model": bi_model_name,
            "cross_model": cross_model_name,
            "subject": subject,
            "num_standards": num_rows,
            "max_samples_per_row": int(max_samples_per_row),
            "total_samples": num_samples,
            "top_k": top_k,
            "top1_acc": round(float(acc_top1), 4),
            "top3_acc": round(float(acc_top3), 4),
            "top10_acc": round(float(acc_top10), 4),
            "top20_acc": round(float(acc_top20), 4),
            "mrr": round(float(mrr), 4),
        }
    )

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except json.JSONDecodeError:
            results = []
    else:
        results = []

    replaced = False
    for i, r in enumerate(results):
        if (
            r["bi_model"] == result_entry["bi_model"]
            and r["cross_model"] == result_entry["cross_model"]
            and r["subject"] == result_entry["subject"]
            and r["num_standards"] == result_entry["num_standards"]
            and r["total_samples"] == result_entry["total_samples"]
        ):
            results[i] = result_entry
            replaced = True
            print(f"Updated existing entry for subject '{subject}'")
            break

    if not replaced:
        results.append(result_entry)
        print(f"Appended new entry for subject '{subject}'")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {json_path}")

    # 잘못 분류된 샘플 랜덤 100개 저장
    if wrong_samples:
        logs_dir = PROJECT_ROOT / "output" / "cross_encoder" / "logs"
        os.makedirs(logs_dir, exist_ok=True)
        csv_name = os.path.splitext(os.path.basename(input_csv))[0]
        wrong_path = logs_dir / f"{csv_name}_wrongs.txt"

        sampled_wrongs = random.sample(wrong_samples, min(100, len(wrong_samples)))
        with open(wrong_path, "w", encoding="utf-8") as f:
            f.write(f"Total wrong samples: {len(wrong_samples)}\n\n")
            for w in sampled_wrongs:
                f.write(f"[Sample #{w['sample_idx']}]\n")
                f.write(f"True Code: {w['true_code']}\n")
                f.write(f"Pred Code: {w['pred_code']}\n")
                f.write(f"Input Text: {w['input_text']}\n")
                f.write(f"True Content: {w['true_content']}\n")
                f.write(f"Pred Content: {w['pred_content']}\n")
                f.write("-" * 60 + "\n")

        print(
            f"\nSaved {len(sampled_wrongs)} randomly selected wrong samples to {wrong_path}"
        )

    return acc_top1, acc_top3, acc_top10, mrr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Bi-Encoder + Cross-Encoder reranking pipeline."
    )
    parser.add_argument(
        "--input_csv", type=str, required=True, help="Path to input CSV file."
    )
    parser.add_argument(
        "--bi_model",
        type=str,
        default="jhgan/ko-sroberta-multitask",
        help="Bi-Encoder model name (default: jhgan/ko-sroberta-multitask).",
    )
    parser.add_argument(
        "--cross_model",
        type=str,
        default=None,
        help="Cross-Encoder model name (default: {PROJECT_ROOT}/model/cross_finetuned).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of top candidates for reranking (default: 20).",
    )
    parser.add_argument(
        "--encoding", type=str, help="CSV encoding (default: auto-detect)."
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default=None,
        help="Path to JSON log file (default: {PROJECT_ROOT}/output/cross_encoder/results_rerank.json).",
    )
    parser.add_argument(
        "--max-samples-per-row",
        type=int,
        default=None,
        help="Max number of text samples to evaluate per row (default: auto-detect).",
    )
    args = parser.parse_args()

    # Handle default values
    cross_model = args.cross_model
    if cross_model is None:
        cross_model = str(PROJECT_ROOT / "model" / "cross_finetuned")

    set_seed(42)
    evaluate_bi_cross_pipeline(
        args.input_csv,
        args.bi_model,
        cross_model,
        top_k=args.top_k,
        encoding=args.encoding,
        json_path=args.json_path,
        max_samples_per_row=args.max_samples_per_row,
    )
