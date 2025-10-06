import argparse
import random

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_cosine_similarity_baseline(
    input_csv: str, model_name: str, encoding: str = "cp949"
):
    # Load CSV
    df = pd.read_csv(input_csv, encoding=encoding)
    required_cols = ["code", "content"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    sample_cols = [c for c in df.columns if c.startswith("text_")]
    if not sample_cols:
        raise ValueError("No text_ columns found for evaluation.")

    # Load embedding model
    print(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)
    model.eval()

    # Encode achievement standards (content)
    contents = df["content"].astype(str).tolist()
    codes = df["code"].astype(str).tolist()
    print("Encoding achievement standards...")
    emb_contents = model.encode(
        contents, convert_to_tensor=True, show_progress_bar=True
    )

    # Flatten sample texts and true codes
    sample_texts, true_codes = [], []
    for _, row in df.iterrows():
        code = str(row["code"])
        for col in sample_cols:
            text = str(row[col]).strip()
            if text and text.lower() != "nan":
                sample_texts.append(text)
                true_codes.append(code)

    print(f"Total evaluation samples: {len(sample_texts)}")

    # Encode all sample texts
    print("Encoding sample texts...")
    emb_samples = model.encode(
        sample_texts, convert_to_tensor=True, show_progress_bar=True
    )

    # 6. Compute cosine similarity matrix
    print("Computing cosine similarity matrix...")
    sims = util.cos_sim(emb_samples, emb_contents)  # [num_samples, num_codes]

    # 7. Evaluate metrics
    preds_top1 = torch.argmax(sims, dim=1).cpu().numpy()
    predicted_codes = [codes[i] for i in preds_top1]

    correct_top1 = np.sum([p == t for p, t in zip(predicted_codes, true_codes)])
    acc_top1 = correct_top1 / len(true_codes)

    # Top-3 accuracy
    topk = 3
    topk_indices = torch.topk(sims, k=topk, dim=1).indices.cpu().numpy()
    correct_top3 = sum(
        t in [codes[i] for i in idxs] for t, idxs in zip(true_codes, topk_indices)
    )
    acc_top3 = correct_top3 / len(true_codes)

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

    # 8. Print summary
    print("\n=== Cosine Similarity Baseline Evaluation ===")
    print(f"Model: {model_name}")
    print(f"Samples evaluated: {len(true_codes)}")
    print(f"Top-1 Accuracy: {acc_top1:.4f}")
    print(f"Top-3 Accuracy: {acc_top3:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")

    return acc_top1, acc_top3, mrr


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
        "--encoding", type=str, default="cp949", help="CSV encoding (default: cp949)."
    )
    args = parser.parse_args()

    set_seed(42)
    evaluate_cosine_similarity_baseline(
        args.input_csv, args.model_name, encoding=args.encoding
    )
