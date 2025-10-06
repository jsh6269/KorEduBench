import argparse
import os

from eval_cosine_similarity import evaluate_cosine_similarity_baseline, set_seed
from tqdm import tqdm


def evaluate_folder(
    folder_path: str,
    model_name: str,
    encoding: str,
    json_path: str,
    max_samples_per_row: int = None,
):
    # --- Find all CSV files in folder ---
    csv_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".csv")
    ]

    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return

    print(f"Found {len(csv_files)} CSV files in {folder_path}")

    # --- Evaluate each CSV ---
    for csv_path in tqdm(csv_files, desc="Evaluating CSV files", unit="file"):
        try:
            print(f"\n=== Processing file: {os.path.basename(csv_path)} ===")
            evaluate_cosine_similarity_baseline(
                input_csv=csv_path,
                model_name=model_name,
                encoding=encoding,
                json_path=json_path,
                max_samples_per_row=max_samples_per_row,
            )
        except Exception as e:
            print(f"Error while evaluating {csv_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch evaluation of cosine-similarity-based curriculum mapping for all CSV files in a folder."
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        required=True,
        help="Path to folder containing CSV files.",
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
    parser.add_argument(
        "--json_path", type=str, default="results.json", help="Path to JSON log file."
    )
    parser.add_argument(
        "--max-samples-per-row",
        type=int,
        default=None,
        help="Maximum number of text samples to evaluate per row (default: auto-detect).",
    )
    args = parser.parse_args()

    set_seed(42)
    evaluate_folder(
        folder_path=args.folder_path,
        model_name=args.model_name,
        encoding=args.encoding,
        json_path=args.json_path,
        max_samples_per_row=args.max_samples_per_row,
    )
