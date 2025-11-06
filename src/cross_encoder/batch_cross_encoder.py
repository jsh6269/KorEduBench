import argparse
import os
import sys
from pathlib import Path

# Get project root (3 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from src.cross_encoder.eval_cross_encoder import evaluate_bi_cross_pipeline
from src.utils.random_seed import set_predict_random_seed


def evaluate_folder(
    folder_path: str,
    bi_model_name: str,
    cross_model_name: str,
    top_k: int = 20,
    encoding: str | None = None,
    json_path: str = None,
    max_samples_per_row: int = None,
):
    if json_path is None:
        json_path = str(
            PROJECT_ROOT / "output" / "cross_encoder" / "results_rerank.json"
        )

    # Find all CSV files in folder
    csv_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".csv")
    ]

    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return

    print(f"Found {len(csv_files)} CSV files in {folder_path}")

    # Evaluate each CSV
    for csv_path in tqdm(csv_files, desc="Evaluating CSV files", unit="file"):
        try:
            print(f"\n=== Processing file: {os.path.basename(csv_path)} ===")
            evaluate_bi_cross_pipeline(
                input_csv=csv_path,
                bi_model_name=bi_model_name,
                cross_model_name=cross_model_name,
                top_k=top_k,
                encoding=encoding,
                json_path=json_path,
                max_samples_per_row=max_samples_per_row,
            )
        except Exception as e:
            import traceback

            print(f"\nError while evaluating {csv_path}: {e}")
            print(f"   Full traceback:")
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch evaluation of Bi-Encoder + Cross-Encoder reranking for all CSV files in a folder."
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        required=True,
        help="Path to folder containing CSV files.",
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
        help="Maximum number of text samples to evaluate per row (default: auto-detect).",
    )
    args = parser.parse_args()

    # Handle default values
    cross_model = args.cross_model
    if cross_model is None:
        cross_model = str(PROJECT_ROOT / "model" / "cross_finetuned")

    set_predict_random_seed(42)
    evaluate_folder(
        folder_path=args.folder_path,
        bi_model_name=args.bi_model,
        cross_model_name=cross_model,
        top_k=args.top_k,
        encoding=args.encoding,
        json_path=args.json_path,
        max_samples_per_row=args.max_samples_per_row,
    )
