import argparse
import os
from pathlib import Path

import pandas as pd

# Get project root (3 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def split_csv_by_subject(
    input_path: str, max_texts: int, encoding: str, output_dir_name: str = None
):
    """
    Split input csv file into subjects, parse contents and save them into specified directory

    Args:
        input_path: Input CSV file path
        max_texts: Number of text columns to include
        encoding: Encoding of CSV file
        output_dir_name: Output directory name (default: subject_text{max_texts})
                         Will be created at PROJECT_ROOT/dataset/{output_dir_name}
    """
    if output_dir_name is None:
        output_dir = PROJECT_ROOT / "dataset" / f"subject_text{max_texts}"
    else:
        output_dir = PROJECT_ROOT / "dataset" / output_dir_name

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path, encoding=encoding)

    # basics
    base_cols = ["subject", "school", "grade", "code", "content"]

    # text columns
    text_cols = [f"text_{i}" for i in range(1, max_texts + 1)]

    # select existing columns
    selected_cols = [col for col in base_cols + text_cols if col in df.columns]
    df = df[selected_cols]

    # save per-subject csv
    for subject, group in df.groupby("subject"):
        safe_name = (
            "".join(c for c in subject if c.isalnum() or c in (" ", "_"))
            .strip()
            .replace(" ", "_")
        )
        output_path = os.path.join(output_dir, f"{safe_name}.csv")

        group.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {output_path} ({len(group)} rows)")

    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split curriculum CSV by subject.")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file path")
    parser.add_argument(
        "--max-texts",
        "-n",
        type=int,
        default=80,
        help="Number of text columns to include (default: 10)",
    )
    parser.add_argument(
        "--encoding", "-e", default="utf-8", help="Encoding of CSV file"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory name (default: subject_text{max_texts}). Will be created at PROJECT_ROOT/dataset/{output_dir_name}",
    )
    args = parser.parse_args()

    split_csv_by_subject(args.input, args.max_texts, args.encoding, args.output)
