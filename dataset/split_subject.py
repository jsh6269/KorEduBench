import argparse
import os

import pandas as pd


def split_csv_by_subject(input_path: str, max_texts: int, encoding: str):
    """
    Split input csv file into subjects, parse contents and save them into subject_text{num} directory
    """
    output_dir = f"subject_text{max_texts}"
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
        default=10,
        help="Number of text columns to include (default: 10)",
    )
    parser.add_argument(
        "--encoding", "-e", default="utf-8", help="Encoding of CSV file"
    )
    args = parser.parse_args()

    split_csv_by_subject(args.input, args.max_texts, args.encoding)
