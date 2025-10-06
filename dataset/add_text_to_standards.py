import argparse
import csv
import json
import os
import zipfile

import chardet
import pandas as pd
from tqdm import tqdm


def append_texts_to_csv(
    label_dir="label",
    csv_path="./unique_achievement_standards.csv",
    output_csv="./text_achievement_standards.csv",
    max_texts=200,
):
    """Append text fields from JSON files inside ZIP archives to a CSV file by matching achievement standard codes."""

    # Detect CSV encoding
    with open(csv_path, "rb") as f:
        enc = chardet.detect(f.read(50000))["encoding"] or "utf-8-sig"
    print(f"Detected CSV encoding: {enc}")

    # Read CSV
    df = pd.read_csv(csv_path, encoding=enc)
    if "code" not in df.columns:
        raise ValueError("CSV must contain a 'code' column.")

    # Ensure enough text columns exist
    existing_texts = [c for c in df.columns if c.startswith("text_")]
    next_idx = len(existing_texts)
    for i in range(next_idx + 1, max_texts + 1):
        col = f"text_{i}"
        if col not in df.columns:
            df[col] = ""

    # Map code to row index
    code_to_idx = {row["code"]: idx for idx, row in df.iterrows()}

    # Collect all ZIP files
    zip_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(label_dir)
        for f in files
        if f.endswith(".zip")
    ]
    print(f"Found {len(zip_files)} ZIP files.\n")

    # Iterate through ZIP files and textct data
    for zip_path in tqdm(zip_files, desc="Processing ZIP files", unit="zip"):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                json_files = [n for n in zf.namelist() if n.endswith(".json")]
                for name in tqdm(
                    json_files,
                    desc=os.path.basename(zip_path),
                    leave=False,
                    unit="json",
                ):
                    try:
                        with zf.open(name) as f:
                            data = json.load(f)

                        src_info = data.get("source_data_info", {})
                        learning = data.get("learning_data_info", {})

                        combined_text = " ".join(
                            str(learning.get(k, "")).strip()
                            for k in ("text_description", "text_qa", "text_an")
                            if str(learning.get(k, "")).strip()
                        ).strip()
                        if not combined_text:
                            continue

                        standards = src_info.get("2022_achievement_standard", [])
                        for s in standards:
                            if "[" not in s or "]" not in s:
                                continue
                            code = s[s.find("[") + 1 : s.find("]")]
                            if code in code_to_idx:
                                idx = code_to_idx[code]
                                for col in [
                                    c for c in df.columns if c.startswith("text_")
                                ]:
                                    if not str(df.at[idx, col]).strip():
                                        df.at[idx, col] = combined_text
                                        break

                    except json.JSONDecodeError:
                        tqdm.write(f"JSON decode error: {name} in {zip_path}")
        except zipfile.BadZipFile:
            tqdm.write(f"Bad ZIP file: {zip_path}")

    # Save the updated CSV
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Update complete → {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Append texts from label JSONs to achievement standards CSV."
    )
    parser.add_argument(
        "label_dir", type=str, help="Path to the label directory containing ZIP files."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="./unique_achievement_standards.csv",
        help="Path to the input CSV file (default: ./unique_achievement_standards.csv).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./text_achievement_standards.csv",
        help="Path to the output CSV file (default: ./updated_achievement_standards.csv).",
    )
    parser.add_argument(
        "--max_texts",
        type=int,
        default=10,
        help="Maximum number of text columns to allocate (default: 10).",
    )
    args = parser.parse_args()

    append_texts_to_csv(
        label_dir=args.label_dir,
        csv_path=args.csv_path,
        output_csv=args.output_csv,
        max_texts=args.max_texts,
    )
