import argparse
import csv
import json
import os
import zipfile
from pathlib import Path

import chardet
import pandas as pd
from tqdm import tqdm

# Get project root (3 levels up from this file: src/preprocessing/script.py -> src -> project_root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_blacklist(blacklist_path):
    """
    Load blacklist JSON file and return a set of (code, content) tuples for fast lookup.
    """
    if not os.path.exists(blacklist_path):
        print(f"Blacklist file not found: {blacklist_path}")
        return set()

    try:
        with open(blacklist_path, "r", encoding="utf-8") as f:
            blacklist_data = json.load(f)

        # Create a set of (code, content) tuples for fast lookup
        blacklist_set = set()
        for code, contents in blacklist_data.items():
            for content in contents:
                # Normalize content (strip whitespace)
                blacklist_set.add((code, content.strip()))

        print(
            f"Blacklist file loaded: {len(blacklist_set)} (code, content) combinations excluded"
        )
        return blacklist_set
    except Exception as e:
        print(f"Error loading Blacklist file: {e}")
        return set()


def append_texts_to_csv(
    label_dir="label",
    csv_path=None,
    output_csv=None,
    max_texts=200,
):
    """Append text fields from JSON files inside ZIP archives to a CSV file by matching achievement standard codes."""
    if csv_path is None:
        csv_path = PROJECT_ROOT / "dataset" / "unique_achievement_standards.csv"
    if output_csv is None:
        output_csv = PROJECT_ROOT / "dataset" / "text_achievement_standards.csv"

    csv_path = str(csv_path)
    output_csv = str(output_csv)

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

    # Load blacklist
    script_dir = Path(__file__).resolve().parent
    blacklist_path = script_dir / "2022_blacklist.json"
    blacklist_set = load_blacklist(blacklist_path)

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
                        try:
                            text = learning.get("text_description", "").strip()
                        except:
                            continue

                        if not text:
                            continue

                        standards = src_info.get("2022_achievement_standard", [])
                        for s in standards:
                            if "[" not in s or "]" not in s:
                                continue
                            code = s[s.find("[") + 1 : s.find("]")]
                            content = s[s.find("]") + 1 :].strip()
                            trimmed_content = content.replace("\n", " ").strip()

                            # Check blacklist: skip if (code, content) combination is in blacklist
                            if (code, content) in blacklist_set:
                                continue

                            if (code, trimmed_content) in blacklist_set:
                                continue

                            if code in code_to_idx:
                                idx = code_to_idx[code]
                                for col in [
                                    c for c in df.columns if c.startswith("text_")
                                ]:
                                    if not str(df.at[idx, col]).strip():
                                        df.at[idx, col] = text
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
        default=None,
        help="Path to the input CSV file (default: {PROJECT_ROOT}/dataset/unique_achievement_standards.csv).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to the output CSV file (default: {PROJECT_ROOT}/dataset/text_achievement_standards.csv).",
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
