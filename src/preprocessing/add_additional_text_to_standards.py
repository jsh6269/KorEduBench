import argparse
import json
import os
import zipfile
from pathlib import Path

import chardet
import pandas as pd
from tqdm import tqdm

# Get project root (3 levels up from this file: src/preprocessing/script.py -> src -> project_root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_insufficient_standards(insufficient_csv_path):
    """
    Load insufficient_text.csv and return a set of (code, content) tuples for fast lookup.

    Args:
        insufficient_csv_path: Path to insufficient_text.csv file

    Returns:
        set: Set of (code, content) tuples
    """
    if not os.path.exists(insufficient_csv_path):
        print(f"Insufficient text CSV not found: {insufficient_csv_path}")
        return set()

    try:
        # Detect encoding
        with open(insufficient_csv_path, "rb") as f:
            enc = chardet.detect(f.read(50000))["encoding"] or "utf-8-sig"

        # Read CSV
        df = pd.read_csv(insufficient_csv_path, encoding=enc)

        if "code" not in df.columns or "content" not in df.columns:
            print(
                "Error: insufficient_text.csv must contain 'code' and 'content' columns"
            )
            return set()

        # Create set of (code, content) tuples
        insufficient_set = set()
        for _, row in df.iterrows():
            code = str(row["code"]).strip()
            content = str(row["content"]).strip()
            if code and content:
                insufficient_set.add((code, content))

        print(
            f"Loaded {len(insufficient_set)} (code, content) combinations from insufficient_text.csv"
        )
        return insufficient_set
    except Exception as e:
        print(f"Error loading insufficient_text.csv: {e}")
        return set()


def add_additional_texts_to_csv(
    label_dir,
    insufficient_csv_path=None,
    text_standards_csv_path=None,
    output_csv_path=None,
    max_texts=200,
):
    """
    Add text_description from JSON files to text_achievement_standards.csv for rows listed in insufficient_text.csv.

    Args:
        label_dir: Directory containing ZIP files with JSON files
        insufficient_csv_path: Path to insufficient_text.csv (default: dataset/insufficient_text.csv)
        text_standards_csv_path: Path to text_achievement_standards.csv (default: dataset/text_achievement_standards.csv)
        output_csv_path: Path to output CSV (default: same as text_standards_csv_path, overwrites)
        max_texts: Maximum number of text columns (default: 200)
    """
    # Set default paths
    if insufficient_csv_path is None:
        insufficient_csv_path = PROJECT_ROOT / "dataset" / "insufficient_text.csv"
    if text_standards_csv_path is None:
        text_standards_csv_path = (
            PROJECT_ROOT / "dataset" / "text_achievement_standards.csv"
        )
    if output_csv_path is None:
        output_csv_path = text_standards_csv_path

    insufficient_csv_path = str(insufficient_csv_path)
    text_standards_csv_path = str(text_standards_csv_path)
    output_csv_path = str(output_csv_path)

    # Load insufficient standards
    insufficient_set = load_insufficient_standards(insufficient_csv_path)
    if not insufficient_set:
        print("No insufficient standards found. Exiting.")
        return

    # Detect and read text_achievement_standards.csv
    with open(text_standards_csv_path, "rb") as f:
        enc = chardet.detect(f.read(50000))["encoding"] or "utf-8-sig"
    print(f"Detected CSV encoding: {enc}")

    df = pd.read_csv(text_standards_csv_path, encoding=enc)

    if "code" not in df.columns:
        raise ValueError("CSV must contain a 'code' column.")

    # Ensure text columns exist
    text_cols = [c for c in df.columns if c.startswith("text_")]
    if not text_cols:
        # Create text columns if they don't exist
        for i in range(1, max_texts + 1):
            df[f"text_{i}"] = ""
        text_cols = [f"text_{i}" for i in range(1, max_texts + 1)]
    else:
        # Ensure we have enough text columns
        existing_indices = [
            int(c.split("_")[1]) for c in text_cols if c.split("_")[1].isdigit()
        ]
        max_idx = max(existing_indices) if existing_indices else 0
        for i in range(max_idx + 1, max_texts + 1):
            col = f"text_{i}"
            if col not in df.columns:
                df[col] = ""
                text_cols.append(col)

    # Create code to row index mapping
    code_to_idx = {str(row["code"]).strip(): idx for idx, row in df.iterrows()}

    print(f"Total rows in CSV: {len(df)}")
    print(f"Text columns available: {len(text_cols)}")

    # Collect all ZIP files
    zip_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(label_dir)
        for f in files
        if f.endswith(".zip")
    ]
    print(f"Found {len(zip_files)} ZIP files.\n")

    added_count = 0
    matched_count = 0

    # Iterate through ZIP files and extract data
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

                        # Extract text_description
                        text_description = str(
                            learning.get("text_description", "")
                        ).strip()
                        if not text_description:
                            continue

                        # Process achievement standards
                        standards = src_info.get("2022_achievement_standard", [])
                        for s in standards:
                            if "[" not in s or "]" not in s:
                                continue

                            code = s[s.find("[") + 1 : s.find("]")]
                            content = s[s.find("]") + 1 :].strip()
                            trimmed_content = content.replace("\n", " ").strip()

                            # Check if (code, content) is in insufficient set
                            if (code, content) in insufficient_set or (
                                code,
                                trimmed_content,
                            ) in insufficient_set:
                                matched_count += 1

                                # Find corresponding row in DataFrame
                                if code in code_to_idx:
                                    idx = code_to_idx[code]

                                    # Check if all text columns are already filled
                                    all_filled = True
                                    for col in text_cols:
                                        value = df.at[idx, col]
                                        if pd.isna(value) or str(value).strip() == "":
                                            all_filled = False
                                            break

                                    # Skip if already fully filled (text_1 to text_200)
                                    if all_filled:
                                        continue

                                    # Find first empty text column and add text_description
                                    for col in text_cols:
                                        value = df.at[idx, col]
                                        # Check if empty (NaN, empty string, or whitespace)
                                        if pd.isna(value) or str(value).strip() == "":
                                            df.at[idx, col] = text_description
                                            added_count += 1
                                            break

                                    # Check again if all text columns are now filled
                                    all_filled = True
                                    for col in text_cols:
                                        value = df.at[idx, col]
                                        if pd.isna(value) or str(value).strip() == "":
                                            all_filled = False
                                            break

                                    if all_filled:
                                        # Remove from insufficient_set to avoid processing again
                                        if (code, content) in insufficient_set:
                                            insufficient_set.remove((code, content))
                                        if (code, trimmed_content) in insufficient_set:
                                            insufficient_set.remove(
                                                (code, trimmed_content)
                                            )

                    except json.JSONDecodeError:
                        tqdm.write(f"JSON decode error: {name} in {zip_path}")
                    except Exception as e:
                        tqdm.write(f"Error processing {name} in {zip_path}: {e}")

        except zipfile.BadZipFile:
            tqdm.write(f"Bad ZIP file: {zip_path}")
        except Exception as e:
            tqdm.write(f"Error processing ZIP file {zip_path}: {e}")

    # Save the updated CSV
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n" + "=" * 70)
    print(f"Update complete → {output_csv_path}")
    print(f"Matched (code, content) combinations: {matched_count}")
    print(f"Text descriptions added: {added_count}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add text_description from label JSONs to text_achievement_standards.csv for rows in insufficient_text.csv."
    )
    parser.add_argument(
        "label_dir", type=str, help="Path to the label directory containing ZIP files."
    )
    parser.add_argument(
        "--insufficient_csv",
        type=str,
        default=None,
        help="Path to insufficient_text.csv (default: {PROJECT_ROOT}/dataset/insufficient_text.csv).",
    )
    parser.add_argument(
        "--text_standards_csv",
        type=str,
        default=None,
        help="Path to text_achievement_standards.csv (default: {PROJECT_ROOT}/dataset/text_achievement_standards.csv).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to output CSV file (default: same as text_standards_csv, overwrites).",
    )
    parser.add_argument(
        "--max_texts",
        type=int,
        default=200,
        help="Maximum number of text columns (default: 200).",
    )
    args = parser.parse_args()

    add_additional_texts_to_csv(
        label_dir=args.label_dir,
        insufficient_csv_path=args.insufficient_csv,
        text_standards_csv_path=args.text_standards_csv,
        output_csv_path=args.output_csv,
        max_texts=args.max_texts,
    )
