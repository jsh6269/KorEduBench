import argparse
import csv
import json
import os
import zipfile
from pathlib import Path

from tqdm import tqdm

# Get project root (3 levels up from this file)
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


def extract_unique_standards(label_dir="label", output_csv=None):
    """
    Extract 2022 achievement standard (code, content) from zip files in the label directory
    CSV column: subject, school, grade, code, content
    Order: subject → code
    Excluded: (code, content) combinations in Blacklist.
    """
    if output_csv is None:
        output_csv = PROJECT_ROOT / "dataset" / "unique_achievement_standards.csv"
    output_csv = str(output_csv)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Load blacklist
    script_dir = Path(__file__).resolve().parent
    blacklist_path = script_dir / "2022_blacklist.json"
    blacklist_set = load_blacklist(blacklist_path)

    # key: code, value: (content, subject, school, grade)
    unique_standards = {}

    # collect every zip files
    zip_files = []
    for root, _, files in os.walk(label_dir):
        for file in files:
            if file.endswith(".zip"):
                zip_files.append(os.path.join(root, file))

    print(f"Total {len(zip_files)} zip files found.\n")

    # iterate zip files
    for zip_path in tqdm(zip_files, desc="Processing zip files", unit="zip"):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                json_files = [n for n in zf.namelist() if n.endswith(".json")]

                # iterate json files
                for name in tqdm(
                    json_files,
                    desc=f"Processing {os.path.basename(zip_path)}",
                    leave=False,
                    unit="json",
                ):
                    try:
                        with zf.open(name) as f:
                            data = json.load(f)

                        src_info = data.get("source_data_info", {})
                        raw_info = data.get("raw_data_info", {})

                        subject = raw_info.get("subject", "").strip()
                        school = raw_info.get("school", "").strip()
                        grade = raw_info.get("grade", "").strip()

                        standards = src_info.get("2022_achievement_standard", [])
                        for s in standards:
                            s = s.strip()
                            if not s or "[" not in s or "]" not in s:
                                continue
                            code = s[s.find("[") + 1 : s.find("]")]
                            content = s[s.find("]") + 1 :].strip()
                            trimmed_content = content.replace("\n", " ").strip()

                            # Check blacklist: skip if (code, content) combination is in blacklist
                            if (code, content) in blacklist_set:
                                continue

                            if (code, trimmed_content) in blacklist_set:
                                continue

                            if code not in unique_standards:
                                unique_standards[code] = (
                                    trimmed_content,
                                    subject,
                                    school,
                                    grade,
                                )

                    except json.JSONDecodeError:
                        tqdm.write(f"JSON decode error: {name} in {zip_path}")
        except zipfile.BadZipFile:
            tqdm.write(f"Bad zip file: {zip_path}")

    # CSV 저장
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "school", "grade", "code", "content"])

        # subject → code 기준 정렬
        sorted_items = sorted(
            unique_standards.items(), key=lambda x: (x[1][1], x[0])  # (subject, code)
        )

        for code, (content, subject, school, grade) in sorted_items:
            writer.writerow([subject, school, grade, code, content])

    print(f"\nTotal {len(unique_standards)} achievement standard saved! → {output_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Extract unique 2022 achievement standards from label zips."
    )
    ap.add_argument(
        "label_dir", type=str, help="Path to the label directory containing zip files."
    )
    ap.add_argument(
        "output_csv",
        type=str,
        nargs="?",
        default=None,
        help="Output CSV filename (default: {PROJECT_ROOT}/dataset/unique_achievement_standards.csv)",
    )
    args = ap.parse_args()

    extract_unique_standards(args.label_dir, args.output_csv)
