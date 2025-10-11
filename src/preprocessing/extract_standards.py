import argparse
import csv
import json
import os
import zipfile
from pathlib import Path

from tqdm import tqdm

# Get project root (3 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def extract_unique_standards(
    label_dir="label", output_csv=None
):
    """
    Extract 2022 achievement standard (code, content) from zip files in the label directory
    CSV column: subject, school, grade, code, content
    Order: subject → code
    """
    if output_csv is None:
        output_csv = PROJECT_ROOT / "dataset" / "unique_achievement_standards.csv"
    output_csv = str(output_csv)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    # key: code, value: (content, subject, school, grade)
    unique_standards = {}

    # collect every zip files
    zip_files = []
    for root, _, files in os.walk(label_dir):
        for file in files:
            if file.endswith(".zip"):
                zip_files.append(os.path.join(root, file))

    print(f"총 {len(zip_files)}개의 zip 파일이 발견되었습니다.\n")

    # iterate zip files
    for zip_path in tqdm(zip_files, desc="ZIP 파일 처리 중", unit="zip"):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                json_files = [n for n in zf.namelist() if n.endswith(".json")]

                # iterate json files
                for name in tqdm(
                    json_files,
                    desc=f"{os.path.basename(zip_path)} 내부",
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
                            if code not in unique_standards:
                                unique_standards[code] = (
                                    content,
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
