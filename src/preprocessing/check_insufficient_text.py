from pathlib import Path

import chardet
import pandas as pd


def check_insufficient_text(input_csv, output_csv, min_texts=None):
    """
    Find and save rows in text_achievement_standards.csv where text_? columns are not sufficiently filled.

    Args:
        input_csv: Input CSV file path
        output_csv: Output CSV file path (insufficient_text.csv)
        min_texts: Minimum number of texts required (default: None, automatically set to total text column count)
    """
    # Detect CSV encoding
    with open(input_csv, "rb") as f:
        enc = chardet.detect(f.read(50000))["encoding"] or "utf-8-sig"
    print(f"Detected CSV encoding: {enc}")

    # Read CSV
    df = pd.read_csv(input_csv, encoding=enc)
    print(f"Total rows in input CSV: {len(df)}")

    # Check if required columns exist
    required_cols = ["subject", "school", "grade", "code", "content"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing: {missing_cols}")

    # Find all text_ columns
    text_cols = [c for c in df.columns if c.startswith("text_")]
    print(f"Found {len(text_cols)} text columns")

    if not text_cols:
        print("Warning: No text_ columns found in the CSV file.")
        return

    # Set default min_texts to specified value or total number of text columns
    if min_texts is None:
        min_texts = len(text_cols)
        print(
            f"Using total text column count: Looking for rows with < {min_texts} texts (i.e., not fully filled)"
        )
    else:
        print(f"Using specified min_texts: Looking for rows with < {min_texts} texts")

    # Count non-empty texts per row
    def count_non_empty_texts(row):
        count = 0
        for col in text_cols:
            value = row[col]
            # Count if value is not empty (not NaN, not empty string, not whitespace)
            if pd.notna(value) and str(value).strip():
                count += 1
        return count

    # Add a column with text count
    df["_text_count"] = df.apply(count_non_empty_texts, axis=1)

    # Print statistics
    print(f"\n=== Text Count Statistics ===")
    print(f"Total text columns available: {len(text_cols)}")

    # Dynamically generate intervals with step size 20
    interval_size = 20
    intervals = []

    # Add 0 texts
    intervals.append((0, 0))

    # Generate intervals with step 20 up to min_texts
    start = 1
    while start < min_texts:
        end = min(start + interval_size - 1, min_texts - 1)
        intervals.append((start, end))
        start += interval_size

    # Print interval statistics
    for start_val, end_val in intervals:
        if start_val == end_val:
            count = (df["_text_count"] == start_val).sum()
            print(f"Rows with {start_val} texts: {count}")
        else:
            count = (
                (df["_text_count"] >= start_val) & (df["_text_count"] <= end_val)
            ).sum()
            print(f"Rows with {start_val}-{end_val} texts: {count}")

    # Print fully filled (>= min_texts)
    fully_filled_count = (df["_text_count"] >= min_texts).sum()
    print(f"Rows with >= {min_texts} texts (fully filled): {fully_filled_count}")

    print(f"Average texts per row: {df['_text_count'].mean():.2f}")
    print(f"Median texts per row: {df['_text_count'].median():.1f}")
    print(f"Min texts: {df['_text_count'].min()}, Max texts: {df['_text_count'].max()}")

    # Filter rows where text count is less than minimum required
    insufficient_rows = df[df["_text_count"] < min_texts]

    # Select only required columns and add text count
    result_df = insufficient_rows[required_cols + ["_text_count"]].copy()
    result_df.rename(columns={"_text_count": "text_count"}, inplace=True)

    print(f"\nFound {len(result_df)} rows with < {min_texts} texts")

    # Save to CSV
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved to: {output_csv}")

    # Print statistics by subject
    print("\n=== Statistics by Subject ===")
    # Get sufficient rows (rows with >= min_texts)
    sufficient_rows = df[df["_text_count"] >= min_texts]

    # Get all subjects from original dataframe
    all_subjects = df["subject"].unique()

    for subject in sorted(all_subjects):
        subject_insufficient = result_df[result_df["subject"] == subject]
        subject_sufficient = sufficient_rows[sufficient_rows["subject"] == subject]
        insufficient_count = len(subject_insufficient)
        sufficient_count = len(subject_sufficient)
        total_count = insufficient_count + sufficient_count
        print(
            f"  - {subject}: {insufficient_count} insufficient, {sufficient_count} sufficient (total: {total_count})"
        )

    print("\n=== Statistics by School ===")
    # Get all schools from original dataframe
    all_schools = df["school"].unique()

    for school in sorted(all_schools):
        school_insufficient = result_df[result_df["school"] == school]
        school_sufficient = sufficient_rows[sufficient_rows["school"] == school]
        insufficient_count = len(school_insufficient)
        sufficient_count = len(school_sufficient)
        total_count = insufficient_count + sufficient_count
        print(
            f"  - {school}: {insufficient_count} insufficient, {sufficient_count} sufficient (total: {total_count})"
        )

    # Show first 10 rows as sample
    if len(result_df) > 0:
        print("\n=== Sample rows (first 10) ===")
        for idx, row in result_df.head(10).iterrows():
            print(
                f"  - Code: {row['code']}, Subject: {row['subject']}, School: {row['school']}, Grade: {row['grade']}, Texts: {row['text_count']}"
            )
    else:
        print(f"\nAll rows have at least {min_texts} text value(s)!")

    # Show distribution of text counts for rows with few texts (< 20)
    print("\n=== Rows with Few Texts (< 20) ===")
    few_texts_df = df[df["_text_count"] < 20]
    if len(few_texts_df) > 0:
        print(f"Total rows with < 20 texts: {len(few_texts_df)}")
        for subject in few_texts_df["subject"].unique():
            subject_data = few_texts_df[few_texts_df["subject"] == subject]
            print(
                f"  - {subject}: {len(subject_data)} rows (avg {subject_data['_text_count'].mean():.1f} texts)"
            )
    else:
        print("No rows with < 20 texts")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Check for rows with insufficient text in achievement standards CSV"
    )
    parser.add_argument(
        "--min_texts",
        type=int,
        default=200,
        help="Minimum number of texts required to be considered sufficient (default: 200)",
    )
    args = parser.parse_args()

    # Get project root (assuming this script is in project root)
    project_root = Path(__file__).resolve().parent.parent.parent
    input_csv = project_root / "dataset" / "text_achievement_standards.csv"
    output_csv = project_root / "dataset" / "insufficient_text.csv"

    print("=" * 70)
    print("Checking for rows with insufficient text")
    print("=" * 70)
    print(f"Input CSV: {input_csv}")
    print(f"Output CSV: {output_csv}")
    print(f"Minimum texts required (sufficient threshold): {args.min_texts}")
    print("=" * 70)

    if not input_csv.exists():
        print(f"Error: Input file not found: {input_csv}")
        print("\nPlease specify the correct path to text_achievement_standards.csv")
    else:
        check_insufficient_text(input_csv, output_csv, args.min_texts)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
