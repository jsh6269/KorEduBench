import argparse
import chardet
from pathlib import Path

import pandas as pd

# Get project root (3 levels up from this file: src/preprocessing/script.py -> src -> project_root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def is_empty(value):
    """Check if a value is empty (NaN, empty string, or whitespace only)."""
    return pd.isna(value) or str(value).strip() == ""


def verify_text_order(input_csv):
    """
    Verify that text columns are filled in order (no gaps).
    If text_i is empty, then text_j (where j > i) should also be empty.
    
    Args:
        input_csv: Input CSV file path
        
    Returns:
        tuple: (is_valid, issues) where is_valid is bool and issues is list of issue descriptions
    """
    # Detect CSV encoding
    with open(input_csv, "rb") as f:
        enc = chardet.detect(f.read(50000))["encoding"] or "utf-8-sig"
    print(f"Detected CSV encoding: {enc}")
    
    # Read CSV
    df = pd.read_csv(input_csv, encoding=enc)
    print(f"Total rows: {len(df)}")
    
    # Find all text_ columns
    text_cols = [c for c in df.columns if c.startswith("text_")]
    
    if not text_cols:
        print("Warning: No text_ columns found in the CSV file.")
        return True, []
    
    # Sort text columns by numeric suffix (text_1, text_2, ..., text_N)
    def get_text_num(col_name):
        try:
            return int(col_name.split("_")[1])
        except (IndexError, ValueError):
            return 0
    
    sorted_text_cols = sorted(text_cols, key=get_text_num)
    print(f"Found {len(sorted_text_cols)} text columns: {sorted_text_cols[0]} to {sorted_text_cols[-1]}")
    
    issues = []
    rows_with_issues = []
    
    # Check each row
    for idx, row in df.iterrows():
        row_issues = []
        
        # Find the first empty text column
        first_empty_idx = None
        for i, col in enumerate(sorted_text_cols):
            if is_empty(row[col]):
                first_empty_idx = i
                break
        
        # If we found an empty column, check if any subsequent columns are filled
        if first_empty_idx is not None:
            for j in range(first_empty_idx + 1, len(sorted_text_cols)):
                col = sorted_text_cols[j]
                if not is_empty(row[col]):
                    # Issue found: earlier column is empty but later column is filled
                    issue_msg = (
                        f"Row {idx} ({row.get('code', 'N/A')}): "
                        f"{sorted_text_cols[first_empty_idx]} is empty but {col} is filled"
                    )
                    row_issues.append(issue_msg)
                    issues.append(issue_msg)
        
        if row_issues:
            rows_with_issues.append({
                'row_index': idx,
                'code': row.get('code', 'N/A'),
                'subject': row.get('subject', 'N/A'),
                'issues': row_issues
            })
    
    is_valid = len(issues) == 0
    
    # Print results
    print("\n" + "=" * 70)
    if is_valid:
        print("✓ VERIFICATION PASSED")
        print("All text columns are filled in order (no gaps found)")
    else:
        print("✗ VERIFICATION FAILED")
        print(f"Found {len(issues)} issue(s) in {len(rows_with_issues)} row(s)")
        print("\n=== Issues Found ===")
        
        # Group by subject for summary
        if len(rows_with_issues) > 0:
            subject_issues = {}
            for row_info in rows_with_issues:
                subject = row_info['subject']
                if subject not in subject_issues:
                    subject_issues[subject] = 0
                subject_issues[subject] += 1
            
            print("\nSummary by Subject:")
            for subject, count in sorted(subject_issues.items()):
                print(f"  - {subject}: {count} row(s) with issues")
            
            # Show first 10 issues
            print("\nFirst 10 issues:")
            for i, issue in enumerate(issues[:10], 1):
                print(f"  {i}. {issue}")
            
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more issue(s)")
    
    print("=" * 70)
    
    return is_valid, issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify that text columns are filled in order (no gaps)."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input CSV file path to verify",
    )
    args = parser.parse_args()
    
    input_csv = Path(args.input)
    
    print("=" * 70)
    print("Text Column Order Verification")
    print("=" * 70)
    print(f"Input CSV: {input_csv}")
    print("=" * 70)
    
    if not input_csv.exists():
        print(f"Error: Input file not found: {input_csv}")
        exit(1)
    
    is_valid, issues = verify_text_order(input_csv)
    
    print("\n" + "=" * 70)
    if is_valid:
        print("Verification completed successfully!")
        exit(0)
    else:
        print("Verification failed. Please fix the issues above.")
        exit(1)

