import pandas as pd
import chardet
from pathlib import Path

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
    required_cols = ['subject', 'school', 'grade', 'code', 'content']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing: {missing_cols}")
    
    # Find all text_ columns
    text_cols = [c for c in df.columns if c.startswith("text_")]
    print(f"Found {len(text_cols)} text columns")
    
    if not text_cols:
        print("Warning: No text_ columns found in the CSV file.")
        return
    
    # Set default min_texts to total number of text columns (find rows that are not fully filled)
    if min_texts is None:
        min_texts = len(text_cols)
        print(f"Auto-detected: Looking for rows with < {min_texts} texts (i.e., not fully filled)")
    
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
    df['_text_count'] = df.apply(count_non_empty_texts, axis=1)
    
    # Print statistics
    print(f"\n=== Text Count Statistics ===")
    print(f"Total text columns available: {len(text_cols)}")
    print(f"Rows with 0 texts: {(df['_text_count'] == 0).sum()}")
    print(f"Rows with 1-10 texts: {((df['_text_count'] >= 1) & (df['_text_count'] <= 10)).sum()}")
    print(f"Rows with 11-50 texts: {((df['_text_count'] >= 11) & (df['_text_count'] <= 50)).sum()}")
    print(f"Rows with 51-100 texts: {((df['_text_count'] >= 51) & (df['_text_count'] <= 100)).sum()}")
    print(f"Rows with 101-150 texts: {((df['_text_count'] >= 101) & (df['_text_count'] <= 150)).sum()}")
    print(f"Rows with 151-199 texts: {((df['_text_count'] >= 151) & (df['_text_count'] <= 199)).sum()}")
    print(f"Rows with {len(text_cols)} texts (fully filled): {(df['_text_count'] == len(text_cols)).sum()}")
    print(f"Average texts per row: {df['_text_count'].mean():.2f}")
    print(f"Median texts per row: {df['_text_count'].median():.1f}")
    print(f"Min texts: {df['_text_count'].min()}, Max texts: {df['_text_count'].max()}")
    
    # Filter rows where text count is less than minimum required
    insufficient_rows = df[df['_text_count'] < min_texts]
    
    # Select only required columns and add text count
    result_df = insufficient_rows[required_cols + ['_text_count']].copy()
    result_df.rename(columns={'_text_count': 'text_count'}, inplace=True)
    
    print(f"\nFound {len(result_df)} rows with < {min_texts} texts")
    
    # Save to CSV
    result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Saved to: {output_csv}")
    
    # Print statistics by subject
    if len(result_df) > 0:
        print("\n=== Statistics by Subject ===")
        subject_counts = result_df['subject'].value_counts()
        for subject, count in subject_counts.items():
            print(f"  - {subject}: {count} rows")
        
        print("\n=== Statistics by School ===")
        school_counts = result_df['school'].value_counts()
        for school, count in school_counts.items():
            print(f"  - {school}: {count} rows")
        
        # Show first 10 rows as sample
        print("\n=== Sample rows (first 10) ===")
        for idx, row in result_df.head(10).iterrows():
            print(f"  - Code: {row['code']}, Subject: {row['subject']}, School: {row['school']}, Grade: {row['grade']}, Texts: {row['text_count']}")
    else:
        print(f"\nAll rows have at least {min_texts} text value(s)!")
    
    # Show distribution of text counts for rows with few texts (< 20)
    print("\n=== Rows with Few Texts (< 20) ===")
    few_texts_df = df[df['_text_count'] < 20]
    if len(few_texts_df) > 0:
        print(f"Total rows with < 20 texts: {len(few_texts_df)}")
        for subject in few_texts_df['subject'].unique():
            subject_data = few_texts_df[few_texts_df['subject'] == subject]
            print(f"  - {subject}: {len(subject_data)} rows (avg {subject_data['_text_count'].mean():.1f} texts)")
    else:
        print("No rows with < 20 texts")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check for rows with insufficient text in achievement standards CSV")
    parser.add_argument("--min_texts", type=int, default=None, 
                      help="Minimum number of texts required (default: None, auto-detects total text columns to find unfilled rows)")
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
    if args.min_texts is None:
        print(f"Minimum texts required: Auto-detect (will find all rows that are not fully filled)")
    else:
        print(f"Minimum texts required: {args.min_texts}")
    print("=" * 70)
    
    if not input_csv.exists():
        print(f"Error: Input file not found: {input_csv}")
        print("\nPlease specify the correct path to text_achievement_standards.csv")
    else:
        check_insufficient_text(input_csv, output_csv, args.min_texts)
        
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
