import argparse
import chardet
import random
from pathlib import Path

import pandas as pd

# Get project root (3 levels up from this file: src/preprocessing/script.py -> src -> project_root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def split_texts_to_train_valid(input_csv, train_csv, valid_csv, num_texts, seed=42):
    """
    Split texts from filtered rows into train and valid CSV files.
    For each row, sample from text_1 to text_{num_texts} and split them:
    - train_csv gets num_texts / 2 texts
    - valid_csv gets num_texts / 2 texts
    - No overlap between train and valid for the same row
    
    Args:
        input_csv: Input CSV file path
        train_csv: Output train CSV file path
        valid_csv: Output valid CSV file path
        num_texts: Number of texts to sample (must be even)
        seed: Random seed for reproducibility
    """
    # Validate num_texts is even
    if num_texts % 2 != 0:
        raise ValueError(f"num_texts must be even, but got {num_texts}")
    
    # Set random seed
    random.seed(seed)
    
    texts_per_split = num_texts // 2
    print(f"Each split will have {texts_per_split} texts per row")
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
    
    # Define target text columns to sample from (text_1 to text_{num_texts})
    target_text_cols = [f"text_{i}" for i in range(1, num_texts + 1)]
    print(f"Sampling from text_1 to text_{num_texts}...")
    
    # Meta columns (non-text columns)
    meta_cols = [c for c in df.columns if not c.startswith("text_")]
    
    # Process each row
    train_rows = []
    valid_rows = []
    filtered_count = 0
    skipped_count = 0
    
    for idx, row in df.iterrows():
        # Find non-empty text columns only from target range (text_1 to text_{num_texts})
        non_empty_texts = []
        for col in target_text_cols:
            if col in df.columns:
                value = row[col]
                if pd.notna(value) and str(value).strip():
                    non_empty_texts.append(col)
        
        # Skip if not enough texts in target range
        if len(non_empty_texts) < num_texts:
            skipped_count += 1
            continue
        
        filtered_count += 1
        
        # Randomly select num_texts texts from non-empty texts in target range
        selected_text_cols = random.sample(non_empty_texts, num_texts)
        
        # Get the actual values from selected columns
        selected_values = [row[col] for col in selected_text_cols]
        
        # Split into train and valid (no overlap)
        train_values = selected_values[:texts_per_split]
        valid_values = selected_values[texts_per_split:]
        
        # Create train row: copy meta columns first
        train_row = row[meta_cols].copy().to_dict()
        # Fill text_1, text_2, ..., text_{texts_per_split} with train values
        for i, value in enumerate(train_values, start=1):
            train_col = f"text_{i}"
            train_row[train_col] = value
        # Fill remaining text columns with empty strings
        for col in text_cols:
            if col not in train_row:
                train_row[col] = ""
        train_rows.append(train_row)
        
        # Create valid row: copy meta columns first
        valid_row = row[meta_cols].copy().to_dict()
        # Fill text_1, text_2, ..., text_{texts_per_split} with valid values
        for i, value in enumerate(valid_values, start=1):
            valid_col = f"text_{i}"
            valid_row[valid_col] = value
        # Fill remaining text columns with empty strings
        for col in text_cols:
            if col not in valid_row:
                valid_row[col] = ""
        valid_rows.append(valid_row)
    
    print(f"\nFiltered rows: {filtered_count} rows with >= {num_texts} non-empty texts in text_1 to text_{num_texts}")
    print(f"Skipped rows: {skipped_count} rows with < {num_texts} non-empty texts in text_1 to text_{num_texts}")
    
    if filtered_count == 0:
        print("No rows found with sufficient texts. Exiting.")
        return
    
    # Create DataFrames
    train_df = pd.DataFrame(train_rows)
    valid_df = pd.DataFrame(valid_rows)
    
    # Ensure all text columns exist in output (fill with empty string if missing)
    for col in text_cols:
        if col not in train_df.columns:
            train_df[col] = ""
        if col not in valid_df.columns:
            valid_df[col] = ""
    
    # Reorder columns: meta columns first, then text columns in numeric order
    # Sort text columns by their numeric suffix (text_1, text_2, ..., text_200)
    def get_text_num(col_name):
        try:
            return int(col_name.split("_")[1])
        except (IndexError, ValueError):
            return 0
    
    sorted_text_cols = sorted([c for c in text_cols if c in train_df.columns], key=get_text_num)
    ordered_cols = meta_cols + sorted_text_cols
    train_df = train_df[ordered_cols]
    valid_df = valid_df[ordered_cols]
    
    # Print statistics
    print("\n=== Statistics by Subject ===")
    train_subject_counts = train_df['subject'].value_counts()
    for subject in sorted(train_df['subject'].unique()):
        train_count = train_subject_counts.get(subject, 0)
        valid_count = len(valid_df[valid_df['subject'] == subject])
        print(f"  - {subject}: train={train_count}, valid={valid_count}, total={train_count + valid_count}")
    
    # Save train and valid CSVs
    train_df.to_csv(train_csv, index=False, encoding="utf-8-sig")
    valid_df.to_csv(valid_csv, index=False, encoding="utf-8-sig")
    
    print(f"\nSaved train CSV to: {train_csv}")
    print(f"Saved valid CSV to: {valid_csv}")
    print(f"Train rows: {len(train_df)}, Valid rows: {len(valid_df)}")
    
    return train_df, valid_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split texts from filtered rows into train and valid CSV files."
    )
    parser.add_argument(
        "--num_texts",
        type=int,
        default=160,
        help="Number of texts to sample from text_1 to text_{num_texts} (must be even, default: 200)",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help=f"Path to input CSV file (default: {PROJECT_ROOT}/dataset/text_achievement_standards.csv)",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default=None,
        help=f"Path to output train CSV file (default: {PROJECT_ROOT}/dataset/train.csv)",
    )
    parser.add_argument(
        "--valid_csv",
        type=str,
        default=None,
        help=f"Path to output valid CSV file (default: {PROJECT_ROOT}/dataset/valid.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()
    
    # Validate num_texts is even
    if args.num_texts % 2 != 0:
        print(f"Error: num_texts must be even, but got {args.num_texts}")
        exit(1)
    
    # Set default paths
    if args.input_csv is None:
        input_csv = PROJECT_ROOT / "dataset" / "text_achievement_standards.csv"
    else:
        input_csv = Path(args.input_csv)
    
    if args.train_csv is None:
        train_csv = PROJECT_ROOT / "dataset" / "train.csv"
    else:
        train_csv = Path(args.train_csv)
    
    if args.valid_csv is None:
        valid_csv = PROJECT_ROOT / "dataset" / "valid.csv"
    else:
        valid_csv = Path(args.valid_csv)
    
    print("=" * 70)
    print("Splitting texts into train and valid CSV files")
    print("=" * 70)
    print(f"Input CSV: {input_csv}")
    print(f"Train CSV: {train_csv}")
    print(f"Valid CSV: {valid_csv}")
    print(f"Number of texts to sample: {args.num_texts} (will split into {args.num_texts // 2} + {args.num_texts // 2})")
    print(f"Sampling range: text_1 to text_{args.num_texts}")
    print(f"Random seed: {args.seed}")
    print("=" * 70)
    
    if not input_csv.exists():
        print(f"Error: Input file not found: {input_csv}")
        print("\nPlease specify the correct path to text_achievement_standards.csv")
    else:
        split_texts_to_train_valid(input_csv, train_csv, valid_csv, args.num_texts, args.seed)
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

