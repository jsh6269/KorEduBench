"""
Common data loading utilities for evaluation scripts.
Provides functions to load and preprocess CSV data for cosine similarity and cross-encoder evaluations.
"""

import os
import re
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from src.utils.common import detect_encoding


@dataclass
class EvaluationData:
    """Container for evaluation data loaded from CSV."""
    
    df: pd.DataFrame
    contents: List[str]  # Achievement standard contents
    codes: List[str]  # Achievement standard codes
    sample_texts: List[str]  # Flattened sample texts for evaluation
    true_codes: List[str]  # True code for each sample text
    subject: str
    num_rows: int
    num_samples: int
    max_samples_per_row: int
    folder_name: Optional[str] = None  # train/valid/test folder name if detected


def load_evaluation_data(
    input_csv: str,
    encoding: Optional[str] = None,
    max_samples_per_row: Optional[int] = None,
) -> EvaluationData:
    """
    Load and preprocess CSV data for evaluation.
    
    Args:
        input_csv: Path to input CSV file
        encoding: CSV encoding (default: auto-detect)
        max_samples_per_row: Maximum number of text samples to use per row (default: auto-detect)
    
    Returns:
        EvaluationData object containing all necessary data for evaluation
    
    Raises:
        ValueError: If required columns are missing or no text_ columns are found
    """
    # Auto-detect encoding if not provided
    if not encoding:
        encoding = detect_encoding(input_csv)

    # Load CSV
    df = pd.read_csv(input_csv, encoding=encoding)
    
    # Validate required columns
    required_cols = ["code", "content"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Detect parent folder name (e.g., train / valid / test)
    parent_folder = os.path.basename(os.path.dirname(os.path.abspath(input_csv)))
    folder_name = None
    if re.search(r"(train|valid|val|test)", parent_folder, re.IGNORECASE):
        folder_name = parent_folder

    # Find sample columns (text_1, text_2, ...)
    sample_cols = [c for c in df.columns if c.startswith("text_")]
    if not sample_cols:
        raise ValueError("No text_ columns found for evaluation.")

    # Extract meta info
    subject = df["subject"].iloc[0] if "subject" in df.columns else "Unknown"
    num_rows = len(df)

    # Auto compute max_samples_per_row if None
    if max_samples_per_row is None:
        max_samples_per_row = int(max((df[sample_cols].notna().sum(axis=1)).max(), 0))
        print(f"Auto-detected max_samples_per_row = {max_samples_per_row}")

    # Extract achievement standards
    contents = df["content"].astype(str).tolist()
    codes = df["code"].astype(str).tolist()

    # Flatten sample texts and true codes
    sample_texts, true_codes = [], []
    for _, row in df.iterrows():
        code = str(row["code"])
        texts = []
        for col in sample_cols:
            text = str(row[col]).strip()
            if text and text.lower() != "nan":
                texts.append(text)

        # Apply max_samples_per_row limit
        if len(texts) > max_samples_per_row:
            texts = texts[:max_samples_per_row]

        for t in texts:
            sample_texts.append(t)
            true_codes.append(code)

    num_samples = len(sample_texts)
    print(f"Total evaluation samples: {num_samples}")

    return EvaluationData(
        df=df,
        contents=contents,
        codes=codes,
        sample_texts=sample_texts,
        true_codes=true_codes,
        subject=subject,
        num_rows=num_rows,
        num_samples=num_samples,
        max_samples_per_row=max_samples_per_row,
        folder_name=folder_name,
    )

