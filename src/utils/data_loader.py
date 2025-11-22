"""
Common data loading utilities for evaluation scripts.
Provides functions to load and preprocess CSV data for cosine similarity and cross-encoder evaluations.
"""

import os
import random
import re
from dataclasses import dataclass
from glob import glob
from typing import List, Optional

import pandas as pd
from datasets import Dataset
from tqdm import tqdm

#from src.test.check_prompt_length import subject
from src.utils.common import detect_encoding
from src.utils.prompt import create_chat_classification_prompt


@dataclass
class EvaluationData:
    """Container for evaluation data loaded from CSV."""

    df: pd.DataFrame
    contents: List[str]  # Achievement standard contents
    codes: List[str]  # Achievement standard codes
    sample_texts: List[str]  # Flattened sample texts for evaluation
    samples_true_codes: List[str]  # True code for each sample text
    samples_candidates: List[
        List[tuple[int, str, str]]
    ]  # Candidates for each sample (list of tuples: (index, code, content))
    subject: str
    num_rows: int
    num_candidates: int
    num_samples: int
    max_samples_per_row: int
    max_candidates: int
    folder_name: Optional[str] = None  # train/valid/test folder name if detected


def load_evaluation_data(
    input_csv: str,
    encoding: Optional[str] = None,
    max_samples_per_row: Optional[int] = None,
    max_total_samples: Optional[int] = None,
    max_candidates: Optional[int] = None,
) -> EvaluationData:
    """
    Load and preprocess CSV data for evaluation.

    Args:
        input_csv: Path to input CSV file
        encoding: CSV encoding (default: auto-detect)
        max_samples_per_row: Maximum number of text samples to use per row (default: auto-detect)
        max_total_samples: Maximum total number of samples across all rows.
                          If specified, randomly samples from all available samples (default: no limit)

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
    num_candidates = num_rows

    # Auto compute max_samples_per_row if None
    if max_samples_per_row is None:
        max_samples_per_row = int(max((df[sample_cols].notna().sum(axis=1)).max(), 0))
        print(f"Auto-detected max_samples_per_row = {max_samples_per_row}")

    # Extract achievement standards
    contents = df["content"].astype(str).tolist()
    codes = df["code"].astype(str).tolist()

    # Prepare full candidates list (with all standards for dynamic sampling)
    full_candidates = [(i + 1, codes[i], contents[i]) for i in range(num_rows)]

    # Flatten sample texts, true codes, and generate candidates per sample
    sample_texts, samples_true_codes, samples_candidates = [], [], []
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
            samples_true_codes.append(code)

            # Generate candidates for this sample (same logic as prepare_training_dataset)
            # Limit candidates if specified (include correct answer + random sample)
            if max_candidates is not None and len(full_candidates) > max_candidates:
                # Find the correct candidate
                correct_idx = codes.index(code)
                correct_candidate = full_candidates[correct_idx]

                # Get other candidates (excluding the correct one)
                other_candidates = [
                    c for i, c in enumerate(full_candidates) if i != correct_idx
                ]

                # Randomly sample max_candidates - 1 other candidates
                sampled_others = random.sample(
                    other_candidates, min(max_candidates - 1, len(other_candidates))
                )

                # Combine correct + sampled candidates and shuffle
                candidates = [correct_candidate] + sampled_others
                random.shuffle(candidates)
            else:
                candidates = full_candidates.copy()

            samples_candidates.append(candidates)

    # Apply max_total_samples limit with random sampling if specified
    if max_total_samples is not None and len(sample_texts) > max_total_samples:
        # Random sampling
        indices = list(range(len(sample_texts)))
        random.shuffle(indices)
        selected_indices = sorted(indices[:max_total_samples])

        sample_texts = [sample_texts[i] for i in selected_indices]
        samples_true_codes = [samples_true_codes[i] for i in selected_indices]
        samples_candidates = [samples_candidates[i] for i in selected_indices]

        print(
            f"Total evaluation samples: {len(sample_texts)} (randomly sampled from {len(indices)} by max_total_samples={max_total_samples})"
        )
    else:
        print(f"Total evaluation samples: {len(sample_texts)}")

    num_samples = len(sample_texts)

    # Update num_candidates based on actual candidates per sample
    # (could vary per sample if max_candidates is used)
    if max_candidates is not None and num_rows > max_candidates:
        num_candidates = max_candidates
    else:
        num_candidates = num_rows

    return EvaluationData(
        df=df,
        contents=contents,
        codes=codes,
        sample_texts=sample_texts,
        samples_true_codes=samples_true_codes,
        samples_candidates=samples_candidates,
        subject=subject,
        num_rows=num_rows,
        num_candidates=num_candidates,
        num_samples=num_samples,
        max_samples_per_row=max_samples_per_row,
        max_candidates=max_candidates,
        folder_name=folder_name,
    )


def prepare_training_dataset(
    train_dir: str,
    tokenizer,
    encoding: str = None,
    max_samples_per_row: int = None,
    max_total_samples: int = None,
    max_candidates: int = None,
    seed: int = 42,
    few_shot: bool = True
):
    """
    Prepare training examples from CSV data in a directory.

    Args:
        train_dir: Directory containing training CSV files
        tokenizer: Tokenizer for applying chat template
        encoding: CSV encoding (default: auto-detect)
        max_samples_per_row: Maximum samples per row (default: None, use all)
        max_total_samples: Maximum total samples (default: None, use all)
        max_candidates: Maximum candidates per prompt (default: None, use all)
        seed: Random seed for shuffling dataset

    Returns:
        Dataset ready for SFTTrainer (with "text" field)
    """
    # Load all CSV files from directory
    csv_files = sorted(glob(os.path.join(train_dir, "*.csv")))
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {train_dir}")

    print(f"Found {len(csv_files)} CSV files in directory: {train_dir}")
    for csv_file in csv_files:
        print(f"  - {os.path.basename(csv_file)}")

    # Collect all training examples from all CSV files
    all_training_examples = []

    for csv_file in csv_files:
        print(f"\nLoading: {os.path.basename(csv_file)}")

        # Load data using existing utility function
        data = load_evaluation_data(
            csv_file,
            encoding,
            max_samples_per_row,
            None,  # Don't apply max_total_samples per file
            max_candidates,  # Pass max_candidates to generate candidates per sample
        )

        # Extract data
        sample_texts = data.sample_texts
        samples_true_codes = data.samples_true_codes
        samples_candidates = data.samples_candidates
        num_rows = data.num_rows

        print(f"  Achievement standards: {num_rows}")
        print(f"  Training samples: {len(sample_texts)}")

        # Create training examples for this file using pre-generated candidates
        for text, code, candidates in tqdm(
            zip(sample_texts, samples_true_codes, samples_candidates),
            desc=f"Creating prompts for {os.path.basename(csv_file)}",
            total=len(sample_texts),
        ):
            # Use the pre-generated candidates from load_evaluation_data
            # (already includes correct answer and random sampling if max_candidates was specified)

            # Create chat prompt for training with completion
            chat_prompt = create_chat_classification_prompt(text, candidates, code, few_shot=few_shot, subject=data.subject)

            all_training_examples.append(chat_prompt)

    print(f"\nTotal training examples from all files: {len(all_training_examples)}")

    # Apply max_total_samples if specified
    if max_total_samples is not None and len(all_training_examples) > max_total_samples:
        print(
            f"Randomly sampling {max_total_samples} examples from {len(all_training_examples)}"
        )
        all_training_examples = random.sample(all_training_examples, max_total_samples)

    # === Prepare dataset ===
    print("\nPreparing dataset...")

    # Convert messages to text using chat template
    print("Converting messages to text format...")
    text_examples = []
    for example in tqdm(all_training_examples, desc="Applying chat template"):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        text_examples.append({"text": text})

    train_dataset = Dataset.from_list(text_examples)
    train_dataset = train_dataset.shuffle(seed=seed)
    print(f"Dataset size: {len(train_dataset)} (shuffled)")

    return train_dataset
 