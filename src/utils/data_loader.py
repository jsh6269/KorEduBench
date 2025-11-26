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

# from src.test.check_prompt_length import subject
from src.utils.common import detect_encoding
from src.utils.prompt import create_chat_classification_prompt
from src.utils.rag_prompt import create_rag_chat_prompt


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
    max_candidates: int
    folder_name: Optional[str] = None  # train/valid/test folder name if detected


def load_evaluation_data(
    input_csv: str,
    encoding: Optional[str] = None,
    num_samples: Optional[int] = None,
    max_candidates: Optional[int] = None,
) -> EvaluationData:
    """
    Load and preprocess CSV data for evaluation.

    Args:
        input_csv: Path to input CSV file
        encoding: CSV encoding (default: auto-detect)
        num_samples: Target number of samples to generate.
                     If None, use all available samples.
                     If num_samples <= num_rows: randomly sample num_samples rows, then 1 sample per row.
                     If num_samples > num_rows: distribute samples across all rows (e.g., 10/3 = 3,3,4).
        max_candidates: Maximum number of candidates per sample (default: None, use all)

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

    # Extract achievement standards
    contents = df["content"].astype(str).tolist()
    codes = df["code"].astype(str).tolist()

    # Prepare full candidates list (with all standards for dynamic sampling)
    full_candidates = [(i + 1, codes[i], contents[i]) for i in range(num_rows)]

    # Collect all available texts per row
    # Use positional index (0-based) instead of DataFrame index
    row_texts_map = {}  # row_idx (0-based) -> list of texts
    for pos_idx, (_, row) in enumerate(df.iterrows()):
        code = str(row["code"])
        texts = []
        for col in sample_cols:
            text = str(row[col]).strip()
            if text and text.lower() != "nan":
                texts.append(text)
        row_texts_map[pos_idx] = texts

    # Apply num_samples sampling logic
    sample_texts, samples_true_codes, samples_candidates = [], [], []

    if num_samples is None:
        # Use all available samples
        for pos_idx, texts in row_texts_map.items():
            code = codes[pos_idx]
            for text in texts:
                sample_texts.append(text)
                samples_true_codes.append(code)

                # Generate candidates for this sample
                if max_candidates is not None and len(full_candidates) > max_candidates:
                    correct_idx = codes.index(code)
                    correct_candidate = full_candidates[correct_idx]
                    other_candidates = [
                        c for i, c in enumerate(full_candidates) if i != correct_idx
                    ]
                    sampled_others = random.sample(
                        other_candidates, min(max_candidates - 1, len(other_candidates))
                    )
                    candidates = [correct_candidate] + sampled_others
                    random.shuffle(candidates)
                else:
                    candidates = full_candidates.copy()
                samples_candidates.append(candidates)
    elif num_samples <= num_rows:
        # Randomly sample num_samples rows, then 1 sample per row
        # Only consider rows that have at least one text
        available_row_indices = [
            idx for idx in range(num_rows) if row_texts_map.get(idx)
        ]
        if len(available_row_indices) < num_samples:
            # If not enough rows with texts, use all available rows
            selected_row_indices = available_row_indices
        else:
            selected_row_indices = random.sample(available_row_indices, num_samples)

        for row_idx in selected_row_indices:
            texts = row_texts_map[row_idx]
            if not texts:
                continue  # Skip rows with no texts (shouldn't happen, but safety check)
            code = codes[row_idx]
            # Randomly sample 1 text from this row
            selected_text = random.choice(texts)
            sample_texts.append(selected_text)
            samples_true_codes.append(code)

            # Generate candidates for this sample
            if max_candidates is not None and len(full_candidates) > max_candidates:
                correct_idx = codes.index(code)
                correct_candidate = full_candidates[correct_idx]
                other_candidates = [
                    c for i, c in enumerate(full_candidates) if i != correct_idx
                ]
                sampled_others = random.sample(
                    other_candidates, min(max_candidates - 1, len(other_candidates))
                )
                candidates = [correct_candidate] + sampled_others
                random.shuffle(candidates)
            else:
                candidates = full_candidates.copy()
            samples_candidates.append(candidates)
    else:
        # num_samples > num_rows: distribute samples across all rows
        samples_per_row_base = num_samples // num_rows
        remainder = num_samples % num_rows

        # Distribute remainder samples (e.g., 10/3 = 3,3,4)
        samples_per_row = [samples_per_row_base] * num_rows
        for i in range(remainder):
            samples_per_row[i] += 1

        for row_idx in range(num_rows):
            texts = row_texts_map.get(row_idx, [])
            if not texts:
                continue  # Skip rows with no texts
            code = codes[row_idx]
            num_to_sample = min(samples_per_row[row_idx], len(texts))
            # Randomly sample num_to_sample texts from this row
            selected_texts = random.sample(texts, num_to_sample)

            for text in selected_texts:
                sample_texts.append(text)
                samples_true_codes.append(code)

                # Generate candidates for this sample
                if max_candidates is not None and len(full_candidates) > max_candidates:
                    correct_idx = codes.index(code)
                    correct_candidate = full_candidates[correct_idx]
                    other_candidates = [
                        c for i, c in enumerate(full_candidates) if i != correct_idx
                    ]
                    sampled_others = random.sample(
                        other_candidates, min(max_candidates - 1, len(other_candidates))
                    )
                    candidates = [correct_candidate] + sampled_others
                    random.shuffle(candidates)
                else:
                    candidates = full_candidates.copy()
                samples_candidates.append(candidates)

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
        num_samples=len(sample_texts),
        max_candidates=max_candidates,
        folder_name=folder_name,
    )


def prepare_training_dataset(
    train_dir: str,
    tokenizer,
    encoding: str = None,
    num_samples: int = None,
    max_total_samples: int = None,
    max_candidates: int = None,
    seed: int = 42,
    few_shot: bool = True,
):
    """
    Prepare training examples from CSV data in a directory.

    Args:
        train_dir: Directory containing training CSV files
        tokenizer: Tokenizer for applying chat template
        encoding: CSV encoding (default: auto-detect)
        num_samples: Target number of samples per CSV file (default: None, use all)
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
            num_samples,  # Pass num_samples to generate samples per file
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
            chat_prompt = create_chat_classification_prompt(
                text, candidates, code, few_shot=few_shot, subject=data.subject
            )

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


def prepare_rag_training_dataset(
    train_dir: str,
    tokenizer,
    train_csv: str,
    model_dir: str,
    top_k: int = 20,
    infer_device: str = "cuda",
    encoding: str = None,
    num_samples: int = None,
    max_total_samples: int = None,
    seed: int = 42,
    few_shot: bool = True,
    num_examples: int = 5,
):
    """
    Prepare training examples from CSV data in a directory using RAG workflow.
    Uses infer_top_k to retrieve top-k candidates for each sample.

    Args:
        train_dir: Directory containing training CSV files
        tokenizer: Tokenizer for applying chat template
        train_csv: Path to train CSV file for infer_top_k
        model_dir: Path to model directory for infer_top_k
        top_k: Number of top candidates to retrieve (default: 20)
        infer_device: Device for infer_top_k execution (default: "cuda")
        encoding: CSV encoding (default: auto-detect)
        num_samples: Target number of samples per CSV file (default: None, use all)
        max_total_samples: Maximum total samples (default: None, use all)
        seed: Random seed for shuffling dataset
        few_shot: Whether to include few-shot examples (default: True)
        num_examples: Number of few-shot examples (default: 5)

    Returns:
        Dataset ready for SFTTrainer (with "text" field)
    """
    # Import here to avoid circular dependencies
    from src.classification.inference import infer_top_k
    from src.classification.predict_multiclass import load_model

    # Load all CSV files from directory
    csv_files = sorted(glob(os.path.join(train_dir, "*.csv")))
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {train_dir}")

    print(f"Found {len(csv_files)} CSV files in directory: {train_dir}")
    for csv_file in csv_files:
        print(f"  - {os.path.basename(csv_file)}")

    # Load classification model for infer_top_k (load once, reuse for all samples)
    print(f"\nLoading classification model for RAG retrieval from: {model_dir}")
    top_k_model, top_k_tokenizer, top_k_config, top_k_mappings = load_model(
        model_dir, infer_device
    )
    print(f"Top-k retrieval model loaded. Using top_k={top_k}")

    # Collect all training examples from all CSV files
    all_training_examples = []

    for csv_file in csv_files:
        print(f"\nLoading: {os.path.basename(csv_file)}")

        # Load data using existing utility function (without max_candidates for RAG)
        data = load_evaluation_data(
            csv_file,
            encoding,
            num_samples,  # Pass num_samples to generate samples per file
            max_candidates=None,  # RAG uses infer_top_k instead
        )

        # Extract data
        sample_texts = data.sample_texts
        samples_true_codes = data.samples_true_codes
        num_rows = data.num_rows

        print(f"  Achievement standards: {num_rows}")
        print(f"  Training samples: {len(sample_texts)}")

        # Create training examples for this file using RAG workflow
        for text, code in tqdm(
            zip(sample_texts, samples_true_codes),
            desc=f"Creating RAG prompts for {os.path.basename(csv_file)}",
            total=len(sample_texts),
        ):
            # Call infer_top_k to retrieve top-k candidates
            infer_result = infer_top_k(
                text=text,
                top_k=top_k,
                train_csv=train_csv,
                model=top_k_model,
                tokenizer=top_k_tokenizer,
                config=top_k_config,
                mappings=top_k_mappings,
                device=infer_device,
                random=False,  # Keep probability order for training
            )

            # Convert infer_top_k result to (rank, code, content) tuple list
            candidates = [
                (idx + 1, item["code"], item["content"])
                for idx, item in enumerate(infer_result["top-k"])
            ]

            # Create RAG chat prompt for training with completion
            chat_prompt = create_rag_chat_prompt(
                text=text,
                candidates=candidates,
                completion=code,
                for_inference=False,
                few_shot=few_shot,
                subject=data.subject,
                num_examples=num_examples,
            )

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
