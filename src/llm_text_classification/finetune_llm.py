"""
LLM Fine-tuning script using Unsloth.
Fine-tunes a language model for educational achievement standard classification.

Requirements:
    - unsloth (install via: pip install unsloth)
    - transformers==4.56.2
    - trl==0.22.2
    - torch==2.8.0 with xformers==0.0.32.post2
    
For installation instructions, see README.md
"""

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

# isort: off
from unsloth import FastLanguageModel  # import unsloth before trl to avoid conflicts

# isort: on

import torch
from datasets import Dataset
from tqdm import tqdm
from trl import SFTConfig, SFTTrainer

# Get project root (3 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import load_evaluation_data
from src.utils.prompt import (
    create_chat_classification_prompt,
    create_classification_prompt,
)
from src.utils.random_seed import set_train_random_seed


def prepare_training_dataset(
    train_dir: str,
    tokenizer,
    encoding: str = None,
    max_samples_per_row: int = None,
    max_total_samples: int = None,
    max_candidates: int = None,
    seed: int = 42,
    return_codes: bool = False,
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
        return_codes: If True, also return list of unique achievement standard codes

    Returns:
        If return_codes=False: Dataset ready for SFTTrainer (with "text" field)
        If return_codes=True: Tuple of (Dataset, list of unique codes)
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
    all_codes = []  # Collect codes if needed

    for csv_file in csv_files:
        print(f"\nLoading: {os.path.basename(csv_file)}")

        # Load data using existing utility function
        data = load_evaluation_data(
            csv_file,
            encoding,
            max_samples_per_row,
            None,  # Don't apply max_total_samples per file
        )

        # Extract data
        contents = data.contents
        codes = data.codes
        sample_texts = data.sample_texts
        true_codes = data.true_codes
        num_rows = data.num_rows

        print(f"  Achievement standards: {num_rows}")
        print(f"  Training samples: {len(sample_texts)}")

        # Collect codes for tokenizer
        all_codes.extend(codes)

        # Prepare full candidates list
        full_candidates = [(i + 1, codes[i], contents[i]) for i in range(num_rows)]

        # Create training examples for this file
        for text, code in tqdm(
            zip(sample_texts, true_codes),
            desc=f"Creating prompts for {os.path.basename(csv_file)}",
            total=len(sample_texts),
        ):
            # Limit candidates if specified (include correct answer + random sample)
            if max_candidates is not None and len(full_candidates) > max_candidates:
                import random

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
                candidates = full_candidates

            # Create chat prompt for training with completion
            chat_prompt = create_chat_classification_prompt(text, candidates, code)

            all_training_examples.append(chat_prompt)

    print(f"\nTotal training examples from all files: {len(all_training_examples)}")

    # Apply max_total_samples if specified
    if max_total_samples is not None and len(all_training_examples) > max_total_samples:
        import random

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

    # Return unique codes if requested
    if return_codes:
        unique_codes_list = sorted(list(set(all_codes)))
        print(f"Unique achievement standard codes: {len(unique_codes_list)}")
        return train_dataset, unique_codes_list

    return train_dataset


def finetune_llm(
    train_dir: str,
    model_name: str = "unsloth/Qwen2.5-1.5B-Instruct",
    output_dir: str = None,
    max_seq_length: int = 2048,
    max_samples_per_row: int = None,
    max_total_samples: int = None,
    max_candidates: int = None,
    encoding: str = None,
    # Training hyperparameters
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_steps: int = 5,
    logging_steps: int = 10,
    save_steps: int = 100,
    # LoRA parameters
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    # Other options
    load_in_4bit: bool = True,
    seed: int = 42,
):
    """
    Fine-tune an LLM for educational achievement standard classification.

    Args:
        train_dir: Directory containing training CSV files
        model_name: Hugging Face model name (unsloth optimized)
        output_dir: Directory to save the fine-tuned model
        max_seq_length: Maximum sequence length
        max_samples_per_row: Maximum samples per row
        max_total_samples: Maximum total samples (random sample if exceeded)
        max_candidates: Maximum candidates per prompt (random sample if exceeded)
        encoding: CSV encoding
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        logging_steps: Logging frequency
        save_steps: Save checkpoint frequency
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        load_in_4bit: Use 4-bit quantization
        use_gradient_checkpointing: Use gradient checkpointing to save memory
        seed: Random seed for reproducibility
    """
    # === Set random seed for reproducibility ===
    print("=" * 80)
    print("LLM Fine-tuning with Unsloth")
    print("=" * 80)
    print(f"Setting random seed: {seed}")
    set_train_random_seed(seed)

    if output_dir is None:
        output_dir = PROJECT_ROOT / "model" / "finetuned_llm"
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Model: {model_name}")
    print(f"Training data directory: {train_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # === Load model with Unsloth ===
    print(f"\nLoading model with Unsloth: {model_name}")
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=load_in_4bit,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    for cfg in (base_model.generation_config, base_model.config):
        cfg.eos_token_id = eos_id
        cfg.pad_token_id = pad_id
        if hasattr(cfg, "eos_token"):
            cfg.eos_token = tokenizer.eos_token

    # === Load training data ===
    train_dataset, unique_codes = prepare_training_dataset(
        train_dir,
        tokenizer,
        encoding,
        max_samples_per_row,
        max_total_samples,
        max_candidates,
        seed,
        return_codes=True,
    )

    # === Add achievement standard codes as special tokens ===
    print(
        f"\nAdding {len(unique_codes)} achievement standard codes as special tokens..."
    )
    num_added_tokens = tokenizer.add_tokens(unique_codes)
    print(f"  Added {num_added_tokens} new tokens to tokenizer")
    print(f"  New vocabulary size: {len(tokenizer)}")

    # Resize model embeddings to accommodate new tokens
    # Only resize if needed, use mean_resizing=False to save memory
    current_embed_size = base_model.get_input_embeddings().num_embeddings
    new_size = len(tokenizer)
    if new_size > current_embed_size:
        base_model.resize_token_embeddings(new_size, mean_resizing=False)

    # === Add LoRA adapters ===
    print("\nAdding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        base_model,
        r=lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    # === Create trainer with SFTConfig ===
    print("\nInitializing trainer...")
    # Try using processing_class instead of tokenizer for newer TRL versions

    sft_config = SFTConfig(
        output_dir=output_dir,
        overwrite_output_dir=True,
        # dataset_text_field removed - SFTTrainer auto-detects "messages" field for chat format
        max_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=1,
        warmup_steps=warmup_steps,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=seed,
        report_to="none",
        # completion_only_loss="<|im_start|>assistant\n",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=sft_config,
    )

    # === Train ===
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    trainer.train()

    # === Save model ===
    print("\n" + "=" * 80)
    print("Saving fine-tuned model...")
    print("=" * 80)

    # Save LoRA adapters
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"LoRA adapters saved to: {output_dir}")

    # === Save training info ===
    training_info = {
        "model_name": model_name,
        "train_dir": train_dir,
        "num_examples": len(train_dataset),
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "max_seq_length": max_seq_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "seed": seed,
        "load_in_4bit": load_in_4bit,
        "use_gradient_checkpointing": "unsloth",
    }

    info_path = os.path.join(output_dir, "training_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(training_info, f, ensure_ascii=False, indent=4)
    print(f"\nTraining info saved to: {info_path}")

    print("\n" + "=" * 80)
    print("Fine-tuning completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune LLM for educational achievement standard classification using Unsloth."
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="Directory containing training CSV files.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/Qwen2.5-1.5B-Instruct",
        help="Hugging Face model name (default: unsloth/Qwen2.5-1.5B-Instruct).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for fine-tuned model (default: {PROJECT_ROOT}/model/finetuned_llm).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048).",
    )
    parser.add_argument(
        "--max-samples-per-row",
        type=int,
        default=None,
        help="Maximum samples per row (default: use all).",
    )
    parser.add_argument(
        "--max-total-samples",
        type=int,
        default=None,
        help="Maximum total samples, randomly sampled if exceeded (default: use all).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Maximum candidates per prompt, randomly sampled if exceeded (default: use all).",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default=None,
        help="CSV encoding (default: auto-detect).",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3).",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=2,
        help="Batch size per device (default: 2).",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Number of warmup steps (default: 5).",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Logging frequency (default: 10).",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Save checkpoint frequency (default: 100).",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha (default: 16).",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout (default: 0.0).",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (default: enabled).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()

    finetune_llm(
        train_dir=args.train_dir,
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        max_samples_per_row=args.max_samples_per_row,
        max_total_samples=args.max_total_samples,
        max_candidates=args.max_candidates,
        encoding=args.encoding,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        load_in_4bit=not args.no_4bit,
        seed=args.seed,
    )
