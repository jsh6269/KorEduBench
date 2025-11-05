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
from pathlib import Path

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

# Get project root (3 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import load_evaluation_data
from src.utils.prompt import create_classification_prompt
from src.utils.random_seed import set_train_random_seed


def prepare_training_examples(
    train_csv: str,
    encoding: str = None,
    max_samples_per_row: int = None,
    max_total_samples: int = None,
):
    """
    Prepare training examples from CSV data.

    Args:
        train_csv: Path to training CSV file
        encoding: CSV encoding (default: auto-detect)
        max_samples_per_row: Maximum samples per row (default: None, use all)
        max_total_samples: Maximum total samples (default: None, use all)

    Returns:
        List of training examples with prompts and completions
    """
    # Load data using existing utility function
    print("Loading training data...")
    data = load_evaluation_data(
        train_csv, encoding, max_samples_per_row, max_total_samples
    )

    # Extract data
    contents = data.contents
    codes = data.codes
    sample_texts = data.sample_texts
    true_codes = data.true_codes
    num_rows = data.num_rows

    print(f"Total achievement standards: {num_rows}")
    print(f"Total training samples: {len(sample_texts)}")

    # Prepare candidates list (for prompt generation)
    candidates = [(i + 1, codes[i], contents[i]) for i in range(num_rows)]

    # Create training examples
    training_examples = []
    print("Preparing training examples...")
    for text, code in tqdm(
        zip(sample_texts, true_codes), desc="Creating prompts", total=len(sample_texts)
    ):
        # Create prompt using the same format as evaluation
        prompt = create_classification_prompt(text, candidates)

        # The completion is just the code
        completion = code

        training_examples.append(
            {
                "prompt": prompt,
                "completion": completion,
                "text": text,
                "code": code,
            }
        )

    print(f"Total training examples prepared: {len(training_examples)}")
    return training_examples


def format_prompt_for_training(example):
    """
    Format prompt and completion for training.

    Args:
        example: Dictionary with 'prompt' and 'completion' keys

    Returns:
        Formatted text for training
    """
    return f"{example['prompt']}\n{example['completion']}"


def finetune_llm(
    train_csv: str,
    model_name: str = "unsloth/Qwen2.5-1.5B-Instruct",
    output_dir: str = None,
    max_seq_length: int = 2048,
    max_samples_per_row: int = None,
    max_total_samples: int = None,
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
    use_gradient_checkpointing: bool = True,
    seed: int = 42,
):
    """
    Fine-tune an LLM for educational achievement standard classification.

    Args:
        train_csv: Path to training CSV file
        model_name: Hugging Face model name (unsloth optimized)
        output_dir: Directory to save the fine-tuned model
        max_seq_length: Maximum sequence length
        max_samples_per_row: Maximum samples per row
        max_total_samples: Maximum total samples (random sample if exceeded)
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
    print(f"Training data: {train_csv}")
    print(f"Output directory: {output_dir}")
    print()

    # === Load training data ===
    training_examples = prepare_training_examples(
        train_csv, encoding, max_samples_per_row, max_total_samples
    )

    # === Load model with Unsloth ===
    print(f"\nLoading model with Unsloth: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=load_in_4bit,
    )

    # === Add LoRA adapters ===
    print("\nAdding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
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
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    # === Prepare dataset ===
    print("\nPreparing dataset...")

    # Format training data for SFTTrainer
    formatted_examples = []
    for example in training_examples:
        formatted_examples.append({"text": format_prompt_for_training(example)})

    train_dataset = Dataset.from_list(formatted_examples)
    print(f"Dataset size: {len(train_dataset)}")

    # === Set up training arguments ===
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        warmup_steps=warmup_steps,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=seed,
        report_to="none",  # Disable wandb/tensorboard
    )

    # === Create trainer ===
    print("\nInitializing trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
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

    # Save merged model (16-bit)
    print("\nSaving merged 16-bit model...")
    merged_output_dir = os.path.join(output_dir, "merged_16bit")
    model.save_pretrained_merged(
        merged_output_dir,
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Merged 16-bit model saved to: {merged_output_dir}")

    # Optionally save merged model (4-bit quantized)
    print("\nSaving merged 4-bit model...")
    merged_4bit_output_dir = os.path.join(output_dir, "merged_4bit")
    model.save_pretrained_merged(
        merged_4bit_output_dir,
        tokenizer,
        save_method="merged_4bit",
    )
    print(f"Merged 4-bit model saved to: {merged_4bit_output_dir}")

    # === Save training info ===
    training_info = {
        "model_name": model_name,
        "train_csv": train_csv,
        "num_examples": len(training_examples),
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
        "use_gradient_checkpointing": use_gradient_checkpointing,
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
        "--train_csv",
        type=str,
        required=True,
        help="Path to training CSV file.",
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
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (default: enabled).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()

    finetune_llm(
        train_csv=args.train_csv,
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        max_samples_per_row=args.max_samples_per_row,
        max_total_samples=args.max_total_samples,
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
        use_gradient_checkpointing=not args.no_gradient_checkpointing,
        seed=args.seed,
    )
