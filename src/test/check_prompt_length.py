#!/usr/bin/env python3
"""
Check prompt length for LLM classification to ensure it doesn't exceed model limits.
"""

import csv
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer

from src.utils.data_loader import load_evaluation_data
from src.utils.prompt import create_chat_classification_prompt


def check_prompt_length(
    input_csv: str,
    model_name: str,
    few_shot: bool,
    max_samples_per_row: int,
    max_total_samples: int,
    max_candidates: int,
):
    # Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # === Load and preprocess data for the CSV ===
    print(f"\nLoading evaluation data from: {input_csv}")
    data = load_evaluation_data(
        input_csv=input_csv,
        encoding=None,
        max_samples_per_row=max_samples_per_row,
        max_total_samples=max_total_samples,
        max_candidates=max_candidates,
    )

    # Extract data for convenience
    contents = data.contents
    codes = data.codes
    sample_texts = data.sample_texts
    subject = data.subject
    num_rows = data.num_rows
    num_candidates = data.num_candidates
    num_samples = data.num_samples
    candidates = [(i + 1, codes[i], contents[i]) for i in range(num_candidates)]

    print("Prompt statistics:")
    print(f"  Subject: {subject}")
    print(f"  CSV: {Path(input_csv).name}")
    print(f"  Number of candidates: {len(candidates)}")
    print(f"  Max samples per row: {max_samples_per_row}")

    # Build a sample prompt (first sample)
    chat_messages = create_chat_classification_prompt(
        sample_texts[0],
        candidates,
        completion="",
        for_inference=True,
        few_shot=few_shot,
        subject=subject,
    )

    if hasattr(tokenizer, "apply_chat_template"):
        sample_prompt = tokenizer.apply_chat_template(
            chat_messages["messages"], tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback: simple concatenation if no chat template available
        sample_prompt = "\n".join(
            [m.get("content", "") for m in chat_messages.get("messages", [])]
        )

    sample_tokens = tokenizer(sample_prompt, return_tensors="pt")
    sample_prompt_length = int(sample_tokens["input_ids"].shape[1])

    print(f"  Sample prompt token length: {sample_prompt_length}")
    print(f"  Total samples: {num_samples}")

    # Compute total and average token counts across all samples
    total_tokens = 0
    for text in sample_texts:
        chat_messages = create_chat_classification_prompt(
            text,
            candidates,
            completion="",
            for_inference=True,
            few_shot=few_shot,
            subject=subject,
        )
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                chat_messages["messages"], tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = "\n".join(
                [m.get("content", "") for m in chat_messages.get("messages", [])]
            )
        tokenized = tokenizer(prompt, return_tensors="pt")
        total_tokens += int(tokenized["input_ids"].shape[1])

    if num_samples > 0:
        avg_tokens = total_tokens / num_samples
        print(f"  Average prompt token length (all samples): {avg_tokens:.1f}")
        print(f"  Total prompt tokens (sum over all samples): {total_tokens}")
    else:
        avg_tokens = 0.0
        print("  No samples available.")

    return {
        "subject": subject,
        "csv": str(input_csv),
        "num_rows": int(num_rows),
        "num_candidates": int(num_candidates),
        "num_samples": int(num_samples),
        "sample_prompt_tokens": int(sample_prompt_length),
        "total_prompt_tokens": int(total_tokens),
        "avg_prompt_tokens": float(avg_tokens),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Check prompt length for LLM classification across all CSVs in a folder"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to folder containing CSV files",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model name for tokenizer",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to output file (default: {PROJECT_ROOT}/output/check_prompt_length/results.csv).",
    )
    parser.add_argument(
        "--max-samples-per-row",
        type=int,
        default=None,
        help="Max number of text samples to evaluate per row (default: auto-detect).",
    )
    parser.add_argument(
        "--max-total-samples",
        type=int,
        default=None,
        help="Max total number of samples, randomly sampled from all available (default: no limit).",
    )
    # Local model arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for local model (cuda or cpu, ignored for API, default: cuda).",
    )

    parser.add_argument(
        "--few-shot",
        action="store_true",
        help="Use few-shot examples (default: False).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=200,
        help="Maximum number of candidate achievement standards to use (default: 200).",
    )

    args = parser.parse_args()

    # Set default output path if not provided
    if args.output_path is None:
        output_path = PROJECT_ROOT / "output" / "check_prompt_length" / "results.csv"
    else:
        output_path = Path(args.output_path)

    # Iterate CSVs in input_dir and aggregate results
    input_path = Path(args.input_dir)
    csv_files = sorted(input_path.rglob("*.csv"))
    if not csv_files:
        print(f"No CSV files found under: {args.input_dir}")
        sys.exit(0)

    results = []
    for csv_path in csv_files:
        r = check_prompt_length(
            input_csv=str(csv_path),
            model_name=args.model_name,
            few_shot=args.few_shot,
            max_samples_per_row=args.max_samples_per_row,
            max_total_samples=args.max_total_samples,
            max_candidates=args.max_candidates,
        )
        results.append(r)

    # Aggregate by subject and print summary
    from collections import defaultdict

    by_subject = defaultdict(
        lambda: {"num_files": 0, "total_samples": 0, "total_tokens": 0}
    )
    for r in results:
        s = r["subject"]
        by_subject[s]["num_files"] += 1
        by_subject[s]["total_samples"] += r["num_samples"]
        by_subject[s]["total_tokens"] += r["total_prompt_tokens"]

    print("\n=== Per-subject summary ===")
    overall_samples = 0
    overall_tokens = 0
    for subject in sorted(by_subject.keys()):
        info = by_subject[subject]
        avg = (
            (info["total_tokens"] / info["total_samples"])
            if info["total_samples"] > 0
            else 0.0
        )
        overall_samples += info["total_samples"]
        overall_tokens += info["total_tokens"]
        print(
            f"- {subject}: files={info['num_files']}, samples={info['total_samples']}, avg_tokens={avg:.1f}, total_tokens={info['total_tokens']}"
        )

    if overall_samples > 0:
        overall_avg = overall_tokens / overall_samples
        print(
            f"\nOverall: samples={overall_samples}, avg_tokens={overall_avg:.1f}, total_tokens={overall_tokens}"
        )
    else:
        print("\nOverall: No samples found.")

    # Save results to CSV
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "subject",
            "num_rows",
            "num_candidates",
            "num_samples",
            "sample_prompt_tokens",
            "total_prompt_tokens",
            "avg_prompt_tokens",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            # Write each result (exclude 'csv' field)
            row = {k: v for k, v in r.items() if k != "csv"}
            writer.writerow(row)

        # Add total row
        if overall_samples > 0:
            total_row = {
                "subject": "TOTAL",
                "num_rows": sum(r["num_rows"] for r in results),
                "num_candidates": "",  # Not meaningful to sum
                "num_samples": overall_samples,
                "sample_prompt_tokens": "",  # Not meaningful to sum
                "total_prompt_tokens": overall_tokens,
                "avg_prompt_tokens": round(overall_avg, 1),
            }
            writer.writerow(total_row)

    print(f"\n✓ Results saved to: {output_path}")
