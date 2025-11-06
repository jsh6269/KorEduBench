#!/usr/bin/env python3
"""
Check prompt length for LLM classification to ensure it doesn't exceed model limits.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import load_evaluation_data
from src.utils.prompt import create_classification_prompt


def check_prompt_length(csv_path: str, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """Check prompt length for a given CSV file."""
    print(f"Checking prompt length for: {csv_path}")
    print(f"Model: {model_name}\n")

    # Load data
    print("Loading data...")
    data = load_evaluation_data(csv_path)

    contents = data.contents
    codes = data.codes
    sample_texts = data.sample_texts

    # Prepare candidates
    candidates = [(i + 1, codes[i], contents[i]) for i in range(len(codes))]

    print(f"Number of achievement standards: {len(candidates)}")
    print(f"Number of sample texts: {len(sample_texts)}")

    # Load tokenizer
    print(f"\nLoading tokenizer for {model_name}...")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Using character count approximation (1 token ≈ 4 characters)")
        tokenizer = None

    # Check a few sample prompts
    print("\n" + "=" * 80)
    print("Checking sample prompts...")
    print("=" * 80)

    sample_indices = [0, len(sample_texts) // 2, len(sample_texts) - 1]

    for idx in sample_indices:
        if idx >= len(sample_texts):
            continue

        text = sample_texts[idx]
        prompt = create_classification_prompt(text, candidates)

        if tokenizer:
            tokens = tokenizer(prompt, return_tensors="pt")
            token_count = tokens["input_ids"].shape[1]
            char_count = len(prompt)

            print(f"\nSample #{idx}:")
            print(f"  Text length: {len(text)} characters")
            print(f"  Prompt length: {char_count} characters, {token_count} tokens")
            print(f"  Text preview: {text[:100]}...")
        else:
            char_count = len(prompt)
            estimated_tokens = char_count // 4

            print(f"\nSample #{idx}:")
            print(f"  Text length: {len(text)} characters")
            print(f"  Prompt length: {char_count} characters")
            print(f"  Estimated tokens: ~{estimated_tokens} tokens")
            print(f"  Text preview: {text[:100]}...")

    # Show model limits
    print("\n" + "=" * 80)
    print("Model Limits")
    print("=" * 80)

    if tokenizer and hasattr(tokenizer, "model_max_length"):
        print(f"Tokenizer max length: {tokenizer.model_max_length}")

    print("\nCommon model max lengths:")
    print("  - Qwen2.5-1.5B-Instruct: 32768 tokens")
    print("  - Qwen2.5-7B-Instruct: 32768 tokens")
    print("  - GPT-3.5-turbo: 4096 tokens")
    print("  - GPT-4: 8192 tokens")

    print("\n" + "=" * 80)
    print("Recommendations")
    print("=" * 80)

    if tokenizer:
        # Use the actual token count from first sample
        prompt = create_classification_prompt(sample_texts[0], candidates)
        tokens = tokenizer(prompt, return_tensors="pt")
        typical_length = tokens["input_ids"].shape[1]

        if typical_length > 32768:
            print("⚠️  CRITICAL: Prompts exceed most model limits!")
            print(f"   Typical length: {typical_length} tokens")
            print(f"   Recommendation: Use RAG or candidate filtering")
        elif typical_length > 16384:
            print("⚠️  WARNING: Prompts are very long")
            print(f"   Typical length: {typical_length} tokens")
            print(f"   Recommendation: Use --max-input-length 16384 or higher")
        elif typical_length > 8192:
            print("⚠️  CAUTION: Prompts are moderately long")
            print(f"   Typical length: {typical_length} tokens")
            print(f"   Recommendation: Use --max-input-length 8192 or higher")
        else:
            print("✓ Prompts are within reasonable limits")
            print(f"   Typical length: {typical_length} tokens")
            print(f"   Default setting (8192 tokens) should work fine")
    else:
        print("Unable to determine token count without tokenizer")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Check prompt length for LLM classification"
    )
    parser.add_argument("--input_csv", type=str, required=True, help="Path to CSV file")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name for tokenizer",
    )

    args = parser.parse_args()
    check_prompt_length(args.input_csv, args.model_name)
