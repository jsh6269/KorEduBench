"""
RAG-based LLM text classification evaluation script.
Uses retrieval-augmented generation (RAG) to classify textbook excerpts into educational achievement standards.
The retriever directly fetches top-K relevant candidates based on the input text,
then the LLM selects from these retrieved candidates.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Get project root (3 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.inference import infer_top_k
from src.classification.predict_multiclass import load_model
from src.utils.prompt import (
    LLMClassificationResponse,
    create_chat_classification_prompt,
    parse_llm_response,
)
from src.utils.api_model import create_api_client, create_api_generate_function
from src.utils.data_loader import load_evaluation_data
from src.utils.model import create_generate_function, load_llm_model
from src.utils.random_seed import set_predict_random_seed


def evaluate_llm_classification(
    input_csv: str,
    generate_fn: callable,
    model_identifier: str,
    tokenizer=None,
    few_shot: bool = False,
    encoding: str = None,
    json_path: str = None,
    max_samples_per_row: int = None,
    max_total_samples: int = None,
    max_new_tokens: int = 10,
    temperature: float = 0.1,
    max_input_length: int = 8192,
    train_csv: str = None,
    model_dir: str = None,
    top_k: int = 20,
    infer_device: str = "cuda",
    num_examples: int = 5,
    check_token_length: bool = True,
):
    """
    Evaluate LLM-based classification on educational content using RAG workflow.

    Args:
        input_csv: Path to input CSV file
        generate_fn: Function to generate predictions (callable that takes prompt str and returns str)
        model_identifier: Model name or API identifier
        tokenizer: Tokenizer for token length checking (optional, None for API mode)
        few_shot: Use few-shot examples
        encoding: CSV encoding (default: auto-detect)
        json_path: Path to save results JSON
        max_samples_per_row: Maximum samples per row
        max_total_samples: Maximum total samples (randomly sampled if specified)
        max_new_tokens: Maximum tokens to generate (passed to generate_fn)
        temperature: Sampling temperature (passed to generate_fn)
        max_input_length: Maximum input length (will truncate if exceeded, for local models)
        train_csv: Path to train CSV file for infer_top_k
        model_dir: Path to model directory for infer_top_k
        top_k: Number of candidates to retrieve via RAG (default: 20)
        infer_device: Device for infer_top_k execution (default: "cuda")
        num_examples: Number of few-shot examples (default: 5)
        check_token_length: Whether to check token length (False for API mode)
    """
    if json_path is None:
        # Generate filename with date and model_name
        current_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
        # Sanitize model_name for filename (replace / and other invalid chars with _)
        safe_model_name = (
            model_identifier.replace("/", "_").replace("\\", "_").replace(":", "_")
        )
        json_path = (
            PROJECT_ROOT
            / "output"
            / "rag_llm_text_classification"
            / f"results_{safe_model_name}_{current_date}.json"
        )
    json_path = str(json_path)

    # === Load and preprocess data ===
    print("Loading evaluation data...")
    data = load_evaluation_data(
        input_csv=input_csv,
        encoding=encoding,
        max_samples_per_row=max_samples_per_row,
        max_total_samples=max_total_samples,
        max_candidates=None,
    )

    # Extract data for convenience
    contents = data.contents
    codes = data.codes
    sample_texts = data.sample_texts
    samples_true_codes = data.samples_true_codes
    subject = data.subject
    num_rows = data.num_rows
    num_samples = data.num_samples
    max_samples_per_row = data.max_samples_per_row
    folder_name = data.folder_name

    # === Check prompt length ===
    print(f"\nPrompt statistics:")
    print(f"  Top-k candidates to retrieve: {top_k}")

    if tokenizer is not None:
        # Create temporary candidates list for length estimation
        temp_candidates = [
            (i + 1, codes[i], contents[i]) for i in range(min(top_k, len(codes)))
        ]
        sample_messages = create_chat_classification_prompt(
            text=sample_texts[0],
            candidates=temp_candidates,
            completion="",
            for_inference=True,
            few_shot=few_shot,
            subject=subject,
            num_examples=num_examples,
        )
        sample_prompt = tokenizer.apply_chat_template(
            sample_messages["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        sample_tokens = tokenizer(sample_prompt, return_tensors="pt")
        prompt_length = sample_tokens["input_ids"].shape[1]

        print(f"  Sample prompt token length: {prompt_length}")

        if check_token_length:
            # Local model: check against max_input_length
            print(f"  Max input length: {max_input_length}")

            if prompt_length > max_input_length:
                print(
                    f"  ⚠️  WARNING: Prompt length ({prompt_length}) exceeds max_input_length ({max_input_length})"
                )
                print(f"  ⚠️  Prompts will be truncated, which may affect accuracy!")
                print(
                    f"  ⚠️  Consider increasing --max-input-length or reducing --top-k."
                )
            else:
                print(f"  ✓ Prompt length is within limits")
    else:
        print(f"  Token length check: Skipped (no tokenizer provided)")

    # === Load retrieval model ===
    print(f"\nLoading retrieval model from {model_dir}...")
    top_k_model, top_k_tokenizer, top_k_config, top_k_mappings = load_model(
        model_dir, infer_device
    )

    # === Prediction ===
    print(f"\nPredicting classifications for {num_samples} samples using RAG...")
    predictions = []
    wrong_samples = []
    correct_samples = []
    truncated_count = 0
    exact_match_count = 0
    match_type_counts = {}

    for i in tqdm(range(num_samples), desc="Classifying"):
        text = sample_texts[i]
        true_code = samples_true_codes[i]

        # RAG Step 1: Retrieve top-K candidates directly from input text
        infer_result = infer_top_k(
            text=text,  # Use input text directly (not a generated query)
            top_k=top_k,
            train_csv=train_csv,
            model=top_k_model,
            tokenizer=top_k_tokenizer,
            config=top_k_config,
            mappings=top_k_mappings,
            device=infer_device,
            random=False,  # Keep probability order
        )

        # Convert infer_top_k result to (rank, code, content) tuple list
        candidates = [
            (idx + 1, item["code"], item["content"])
            for idx, item in enumerate(infer_result["top-k"])
        ]

        # RAG Step 2: LLM selects from retrieved candidates
        chat_messages = create_chat_classification_prompt(
            text=text,
            candidates=candidates,
            completion="",
            for_inference=True,
            few_shot=few_shot,
            subject=subject,
            num_examples=num_examples,
        )

        if tokenizer is not None:
            # Local models: convert chat messages to string using tokenizer's chat template
            prompt = tokenizer.apply_chat_template(
                chat_messages["messages"], tokenize=False, add_generation_prompt=True
            )
        else:
            # API models: pass messages list directly (API natively supports chat format)
            prompt = chat_messages["messages"]

        # Check if this specific prompt will be truncated (only for local models)
        if check_token_length and tokenizer is not None:
            prompt_tokens = tokenizer(prompt, return_tensors="pt")
            if prompt_tokens["input_ids"].shape[1] > max_input_length:
                truncated_count += 1

        # Generate prediction
        response = generate_fn(prompt)

        # Parse response - now returns LLMClassificationResponse object
        llm_response = parse_llm_response(response, candidates)
        pred_code = llm_response.predicted_code

        # Track match types (exact, partial, invalid)
        match_type_str = llm_response.match_type.value
        match_type_counts[match_type_str] = match_type_counts.get(match_type_str, 0) + 1

        if llm_response.is_exact_match:
            exact_match_count += 1

        predictions.append(pred_code)

        # Track wrong and correct predictions
        true_content = contents[codes.index(true_code)] if true_code in codes else "N/A"
        pred_content = (
            contents[codes.index(pred_code)] if pred_code in codes else "INVALID"
        )

        sample_info = {
            "sample_idx": i,
            "input_text": text,
            "true_code": true_code,
            "pred_code": pred_code,
            "true_content": true_content,
            "pred_content": pred_content,
            "llm_response": response,
            "match_type": match_type_str,
            "confidence": llm_response.confidence,
            "is_exact_match": llm_response.is_exact_match,
            "retrieved_candidates": [
                {"rank": idx + 1, "code": item["code"], "content": item["content"]}
                for idx, item in enumerate(infer_result["top-k"])
            ],
        }

        if pred_code != true_code:
            wrong_samples.append(sample_info)
        else:
            correct_samples.append(sample_info)

    # === Evaluation Metrics ===
    print("\nCalculating metrics...")

    # LLM only produces top-1 predictions
    correct = sum(
        1 for pred, true in zip(predictions, samples_true_codes) if pred == true
    )
    accuracy = correct / num_samples

    # === Summary ===
    print("\n=== RAG-based LLM Text Classification Evaluation ===")
    print(f"Model: {model_identifier}")
    print(f"Subject: {subject}")
    print(f"Total achievement standards: {num_rows}")
    print(f"Top-k candidates retrieved: {top_k}")
    print(f"Samples evaluated: {num_samples}")
    if truncated_count > 0:
        print(
            f"⚠️  Truncated prompts: {truncated_count}/{num_samples} ({truncated_count/num_samples*100:.1f}%)"
        )
    print(f"\nMatch Type Distribution:")
    for match_type, count in sorted(match_type_counts.items()):
        print(f"  {match_type}: {count}/{num_samples} ({count/num_samples*100:.1f}%)")
    print(
        f"\nExact matches: {exact_match_count}/{num_samples} ({exact_match_count/num_samples*100:.1f}%)"
    )
    print(f"Accuracy: {accuracy:.4f} ({correct}/{num_samples})")

    # === JSON Logging ===
    result_entry = {}
    if folder_name:
        result_entry["folder"] = folder_name

    result_entry.update(
        {
            "subject": subject,
            "num_standards": num_rows,
            "top_k": top_k,
            "total_samples": num_samples,
            "correct": correct,
            "accuracy": round(float(accuracy), 4),
            "exact_match_count": exact_match_count,
            "exact_match_rate": round(float(exact_match_count / num_samples), 4),
            "match_type_counts": match_type_counts,
            "truncated_count": truncated_count,
            "model": model_identifier,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "few_shot": few_shot,
            "num_examples": num_examples if few_shot else 0,
        }
    )

    # Save JSON (append or create new)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                existing_results = json.load(f)
                if not isinstance(existing_results, list):
                    existing_results = [existing_results]
            except json.JSONDecodeError:
                existing_results = []
    else:
        existing_results = []

    existing_results.append(result_entry)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to: {json_path}")

    # Save detailed logs
    log_dir = os.path.join(os.path.dirname(json_path), "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_filename = f"{subject}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json"
    log_path = os.path.join(log_dir, log_filename)

    detailed_log = {
        "metadata": result_entry,
        "correct_samples": correct_samples,
        "wrong_samples": wrong_samples,
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(detailed_log, f, indent=2, ensure_ascii=False)

    print(f"✓ Detailed logs saved to: {log_path}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="RAG-based LLM Text Classification Evaluation"
    )

    # Input/output
    parser.add_argument("--input_csv", type=str, required=True, help="Input CSV file")
    parser.add_argument(
        "--encoding", type=str, default=None, help="CSV encoding (default: auto-detect)"
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default=None,
        help="Path to save results JSON (default: auto-generated)",
    )

    # Model selection
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model-name",
        type=str,
        help="Local model name (e.g., Qwen/Qwen2.5-7B-Instruct)",
    )
    model_group.add_argument(
        "--api-provider", type=str, help="API provider (e.g., openrouter, openai)"
    )

    # API-specific settings
    parser.add_argument(
        "--api-model",
        type=str,
        help="API model identifier (required if --api-provider is set)",
    )
    parser.add_argument(
        "--api-delay", type=float, default=0.0, help="Delay between API calls in seconds"
    )

    # RAG-specific settings
    parser.add_argument(
        "--train-csv", type=str, required=True, help="Train CSV file for infer_top_k"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Model directory for retrieval (infer_top_k)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of candidates to retrieve (default: 20)",
    )
    parser.add_argument(
        "--infer-device",
        type=str,
        default="cuda",
        help="Device for infer_top_k execution (default: cuda)",
    )

    # Generation settings
    parser.add_argument(
        "--max-new-tokens", type=int, default=20, help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Sampling temperature"
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=8192,
        help="Maximum input length for local models",
    )

    # Evaluation settings
    parser.add_argument(
        "--max-samples-per-row",
        type=int,
        default=None,
        help="Max samples per row (default: auto-detect)",
    )
    parser.add_argument(
        "--max-total-samples",
        type=int,
        default=None,
        help="Max total samples (default: no limit)",
    )
    parser.add_argument(
        "--few-shot", action="store_true", help="Use few-shot examples"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of few-shot examples (default: 5)",
    )

    # Other settings
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Validate API settings
    if args.api_provider and not args.api_model:
        parser.error("--api-model is required when using --api-provider")

    # Set random seed
    set_predict_random_seed(args.seed)

    # === Model Setup ===
    if args.api_provider:
        # API mode
        print(f"Using API provider: {args.api_provider}")
        print(f"API model: {args.api_model}")

        api_client = create_api_client(args.api_provider)
        generate_fn = create_api_generate_function(
            api_client,
            model=args.api_model,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            delay=args.api_delay,
        )
        model_identifier = f"{args.api_provider}/{args.api_model}"
        tokenizer = None
        check_token_length = False
    else:
        # Local model mode
        print(f"Loading local model: {args.model_name}")
        model, tokenizer = load_llm_model(args.model_name, args.device)

        generate_fn = create_generate_function(
            model,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_length=args.max_input_length,
        )
        model_identifier = args.model_name
        check_token_length = True

    # === Evaluation ===
    accuracy = evaluate_llm_classification(
        input_csv=args.input_csv,
        generate_fn=generate_fn,
        model_identifier=model_identifier,
        tokenizer=tokenizer,
        few_shot=args.few_shot,
        encoding=args.encoding,
        json_path=args.json_path,
        max_samples_per_row=args.max_samples_per_row,
        max_total_samples=args.max_total_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_input_length=args.max_input_length,
        train_csv=args.train_csv,
        model_dir=args.model_dir,
        top_k=args.top_k,
        infer_device=args.infer_device,
        num_examples=args.num_examples,
        check_token_length=check_token_length,
    )

    print(f"\n=== Final Result ===")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
