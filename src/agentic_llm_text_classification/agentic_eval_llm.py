"""
LLM-based text classification evaluation script.
Uses a generative LLM (Qwen-1.5B) to classify textbook excerpts into educational achievement standards.
The LLM directly outputs achievement standard codes (e.g., "10영03-04") instead of content text.
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
from src.utils.agentic_prompt import (
    LLMClassificationResponse,
    create_rag_chat_prompt,
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
    num_samples: int = None,
    max_new_tokens: int = 10,
    temperature: float = 0.1,
    max_input_length: int = 8192,
    train_csv: str = None,
    model_dir: str = None,
    top_k: int = 15,
    infer_device: str = "cuda",
    check_token_length: bool = True,
):
    """
    Evaluate LLM-based classification on educational content using agentic workflow.

    Args:
        input_csv: Path to input CSV file
        generate_fn: Function to generate predictions (callable that takes prompt str and returns str)
        model_identifier: Model name or API identifier
        tokenizer: Tokenizer for token length checking (optional, None for API mode)
        few_shot: Use few-shot examples
        encoding: CSV encoding (default: auto-detect)
        json_path: Path to save results JSON
        num_samples: Target number of samples to generate
        max_new_tokens: Maximum tokens to generate (passed to generate_fn)
        temperature: Sampling temperature (passed to generate_fn)
        max_input_length: Maximum input length (will truncate if exceeded, for local models)
        train_csv: Path to train CSV file for infer_top_k
        model_dir: Path to model directory for infer_top_k
        top_k: Number of candidates to retrieve in Step 1 (default: 15)
        infer_device: Device for infer_top_k execution (default: "cuda")
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
            / "agentic_llm_text_classification"
            / f"results_{safe_model_name}_{current_date}.json"
        )
    json_path = str(json_path)

    # === Load and preprocess data ===
    print("Loading evaluation data...")
    data = load_evaluation_data(
        input_csv=input_csv,
        encoding=encoding,
        num_samples=num_samples,
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
    folder_name = data.folder_name

    # === Check prompt length ===
    print(f"\nPrompt statistics:")
    print(f"  Top-k candidates to retrieve: {top_k}")

    if tokenizer is not None:
        # Check Step 2 prompt length (use sample candidates for estimation)
        # Create temporary candidates list for length estimation
        temp_candidates = [
            (i + 1, codes[i], contents[i]) for i in range(min(top_k, len(codes)))
        ]
        sample_step2_messages = create_rag_chat_prompt(
            text=sample_texts[0],
            candidates=temp_candidates,
            completion="",
            for_inference=True,
            few_shot=few_shot,
            subject=subject,
            num_examples=5,
        )
        sample_step2_prompt = tokenizer.apply_chat_template(
            sample_step2_messages["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        sample_step2_tokens = tokenizer(sample_step2_prompt, return_tensors="pt")
        sample_step2_length = sample_step2_tokens["input_ids"].shape[1]

        print(f"  LLM prompt token length: {sample_step2_length}")

        if check_token_length:
            # Local model: check against max_input_length
            print(f"  Max input length: {max_input_length}")

            if sample_step2_length > max_input_length:
                print(
                    f"  ⚠️  WARNING: Prompt length ({sample_step2_length}) exceeds max_input_length ({max_input_length})"
                )
                print(f"  ⚠️  Prompts will be truncated, which may affect accuracy!")
                print(
                    f"  ⚠️  Consider increasing --max-input-length or reducing --top-k."
                )
            else:
                print(f"  ✓ Prompt length is within limits")
    else:
        print(f"  Token length check: Skipped (no tokenizer provided)")

    # === Prediction ===
    print(f"\nPredicting classifications for {num_samples} samples...")
    predictions = []
    wrong_samples = []
    correct_samples = []
    truncated_count = 0
    exact_match_count = 0
    match_type_counts = {}

    top_k_model, top_k_tokenizer, top_k_config, top_k_mappings = load_model(
        model_dir, infer_device
    )

    for i in tqdm(range(num_samples), desc="Classifying"):
        text = sample_texts[i]
        true_code = samples_true_codes[i]

        # Call infer_top_k with generated query
        infer_result = infer_top_k(
            text=text,
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

        # Step 2: Final Selection
        step2_messages = create_rag_chat_prompt(
            text=text,
            candidates=candidates,
            completion="",
            for_inference=True,
            few_shot=few_shot,
            subject=subject,
        )

        if tokenizer is not None:
            # Local models: convert chat messages to string using tokenizer's chat template
            step2_prompt = tokenizer.apply_chat_template(
                step2_messages["messages"], tokenize=False, add_generation_prompt=True
            )
        else:
            # API models: pass messages list directly (API natively supports chat format)
            step2_prompt = step2_messages["messages"]

        # Check if Step 2 prompt will be truncated (only for local models)
        if check_token_length and tokenizer is not None:
            step2_tokens = tokenizer(step2_prompt, return_tensors="pt")
            if step2_tokens["input_ids"].shape[1] > max_input_length:
                truncated_count += 1

        # Generate final prediction
        step2_response = generate_fn(step2_prompt)

        # Parse response - now returns LLMClassificationResponse object
        llm_response = parse_llm_response(step2_response, candidates)
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
            "llm_response": step2_response,
            "match_type": match_type_str,
            "confidence": llm_response.confidence,
            "is_exact_match": llm_response.is_exact_match,
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
    print("\n=== LLM Text Classification Evaluation ===")
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
            "exact_match_percentage": (
                round(exact_match_count / num_samples * 100, 2)
                if num_samples > 0
                else 0
            ),
            "match_type_distribution": {
                k: round(v / num_samples * 100, 2) for k, v in match_type_counts.items()
            },
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "max_input_length": max_input_length,
            "truncated_count": truncated_count,
            "truncated_percentage": (
                round(truncated_count / num_samples * 100, 2) if num_samples > 0 else 0
            ),
        }
    )

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except json.JSONDecodeError:
            results = []
    else:
        results = []

    replaced = False
    for i, r in enumerate(results):
        if (
            r["subject"] == result_entry["subject"]
            and r["num_standards"] == result_entry["num_standards"]
            and r.get("top_k", result_entry["top_k"]) == result_entry["top_k"]
            and r["total_samples"] == result_entry["total_samples"]
        ):
            results[i] = result_entry
            replaced = True
            print(f"Updated existing entry for subject '{subject}'")
            break

    if not replaced:
        results.append(result_entry)
        print(f"Appended new entry for subject '{subject}'")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {json_path}")

    # Save wrong samples
    logs_dir = PROJECT_ROOT / "output" / "agentic_llm_text_classification" / "logs"
    os.makedirs(logs_dir, exist_ok=True)
    csv_name = os.path.splitext(os.path.basename(input_csv))[0]

    if wrong_samples:
        wrong_path = logs_dir / f"{csv_name}_wrongs.txt"
        sampled_wrongs = random.sample(wrong_samples, min(100, len(wrong_samples)))
        with open(wrong_path, "w", encoding="utf-8") as f:
            f.write(f"Total wrong samples: {len(wrong_samples)}\n\n")
            for w in sampled_wrongs:
                f.write(f"[Sample #{w['sample_idx']}]\n")
                f.write(f"True Code: {w['true_code']}\n")
                f.write(f"Pred Code: {w['pred_code']}\n")
                f.write(f"Match Type: {w['match_type']}\n")
                f.write(f"Confidence: {w['confidence']:.2f}\n")
                f.write(f"Exact Match: {'YES' if w['is_exact_match'] else 'NO'}\n")
                f.write(f"Input Text: {w['input_text']}\n")
                f.write(f"True Content: {w['true_content']}\n")
                f.write(f"Pred Content: {w['pred_content']}\n")
                f.write(f"True Code: {w['true_code']}\n")
                f.write(f"LLM Response: {w['llm_response']}\n")
                f.write("-" * 60 + "\n")
        print(
            f"\nSaved {len(sampled_wrongs)} randomly selected wrong samples to {wrong_path}"
        )

    # Save correct samples
    if correct_samples:
        correct_path = logs_dir / f"{csv_name}_corrects.txt"
        sampled_corrects = random.sample(
            correct_samples, min(100, len(correct_samples))
        )
        with open(correct_path, "w", encoding="utf-8") as f:
            f.write(f"Total correct samples: {len(correct_samples)}\n\n")
            for c in sampled_corrects:
                f.write(f"[Sample #{c['sample_idx']}]\n")
                f.write(f"Code: {c['true_code']}\n")
                f.write(f"Match Type: {c['match_type']}\n")
                f.write(f"Confidence: {c['confidence']:.2f}\n")
                f.write(f"Exact Match: {'YES' if c['is_exact_match'] else 'NO'}\n")
                f.write(f"Input Text: {c['input_text']}\n")
                f.write(f"Step 1 Query: {c.get('step1_query', 'N/A')}\n")
                f.write(f"Content: {c['true_content']}\n")
                f.write(f"True Content: {c['true_content']}\n")
                f.write(f"Pred Content: {c['pred_content']}\n")
                f.write(f"LLM Response: {c['llm_response']}\n")
                f.write("-" * 60 + "\n")
        print(
            f"Saved {len(sampled_corrects)} randomly selected correct samples to {correct_path}"
        )

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate LLM-based text classification for educational content."
    )
    parser.add_argument(
        "--input_csv", type=str, required=True, help="Path to input CSV file."
    )

    # Model selection: mutually exclusive group for API vs Local
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model_name",
        type=str,
        help="Local Hugging Face model name (e.g., Qwen/Qwen2.5-1.5B-Instruct).",
    )
    model_group.add_argument(
        "--api-provider",
        type=str,
        choices=["openai", "anthropic", "google", "openrouter"],
        help="API provider (openai, anthropic, google, openrouter).",
    )

    # API-specific arguments
    parser.add_argument(
        "--api-model",
        type=str,
        help="API model name (required with --api-provider, e.g., gpt-4, claude-3-5-sonnet-20241022, gemini-pro).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (optional, will use .env if not provided).",
    )
    parser.add_argument(
        "--api-delay",
        type=float,
        default=0.5,
        help="Delay in seconds between API calls to avoid rate limits (default: 0.5).",
    )

    # Common arguments
    parser.add_argument(
        "--encoding", type=str, help="CSV encoding (default: auto-detect)."
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default=None,
        help="Path to JSON log file (default: {PROJECT_ROOT}/output/agentic_llm_text_classification/results.json).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Target number of samples to generate. If None, use all available samples. If num_samples <= num_rows: randomly sample num_samples rows, then 1 sample per row. If num_samples > num_rows: distribute samples across all rows.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate (default: 50).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1).",
    )
    # Agentic LLM arguments
    parser.add_argument(
        "--train-csv",
        type=str,
        required=True,
        help="Path to train CSV file for infer_top_k.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to model directory for infer_top_k.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of top candidates to retrieve in Step 1 (default: 15).",
    )
    parser.add_argument(
        "--infer-device",
        type=str,
        default="cuda",
        help="Device for infer_top_k execution (default: cuda).",
    )

    # Local model arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for local model (cuda or cpu, ignored for API, default: cuda).",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=2048,
        help="Maximum input token length for local model (ignored for API, default: 2048).",
    )

    parser.add_argument(
        "--few-shot",
        action="store_true",
        help="Use few-shot examples (default: False).",
    )

    args = parser.parse_args()

    # Validation
    if args.api_provider and not args.api_model:
        parser.error("--api-model is required when using --api-provider")

    set_predict_random_seed(42)

    # === Create generate_prediction function based on mode ===
    if args.api_provider:
        # API mode
        print(f"=" * 80)
        print(f"Using API provider: {args.api_provider}")
        print(f"API model: {args.api_model}")
        print(f"=" * 80)

        api_client = create_api_client(args.api_provider, args.api_key)
        generate_fn = create_api_generate_function(
            api_client,
            args.api_model,
            args.max_new_tokens,
            args.temperature,
            args.api_delay,
        )
        print(f"API delay: {args.api_delay}s between requests")
        print(f"API retry: automatic retry up to 10 times on rate limit errors")

        # Load a tokenizer for approximate token counting
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        model_identifier = f"{args.api_provider}/{args.api_model}"
        check_token_length = False

    else:
        # Local model mode
        print(f"=" * 80)
        print(f"Using local model: {args.model_name}")
        print(f"Device: {args.device}")
        print(f"=" * 80)

        model, tokenizer = load_llm_model(args.model_name, args.device)
        generate_fn = create_generate_function(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        model_identifier = args.model_name
        check_token_length = True

    # === Call evaluation function ===
    evaluate_llm_classification(
        input_csv=args.input_csv,
        generate_fn=generate_fn,
        model_identifier=model_identifier,
        tokenizer=tokenizer,
        few_shot=args.few_shot,
        encoding=args.encoding,
        json_path=args.json_path,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_input_length=args.max_input_length,
        train_csv=args.train_csv,
        model_dir=args.model_dir,
        top_k=args.top_k,
        infer_device=args.infer_device,
        check_token_length=check_token_length,
    )
