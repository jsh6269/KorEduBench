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
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Get project root (3 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import load_evaluation_data
from src.utils.prompt import (
    LLMClassificationResponse,
    create_classification_prompt,
    parse_llm_response,
)
from src.utils.random_seed import set_predict_random_seed


def load_llm_model(model_name: str, device: str = "cuda"):
    """
    Load LLM model and tokenizer.

    Args:
        model_name: Hugging Face model name
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"\nLoading LLM model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True,
    )

    if device != "cuda":
        model = model.to(device)

    model.eval()

    print(f"Model loaded successfully on {device}")
    return model, tokenizer


def generate_prediction(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.1,
    device: str = "cuda",
    max_input_length: int = 2048,
) -> str:
    """
    Generate prediction from LLM.

    Args:
        model: LLM model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        device: Device
        max_input_length: Maximum input length (will truncate if exceeded)

    Returns:
        Generated text
    """
    # Tokenize with truncation to prevent exceeding max length
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )

    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part (exclude input prompt)
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    return generated_text.strip()


def evaluate_llm_classification(
    input_csv: str,
    model_name: str,
    few_shot: bool = False,
    encoding: str = None,
    json_path: str = None,
    max_samples_per_row: int = None,
    max_total_samples: int = None,
    max_new_tokens: int = 10,
    temperature: float = 0.1,
    device: str = "cuda",
    max_input_length: int = 8192,
    max_candidates: int = 200,
):
    """
    Evaluate LLM-based classification on educational content.

    Args:
        input_csv: Path to input CSV file
        model_name: Hugging Face model name
        encoding: CSV encoding (default: auto-detect)
        json_path: Path to save results JSON
        max_samples_per_row: Maximum samples per row
        max_total_samples: Maximum total samples (randomly sampled if specified)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        device: Device to use
        max_input_length: Maximum input length (will truncate if exceeded)
        max_candidates: Maximum number of candidate achievement standards to use (default: 60)
    """
    if json_path is None:
        json_path = PROJECT_ROOT / "output" / "llm_text_classification" / "results.json"
    json_path = str(json_path)

    # === Load and preprocess data ===
    print("Loading evaluation data...")
    data = load_evaluation_data(
        input_csv, encoding, max_samples_per_row, max_total_samples
    )

    # Extract data for convenience
    contents = data.contents
    codes = data.codes
    sample_texts = data.sample_texts
    true_codes = data.true_codes
    subject = data.subject
    num_rows = data.num_rows
    num_samples = data.num_samples
    max_samples_per_row = data.max_samples_per_row
    folder_name = data.folder_name

    # Prepare candidates list (limit to max_candidates)
    num_candidates = min(num_rows, max_candidates)
    if num_rows > max_candidates:
        print(
            f"⚠️  Warning: Number of achievement standards ({num_rows}) exceeds max_candidates ({max_candidates})"
        )
        print(f"⚠️  Randomly sampling {max_candidates} standards as candidates")
        # Randomly sample indices and sort them
        selected_indices = sorted(random.sample(range(num_rows), max_candidates))
    else:
        selected_indices = list(range(num_candidates))

    candidates = [
        (i + 1, codes[selected_indices[i]], contents[selected_indices[i]])
        for i in range(num_candidates)
    ]

    # === Load LLM Model ===
    model, tokenizer = load_llm_model(model_name, device)

    # === Check prompt length ===
    # Create a sample prompt to check length
    sample_prompt = create_classification_prompt(sample_texts[0], candidates)
    sample_tokens = tokenizer(sample_prompt, return_tensors="pt")
    prompt_length = sample_tokens["input_ids"].shape[1]

    print(f"\nPrompt statistics:")
    print(f"  Number of candidates: {len(candidates)}")
    print(f"  Sample prompt token length: {prompt_length}")
    print(f"  Max input length: {max_input_length}")

    if prompt_length > max_input_length:
        print(
            f"  ⚠️  WARNING: Prompt length ({prompt_length}) exceeds max_input_length ({max_input_length})"
        )
        print(f"  ⚠️  Prompts will be truncated, which may affect accuracy!")
        print(
            f"  ⚠️  Consider increasing --max-input-length or reducing the number of candidates."
        )
    else:
        print(f"  ✓ Prompt length is within limits")

    # === Prediction ===
    print(f"\nPredicting classifications for {num_samples} samples...")
    predictions = []
    wrong_samples = []
    correct_samples = []
    truncated_count = 0
    exact_match_count = 0
    match_type_counts = {}

    for i in tqdm(range(num_samples), desc="Classifying"):
        text = sample_texts[i]
        true_code = true_codes[i]

        # Create prompt
        prompt = create_classification_prompt(text, candidates)

        # Check if this specific prompt will be truncated
        prompt_tokens = tokenizer(prompt, return_tensors="pt")
        if prompt_tokens["input_ids"].shape[1] > max_input_length:
            truncated_count += 1

        # Generate prediction
        response = generate_prediction(
            model,
            tokenizer,
            prompt,
            max_new_tokens,
            temperature,
            device,
            max_input_length,
        )

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
        }

        if pred_code != true_code:
            wrong_samples.append(sample_info)
        else:
            correct_samples.append(sample_info)

    # === Evaluation Metrics ===
    print("\nCalculating metrics...")

    # LLM only produces top-1 predictions
    correct = sum(1 for pred, true in zip(predictions, true_codes) if pred == true)
    accuracy = correct / num_samples

    # Calculate MRR (for single predictions, MRR = accuracy)
    reciprocal_ranks = [
        1.0 if pred == true else 0.0 for pred, true in zip(predictions, true_codes)
    ]
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    # === Summary ===
    print("\n=== LLM Text Classification Evaluation ===")
    print(f"Model: {model_name}")
    print(f"Subject: {subject}")
    print(f"Total achievement standards: {num_rows}")
    print(f"Number of candidates used: {num_candidates} (max: {max_candidates})")
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
    print(f"MRR: {mrr:.4f}")

    # === JSON Logging ===
    result_entry = {}
    if folder_name:
        result_entry["folder"] = folder_name

    result_entry.update(
        {
            "model_name": model_name,
            "subject": subject,
            "num_standards": num_rows,
            "num_candidates": num_candidates,
            "max_candidates": max_candidates,
            "max_samples_per_row": int(max_samples_per_row),
            "total_samples": num_samples,
            "correct": correct,
            "accuracy": round(float(accuracy), 4),
            "mrr": round(float(mrr), 4),
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
            r["model_name"] == result_entry["model_name"]
            and r["subject"] == result_entry["subject"]
            and r["num_standards"] == result_entry["num_standards"]
            and r.get("max_candidates", result_entry["max_candidates"])
            == result_entry["max_candidates"]
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
    logs_dir = PROJECT_ROOT / "output" / "llm_text_classification" / "logs"
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
                f.write(f"Content: {c['true_content']}\n")
                f.write(f"True Content: {c['true_content']}\n")
                f.write(f"Pred Content: {c['pred_content']}\n")
                f.write(f"LLM Response: {c['llm_response']}\n")
                f.write("-" * 60 + "\n")
        print(
            f"Saved {len(sampled_corrects)} randomly selected correct samples to {correct_path}"
        )

    return accuracy, mrr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate LLM-based text classification for educational content."
    )
    parser.add_argument(
        "--input_csv", type=str, required=True, help="Path to input CSV file."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Hugging Face model name (default: Qwen/Qwen2.5-1.5B-Instruct).",
    )
    parser.add_argument(
        "--encoding", type=str, help="CSV encoding (default: auto-detect)."
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default=None,
        help="Path to JSON log file (default: {PROJECT_ROOT}/output/llm_text_classification/results.json).",
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
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu, default: cuda).",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=2048,
        help="Maximum input token length. Prompts exceeding this will be truncated (default: 2048).",
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

    set_predict_random_seed(42)
    evaluate_llm_classification(
        args.input_csv,
        args.model_name,
        encoding=args.encoding,
        json_path=args.json_path,
        max_samples_per_row=args.max_samples_per_row,
        max_total_samples=args.max_total_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
        max_input_length=args.max_input_length,
        few_shot=args.few_shot,
        max_candidates=args.max_candidates,
    )
