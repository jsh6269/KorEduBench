"""
Fine-tuned LLM evaluation script with RAG workflow.
Evaluates a fine-tuned language model for educational achievement standard classification using RAG.
Loads models trained with rag_finetune_llm.py and evaluates them using RAG workflow (infer_top_k + LLM).
"""

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

# Get project root (3 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.inference import infer_top_k
from src.classification.predict_multiclass import load_model
from src.utils.data_loader import load_evaluation_data
from src.utils.model import generate_prediction
from src.utils.rag_prompt import (
    LLMClassificationResponse,
    create_rag_chat_prompt,
    parse_llm_response,
)
from src.utils.random_seed import set_predict_random_seed
from src.utils.unsloth_model import load_finetuned_model


def evaluate_finetuned_rag_llm(
    input_csv: str,
    model_path: str,
    train_csv: str,
    model_dir: str,
    encoding: str = None,
    json_path: str = None,
    num_samples: int = None,
    max_new_tokens: int = 10,
    temperature: float = 0.1,
    device: str = "cuda",
    max_input_length: int = 6144,
    top_k: int = 20,
    infer_device: str = "cuda",
    few_shot: bool = False,
    num_examples: int = 5,
):
    """
    Evaluate fine-tuned LLM classification on educational content using RAG workflow.

    Args:
        input_csv: Path to input CSV file
        model_path: Path to fine-tuned model directory (LoRA adapters)
        train_csv: Path to train CSV file for infer_top_k
        model_dir: Path to model directory for infer_top_k
        encoding: CSV encoding (default: auto-detect)
        json_path: Path to save results JSON
        num_samples: Target number of samples to generate (default: None, use all available samples)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        device: Device to use for fine-tuned model
        max_input_length: Maximum input length (will truncate if exceeded)
        top_k: Number of candidates to retrieve (default: 20)
        infer_device: Device for infer_top_k execution (default: "cuda")
        few_shot: Use few-shot examples (default: True)
        num_examples: Number of few-shot examples to use (default: 5)
    """
    # Load training info first (needed for json_path generation)
    training_info = {}
    info_path = os.path.join(model_path, "training_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            training_info = json.load(f)

    # Create output folder with model_name_yy-mm-dd format
    current_date = datetime.now().strftime("%y-%m-%d")
    # Use base_model from training_info, or model_path basename as fallback
    model_name = training_info.get("model_name", os.path.basename(model_path))
    # Sanitize model_name for filename (replace / and other invalid chars with _)
    safe_model_name = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    output_folder = (
        PROJECT_ROOT
        / "output"
        / "rag_finetuned_llm_text_classification"
        / f"{safe_model_name}_{current_date}"
    )
    os.makedirs(output_folder, exist_ok=True)

    if json_path is None:
        json_path = output_folder / f"finetuned_results.json"
    else:
        # If json_path is provided, still use the organized folder structure
        json_path = output_folder / os.path.basename(json_path)
    json_path = str(json_path)

    # === Load and preprocess data ===
    print("Loading evaluation data...")
    data = load_evaluation_data(
        input_csv=input_csv,
        encoding=encoding,
        num_samples=num_samples,
        max_candidates=None,  # RAG uses infer_top_k instead
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

    # === Load Fine-tuned Model ===
    model, tokenizer = load_finetuned_model(model_path, device, max_input_length)

    # Print training info if available
    if training_info:
        print(f"\nTraining info loaded:")
        print(f"  Base model: {training_info.get('model_name', 'N/A')}")
        print(f"  Training examples: {training_info.get('num_examples', 'N/A')}")
        print(f"  Epochs: {training_info.get('num_train_epochs', 'N/A')}")
        print(f"  Learning rate: {training_info.get('learning_rate', 'N/A')}")

    # === Check prompt length ===
    print(f"\nPrompt statistics:")
    print(f"  Top-k candidates to retrieve: {top_k}")

    # Create temporary candidates list for length estimation
    temp_candidates = [
        (i + 1, codes[i], contents[i]) for i in range(min(top_k, len(codes)))
    ]
    sample_messages = create_rag_chat_prompt(
        text=sample_texts[0],
        candidates=temp_candidates,
        completion="",
        for_inference=True,
        few_shot=few_shot,
        subject=subject,
        num_examples=num_examples,
    )
    sample_prompt = tokenizer.apply_chat_template(
        sample_messages["messages"], tokenize=False, add_generation_prompt=True
    )
    sample_tokens = tokenizer(sample_prompt, return_tensors="pt")
    prompt_length = sample_tokens["input_ids"].shape[1]

    print(f"  LLM prompt token length: {prompt_length}")
    print(f"  Max input length: {max_input_length}")

    if prompt_length > max_input_length:
        print(
            f"  ⚠️  WARNING: Prompt length ({prompt_length}) exceeds max_input_length ({max_input_length})"
        )
        print(f"  ⚠️  Prompts will be truncated, which may affect accuracy!")
        print(f"  ⚠️  Consider increasing --max-input-length or reducing --top-k.")
    else:
        print(f"  ✓ Prompt length is within limits")

    # === Load classification model for infer_top_k ===
    top_k_model, top_k_tokenizer, top_k_config, top_k_mappings = load_model(
        model_dir, infer_device
    )

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

        # Create RAG prompt with candidates
        messages = create_rag_chat_prompt(
            text=text,
            candidates=candidates,
            completion="",
            for_inference=True,
            few_shot=few_shot,
            subject=subject,
            num_examples=num_examples,
        )
        prompt = tokenizer.apply_chat_template(
            messages["messages"], tokenize=False, add_generation_prompt=True
        )

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
    correct = sum(
        1 for pred, true in zip(predictions, samples_true_codes) if pred == true
    )
    accuracy = correct / num_samples

    # === Summary ===
    print("\n=== Fine-tuned RAG LLM Evaluation ===")
    print(f"Model path: {model_path}")
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
            "model_path": model_path,
            "base_model": training_info.get("model_name", "N/A"),
            "subject": subject,
            "num_standards": num_rows,
            "top_k": top_k,
            "total_samples": num_samples,
            "num_examples": num_examples,
            "correct": correct,
            "accuracy": round(float(accuracy), 4),
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
            "training_info": training_info,
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
            r.get("model_path") == result_entry["model_path"]
            and r["subject"] == result_entry["subject"]
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
    logs_dir = output_folder / "logs"
    os.makedirs(logs_dir, exist_ok=True)
    csv_name = os.path.splitext(os.path.basename(input_csv))[0]

    if wrong_samples:
        # Save wrong samples as text file (sampled)
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
                f.write(f"LLM Response: {w['llm_response']}\n")
                f.write("-" * 60 + "\n")
        print(
            f"\nSaved {len(sampled_wrongs)} randomly selected wrong samples to {wrong_path}"
        )

        # Save wrong samples as CSV file (all samples)
        wrong_csv_path = logs_dir / f"{csv_name}_wrongs.csv"
        with open(wrong_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "true_code",
                    "pred_code",
                    "match_type",
                    "true_content",
                    "pred_content",
                ],
            )
            writer.writeheader()
            for w in wrong_samples:
                writer.writerow(
                    {
                        "true_code": w["true_code"],
                        "pred_code": w["pred_code"],
                        "match_type": w["match_type"],
                        "true_content": w["true_content"],
                        "pred_content": w["pred_content"],
                    }
                )
        print(f"Saved {len(wrong_samples)} wrong samples to {wrong_csv_path}")

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
                f.write(f"Pred Content: {c['pred_content']}\n")
                f.write(f"LLM Response: {c['llm_response']}\n")
                f.write("-" * 60 + "\n")
        print(
            f"Saved {len(sampled_corrects)} randomly selected correct samples to {correct_path}"
        )

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned LLM for educational content classification using RAG workflow."
    )
    parser.add_argument(
        "--input_csv", type=str, required=True, help="Path to input CSV file."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model directory (containing LoRA adapters).",
    )
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
        "--encoding", type=str, help="CSV encoding (default: auto-detect)."
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default=None,
        help="Path to JSON log file (default: {PROJECT_ROOT}/output/rag_llm_text_classification/finetuned_results.json).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Target number of samples to generate (default: None, use all available samples).",
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
        help="Device to use for fine-tuned model (cuda or cpu, default: cuda).",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=4096,
        help="Maximum input token length. Prompts exceeding this will be truncated (default: 4096).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top candidates to retrieve in Step 1 (default: 20).",
    )
    parser.add_argument(
        "--infer-device",
        type=str,
        default="cuda",
        help="Device for infer_top_k execution (default: cuda).",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of few-shot examples to use (default: 5).",
    )
    args = parser.parse_args()

    set_predict_random_seed(42)
    evaluate_finetuned_rag_llm(
        args.input_csv,
        args.model_path,
        train_csv=args.train_csv,
        model_dir=args.model_dir,
        encoding=args.encoding,
        json_path=args.json_path,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
        max_input_length=args.max_input_length,
        top_k=args.top_k,
        infer_device=args.infer_device,
        few_shot=True if args.num_examples > 0 else False,
        num_examples=args.num_examples,
    )
