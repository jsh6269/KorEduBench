"""
Test script to verify GPT-5-mini reasoning_mode works with actual RAG prompt.

This script:
1. Loads a single sample from CSV
2. Creates RAG prompt (same as rag_eval_llm.py)
3. Calls API with reasoning_mode parameter
4. Displays the response to verify it's not empty
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.classification.inference import infer_top_k
from src.classification.predict_multiclass import load_model
from src.utils.api_model import create_api_client, create_api_generate_function
from src.utils.data_loader import load_evaluation_data
from src.utils.rag_prompt import create_rag_chat_prompt, parse_llm_response


def test_single_sample():
    """Test a single sample with actual API call."""
    print("=" * 80)
    print("Testing GPT-4.1-mini with reasoning_effort='none' parameter")
    print("=" * 80)

    # Configuration (same as api_rag_llm_text_classification.sh)
    API_PROVIDER = "openrouter"
    API_MODEL = "openai/gpt-4.1-mini"  # Changed from gpt-5-mini
    API_DELAY = 0.2
    MAX_NEW_TOKENS = 20
    TEMPERATURE = 0.1
    TOP_K = 20
    NUM_EXAMPLES = 5

    # Paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    VALID_CSV = PROJECT_ROOT / "dataset" / "valid_80" / "과학.csv"
    TRAIN_CSV = PROJECT_ROOT / "dataset" / "train_80" / "과학.csv"
    MODEL_DIR = PROJECT_ROOT / "model" / "achievement_classifier" / "best_model"
    INFER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Check if files exist
    if not VALID_CSV.exists():
        print(f"Error: {VALID_CSV} not found")
        return 1

    if not TRAIN_CSV.exists():
        print(f"Error: {TRAIN_CSV} not found")
        return 1

    if not MODEL_DIR.exists():
        print(f"Error: {MODEL_DIR} not found")
        return 1

    print(f"\nInput CSV: {VALID_CSV}")
    print(f"Train CSV: {TRAIN_CSV}")
    print(f"Model dir: {MODEL_DIR}")
    print(f"Device: {INFER_DEVICE}")

    # Load evaluation data (just 1 sample)
    print("\nLoading evaluation data...")
    data = load_evaluation_data(
        input_csv=str(VALID_CSV),
        encoding=None,
        num_samples=1,  # Just one sample
        max_candidates=None,
    )

    if len(data.sample_texts) == 0:
        print("Error: No samples found in CSV")
        return 1

    # Get first sample
    text = data.sample_texts[0]
    true_code = data.samples_true_codes[0]
    subject = data.subject

    print(f"\nSubject: {subject}")
    print(f"True Code: {true_code}")
    print(
        f"Sample Text: {text[:100]}..." if len(text) > 100 else f"Sample Text: {text}"
    )

    # Load top-k retrieval model
    print(f"\nLoading top-k retrieval model from {MODEL_DIR}...")
    top_k_model, top_k_tokenizer, top_k_config, top_k_mappings = load_model(
        str(MODEL_DIR), INFER_DEVICE
    )

    # Get top-k candidates
    print(f"\nRetrieving top-{TOP_K} candidates...")
    infer_result = infer_top_k(
        text=text,
        top_k=TOP_K,
        train_csv=str(TRAIN_CSV),
        model=top_k_model,
        tokenizer=top_k_tokenizer,
        config=top_k_config,
        mappings=top_k_mappings,
        device=INFER_DEVICE,
        random=False,
    )

    # Convert to candidates format
    candidates = [
        (idx + 1, item["code"], item["content"])
        for idx, item in enumerate(infer_result["top-k"])
    ]

    print(f"Retrieved {len(candidates)} candidates")
    print("Top 5 candidates:")
    for rank, code, content in candidates[:5]:
        print(f"  {rank}: {code} - {content[:50]}...")

    # Check if response "9과17" matches any candidate
    print(f"\nChecking if '9과17' is in candidates...")
    matching_candidates = [c for c in candidates if "9과17" in c[1]]
    if matching_candidates:
        print(f"  Found {len(matching_candidates)} candidates containing '9과17':")
        for rank, code, content in matching_candidates:
            print(f"    {rank}: {code}")
    else:
        print(f"  No candidates found containing '9과17'")
        print(f"  This suggests the response was truncated or incorrect")

    # Check if model is qwen3 (for /no_think prompt command)
    model_identifier = f"{API_PROVIDER}/{API_MODEL}"
    is_qwen3 = "qwen3" in model_identifier.lower()

    # Create RAG prompt
    print(
        f"\nCreating RAG prompt (few_shot={NUM_EXAMPLES > 0}, is_qwen3={is_qwen3})..."
    )
    messages = create_rag_chat_prompt(
        text=text,
        candidates=candidates,
        completion="",
        for_inference=True,
        few_shot=NUM_EXAMPLES > 0,
        subject=subject,
        num_examples=NUM_EXAMPLES,
        is_qwen3=is_qwen3,
    )

    # Display prompt structure
    print("\nPrompt structure:")
    for msg in messages["messages"]:
        role = msg["role"]
        content_preview = (
            msg["content"][:200] + "..."
            if len(msg["content"]) > 200
            else msg["content"]
        )
        print(f"  {role}: {len(msg['content'])} chars - {content_preview}")

    # Create API client and generate function
    print(f"\nCreating API client ({API_PROVIDER})...")
    api_client = create_api_client(API_PROVIDER, None)

    print(f"Creating generate function (model: {API_MODEL})...")
    generate_fn = create_api_generate_function(
        api_client,
        API_MODEL,
        MAX_NEW_TOKENS,
        TEMPERATURE,
        API_DELAY,
        seed=42,
    )

    # Make API call
    print(f"\nCalling API (this may take a moment)...")
    print("Note: Check if reasoning_effort='none' is in the request parameters")

    try:
        # Call API directly to get full response object
        print("\nMaking direct API call to inspect full response...")
        response_obj = api_client.chat.completions.create(
            model=API_MODEL,
            messages=messages["messages"],
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            seed=42,
            reasoning_effort="none",  # Use "none" for gpt-4.1-mini
        )

        # Extract text response
        response = (
            response_obj.choices[0].message.content.strip()
            if response_obj.choices[0].message.content
            else ""
        )

        print("\n" + "=" * 80)
        print("Full API Response Object:")
        print("=" * 80)
        print(f"Response ID: {response_obj.id}")
        print(f"Model: {response_obj.model}")
        print(f"Created: {response_obj.created}")

        # Check usage information
        if hasattr(response_obj, "usage") and response_obj.usage:
            usage = response_obj.usage
            print(f"\nUsage Information:")
            print(f"  Prompt tokens: {usage.prompt_tokens}")
            print(f"  Completion tokens: {usage.completion_tokens}")
            print(f"  Total tokens: {usage.total_tokens}")

        # Check for reasoning information
        message = response_obj.choices[0].message
        print(f"\nReasoning Information:")

        # Check reasoning field
        if hasattr(message, "reasoning"):
            reasoning = message.reasoning
            print(f"  Reasoning field: {reasoning is not None}")
            if reasoning:
                print(f"  Reasoning length: {len(str(reasoning))} chars")
                print(f"  Reasoning preview: {str(reasoning)[:200]}...")

        # Check reasoning_details (encrypted reasoning data)
        if hasattr(message, "reasoning_details") and message.reasoning_details:
            print(f"  Reasoning details present: True")
            print(f"  Number of reasoning details: {len(message.reasoning_details)}")
            for idx, detail in enumerate(message.reasoning_details):
                print(f"    Detail {idx}:")
                print(f"      Type: {detail.get('type', 'N/A')}")
                print(f"      Format: {detail.get('format', 'N/A')}")
                print(f"      Data length: {len(detail.get('data', ''))} chars")
                print(
                    f"      ⚠️  This encrypted reasoning data counts toward max_tokens!"
                )
        else:
            print(f"  Reasoning details: None")

        # Calculate estimated reasoning tokens
        if hasattr(message, "reasoning_details") and message.reasoning_details:
            total_reasoning_data = sum(
                len(str(detail.get("data", ""))) for detail in message.reasoning_details
            )
            # Rough estimate: 1 token ≈ 4 characters for Korean/English mixed
            estimated_reasoning_tokens = total_reasoning_data // 4
            print(f"  Estimated reasoning tokens: ~{estimated_reasoning_tokens}")
            print(
                f"  Completion tokens (text only): {usage.completion_tokens if hasattr(response_obj, 'usage') and response_obj.usage else 'N/A'}"
            )
            print(
                f"  Total estimated tokens: ~{estimated_reasoning_tokens + (usage.completion_tokens if hasattr(response_obj, 'usage') and response_obj.usage else 0)}"
            )
            print(f"  Max tokens limit: {MAX_NEW_TOKENS}")
            if (
                estimated_reasoning_tokens
                + (
                    usage.completion_tokens
                    if hasattr(response_obj, "usage") and response_obj.usage
                    else 0
                )
                > MAX_NEW_TOKENS
            ):
                print(f"  ⚠️  Total tokens exceed max_tokens limit!")

        # Check for finish_reason
        if hasattr(response_obj.choices[0], "finish_reason"):
            finish_reason = response_obj.choices[0].finish_reason
            print(f"\nFinish Reason: {finish_reason}")
            print(f"  Max tokens limit: {MAX_NEW_TOKENS}")
            print(
                f"  Completion tokens: {usage.completion_tokens if hasattr(response_obj, 'usage') and response_obj.usage else 'N/A'}"
            )

            if finish_reason == "length":
                print(f"  ⚠️  Response was truncated due to max_tokens limit!")
                print(
                    f"  This is strange if completion_tokens ({usage.completion_tokens if hasattr(response_obj, 'usage') and response_obj.usage else 'N/A'}) < max_tokens ({MAX_NEW_TOKENS})"
                )
                print(f"  Possible reasons:")
                print(f"    - Reasoning tokens might be counted separately")
                print(f"    - API might have internal token counting differences")
                print(f"    - Response might have been cut off for other reasons")

        # Check logprobs if available (might show token details)
        if (
            hasattr(response_obj.choices[0], "logprobs")
            and response_obj.choices[0].logprobs
        ):
            logprobs = response_obj.choices[0].logprobs
            print(f"\nLogprobs Information:")
            print(f"  Logprobs available: {logprobs is not None}")
            if hasattr(logprobs, "tokens"):
                print(
                    f"  Number of tokens in logprobs: {len(logprobs.tokens) if logprobs.tokens else 0}"
                )

        # Check the actual choice object
        choice = response_obj.choices[0]
        print(f"\nChoice Object Details:")
        print(f"  Choice index: {choice.index}")
        print(f"  Finish reason: {choice.finish_reason}")
        print(f"  Message role: {choice.message.role}")
        print(
            f"  Message content length: {len(choice.message.content) if choice.message.content else 0}"
        )

        # Check if there's any stop sequence or other finish reasons
        print(f"\nFull Choice Object:")
        print(f"  {choice}")

        print("\n" + "=" * 80)
        print("Text Response:")
        print("=" * 80)
        print(f"Response: {response}")
        print(f"Response length: {len(response)} chars")
        print(f"Response is empty: {len(response.strip()) == 0}")

        # Parse response
        llm_response = parse_llm_response(response, candidates)
        print(f"\nParsed Response:")
        print(f"  Predicted Code: {llm_response.predicted_code}")
        print(f"  Match Type: {llm_response.match_type.value}")
        print(f"  Confidence: {llm_response.confidence:.2f}")
        print(f"  Is Valid: {llm_response.is_valid}")
        print(f"  Is Exact Match: {llm_response.is_exact_match}")

        # Check if response is valid
        if llm_response.is_valid:
            print(f"\n✓ SUCCESS: Got valid response!")
            if llm_response.is_exact_match:
                print(f"✓ Exact match with true code: {true_code}")
            else:
                print(f"  (Partial match, true code was: {true_code})")
        else:
            print(f"\n✗ WARNING: Response is INVALID (empty or no match found)")
            print(f"  This suggests reasoning_mode may not be working correctly")

        print("=" * 80)
        return 0 if llm_response.is_valid else 1

    except Exception as e:
        print(f"\n✗ ERROR: API call failed")
        print(f"  Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(test_single_sample())
