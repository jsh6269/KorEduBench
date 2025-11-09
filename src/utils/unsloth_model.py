from unsloth import FastLanguageModel


def load_finetuned_model(
    model_path: str,
    device: str = "cuda",
    max_seq_length: int = 2048,
):
    """
    Load fine-tuned LLM model and tokenizer using Unsloth.

    Args:
        model_path: Path to fine-tuned model directory (LoRA adapters)
        device: Device to load model on
        max_seq_length: Maximum sequence length

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"\nLoading fine-tuned model from: {model_path}")
    print("Loading LoRA adapters...")

    # Use Unsloth's FastLanguageModel for loading LoRA adapters
    print(f"Loading model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,  # Use 4-bit to match training configuration
    )

    # Enable native 2x faster inference
    FastLanguageModel.for_inference(model)

    print(f"Model loaded successfully")
    return model, tokenizer
