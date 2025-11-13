from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    max_input_length: int = 6144,
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


def create_generate_function(
    model,
    tokenizer,
    device: str,
    max_input_length: int,
    max_new_tokens: int,
    temperature: float,
) -> Callable[[str], str]:
    """
    Create a generate_prediction function for local model.

    Args:
        model: LLM model
        tokenizer: Tokenizer
        device: Device
        max_input_length: Maximum input length (will truncate if exceeded)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Function that generates predictions using the local model (takes only prompt)
    """

    def generate_fn(prompt: str) -> str:
        """
        Generate prediction from local model.

        Args:
            prompt: Input prompt text

        Returns:
            Generated text response
        """
        return generate_prediction(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
            max_input_length=max_input_length,
        )

    return generate_fn
