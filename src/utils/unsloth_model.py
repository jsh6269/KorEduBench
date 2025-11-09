import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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

    # Load PEFT config to get base model path
    peft_config = PeftConfig.from_pretrained(model_path)
    peft_config.use_cache = False
    base_model_id = peft_config.base_model_name_or_path

    # Load tokenizer from adapter path (includes added tokens)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_storage=torch.float16,
    )

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        config=peft_config,
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Resize model embeddings to match tokenizer vocab size
    current_vocab_size = base_model.get_input_embeddings().num_embeddings
    target_vocab_size = len(tokenizer)
    if current_vocab_size != target_vocab_size:
        base_model.resize_token_embeddings(target_vocab_size, mean_resizing=False)

    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, model_path)

    print(f"Model loaded successfully")
    return model, tokenizer
