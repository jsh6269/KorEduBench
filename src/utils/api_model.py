"""
API Model utility for LLM inference using various API providers.
Provides unified interface for OpenAI, Anthropic, Google AI Studio, and OpenRouter.
"""

from typing import Callable, Optional

from openai import OpenAI

from src.utils.env_loader import get_api_key

# Provider configurations: API key names and base URLs
PROVIDER_CONFIGS = {
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "base_url": None,
    },
    "openrouter": {
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
    },
    "google": {
        "env_key": "GOOGLE_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    "anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com/v1",
    },
}


def create_api_client(provider: str, api_key: Optional[str] = None) -> OpenAI:
    """
    Create OpenAI-compatible client for different API providers.

    Args:
        provider: API provider name (openai, anthropic, google, openrouter)
        api_key: API key (optional, will load from environment if not provided)

    Returns:
        OpenAI client configured for the specified provider

    Raises:
        ValueError: If provider is not supported
    """
    # Check if provider is supported
    if provider not in PROVIDER_CONFIGS:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported: {', '.join(PROVIDER_CONFIGS.keys())}"
        )

    config = PROVIDER_CONFIGS[provider]

    # Load from env if not provided
    if api_key is None:
        api_key = get_api_key(config["env_key"], required=True)

    # Build client arguments
    client_kwargs = {"api_key": api_key}
    if config["base_url"] is not None:
        client_kwargs["base_url"] = config["base_url"]

    return OpenAI(**client_kwargs)


def create_api_generate_function(
    client: OpenAI,
    model: str,
    max_new_tokens: int = 50,
    temperature: float = 0.1,
) -> Callable[[str], str]:
    """
    Create a generate_prediction function for API calls.

    Args:
        client: OpenAI-compatible API client
        model: Model name/identifier
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Function that generates predictions using the API (takes only prompt)
    """

    def generate_prediction(prompt: str) -> str:
        """
        Generate prediction using API.

        Args:
            prompt: Input prompt text

        Returns:
            Generated text response
        """
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    return generate_prediction
