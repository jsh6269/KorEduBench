"""
API Model utility for LLM inference using various API providers.
Provides unified interface for OpenAI, Anthropic, Google AI Studio, and OpenRouter.
"""

import time
from typing import Callable, Optional

from openai import OpenAI, RateLimitError

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
    delay_seconds: float = 0.0,
    seed: Optional[int] = 42,
) -> Callable:
    """
    Create a generate_prediction function for API calls.

    Args:
        client: OpenAI-compatible API client
        model: Model name/identifier
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        delay_seconds: Delay in seconds after each API call (to avoid rate limits)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Function that generates predictions using the API (takes messages list or string prompt)
    """

    MAX_RETRIES = 10  # Fixed retry count
    retry_delay = 2 * delay_seconds

    def generate_prediction(prompt) -> str:
        """
        Generate prediction using API with automatic retry on rate limits.

        Args:
            prompt: Input prompt - either a string (wrapped as user message) or list of message dicts

        Returns:
            Generated text response

        Raises:
            RateLimitError: If rate limit persists after all retries
        """
        # Convert prompt to messages format
        if isinstance(prompt, str):
            # String prompt: wrap as user message for backward compatibility
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            # Already in messages format
            messages = prompt
        else:
            raise ValueError(f"Prompt must be str or list, got {type(prompt)}")

        last_error = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                # Build request parameters
                request_params = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_new_tokens,
                    "temperature": temperature,
                }
                # Add seed if provided
                if seed is not None:
                    request_params["seed"] = seed

                # Add reasoning_effort for GPT models to minimize thinking mode
                # GPT-5-mini supports: "minimal" (lowest), "low", "medium", "high"
                # GPT-4.1-mini supports: "none", "low", "medium", "high"
                elif (
                    "gpt-4.1" in model.lower()
                    or "gpt-4" in model.lower()
                    or "gpt-5" in model.lower()
                ):
                    request_params["reasoning_effort"] = "none"

                response = client.chat.completions.create(**request_params)

                # Add delay to avoid rate limits
                if delay_seconds > 0:
                    time.sleep(delay_seconds)

                return response.choices[0].message.content.strip()

            except RateLimitError as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    # Linear backoff: wait longer each retry
                    wait_time = retry_delay * (attempt + 1)
                    time.sleep(wait_time)
                else:
                    # All retries exhausted
                    raise

        # This should never be reached, but just in case
        if last_error:
            raise last_error
        raise RuntimeError("Unexpected error in API call")

    return generate_prediction
