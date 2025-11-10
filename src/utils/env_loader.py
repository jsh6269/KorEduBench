"""
Environment variable loader utility.
Loads API keys and configuration from .env file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


def load_env():
    """
    Load environment variables from .env file.

    Searches for .env file in the project root directory and loads it.
    If the file doesn't exist, it will silently continue (allowing env vars
    to be set through other means like system environment or CI/CD).
    """
    # Get project root (3 levels up from this file)
    project_root = Path(__file__).resolve().parent.parent.parent
    env_file = project_root / ".env"

    if env_file.exists():
        load_dotenv(env_file)
        print(f"✓ Loaded environment variables from: {env_file}")
    else:
        print(f"ℹ No .env file found at: {env_file}")
        print("  Using system environment variables or defaults")


def get_api_key(key_name: str, required: bool = False) -> str | None:
    """
    Get an API key from environment variables.

    Args:
        key_name: Name of the environment variable (e.g., 'OPENAI_API_KEY')
        required: If True, raises ValueError when key is not found

    Returns:
        API key value or None if not found (when not required)

    Raises:
        ValueError: If required=True and key is not found
    """
    value = os.getenv(key_name)

    if required and not value:
        raise ValueError(
            f"Required API key '{key_name}' not found. "
            f"Please set it in .env file or environment variables."
        )

    return value


# Automatically load .env when this module is imported
load_env()
