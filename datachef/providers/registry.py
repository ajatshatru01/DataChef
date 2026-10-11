"""
Provider registry — selects and instantiates the configured provider.
"""

from __future__ import annotations

from datachef import config
from datachef.providers.base import BaseProvider
from datachef.providers.huggingface import HuggingFaceProvider
from datachef.providers.ollama import OllamaProvider


def get_provider() -> BaseProvider:
    """
    Return the configured inference provider based on DATACHEF_PROVIDER.
    Supports: 'ollama' (local open-source) and 'huggingface' (open-weight hosted).
    """
    if config.PROVIDER == "ollama":
        return OllamaProvider(
            host=config.OLLAMA_HOST,
            model_id=config.OLLAMA_MODEL,
            timeout=config.REQUEST_TIMEOUT,
            max_retries=config.MAX_RETRIES,
        )

    if config.PROVIDER == "huggingface":
        return HuggingFaceProvider(
            api_key=config.HF_API_KEY,
            model_id=config.HF_MODEL_ID,
            inference_provider=config.HF_INFERENCE_PROVIDER,
            timeout=config.REQUEST_TIMEOUT,
            max_retries=config.MAX_RETRIES,
        )

    raise ValueError(
        f"Unknown provider '{config.PROVIDER}'. Supported providers: ollama, huggingface."
    )
