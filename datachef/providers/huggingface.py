"""
HuggingFaceProvider — concrete inference backend using huggingface_hub.

Uses huggingface_hub.InferenceClient (Apache 2.0) with the Messages API,
which provides OpenAI-compatible tool calling without depending on any
proprietary SDK.

Library: https://github.com/huggingface/huggingface_hub
License: Apache 2.0
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

from datachef import config
from datachef.providers.base import (
    BaseProvider,
    CreditExhaustedError,
    ParsedToolCall,
    ProviderError,
    ProviderResponse,
    RateLimitError,
    TokenUsage,
)


class HuggingFaceProvider(BaseProvider):
    """
    Inference provider using Hugging Face InferenceClient (Apache 2.0).

    Supports:
      • Chat completion with tool calling via the Messages API.
      • Configurable inference provider (together, fireworks-ai, novita, auto).
      • Exponential backoff on rate limits and 5xx errors.
      • Graceful credit-exhaustion detection.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str | None = None,
        inference_provider: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        self._api_key = api_key or config.HF_API_KEY
        self._model = model_id or config.HF_MODEL_ID
        self._inference_provider = inference_provider or config.HF_INFERENCE_PROVIDER
        self._timeout = timeout or config.REQUEST_TIMEOUT
        self._max_retries = max_retries if max_retries is not None else config.MAX_RETRIES

        provider_arg = (
            None if self._inference_provider == "auto" else self._inference_provider
        )
        self._client = InferenceClient(
            provider=provider_arg,
            api_key=self._api_key,
            timeout=float(self._timeout),
        )

    # ------------------------------------------------------------------
    # BaseProvider implementation
    # ------------------------------------------------------------------

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> ProviderResponse:
        """Send a chat completion request with retry logic."""
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                wait = 2 ** attempt  # 2s, 4s, 8s ...
                time.sleep(wait)

            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,  # type: ignore[arg-type]
                    tools=tools if tools else None,  # type: ignore[arg-type]
                    tool_choice=tool_choice if tools else None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return self._normalize_response(response)

            except HfHubHTTPError as exc:
                status = getattr(exc, "response", None)
                status_code = getattr(status, "status_code", None) if status else None

                if status_code == 429:
                    last_exc = RateLimitError(str(exc), status_code=429)
                    continue  # Retry after backoff

                if status_code and status_code >= 500:
                    last_exc = ProviderError(str(exc), status_code=status_code)
                    continue  # Retry on server errors

                # Credit exhaustion
                if _is_credit_error(str(exc)):
                    raise CreditExhaustedError(
                        "Hugging Face inference credits exhausted. "
                        "Please check your account at https://huggingface.co/settings/billing"
                    ) from exc

                # Non-retryable client error
                raise ProviderError(str(exc), status_code=status_code) from exc

            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    continue
                break

        if isinstance(last_exc, (RateLimitError, CreditExhaustedError)):
            raise last_exc
        raise ProviderError(
            f"All {self._max_retries + 1} attempts failed. Last error: {last_exc}"
        ) from last_exc

    def model_id(self) -> str:
        return self._model

    def provider_name(self) -> str:
        return f"huggingface/{self._inference_provider}"

    # ------------------------------------------------------------------
    # Response normalization
    # ------------------------------------------------------------------

    def _normalize_response(self, response: Any) -> ProviderResponse:
        """Convert an InferenceClient chat completion response to ProviderResponse."""
        choice = response.choices[0]
        message = choice.message
        finish_reason = choice.finish_reason or "stop"

        content: str | None = getattr(message, "content", None)
        raw_tool_calls = getattr(message, "tool_calls", None) or []

        parsed_tool_calls: list[ParsedToolCall] = []
        for tc in raw_tool_calls:
            try:
                fn = tc.function
                args_raw = fn.arguments
                if isinstance(args_raw, str):
                    args = json.loads(args_raw)
                elif isinstance(args_raw, dict):
                    args = args_raw
                else:
                    args = {}
                parsed_tool_calls.append(ParsedToolCall(
                    call_id=getattr(tc, "id", None) or str(uuid.uuid4()),
                    tool_name=fn.name,
                    arguments=args,
                ))
            except Exception:
                # Skip malformed tool calls — ToolCallParser handles fallback
                pass

        usage_obj = getattr(response, "usage", None)
        usage = TokenUsage(
            prompt_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage_obj, "total_tokens", 0) or 0,
        )

        return ProviderResponse(
            content=content,
            tool_calls=parsed_tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            raw_response={},  # Intentionally omitted to avoid logging sensitive data
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_credit_error(message: str) -> bool:
    """Heuristically detect credit-exhaustion error messages."""
    keywords = ["credit", "quota", "billing", "payment", "exceeded", "insufficient"]
    lower = message.lower()
    return any(k in lower for k in keywords)
