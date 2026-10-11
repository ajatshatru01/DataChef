"""
OllamaProvider — local inference backend for Ollama (Gemma, Llama, Qwen, etc.).

Runs 100% locally on user machine without cloud API keys or external network traffic.
Supports structured tool-calling via Ollama's native /api/chat and /v1/chat/completions.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import requests

from datachef import config
from datachef.providers.base import (
    BaseProvider,
    ParsedToolCall,
    ProviderError,
    ProviderResponse,
    TokenUsage,
)


class OllamaProvider(BaseProvider):
    """
    Local inference provider using Ollama.
    """

    def __init__(
        self,
        host: str | None = None,
        model_id: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        self._host = (host or config.OLLAMA_HOST).rstrip("/")
        self._model = model_id or config.OLLAMA_MODEL
        self._timeout = timeout or config.REQUEST_TIMEOUT
        self._max_retries = max_retries if max_retries is not None else config.MAX_RETRIES

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
        """
        Send a chat completion request to Ollama with retry logic.
        """
        url = f"{self._host}/api/chat"
        payload: dict[str, Any] = {
            "model": self._model,
            # The agent stores assistant tool calls in OpenAI-compatible form,
            # where function.arguments is a JSON string.  Ollama's native
            # /api/chat endpoint requires arguments to be a JSON object and
            # identifies tool-result messages with tool_name.
            "messages": self._normalize_request_messages(messages),
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        # If tools provided and tool_choice is not "none"
        if tools and tool_choice != "none":
            payload["tools"] = tools

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                time.sleep(2 ** attempt)

            try:
                resp = requests.post(url, json=payload, timeout=self._timeout)
                if resp.status_code == 200:
                    return self._normalize_response(resp.json())

                error = ProviderError(
                    f"Ollama request failed with status {resp.status_code}: {resp.text}",
                    status_code=resp.status_code,
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_exc = error
                    continue
                raise error

            except requests.exceptions.ConnectionError as exc:
                last_exc = ProviderError(
                    f"Could not connect to Ollama at {self._host}. "
                    "Make sure Ollama is running (`ollama serve`). "
                    f"Original error: {exc}"
                )
                continue
            except requests.exceptions.Timeout as exc:
                last_exc = ProviderError(
                    f"Ollama request timed out after {self._timeout} seconds: {exc}"
                )
                continue
            except Exception as exc:
                if isinstance(exc, ProviderError):
                    raise
                raise ProviderError(f"Unexpected error communicating with Ollama: {exc}") from exc

        if last_exc:
            raise last_exc
        raise ProviderError("Ollama request failed after retries.")

    def model_id(self) -> str:
        return self._model

    def provider_name(self) -> str:
        return "ollama"

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_request_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Translate OpenAI-style tool history to Ollama's native schema.

        A new message list is returned so callers can safely reuse the original
        conversation with another provider.
        """
        normalized_messages: list[dict[str, Any]] = []
        tool_names_by_id: dict[str, str] = {}

        for message in messages:
            normalized = dict(message)
            raw_tool_calls = message.get("tool_calls")

            if isinstance(raw_tool_calls, list):
                normalized_calls: list[dict[str, Any]] = []
                for raw_call in raw_tool_calls:
                    if not isinstance(raw_call, dict):
                        continue

                    raw_function = raw_call.get("function")
                    if not isinstance(raw_function, dict):
                        continue

                    function_name = raw_function.get("name")
                    arguments = raw_function.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError as exc:
                            raise ProviderError(
                                "Cannot send tool history to Ollama because a tool call "
                                "contains malformed JSON arguments."
                            ) from exc

                    if not isinstance(arguments, dict):
                        raise ProviderError(
                            "Cannot send tool history to Ollama because tool-call "
                            "arguments must decode to a JSON object."
                        )

                    native_call: dict[str, Any] = {
                        "function": {
                            "name": function_name,
                            "arguments": arguments,
                        }
                    }
                    call_id = raw_call.get("id")
                    if call_id:
                        native_call["id"] = call_id
                        if isinstance(function_name, str):
                            tool_names_by_id[str(call_id)] = function_name

                    normalized_calls.append(native_call)

                normalized["tool_calls"] = normalized_calls

            if normalized.get("role") == "tool" and not normalized.get("tool_name"):
                call_id = normalized.get("tool_call_id")
                tool_name = tool_names_by_id.get(str(call_id)) if call_id else None
                if tool_name:
                    normalized["tool_name"] = tool_name

            normalized_messages.append(normalized)

        return normalized_messages

    def _normalize_response(self, raw: dict[str, Any]) -> ProviderResponse:
        message = raw.get("message", {})
        content = message.get("content", "")
        raw_tool_calls = message.get("tool_calls", [])

        parsed_calls: list[ParsedToolCall] = []
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            if name:
                parsed_calls.append(
                    ParsedToolCall(
                        call_id=str(tc.get("id") or uuid.uuid4()),
                        tool_name=name,
                        arguments=args if isinstance(args, dict) else {},
                    )
                )

        usage = TokenUsage(
            prompt_tokens=raw.get("prompt_eval_count", 0),
            completion_tokens=raw.get("eval_count", 0),
            total_tokens=raw.get("prompt_eval_count", 0) + raw.get("eval_count", 0),
        )

        finish_reason = "tool_calls" if parsed_calls else "stop"

        return ProviderResponse(
            content=content,
            tool_calls=parsed_calls,
            finish_reason=finish_reason,
            usage=usage,
            raw_response=raw,
        )
