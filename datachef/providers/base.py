"""
BaseProvider — abstract interface for all inference backends.

The AI reasoning layer interacts exclusively with this interface.
New inference backends (Ollama, llama.cpp, other APIs) are added by
implementing BaseProvider without touching any other layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ParsedToolCall(BaseModel):
    """A single parsed tool call from a model response."""

    call_id: str
    tool_name: str
    arguments: dict[str, Any]


class ProviderResponse(BaseModel):
    """Normalized response from any inference provider."""

    content: str | None = None
    tool_calls: list[ParsedToolCall] = Field(default_factory=list)
    finish_reason: str = "stop"
    usage: TokenUsage = Field(default_factory=TokenUsage)
    raw_response: dict[str, Any] = Field(default_factory=dict)


class ProviderError(Exception):
    """Raised when the inference provider returns a non-recoverable error."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(ProviderError):
    """Raised on HTTP 429 — rate limit exceeded."""


class CreditExhaustedError(ProviderError):
    """Raised when the account has no remaining inference credits."""


class BaseProvider(ABC):
    """Abstract base class for DataChef inference providers."""

    @abstractmethod
    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> ProviderResponse:
        """
        Send a chat completion request and return a normalized ProviderResponse.

        Parameters
        ----------
        messages:
            Conversation history in OpenAI/HF Messages API format.
        tools:
            Tool definitions in OpenAI function-calling JSON schema format.
        tool_choice:
            "auto" (model decides), "none" (force text response), or a
            specific tool name.
        temperature:
            Sampling temperature. Lower values → more deterministic.
        max_tokens:
            Maximum tokens in the model's response.
        """

    @abstractmethod
    def model_id(self) -> str:
        """Return the model identifier being used."""

    @abstractmethod
    def provider_name(self) -> str:
        """Return a human-readable provider name."""
