"""
ToolSchema — Pydantic schema representing a registered analytical tool.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field


class ToolSchema(BaseModel):
    """
    Schema representation of a registered analytical tool,
    matching OpenAI / HuggingFace function calling schema format.
    """

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    handler: Callable[..., Any] | None = None

    def to_function_definition(self) -> dict[str, Any]:
        """Format as tool schema for provider API."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
