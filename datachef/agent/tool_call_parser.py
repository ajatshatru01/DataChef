"""
ToolCallParser — extracts structured tool calls from model responses.

Handles:
  1. Structured tool calls already parsed by the provider (preferred path).
  2. JSON embedded in markdown code blocks (fallback for models that
     emit tool calls as text rather than using the structured API field).
  3. Plain JSON objects in the response text.

Malformed outputs are returned as parse errors, not raised as exceptions,
so the agent can feed the error back to the model.
"""

from __future__ import annotations

import json
import re
from typing import Any

from datachef.providers.base import ParsedToolCall, ProviderResponse


class ToolCallParseError(Exception):
    """Raised when a tool call cannot be parsed from the model response."""


class ToolCallParser:
    """
    Parses tool calls from ProviderResponse objects.

    Primary path: read from response.tool_calls (already structured by provider).
    Fallback path: search response.content for embedded JSON.
    """

    def parse(self, response: ProviderResponse) -> list[ParsedToolCall]:
        """
        Extract tool calls from *response*.

        Returns an empty list if the response contains no tool calls
        (i.e. it is a text conclusion).

        Raises ToolCallParseError if the response appears to contain a
        tool call but it cannot be decoded.
        """
        # Primary path: provider already parsed the tool calls
        if response.tool_calls:
            return response.tool_calls

        # Fallback: search content for embedded JSON tool call
        content = response.content or ""
        if not content.strip():
            if response.finish_reason == "tool_calls":
                raise ToolCallParseError(
                    "The provider reported a tool call but returned no parseable tool-call payload."
                )
            return []

        # Heuristic: only attempt JSON extraction if the content looks like
        # it might contain a tool call (has "name" and "arguments" keys)
        if '"name"' not in content or '"arguments"' not in content:
            if response.finish_reason == "tool_calls":
                raise ToolCallParseError(
                    "The provider reported a tool call but its text did not contain name/arguments."
                )
            return []

        parsed = self._extract_from_text(content)
        if not parsed:
            raise ToolCallParseError(
                "Tool-call JSON was malformed. Return a valid tool name and arguments object."
            )
        return parsed

    def extract_json_array(self, content: str) -> list[int] | None:
        """
        Extract a JSON integer array from *content*.
        Used for parsing the model's discovery prioritization response.
        Returns None if no valid array is found.
        """
        # Try to find [...] pattern
        match = re.search(r"\[[\d\s,]+\]", content)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list) and all(isinstance(i, int) for i in result):
                    return result
            except json.JSONDecodeError:
                pass
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_from_text(self, content: str) -> list[ParsedToolCall]:
        """
        Attempt to extract a tool call from raw text content.
        Tries markdown code blocks first, then bare JSON objects.
        """
        # 1. Markdown code block: ```json ... ``` or ``` ... ```
        code_block_pattern = re.compile(
            r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL
        )
        for match in code_block_pattern.finditer(content):
            try:
                obj = json.loads(match.group(1))
                tc = self._dict_to_tool_call(obj)
                if tc:
                    return [tc]
            except (json.JSONDecodeError, KeyError):
                continue

        # 2. Bare JSON object anywhere in content
        brace_pattern = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", re.DOTALL)
        for match in brace_pattern.finditer(content):
            try:
                obj = json.loads(match.group())
                tc = self._dict_to_tool_call(obj)
                if tc:
                    return [tc]
            except (json.JSONDecodeError, KeyError):
                continue

        return []

    @staticmethod
    def _dict_to_tool_call(obj: dict[str, Any]) -> ParsedToolCall | None:
        """
        Convert a raw dict to a ParsedToolCall if it has the expected shape.
        Returns None if the dict does not look like a tool call.
        """
        import uuid

        name = obj.get("name") or obj.get("tool") or obj.get("function", {}).get("name")
        arguments = (
            obj.get("arguments")
            or obj.get("parameters")
            or obj.get("function", {}).get("arguments")
            or {}
        )
        if not name:
            return None
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if not isinstance(arguments, dict):
            return None
        return ParsedToolCall(
            call_id=str(uuid.uuid4()),
            tool_name=name,
            arguments=arguments,
        )
