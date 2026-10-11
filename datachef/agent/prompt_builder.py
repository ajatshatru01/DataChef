"""
PromptBuilder — constructs system prompts and per-turn user messages.

The system prompt is built once per session and is stable across turns.
Per-turn messages are assembled by ContextBuilder for each model call.
"""

from __future__ import annotations

from typing import Any

from datachef.context.context_builder import ContextBuilder
from datachef.context.models import DatasetProfile, InvestigationState
from datachef.tools.registry import ToolRegistry

_SYSTEM_PROMPT_TEMPLATE = """\
You are DataChef, a precise and rigorous data investigation agent.

## Your Role
You investigate tabular datasets by selecting and executing analytical tools.
You formulate hypotheses, choose appropriate tools, interpret computed results,
and produce evidence-backed conclusions.

## Core Rules — NEVER violate these
1. NEVER cite a number you have not obtained from a tool call in this investigation.
   Every metric, percentage, mean, count, and correlation must come from a tool result.
2. NEVER invent column names. Use only column names listed in the dataset schema below.
3. ALWAYS distinguish between computed facts and your interpretation of them.
4. CONCLUDE as soon as you have sufficient evidence. Do not make unnecessary tool calls.
5. ACKNOWLEDGE uncertainty and limitations. Do not claim causation from correlation.
6. If the data is insufficient to answer the question, say so clearly.

## Dataset Schema
{schema_block}

## Dataset Profile Summary
{profile_summary}

## Available Tools
You may call any of the following tools. Call them with valid arguments only.
{tool_schemas}

## Investigation Guidelines
- Start by identifying the most directly relevant tool for the question.
- After each result, decide: is this sufficient to conclude, or is follow-up needed?
- Keep follow-up calls focused — each should add new evidence, not repeat prior calls.
- When concluding, structure your response as:
  CONCLUSION: <clear answer to the question>
  INTERPRETATION: <what the evidence suggests>
  LIMITATIONS: <what the evidence does not show, and what caveats apply>
"""

_DISCOVERY_SYSTEM_ADDENDUM = """
## Discovery Mode Instructions
You are running in autonomous discovery mode.
You will receive a list of candidate findings ranked by heuristic score.
Select the most analytically interesting candidates to investigate.
Order your selection from most to least promising.
Return your selection as a JSON array of candidate indices (0-based).
Example: [2, 0, 4]
"""


class PromptBuilder:
    """Builds system prompts and per-turn messages for the investigation agent."""

    def __init__(
        self,
        profile: DatasetProfile,
        registry: ToolRegistry,
    ) -> None:
        self._profile = profile
        self._registry = registry
        self._ctx_builder = ContextBuilder(profile)
        self._system_prompt: str | None = None

    # ------------------------------------------------------------------
    # System prompt (built once, reused)
    # ------------------------------------------------------------------

    def build_system_prompt(self, discovery_mode: bool = False) -> str:
        """Build and cache the system prompt for a session."""
        if self._system_prompt is None:
            schema_block = self._ctx_builder.build_schema_block()
            profile_summary = self._ctx_builder.build_profile_summary()
            tool_schemas = _format_tool_schemas(self._registry.get_schemas())

            self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
                schema_block=schema_block,
                profile_summary=profile_summary,
                tool_schemas=tool_schemas,
            )

        if discovery_mode:
            return self._system_prompt + _DISCOVERY_SYSTEM_ADDENDUM
        return self._system_prompt

    def invalidate_cache(self) -> None:
        """Force system prompt rebuild on next call (e.g., after tool set change)."""
        self._system_prompt = None

    # ------------------------------------------------------------------
    # Per-turn user message
    # ------------------------------------------------------------------

    def build_turn_message(
        self,
        question: str,
        state: InvestigationState,
        is_final_turn: bool = False,
        from_config: dict | None = None,
    ) -> str:
        """
        Build the user-role message for a single investigation turn.

        Parameters
        ----------
        question:
            The investigation question or discovery candidate description.
        state:
            Current InvestigationState (used to extract recent tool results).
        is_final_turn:
            If True, instructs the model to produce a conclusion immediately.
        from_config:
            Optional overrides for loop limits (max_tool_calls, max_turns).
        """
        from datachef import config as cfg

        max_tool_calls = (from_config or {}).get("max_tool_calls", cfg.MAX_TOOL_CALLS)
        max_turns = (from_config or {}).get("max_turns", cfg.MAX_TURNS)

        recent_records = state.recent_tool_results(cfg.CONTEXT_HISTORY_N)
        tool_results_block = self._ctx_builder.build_tool_results_block(recent_records)

        turn_instruction = self._ctx_builder.build_turn_instruction(
            tool_calls_used=state.tool_call_count,
            max_tool_calls=max_tool_calls,
            turns_used=state.turn_count,
            max_turns=max_turns,
            is_final_turn=is_final_turn,
        )

        parts = [f"## Investigation Question\n{question}"]
        if tool_results_block:
            parts.append(tool_results_block)
        parts.append(turn_instruction)

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Discovery prioritization message
    # ------------------------------------------------------------------

    def build_discovery_prioritization_message(
        self, candidates: list[Any], max_candidates: int
    ) -> str:
        """Build the message asking the model to prioritize discovery candidates."""
        lines = [
            f"Select up to {max_candidates} candidates to investigate in depth, "
            "ordered from most to least interesting. "
            "Return a JSON array of candidate indices (0-based).\n",
            "Candidates:",
        ]
        for i, c in enumerate(candidates):
            lines.append(
                f"[{i}] ({c.check_type}) {c.description} "
                f"[score={c.heuristic_score:.2f}]"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_tool_schemas(schemas: list[dict]) -> str:
    """Format tool schemas as a concise text block for the system prompt."""
    lines: list[str] = []
    for schema in schemas:
        fn = schema["function"]
        name = fn["name"]
        desc = fn["description"]
        params = fn.get("parameters", {}).get("properties", {})
        required = fn.get("parameters", {}).get("required", [])
        param_strs = []
        for p_name, p_schema in params.items():
            req_marker = "*" if p_name in required else ""
            p_type = p_schema.get("type", "any")
            p_desc = p_schema.get("description", "")
            param_strs.append(f"    {p_name}{req_marker} ({p_type}): {p_desc}")
        lines.append(f"• {name}: {desc}")
        if param_strs:
            lines.append("  Parameters (* = required):")
            lines.extend(param_strs)
        lines.append("")
    return "\n".join(lines)
