"""
InvestigationAgent — the core AI orchestration loop.

Manages the bounded iterative investigation state machine:
  IDLE → PLANNING → AWAITING_RESPONSE → VALIDATING → EXECUTING
       → EVALUATING → [loop or COMPOSING] → DONE

Exposes two entry points:
  • investigate(user_message) → Finding   (conversational mode)
  • discover() → list[Finding]            (autonomous discovery mode)
"""

from __future__ import annotations

import json
import time
from typing import Any

import polars as pl

from datachef import config
from datachef.agent.finding_composer import FindingComposer
from datachef.agent.prompt_builder import PromptBuilder
from datachef.agent.tool_call_parser import ToolCallParseError, ToolCallParser
from datachef.analytics.chart_builder import ChartBuilder
from datachef.analytics.discovery_engine import DiscoveryEngine
from datachef.context.models import (
    CandidateFinding,
    Finding,
    InvestigationState,
    ToolCallRecord,
)
from datachef.providers.base import (
    BaseProvider,
    CreditExhaustedError,
    ProviderError,
)
from datachef.tools.registry import ToolRegistry, ToolValidator


class InvestigationAgent:
    """
    Orchestrates iterative AI investigations over a loaded dataset.

    One agent instance is created per session and reused across multiple
    investigate() calls (each call gets its own InvestigationState).
    """

    def __init__(
        self,
        provider: BaseProvider,
        registry: ToolRegistry,
        prompt_builder: PromptBuilder,
        df: pl.DataFrame,
        state: InvestigationState,
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._prompt_builder = prompt_builder
        self._df = df
        self._state = state
        self._parser = ToolCallParser()
        self._composer = FindingComposer()
        self._chart_builder = ChartBuilder()
        self._discovery_engine = DiscoveryEngine()

    # ------------------------------------------------------------------
    # Entry point: conversational investigation
    # ------------------------------------------------------------------

    def investigate(
        self,
        user_message: str,
        on_step: Any | None = None,
    ) -> Finding:
        """
        Investigate a single user question.

        Runs the bounded loop and returns a structured Finding.
        Thread-safety: each call creates a local sub-state snapshot.
        """
        # Validate tool validator with current column names
        validator = ToolValidator(self._registry, self._df.columns)
        system_prompt = self._prompt_builder.build_system_prompt(discovery_mode=False)

        # Build the messages list for this investigation
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        # Include existing conversational history (not tool records — those go in context)
        for msg in self._state.conversation_history:
            if msg.role in ("user", "assistant") and msg.tool_calls is None:
                messages.append({"role": msg.role, "content": msg.content})

        return self._run_loop(
            question=user_message,
            trigger="conversational",
            messages=messages,
            validator=validator,
            on_step=on_step,
        )

    # ------------------------------------------------------------------
    # Entry point: autonomous discovery
    # ------------------------------------------------------------------

    def discover(self, on_step: Any | None = None) -> list[Finding]:
        """
        Run autonomous discovery mode.

        1. Generate candidates via DiscoveryEngine (deterministic, no API).
        2. Ask the model to prioritize candidates (1 API call).
        3. Investigate each selected candidate via the standard loop.
        """
        if self._state.dataset_profile is None:
            return []

        candidates = self._discovery_engine.run(self._state.dataset_profile, df=self._df)
        if not candidates:
            return []

        selected_candidates = self._prioritize_candidates(candidates)
        findings: list[Finding] = []
        for candidate in selected_candidates:
            finding = self._investigate_candidate(candidate, on_step=on_step)
            findings.append(finding)

        return findings

    # ------------------------------------------------------------------
    # Internal: investigation loop
    # ------------------------------------------------------------------

    def _run_loop(
        self,
        question: str,
        trigger: str,
        messages: list[dict[str, Any]],
        validator: ToolValidator,
        suggested_tools: list[str] | None = None,
        on_step: Any | None = None,
    ) -> Finding:
        """
        Core bounded investigation loop.
        Returns a Finding regardless of outcome (including errors / limit hits).
        """
        # Create a fresh sub-state for this investigation
        sub_state = InvestigationState(
            session_id=self._state.session_id,
            dataset_profile=self._state.dataset_profile,
        )

        tool_schemas = self._registry.get_schemas()
        start_time = time.perf_counter()
        hit_limit = False
        limit_reason: str | None = None

        while True:
            # --- Check limits ---
            elapsed = time.perf_counter() - start_time
            if elapsed >= config.MAX_WALL_SECONDS:
                return self._finish_without_model(
                    sub_state, question, trigger, "timeout", on_step
                )
            if config.MAX_TURNS <= 0:
                return self._finish_without_model(
                    sub_state, question, trigger, "max_turns", on_step
                )

            if sub_state.tool_call_count >= config.MAX_TOOL_CALLS:
                hit_limit, limit_reason = True, "max_tool_calls"
            elif sub_state.turn_count >= config.MAX_TURNS - 1:
                hit_limit, limit_reason = True, "max_turns"

            is_final = hit_limit

            # --- Build turn message ---
            turn_msg = self._prompt_builder.build_turn_message(
                question=question,
                state=sub_state,
                is_final_turn=is_final,
            )
            if suggested_tools and sub_state.turn_count == 0:
                turn_msg += (
                    f"\n\nSuggested starting tools: {', '.join(suggested_tools)}"
                )

            messages_copy = list(messages)  # don't mutate shared list
            messages_copy.append({"role": "user", "content": turn_msg})

            if on_step:
                on_step("planning", {"turn": sub_state.turn_count + 1, "question": question})

            # --- API call ---
            tool_choice = "none" if is_final else "auto"
            try:
                response = self._provider.chat_completion(
                    messages=messages_copy,
                    tools=tool_schemas if not is_final else [],
                    tool_choice=tool_choice,
                    temperature=config.TEMPERATURE,
                    max_tokens=config.MAX_RESPONSE_TOKENS,
                )
            except CreditExhaustedError as exc:
                finding = self._composer.compose_error_finding(
                    state=sub_state,
                    question=question,
                    trigger=trigger,
                    error_message=str(exc),
                    model_id=self._provider.model_id(),
                    provider=self._provider.provider_name(),
                    temperature=config.TEMPERATURE,
                )
                return self._remember_finding(finding, question, trigger)
            except ProviderError as exc:
                finding = self._composer.compose_error_finding(
                    state=sub_state,
                    question=question,
                    trigger=trigger,
                    error_message=f"Inference API error: {exc}",
                    model_id=self._provider.model_id(),
                    provider=self._provider.provider_name(),
                    temperature=config.TEMPERATURE,
                )
                return self._remember_finding(finding, question, trigger)

            sub_state.turn_count += 1

            if time.perf_counter() - start_time >= config.MAX_WALL_SECONDS:
                content = response.content or ""
                if response.tool_calls or not content.strip():
                    return self._finish_without_model(
                        sub_state, question, trigger, "timeout", on_step
                    )
                hit_limit, limit_reason = True, "timeout"

            if response.finish_reason == "content_filter":
                finding = self._composer.compose_error_finding(
                    state=sub_state,
                    question=question,
                    trigger=trigger,
                    error_message="The inference provider filtered the response.",
                    model_id=self._provider.model_id(),
                    provider=self._provider.provider_name(),
                    temperature=config.TEMPERATURE,
                )
                return self._remember_finding(finding, question, trigger)

            # --- Parse response ---
            try:
                tool_calls = self._parser.parse(response)
            except ToolCallParseError as exc:
                messages.append({"role": "assistant", "content": response.content or ""})
                messages.append(
                    {
                        "role": "user",
                        "content": f"ERROR: Tool call rejected. Reason: {exc}",
                    }
                )
                continue

            # Text conclusion (no tool calls)
            if not tool_calls or is_final:
                if on_step:
                    on_step("composing", {"turn": sub_state.turn_count})
                content = response.content or ""
                if response.finish_reason == "length":
                    content += (
                        "\n\nLIMITATIONS:\n- The provider truncated the final response "
                        "because its response-token limit was reached."
                    )
                if not content.strip():
                    content = _forced_conclusion(sub_state, limit_reason or "empty_response")
                finding = self._composer.compose(
                    state=sub_state,
                    question=question,
                    trigger=trigger,
                    final_response_content=content,
                    model_id=self._provider.model_id(),
                    provider=self._provider.provider_name(),
                    temperature=config.TEMPERATURE,
                    hit_limit=hit_limit,
                    limit_reason=limit_reason,
                )
                # Append to shared state
                return self._remember_finding(finding, question, trigger)

            # --- Process tool calls ---
            messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in tool_calls
                ],
            })
            for tc in tool_calls:
                if sub_state.tool_call_count >= config.MAX_TOOL_CALLS:
                    error_msg = (
                        "Tool call skipped because the investigation tool-call limit was reached."
                    )
                    sub_state.add_tool_record(
                        ToolCallRecord(
                            call_id=tc.call_id,
                            turn_index=sub_state.turn_count,
                            tool_name=tc.tool_name,
                            arguments=tc.arguments,
                            result_summary=error_msg,
                            validation_passed=False,
                            execution_succeeded=False,
                            error_message=error_msg,
                        )
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.call_id,
                            "content": f"ERROR: {error_msg}",
                        }
                    )
                    continue

                if on_step:
                    on_step("validating", {"tool": tc.tool_name, "arguments": tc.arguments})
                error_msg = validator.validate_and_format_error(tc.tool_name, tc.arguments)
                if error_msg:
                    record = ToolCallRecord(
                        call_id=tc.call_id,
                        turn_index=sub_state.turn_count,
                        tool_name=tc.tool_name,
                        arguments=tc.arguments,
                        result_summary=error_msg,
                        validation_passed=False,
                        execution_succeeded=False,
                        error_message=error_msg,
                    )
                    sub_state.add_tool_record(record)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.call_id,
                        "content": error_msg,
                    })
                    continue

                if on_step:
                    on_step("executing", {"tool": tc.tool_name, "arguments": tc.arguments})
                result_dict, exec_ms = self._registry.dispatch(
                    tc.tool_name, tc.arguments, self._df
                )
                full_summary = _result_to_summary(result_dict, tc.tool_name)
                max_chars = max(config.MAX_RESULT_TOKENS, 1) * 4
                result_summary = _truncate(full_summary, max_chars)
                result_truncated = len(full_summary) > max_chars
                execution_succeeded = "error" not in result_dict

                if on_step:
                    on_step("evaluating", {"tool": tc.tool_name, "summary": result_summary, "exec_ms": exec_ms})

                chart_json = None
                if execution_succeeded:
                    chart_json = self._try_build_chart(tc.tool_name, tc.arguments, result_dict)

                record = ToolCallRecord(
                    call_id=tc.call_id,
                    turn_index=sub_state.turn_count,
                    tool_name=tc.tool_name,
                    arguments=tc.arguments,
                    result_summary=result_summary,
                    result_data=result_dict,
                    chart_json=chart_json,
                    execution_time_ms=exec_ms,
                    validation_passed=True,
                    execution_succeeded=execution_succeeded,
                    result_truncated=result_truncated,
                    error_message=result_dict.get("error"),
                )
                sub_state.add_tool_record(record)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.call_id,
                    "content": result_summary,
                })

    def _finish_without_model(
        self,
        sub_state: InvestigationState,
        question: str,
        trigger: str,
        reason: str,
        on_step: Any | None,
    ) -> Finding:
        """Compose a safe local conclusion when no model round-trip remains."""
        if on_step:
            on_step("composing", {"turn": sub_state.turn_count})
        finding = self._composer.compose(
            state=sub_state,
            question=question,
            trigger=trigger,
            final_response_content=_forced_conclusion(sub_state, reason),
            model_id=self._provider.model_id(),
            provider=self._provider.provider_name(),
            temperature=config.TEMPERATURE,
            hit_limit=True,
            limit_reason=reason,
        )
        return self._remember_finding(finding, question, trigger)

    def _remember_finding(
        self, finding: Finding, question: str, trigger: str
    ) -> Finding:
        """Store a completed finding without polluting chat history from discovery."""
        self._state.findings.append(finding)
        if trigger == "conversational":
            self._state.add_message("user", question)
            self._state.add_message("assistant", finding.conclusion)
        return finding

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    def _prioritize_candidates(
        self, candidates: list[CandidateFinding]
    ) -> list[CandidateFinding]:
        """Ask the model to select and order the top candidates."""
        max_n = config.MAX_DISCOVERY_CANDIDATES
        if not candidates:
            return []

        system_prompt = self._prompt_builder.build_system_prompt(discovery_mode=True)
        prio_msg = self._prompt_builder.build_discovery_prioritization_message(
            candidates, max_n
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prio_msg},
        ]

        try:
            response = self._provider.chat_completion(
                messages=messages,
                tools=[],
                tool_choice="none",
                temperature=0.1,  # Very low temp for deterministic ordering
                max_tokens=128,
            )
            content = response.content or ""
            indices = self._parser.extract_json_array(content)
            if indices:
                selected = []
                for idx in indices[:max_n]:
                    if 0 <= idx < len(candidates):
                        selected.append(candidates[idx])
                if selected:
                    return selected
        except Exception:
            pass

        # Fallback: use top N by heuristic score
        return candidates[:max_n]

    def _investigate_candidate(
        self, candidate: CandidateFinding, on_step: Any | None = None
    ) -> Finding:
        """Run the standard investigation loop for a single discovery candidate."""
        validator = ToolValidator(self._registry, self._df.columns)
        system_prompt = self._prompt_builder.build_system_prompt(discovery_mode=False)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        return self._run_loop(
            question=candidate.description,
            trigger="discovery",
            messages=messages,
            validator=validator,
            suggested_tools=candidate.suggested_tools,
            on_step=on_step,
        )

    # ------------------------------------------------------------------
    # Chart building
    # ------------------------------------------------------------------

    def _try_build_chart(
        self, tool_name: str, arguments: dict, result: dict
    ) -> str | None:
        """Attempt to build a chart for the tool result. Returns JSON or None."""
        try:
            if tool_name == "compare_groups" and "group_stats" in result:
                return self._chart_builder.bar_group_means(
                    result["group_stats"],
                    arguments.get("numeric_col", "value"),
                    arguments.get("group_col", "group"),
                )
            if tool_name == "get_group_distribution" and "groups" in result:
                return self._chart_builder.bar_group_means(
                    result["groups"],
                    arguments.get("numeric_col", "value"),
                    arguments.get("group_col", "group"),
                )
            if tool_name == "analyze_trend" and "chart_data" in result:
                return self._chart_builder.line_trend(
                    result["chart_data"],
                    result.get("value_col", "value"),
                    result.get("time_col", "time"),
                )
            if tool_name == "get_missing_summary":
                return self._chart_builder.missing_bar(result)
            if tool_name == "get_correlation" and "matrix" in result:
                return self._chart_builder.correlation_heatmap(result["matrix"])
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_to_summary(result: dict, tool_name: str) -> str:
    """Convert a tool result dict to a concise text summary."""
    if "error" in result:
        return f"Error: {result['error']}"
    try:
        return json.dumps(result, indent=2, default=str)
    except Exception:
        return str(result)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [result truncated to fit token budget]"


def _forced_conclusion(state: InvestigationState, reason: str) -> str:
    if state.tool_call_log:
        conclusion = "The investigation stopped at its configured limit; review the computed evidence below."
    else:
        conclusion = "The investigation stopped before deterministic evidence could be collected."
    return (
        f"CONCLUSION: {conclusion}\n\n"
        "INTERPRETATION: No additional model interpretation was generated.\n\n"
        f"LIMITATIONS:\n- Investigation ended because of: {reason}."
    )
