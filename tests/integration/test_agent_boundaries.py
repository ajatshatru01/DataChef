"""Integration coverage for bounded turns, batched calls, and discovery state."""

from typing import Any

import polars as pl

from datachef import config
from datachef.agent.investigation_agent import InvestigationAgent
from datachef.agent.prompt_builder import PromptBuilder
from datachef.analytics.profiler import DatasetProfiler
from datachef.context.models import InvestigationState
from datachef.providers.base import BaseProvider, ParsedToolCall, ProviderResponse
from datachef.tools.registry import ToolRegistry


class SequenceProvider(BaseProvider):
    def __init__(self, responses: list[ProviderResponse]) -> None:
        self.responses = responses
        self.call_count = 0

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> ProviderResponse:
        response = self.responses[self.call_count]
        self.call_count += 1
        return response

    def model_id(self) -> str:
        return "sequence-model"

    def provider_name(self) -> str:
        return "sequence-provider"


def _agent(provider: BaseProvider, df: pl.DataFrame) -> tuple[InvestigationAgent, InvestigationState]:
    profile = DatasetProfiler().profile(df, "memory.csv")
    state = InvestigationState(dataset_profile=profile)
    registry = ToolRegistry()
    return (
        InvestigationAgent(provider, registry, PromptBuilder(profile, registry), df, state),
        state,
    )


def test_max_turns_is_a_true_model_round_trip_bound(monkeypatch) -> None:
    monkeypatch.setattr(config, "MAX_TURNS", 1)
    provider = SequenceProvider([ProviderResponse(content="Final answer.")])
    agent, _ = _agent(provider, pl.DataFrame({"sales": [1.0, 2.0, 3.0]}))

    finding = agent.investigate("Summarize sales")

    assert provider.call_count == 1
    assert finding.turn_count == 1
    assert finding.hit_limit is True
    assert finding.limit_reason == "max_turns"


def test_batched_tool_calls_respect_call_limit(monkeypatch) -> None:
    monkeypatch.setattr(config, "MAX_TOOL_CALLS", 1)
    provider = SequenceProvider(
        [
            ProviderResponse(
                tool_calls=[
                    ParsedToolCall(
                        call_id="one",
                        tool_name="get_column_stats",
                        arguments={"columns": ["sales"]},
                    ),
                    ParsedToolCall(
                        call_id="two",
                        tool_name="get_column_stats",
                        arguments={"columns": ["profit"]},
                    ),
                ],
                finish_reason="tool_calls",
            ),
            ProviderResponse(content="CONCLUSION: Evidence collected."),
        ]
    )
    agent, _ = _agent(
        provider,
        pl.DataFrame({"sales": [1.0, 2.0, 3.0], "profit": [2.0, 4.0, 6.0]}),
    )

    finding = agent.investigate("Compare metrics")

    assert finding.tool_call_count == 1
    assert len(finding.evidence) == 1
    assert len(finding.tool_call_log) == 2
    assert finding.tool_call_log[1].validation_passed is False


def test_discovery_adds_each_finding_once_and_does_not_pollute_chat() -> None:
    provider = SequenceProvider(
        [
            ProviderResponse(content="[0]"),
            ProviderResponse(content="CONCLUSION: Candidate reviewed."),
        ]
    )
    df = pl.DataFrame({"value": list(range(20)), "group": ["a"] * 18 + [None, None]})
    agent, state = _agent(provider, df)

    findings = agent.discover()

    assert len(findings) == 1
    assert len(state.findings) == 1
    assert state.conversation_history == []
