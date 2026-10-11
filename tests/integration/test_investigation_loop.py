"""
Integration test for the InvestigationAgent iterative state loop.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from datachef.agent.investigation_agent import InvestigationAgent
from datachef.agent.prompt_builder import PromptBuilder
from datachef.analytics.profiler import DatasetProfiler
from datachef.context.models import InvestigationState
from datachef.providers.base import BaseProvider, ParsedToolCall, ProviderResponse
from datachef.tools.registry import ToolRegistry


class MockProvider(BaseProvider):
    """Deterministic mock provider that simulates a 2-turn investigation."""

    def __init__(self) -> None:
        self.call_count = 0

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> ProviderResponse:
        self.call_count += 1
        if self.call_count == 1:
            # Turn 1: Call tool
            return ProviderResponse(
                content="I should check statistics for the sales column.",
                tool_calls=[
                    ParsedToolCall(
                        call_id="call_mock_1",
                        tool_name="get_column_stats",
                        arguments={"columns": ["sales"]},
                    )
                ],
                finish_reason="tool_calls",
            )
        else:
            # Turn 2: Synthesize conclusion
            conclusion_text = (
                "### CONCLUSION\n"
                "Average sales is 150 with a maximum of 300.\n\n"
                "### INTERPRETATION\n"
                "The distribution is moderately right-skewed.\n\n"
                "### LIMITATIONS\n"
                "- Only 5 rows analyzed in this test."
            )
            return ProviderResponse(
                content=conclusion_text,
                tool_calls=[],
                finish_reason="stop",
            )

    def model_id(self) -> str:
        return "mock-test-model"

    def provider_name(self) -> str:
        return "mock-provider"


def test_investigation_loop_e2e():
    df = pl.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "sales": [100.0, 120.0, 150.0, 180.0, 300.0],
    })

    profiler = DatasetProfiler()
    profile = profiler.profile(df, source_path="test_orders.csv")

    registry = ToolRegistry()
    prompt_builder = PromptBuilder(profile, registry)
    state = InvestigationState(dataset_profile=profile)
    mock_provider = MockProvider()

    agent = InvestigationAgent(
        provider=mock_provider,
        registry=registry,
        prompt_builder=prompt_builder,
        df=df,
        state=state,
    )

    steps_recorded = []

    def on_step(step_name: str, payload: dict[str, Any]) -> None:
        steps_recorded.append((step_name, payload))

    finding = agent.investigate(
        user_message="What are the sales metrics?",
        on_step=on_step,
    )

    assert finding is not None
    assert finding.question == "What are the sales metrics?"
    assert "Average sales is 150" in finding.conclusion
    assert len(finding.evidence) == 1
    assert finding.evidence[0].tool_name == "get_column_stats"
    assert finding.tool_call_count == 1
    assert finding.turn_count == 2
    assert len(steps_recorded) > 0
