"""
Unit tests for FindingComposer.
"""

from __future__ import annotations

from datachef.agent.finding_composer import FindingComposer
from datachef.context.models import InvestigationState, ToolCallRecord


def test_finding_composer_parsing():
    composer = FindingComposer()
    state = InvestigationState(session_id="test-session")
    state.add_tool_record(
        ToolCallRecord(
            turn_index=1,
            tool_name="compare_groups",
            arguments={"numeric_col": "sales", "group_col": "region"},
            result_summary="East region has higher mean sales (150 vs 80).",
            validation_passed=True,
        )
    )

    response_text = """
### CONCLUSION
The East region has significantly higher sales than other regions.

### INTERPRETATION
This disparity is driven by a higher volume of commercial orders in the East metropolitan area.

### LIMITATIONS
- Sample size in the West region is limited.
- Seasonality was not controlled for.
"""

    finding = composer.compose(
        state=state,
        question="Compare sales across regions",
        trigger="conversational",
        final_response_content=response_text,
        model_id="meta-llama/Llama-3.1-70B-Instruct",
        provider="huggingface",
        temperature=0.2,
    )

    assert finding.question == "Compare sales across regions"
    assert "East region" in finding.conclusion
    assert "disparity" in finding.interpretation
    assert len(finding.limitations) >= 1
    assert len(finding.evidence) == 1
    assert finding.evidence[0].tool_name == "compare_groups"
    assert finding.tool_call_count == 1




def _compose_with(*records: ToolCallRecord):
    state = InvestigationState(session_id="test-session")
    for record in records:
        state.add_tool_record(record)
    return FindingComposer().compose(
        state=state,
        question="Compare sales across regions",
        trigger="conversational",
        final_response_content="### CONCLUSION\nEast leads.",
        model_id="test-model",
        provider="test",
        temperature=0.2,
    )


def test_failed_execution_excluded_from_evidence():
    good = ToolCallRecord(
        turn_index=1,
        tool_name="compare_groups",
        arguments={},
        result_summary="East mean is 150.",
    )
    failed = ToolCallRecord(
        turn_index=2,
        tool_name="describe_column",
        arguments={},
        result_summary="Tool crashed.",
        validation_passed=True,
        execution_succeeded=False,
    )

    finding = _compose_with(good, failed)

    assert [e.call_id for e in finding.evidence] == [good.call_id]


def test_validation_failure_excluded_from_evidence():
    good = ToolCallRecord(
        turn_index=1,
        tool_name="compare_groups",
        arguments={},
        result_summary="East mean is 150.",
    )
    invalid = ToolCallRecord(
        turn_index=2,
        tool_name="describe_column",
        arguments={},
        validation_passed=False,
    )

    finding = _compose_with(good, invalid)

    assert [e.call_id for e in finding.evidence] == [good.call_id]


def test_primary_evidence_is_last_successful_call():
    first = ToolCallRecord(turn_index=1, tool_name="compare_groups", arguments={})
    second = ToolCallRecord(turn_index=2, tool_name="describe_column", arguments={})
    failed_last = ToolCallRecord(
        turn_index=3,
        tool_name="describe_column",
        arguments={},
        execution_succeeded=False,
    )

    finding = _compose_with(first, second, failed_last)

    primary = [e.call_id for e in finding.evidence if e.is_primary]
    assert primary == [second.call_id]


def test_recent_tool_results_excludes_failed_executions():
    state = InvestigationState(session_id="test-session")
    good = ToolCallRecord(turn_index=1, tool_name="compare_groups", arguments={})
    failed = ToolCallRecord(
        turn_index=2,
        tool_name="describe_column",
        arguments={},
        execution_succeeded=False,
    )
    state.add_tool_record(good)
    state.add_tool_record(failed)

    recent = state.recent_tool_results(5)

    assert [r.call_id for r in recent] == [good.call_id]