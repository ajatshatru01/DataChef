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
