"""
Unit tests for the NumericalClaim Pydantic model and its integration with Finding.

Covers:
- Claim with all fields populated
- Claim without unit (defaults to None)
- Missing required field raises ValidationError
- Old-style Finding payload without numerical_claims loads with an empty list
- Finding with claims survives a JSON round trip
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from datachef.context.models import Finding, NumericalClaim


def test_numerical_claim_all_fields():
    """A claim with all fields specified instantiates and retains all attributes."""
    claim = NumericalClaim(
        metric_name="mean",
        value=42.5,
        unit="USD",
        source_call_id="call-123",
        result_path="groups.East.mean",
    )
    assert claim.metric_name == "mean"
    assert claim.value == 42.5
    assert claim.unit == "USD"
    assert claim.source_call_id == "call-123"
    assert claim.result_path == "groups.East.mean"


def test_numerical_claim_without_unit():
    """unit is optional and defaults to None when omitted."""
    claim = NumericalClaim(
        metric_name="count",
        value=100.0,
        source_call_id="call-456",
        result_path="total_count",
    )
    assert claim.metric_name == "count"
    assert claim.value == 100.0
    assert claim.unit is None
    assert claim.source_call_id == "call-456"
    assert claim.result_path == "total_count"


@pytest.mark.parametrize(
    "missing_field",
    ["metric_name", "value", "source_call_id", "result_path"],
)
def test_numerical_claim_missing_required_field_raises(missing_field: str):
    """Omitting any required field must raise a Pydantic ValidationError."""
    valid_data = {
        "metric_name": "median",
        "value": 15.0,
        "unit": "kg",
        "source_call_id": "call-789",
        "result_path": "stats.median",
    }
    del valid_data[missing_field]
    with pytest.raises(ValidationError):
        NumericalClaim(**valid_data)


def test_old_style_finding_payload_loads_empty_claims():
    """An old-style Finding payload without numerical_claims loads with an empty list."""
    legacy_payload = {
        "session_id": "sess-test-1",
        "trigger": "conversational",
        "question": "What is the average sales in the East region?",
        "conclusion": "Average sales is $42.5.",
    }
    finding = Finding.model_validate(legacy_payload)
    assert finding.numerical_claims == []


def test_finding_with_claims_survives_json_round_trip():
    """A Finding with claims survives JSON serialization and deserialization."""
    claim = NumericalClaim(
        metric_name="mean_sales",
        value=123.45,
        unit="USD",
        source_call_id="call-abc-123",
        result_path="groups.East.mean",
    )
    finding = Finding(
        session_id="sess-test-2",
        trigger="conversational",
        question="What is the average sales in the East region?",
        conclusion="Average sales is 123.45 USD.",
        numerical_claims=[claim],
    )
    json_data = finding.model_dump_json()
    reloaded = Finding.model_validate_json(json_data)

    assert len(reloaded.numerical_claims) == 1
    assert reloaded.numerical_claims[0].metric_name == "mean_sales"
    assert reloaded.numerical_claims[0].value == 123.45
    assert reloaded.numerical_claims[0].unit == "USD"
    assert reloaded.numerical_claims[0].source_call_id == "call-abc-123"
    assert reloaded.numerical_claims[0].result_path == "groups.East.mean"
