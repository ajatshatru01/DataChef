"""Persistence contract and path-safety tests."""

import shutil
import uuid
from pathlib import Path

import pytest

from datachef.agent.finding_composer import FindingComposer
from datachef.context.models import DatasetProfile, InvestigationState, ToolCallRecord
from datachef.persistence.investigation_record import InvestigationRecord


def _record_dir() -> Path:
    path = Path(".pytest-records") / str(uuid.uuid4())
    path.mkdir(parents=True)
    return path


def test_record_contains_reproducible_tool_data() -> None:
    record_dir = _record_dir()
    profile = DatasetProfile(
        source_path="orders.csv",
        dataset_fingerprint="sha256:test",
        row_count=2,
        column_count=1,
    )
    state = InvestigationState(session_id="session-1", dataset_profile=profile)
    state.add_tool_record(
        ToolCallRecord(
            call_id="call-1",
            turn_index=1,
            tool_name="get_column_stats",
            arguments={"columns": ["sales"]},
            result_summary='{"sales": {"mean": 15}}',
            result_data={"sales": {"mean": 15}},
            execution_time_ms=7,
        )
    )
    finding = FindingComposer().compose(
        state=state,
        question="Average sales?",
        trigger="conversational",
        final_response_content="CONCLUSION: The computed mean is 15.",
        model_id="test-model",
        provider="test-provider",
        temperature=0.2,
    )

    try:
        records = InvestigationRecord(record_dir)
        records.save(finding)
        saved = records.load_session("session-1")[0]

        assert saved["dataset_fingerprint"] == "sha256:test"
        assert saved["tool_calls"][0]["call_id"] == "call-1"
        assert saved["tool_calls"][0]["result_data"] == {"sales": {"mean": 15}}
        assert saved["tool_calls"][0]["execution_time_ms"] == 7
        assert saved["finding"]["conclusion"] == "The computed mean is 15."
    finally:
        shutil.rmtree(record_dir)


def test_record_rejects_path_traversal() -> None:
    record_dir = _record_dir()
    try:
        records = InvestigationRecord(record_dir)
        with pytest.raises(ValueError):
            records.load_session("../outside")
    finally:
        shutil.rmtree(record_dir)
