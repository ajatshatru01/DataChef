"""
InvestigationRecord — immutable serialized record of a completed investigation.

Stored as newline-delimited JSON (NDJSON) under ~/.datachef/investigations/.
Each line is one complete investigation record.

Records are designed to be:
  • Human-readable (plain JSON)
  • Reproducible (tool names + arguments are stored verbatim)
  • Safe (no raw dataset rows, no API keys)
"""

from __future__ import annotations

import json
import math
import re
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

from datachef import config
from datachef.context.models import Finding


class InvestigationRecord:
    """
    Writes and reads investigation records to/from disk.

    File layout: one NDJSON file per session under INVESTIGATION_DIR.
    File name: <session_id>.ndjson
    """

    def __init__(self, investigation_dir: Path | None = None) -> None:
        self._dir = investigation_dir or config.INVESTIGATION_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, finding: Finding, session_id: str | None = None) -> Path:
        """
        Append *finding* as a JSON record to the session's NDJSON file.
        Returns the path of the written file.
        """
        sid = session_id or finding.session_id
        record = self._finding_to_record(finding)
        if session_id:
            record["session_id"] = session_id
        output_path = self._session_path(sid)
        with output_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(_json_safe(record), default=str, allow_nan=False) + "\n")
        return output_path

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_session(self, session_id: str) -> list[dict[str, Any]]:
        """Return all records for a given session_id."""
        path = self._session_path(session_id)
        if not path.exists():
            return []
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                with suppress(json.JSONDecodeError):
                    records.append(json.loads(line))
        return records

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return a summary list of all recorded sessions."""
        sessions = []
        for path in sorted(self._dir.glob("*.ndjson"), key=lambda p: p.stat().st_mtime, reverse=True):
            session_id = path.stem
            records = self.load_session(session_id)
            if records:
                sessions.append({
                    "session_id": session_id,
                    "n_findings": len(records),
                    "first_question": records[0].get("question", ""),
                    "created_at": records[0].get("created_at", ""),
                    "path": str(path),
                })
        return sessions

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _finding_to_record(finding: Finding) -> dict[str, Any]:
        """Convert a Finding to a safe, serializable record dict."""
        return {
            "record_id": str(uuid.uuid4()),
            "session_id": finding.session_id,
            "dataset_fingerprint": finding.dataset_fingerprint,
            "trigger": finding.trigger,
            "question": finding.question,
            "tool_calls": [
                {
                    "call_id": call.call_id,
                    "turn_index": call.turn_index,
                    "tool_name": call.tool_name,
                    "arguments": call.arguments,
                    "result_summary": call.result_summary,
                    "result_data": call.result_data,
                    "execution_time_ms": call.execution_time_ms,
                    "validation_passed": call.validation_passed,
                    "execution_succeeded": call.execution_succeeded,
                    "result_truncated": call.result_truncated,
                    "error_message": call.error_message,
                    "timestamp": call.timestamp.isoformat(),
                    # chart_json intentionally excluded (too large, reconstructible)
                }
                for call in finding.tool_call_log
            ],
            "conclusion": finding.conclusion,
            "interpretation": finding.interpretation,
            "limitations": finding.limitations,
            "unverified_numerical_claims": finding.unverified_numerical_claims,
            "model_config": {
                "model_id": finding.model_id,
                "provider": finding.provider,
                "temperature": finding.temperature,
            },
            "hit_limit": finding.hit_limit,
            "limit_reason": finding.limit_reason,
            "turn_count": finding.turn_count,
            "tool_call_count": finding.tool_call_count,
            "error": finding.error,
            "created_at": finding.created_at.isoformat() if finding.created_at else None,
            "finding": {
                "conclusion": finding.conclusion,
                "evidence": [e.model_dump(mode="json", exclude={"chart_json"}) for e in finding.evidence],
                "interpretation": finding.interpretation,
                "limitations": finding.limitations,
                "unverified_numerical_claims": finding.unverified_numerical_claims,
            },
        }

    def _session_path(self, session_id: str) -> Path:
        """Resolve a session file while preventing path traversal."""
        if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
            raise ValueError("session_id may contain only letters, numbers, '_' and '-'.")
        return self._dir / f"{session_id}.ndjson"


def _json_safe(value: Any) -> Any:
    """Recursively convert non-finite floats to strict-JSON null values."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value
