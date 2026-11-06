"""
Core data models for DataChef.

Defines DatasetProfile, InvestigationState, Finding, and all supporting types.
These are the shared vocabulary used across every layer of the system.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Column-level profile models
# ---------------------------------------------------------------------------


class NumericStats(BaseModel):
    """Descriptive statistics for a numeric column."""

    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    q25: float | None = None
    q75: float | None = None
    iqr: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    n_outliers_iqr: int = 0


class CategoricalStats(BaseModel):
    """Statistics for a categorical column."""

    top_values: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of {value, count, pct} dicts for top categories",
    )
    cardinality: int = 0


ColumnRole = Literal["numeric", "categorical", "datetime", "text", "id", "unknown"]


class ColumnProfile(BaseModel):
    """Profile of a single dataset column."""

    name: str
    dtype: str  # polars dtype name string
    inferred_role: ColumnRole = "unknown"
    null_count: int = 0
    null_pct: float = 0.0
    n_unique: int = 0
    stats: NumericStats | CategoricalStats | None = None
    sample_values: list[Any] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Dataset-level profile
# ---------------------------------------------------------------------------


class DatasetProfile(BaseModel):
    """Full statistical profile of an uploaded dataset."""

    source_path: str
    dataset_fingerprint: str = ""
    row_count: int
    column_count: int
    columns: list[ColumnProfile] = Field(default_factory=list)
    duplicate_row_count: int = 0
    correlation_matrix: dict[str, dict[str, float | None]] | None = None
    profiled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def get_column(self, name: str) -> ColumnProfile | None:
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def numeric_columns(self) -> list[ColumnProfile]:
        return [c for c in self.columns if c.inferred_role == "numeric"]

    def categorical_columns(self) -> list[ColumnProfile]:
        return [c for c in self.columns if c.inferred_role == "categorical"]

    def datetime_columns(self) -> list[ColumnProfile]:
        return [c for c in self.columns if c.inferred_role == "datetime"]

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


# ---------------------------------------------------------------------------
# Tool call records
# ---------------------------------------------------------------------------


class ToolCallRecord(BaseModel):
    """Immutable record of a single tool invocation."""

    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    turn_index: int
    tool_name: str
    arguments: dict[str, Any]
    result_summary: str = ""
    result_data: dict[str, Any] = Field(default_factory=dict)
    chart_json: str | None = None  # Plotly figure JSON string
    execution_time_ms: int = 0
    validation_passed: bool = True
    execution_succeeded: bool = True
    result_truncated: bool = False
    error_message: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Message types (mirrors OpenAI/HF Messages API format)
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single message in the conversation history."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


# ---------------------------------------------------------------------------
# Evidence and findings
# ---------------------------------------------------------------------------


class Evidence(BaseModel):
    """A single piece of computed evidence supporting a finding."""

    call_id: str = ""
    turn_index: int = 0
    tool_name: str
    arguments: dict[str, Any]
    result_summary: str
    result_data: dict[str, Any] = Field(default_factory=dict)
    chart_json: str | None = None
    execution_time_ms: int = 0
    execution_succeeded: bool = True
    result_truncated: bool = False
    timestamp: datetime | None = None
    is_primary: bool = False


class NumericalClaim(BaseModel):
    """A structured numerical claim derived from a tool call."""

    metric_name: str
    value: float
    unit: str | None = None
    source_call_id: str
    result_path: str


class Finding(BaseModel):
    """The output of a completed investigation."""

    finding_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    dataset_fingerprint: str = ""
    trigger: Literal["conversational", "discovery"]
    question: str
    conclusion: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    tool_call_log: list[ToolCallRecord] = Field(default_factory=list)
    interpretation: str = ""
    limitations: list[str] = Field(default_factory=list)
    unverified_numerical_claims: list[str] = Field(default_factory=list)
    numerical_claims: list[NumericalClaim] = Field(default_factory=list)
    tool_call_count: int = 0
    turn_count: int = 0
    hit_limit: bool = False
    limit_reason: str | None = None
    model_id: str = ""
    provider: str = ""
    temperature: float = 0.2
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = None  # populated only on fatal failure


# ---------------------------------------------------------------------------
# Investigation state (mutable session object)
# ---------------------------------------------------------------------------


class InvestigationState(BaseModel):
    """Tracks the full state of an active investigation session."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dataset_profile: DatasetProfile | None = None
    conversation_history: list[Message] = Field(default_factory=list)
    tool_call_log: list[ToolCallRecord] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    turn_count: int = 0
    tool_call_count: int = 0

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        self.conversation_history.append(Message(role=role, content=content, **kwargs))

    def add_tool_record(self, record: ToolCallRecord) -> None:
        self.tool_call_log.append(record)
        if record.validation_passed:
            self.tool_call_count += 1

    def recent_tool_results(self, n: int) -> list[ToolCallRecord]:
        """Return the last n tool call records that passed validation and executed successfully."""
        successful = [
            r for r in self.tool_call_log
            if r.validation_passed and r.execution_succeeded
        ]
        return successful[-n:]

# ---------------------------------------------------------------------------
# Discovery candidate
# ---------------------------------------------------------------------------


class CandidateFinding(BaseModel):
    """A candidate finding produced by the DiscoveryEngine."""

    check_type: str
    description: str
    involved_columns: list[str] = Field(default_factory=list)
    heuristic_score: float = 0.0  # 0.0 – 1.0
    suggested_tools: list[str] = Field(default_factory=list)
    context_data: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Structured final-turn output (replaces heading-based regex parsing)
# ---------------------------------------------------------------------------


class FindingDraft(BaseModel):
    """
    Structured conclusion produced by the model on the final investigation turn.

    The agent is asked to return a JSON object matching this schema instead of
    the legacy CONCLUSION/INTERPRETATION/LIMITATIONS heading format.
    FindingComposer validates and consumes this before falling back to regex.
    """

    conclusion: str = Field(
        ..., min_length=1, description="Direct, evidence-backed answer to the question."
    )
    interpretation: str = Field(
        default="", description="What the evidence implies beyond the raw numbers."
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Caveats, confounders, and what the evidence does not show.",
    )
    primary_evidence_call_ids: list[str] = Field(
        default_factory=list,
        description="call_id values of the tool calls whose results were the decisive evidence.",
    )

    @field_validator("conclusion")
    @classmethod
    def conclusion_not_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("conclusion must not be empty or whitespace only")
        return cleaned

    @field_validator("limitations", mode="before")
    @classmethod
    def normalize_limitations(cls, v: Any) -> list[str]:
        """Accept a bare string (one block of text) or a list; always return list[str]."""
        if isinstance(v, str):
            return [line.strip() for line in v.splitlines() if line.strip()]
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        return []
