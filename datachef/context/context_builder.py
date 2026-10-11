"""
ContextBuilder — constructs the minimal context snapshot sent to the inference API.

Raw dataset rows are NEVER included. Only schema, profile summaries, and
condensed tool results cross the network boundary.
"""

from __future__ import annotations

import json

from datachef.context.models import (
    DatasetProfile,
    ToolCallRecord,
)


class ContextBuilder:
    """
    Builds per-turn context snapshots for the AI reasoning layer.

    Applies a token budget: if the assembled context exceeds the budget,
    tool result summaries are truncated from oldest to newest until it fits.
    """

    # Rough chars-per-token estimate (conservative)
    _CHARS_PER_TOKEN = 4
    # Budget for context sent to the model (leaves room for response)
    _CONTEXT_TOKEN_BUDGET = 6_000

    def __init__(self, profile: DatasetProfile) -> None:
        self._profile = profile

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_schema_block(self) -> str:
        """
        Return a compact schema string: column name, dtype, role, null%.
        Included in every system prompt.
        """
        lines = [
            f"Dataset: {self._profile.row_count:,} rows × "
            f"{self._profile.column_count} columns | "
            f"Source: {self._profile.source_path}",
            "",
            "Columns:",
        ]
        for col in self._profile.columns:
            null_str = f"  null={col.null_pct:.1f}%" if col.null_pct > 0 else ""
            lines.append(
                f"  {col.name!r:30s}  dtype={col.dtype:15s}  "
                f"role={col.inferred_role:12s}{null_str}"
            )
        return "\n".join(lines)

    def build_profile_summary(self) -> str:
        """
        Return a condensed profile summary.
        Included in the system prompt; kept intentionally brief.
        """
        parts: list[str] = []

        # Missingness
        null_cols = [
            (c.name, c.null_pct) for c in self._profile.columns if c.null_pct > 0
        ]
        if null_cols:
            null_str = ", ".join(
                f"{name} ({pct:.1f}%)" for name, pct in sorted(null_cols, key=lambda x: -x[1])
            )
            parts.append(f"Columns with missing values: {null_str}")

        # Duplicates
        if self._profile.duplicate_row_count > 0:
            dup_pct = self._profile.duplicate_row_count / max(self._profile.row_count, 1) * 100
            parts.append(
                f"Duplicate rows: {self._profile.duplicate_row_count:,} ({dup_pct:.1f}%)"
            )

        # Numeric summaries
        numeric = self._profile.numeric_columns()
        if numeric:
            parts.append("Numeric columns (mean / std / min / max):")
            for col in numeric[:8]:  # cap at 8 to save tokens
                if col.stats and hasattr(col.stats, "mean") and col.stats.mean is not None:
                    s = col.stats
                    parts.append(
                        f"  {col.name}: mean={s.mean:.3g}  std={s.std:.3g}  "
                        f"min={s.min:.3g}  max={s.max:.3g}"
                    )

        # Categorical top values
        categorical = self._profile.categorical_columns()
        if categorical:
            parts.append("Categorical columns (top values):")
            for col in categorical[:6]:
                if col.stats and hasattr(col.stats, "top_values") and col.stats.top_values:
                    top = col.stats.top_values[:3]
                    top_str = ", ".join(
                        f"{t['value']} ({t['pct']:.1f}%)" for t in top
                    )
                    parts.append(f"  {col.name}: {top_str} | cardinality={col.n_unique}")

        # Correlations (top 5 strongest)
        if self._profile.correlation_matrix:
            corrs = _flatten_correlation_matrix(self._profile.correlation_matrix)
            corrs.sort(key=lambda x: -abs(x[2]))
            if corrs:
                parts.append("Strongest correlations (Pearson r):")
                for col_a, col_b, r in corrs[:5]:
                    parts.append(f"  {col_a} ↔ {col_b}: r={r:.3f}")

        return "\n".join(parts)

    def build_tool_results_block(self, recent_records: list[ToolCallRecord]) -> str:
        """
        Format recent tool results for inclusion in the per-turn user message.
        Truncates individual summaries if total length exceeds token budget.
        """
        if not recent_records:
            return ""

        parts: list[str] = ["Recent tool results:"]
        for i, record in enumerate(recent_records, 1):
            header = (
                f"[{i}] Tool: {record.tool_name}  "
                f"Args: {json.dumps(record.arguments, ensure_ascii=False)}"
            )
            summary = _truncate(record.result_summary, max_chars=800)
            parts.append(f"{header}\nResult: {summary}")

        block = "\n\n".join(parts)
        # Apply overall budget
        budget_chars = self._CONTEXT_TOKEN_BUDGET * self._CHARS_PER_TOKEN
        if len(block) > budget_chars:
            block = block[:budget_chars] + "\n... (truncated)"
        return block

    def build_turn_instruction(
        self,
        tool_calls_used: int,
        max_tool_calls: int,
        turns_used: int,
        max_turns: int,
        is_final_turn: bool,
    ) -> str:
        """Return the turn-end instruction appended to every user message."""
        if is_final_turn:
            return (
                "This is your final turn. "
                "Do NOT call any more tools. "
                "Provide your final conclusion based on the evidence collected so far."
            )
        remaining_calls = max_tool_calls - tool_calls_used
        remaining_turns = max_turns - turns_used
        return (
            f"You have used {tool_calls_used}/{max_tool_calls} tool calls "
            f"and {turns_used}/{max_turns} turns. "
            f"Remaining budget: {remaining_calls} tool calls, {remaining_turns} turns. "
            "Call a tool if more evidence is needed, or provide your final conclusion."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_correlation_matrix(
    matrix: dict[str, dict[str, float | None]],
) -> list[tuple[str, str, float]]:
    """Flatten a symmetric correlation matrix to (col_a, col_b, r) triples (upper triangle)."""
    result: list[tuple[str, str, float]] = []
    cols = list(matrix.keys())
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            r = matrix[col_a].get(col_b)
            if r is not None:
                result.append((col_a, col_b, r))
    return result


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ... [truncated]"
