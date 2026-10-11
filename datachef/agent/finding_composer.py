"""
FindingComposer — synthesizes a completed investigation into a structured Finding.

Separates computed evidence from the model's interpretation and extracts
stated limitations from the final model response.
"""

from __future__ import annotations

import json
import math
import re

from datachef.context.models import Evidence, Finding, InvestigationState


class FindingComposer:
    """Composes a Finding from an InvestigationState at the end of the loop."""

    def compose(
        self,
        state: InvestigationState,
        question: str,
        trigger: str,
        final_response_content: str,
        model_id: str,
        provider: str,
        temperature: float,
        hit_limit: bool = False,
        limit_reason: str | None = None,
        error: str | None = None,
    ) -> Finding:
        """
        Build a Finding from the accumulated investigation state.

        Parameters
        ----------
        state:
            The current InvestigationState containing tool call records.
        question:
            The original question or discovery candidate description.
        trigger:
            "conversational" or "discovery".
        final_response_content:
            The model's final text response (the conclusion turn).
        model_id, provider, temperature:
            Model configuration metadata for reproducibility.
        hit_limit:
            Whether the loop was terminated by a limit rather than natural conclusion.
        limit_reason:
            Which limit was hit (e.g., "max_tool_calls", "max_turns", "timeout").
        error:
            Set if the investigation failed with a non-recoverable error.
        """
        conclusion, interpretation, limitations = _parse_conclusion(final_response_content)

        # Build evidence list from successful tool call records
        successful_records = [r for r in state.tool_call_log if r.validation_passed]
        evidence: list[Evidence] = []
        for i, record in enumerate(successful_records):
            evidence.append(Evidence(
                call_id=record.call_id,
                turn_index=record.turn_index,
                tool_name=record.tool_name,
                arguments=record.arguments,
                result_summary=record.result_summary,
                result_data=record.result_data,
                chart_json=record.chart_json,
                execution_time_ms=record.execution_time_ms,
                execution_succeeded=record.execution_succeeded,
                result_truncated=record.result_truncated,
                timestamp=record.timestamp,
                is_primary=(i == len(successful_records) - 1),  # Last result = primary
            ))

        unverified_claims = _find_unverified_numbers(
            f"{conclusion}\n{interpretation}", successful_records
        )
        if unverified_claims:
            limitations.append(
                "Potential unverified numerical claim(s) were not found in this "
                f"investigation's tool results: {', '.join(unverified_claims)}."
            )

        return Finding(
            session_id=state.session_id,
            dataset_fingerprint=(
                state.dataset_profile.dataset_fingerprint if state.dataset_profile else ""
            ),
            trigger=trigger,  # type: ignore[arg-type]
            question=question,
            conclusion=conclusion or final_response_content,
            evidence=evidence,
            tool_call_log=list(state.tool_call_log),
            interpretation=interpretation,
            limitations=limitations,
            unverified_numerical_claims=unverified_claims,
            tool_call_count=state.tool_call_count,
            turn_count=state.turn_count,
            hit_limit=hit_limit,
            limit_reason=limit_reason,
            model_id=model_id,
            provider=provider,
            temperature=temperature,
            error=error,
        )

    def compose_error_finding(
        self,
        state: InvestigationState,
        question: str,
        trigger: str,
        error_message: str,
        model_id: str,
        provider: str,
        temperature: float,
    ) -> Finding:
        """Create a minimal Finding for a failed investigation."""
        return self.compose(
            state=state,
            question=question,
            trigger=trigger,
            final_response_content=(
                f"Investigation could not be completed due to an error: {error_message}"
            ),
            model_id=model_id,
            provider=provider,
            temperature=temperature,
            hit_limit=False,
            error=error_message,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_conclusion(
    content: str,
) -> tuple[str, str, list[str]]:
    """
    Extract CONCLUSION, INTERPRETATION, and LIMITATIONS sections from the
    model's final response.

    Falls back gracefully if the model did not follow the structured format.
    """
    content = content.strip()

    conclusion = ""
    interpretation = ""
    limitations: list[str] = []

    # Try to extract structured sections (handles "### CONCLUSION", "CONCLUSION:", "**CONCLUSION:**")
    conclusion_match = re.search(
        r"(?:^|\n)(?:#+\s*|\*\*)?CONCLUSION(?:\*\*)?[:\s]*\n*(.+?)(?=(?:\n(?:#+\s*|\*\*)?INTERPRETATION)|(?:\n(?:#+\s*|\*\*)?LIMITATIONS)|\Z)",
        content,
        re.IGNORECASE | re.DOTALL,
    )
    interp_match = re.search(
        r"(?:^|\n)(?:#+\s*|\*\*)?INTERPRETATION(?:\*\*)?[:\s]*\n*(.+?)(?=(?:\n(?:#+\s*|\*\*)?LIMITATIONS)|\Z)",
        content,
        re.IGNORECASE | re.DOTALL,
    )
    limits_match = re.search(
        r"(?:^|\n)(?:#+\s*|\*\*)?LIMITATIONS(?:\*\*)?[:\s]*\n*(.+?)\Z",
        content,
        re.IGNORECASE | re.DOTALL,
    )

    if conclusion_match:
        conclusion = conclusion_match.group(1).strip()
    if interp_match:
        interpretation = interp_match.group(1).strip()
    if limits_match:
        raw_limits = limits_match.group(1).strip()
        # Split by bullet points or newlines
        limitations = []
        for line in raw_limits.splitlines():
            cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            if cleaned:
                limitations.append(cleaned)

    # If no structured sections found, use the whole content as conclusion
    if not conclusion:
        conclusion = content
        # Try to find any limitation caveats in the text
        caveat_keywords = ["note that", "however", "caveat", "limitation", "important to note"]
        for sentence in re.split(r"[.!?]", content):
            if any(k in sentence.lower() for k in caveat_keywords):
                limitations.append(sentence.strip())

    return conclusion, interpretation, limitations


_NUMBER_RE = re.compile(
    r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?%?"
)


def _find_unverified_numbers(
    text: str, records: list
) -> list[str]:
    """Flag model numbers that cannot be matched to deterministic tool output."""
    evidence_text = "\n".join(
        f"{record.result_summary}\n{json.dumps(record.result_data, default=str)}"
        for record in records
    )
    evidence_values = [_number_value(token) for token in _NUMBER_RE.findall(evidence_text)]
    evidence_values = [value for value in evidence_values if value is not None]

    unverified: list[str] = []
    for token in _NUMBER_RE.findall(text):
        value = _number_value(token)
        if value is None:
            continue
        matched = any(
            math.isclose(value, candidate, rel_tol=1e-4, abs_tol=1e-6)
            for candidate in evidence_values
        )
        if not matched and token not in unverified:
            unverified.append(token)
    return unverified


def _number_value(token: str) -> float | None:
    try:
        return float(token.rstrip("%").replace(",", ""))
    except ValueError:
        return None
