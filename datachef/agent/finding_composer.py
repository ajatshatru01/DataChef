"""
FindingComposer — synthesizes a completed investigation into a structured Finding.

Separates computed evidence from the model's interpretation and extracts
stated limitations from the final model response.
"""

from __future__ import annotations

import json
import math
import re

from datachef.context.models import Evidence, Finding, FindingDraft, InvestigationState


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
        draft = _parse_finding_draft(final_response_content)
        validated_primary_ids: set[str] = set()

        successful_records = [
            r for r in state.tool_call_log
            if r.validation_passed and r.execution_succeeded
        ]

        if draft is not None:
            conclusion = draft.conclusion
            interpretation = draft.interpretation
            limitations = list(draft.limitations)

            known_ids = {r.call_id for r in successful_records}
            for cid in draft.primary_evidence_call_ids:
                if cid in known_ids:
                    validated_primary_ids.add(cid)
                else:
                    limitations.append(
                        f"Note: Model referenced an unknown tool call ID '{cid}', "
                        "which was excluded from verified evidence."
                    )
        else:
            conclusion, interpretation, limitations = _parse_conclusion(final_response_content)

        evidence: list[Evidence] = []
        for i, record in enumerate(successful_records):
            is_primary = (
                record.call_id in validated_primary_ids
                if validated_primary_ids
                else (i == len(successful_records) - 1)
            )
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
                is_primary=is_primary,
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


def _parse_finding_draft(content: str) -> FindingDraft | None:
    """
    Try to parse model output as a FindingDraft JSON object.

    Handles: pure JSON, JSON inside markdown code fences, and JSON embedded in prose.
    Returns None if parsing fails completely. Never raises.
    """
    content = content.strip()
    if not content:
        return None

    # 1. Direct JSON parse
    try:
        return FindingDraft.model_validate_json(content)
    except Exception:
        pass

    # 2. Extract from markdown code fence
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL | re.IGNORECASE)
    if fence_match:
        try:
            return FindingDraft.model_validate_json(fence_match.group(1).strip())
        except Exception:
            pass

    # 3. Extract outermost { ... } block from surrounding prose
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return FindingDraft.model_validate_json(content[start : end + 1])
        except Exception:
            pass

    return None


def _has_conclusion_heading(content: str) -> bool:
    """Return True if content contains an explicit legacy CONCLUSION heading."""
    return bool(
        re.search(
            r"(?:^|\n)(?:#+\s*|\*\*)?CONCLUSION(?:\*\*)?[:\s]*",
            content.strip(),
            re.IGNORECASE,
        )
    )


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
    evidence_values: list[float] = [
        val
        for token in _NUMBER_RE.findall(evidence_text)
        if (val := _number_value(token)) is not None
    ]

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
