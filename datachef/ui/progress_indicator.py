"""
ProgressIndicator — live status indicator during agent investigations.

Provides a clean Streamlit status container that updates dynamically
as the agent moves through the bounded loop states:
PLANNING → VALIDATING → EXECUTING → EVALUATING → COMPOSING.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


class ProgressIndicator:
    """
    Manages live feedback in Streamlit during an active investigation turn.
    """

    STEP_ICONS = {
        "planning": "🧠",
        "validating": "🛡️",
        "executing": "⚙️",
        "evaluating": "📊",
        "composing": "📝",
    }

    STEP_TITLES = {
        "planning": "Planning investigation strategy...",
        "validating": "Validating tool schema & parameters...",
        "executing": "Executing deterministic analytics...",
        "evaluating": "Evaluating computed evidence...",
        "composing": "Synthesizing evidence-backed finding...",
    }

    def __init__(self, label: str = "Investigating dataset...") -> None:
        self._status = st.status(label, expanded=True)
        self._steps: list[str] = []

    def on_step(self, step_name: str, payload: dict[str, Any]) -> None:
        """Callback invoked by InvestigationAgent."""
        icon = self.STEP_ICONS.get(step_name, "🔍")
        title = self.STEP_TITLES.get(step_name, f"Step: {step_name}")

        detail = ""
        if step_name == "planning":
            turn = payload.get("turn", 1)
            detail = f"Turn {turn} — formulating hypothesis & tool selection"
        elif step_name == "validating":
            tool = payload.get("tool", "")
            detail = f"Checking constraints for tool `{tool}`"
        elif step_name == "executing":
            tool = payload.get("tool", "")
            args = payload.get("arguments", {})
            args_str = ", ".join(f"{k}={v}" for k, v in args.items()) if args else "no args"
            detail = f"Running `{tool}({args_str})` on Polars engine"
        elif step_name == "evaluating":
            tool = payload.get("tool", "")
            exec_ms = payload.get("exec_ms", 0)
            detail = f"Received `{tool}` result in {exec_ms}ms"
        elif step_name == "composing":
            detail = "Structuring conclusion, separating facts from inference"

        log_entry = f"{icon} **{title}**\n\n_{detail}_" if detail else f"{icon} **{title}**"
        self._steps.append(log_entry)

        with self._status:
            st.markdown(log_entry)

    def complete(self, label: str = "Investigation completed!") -> None:
        """Mark status container as finished and collapse."""
        self._status.update(label=label, state="complete", expanded=False)

    def error(self, label: str = "Investigation halted with error.") -> None:
        """Mark status container as failed."""
        self._status.update(label=label, state="error", expanded=True)
