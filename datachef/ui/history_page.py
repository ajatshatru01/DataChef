"""
HistoryPage — view past investigations with expandable step-by-step records and export.
"""

from __future__ import annotations

import json
from datetime import datetime

import streamlit as st

from datachef.persistence import session_store
from datachef.ui.finding_card import render_finding_card


def render_history_page() -> None:
    st.markdown("## 📜 Investigation History & Reproducibility Audit")
    st.caption("Inspect immutable step-by-step logs, tool execution traces, and exported findings.")

    record_store = session_store.get_record_store()
    current_state = session_store.get_state()
    saved_sessions = record_store.list_sessions()

    # Session Selector
    session_options = {"Current Active Session": current_state.session_id}
    for s in saved_sessions:
        label = f"{s['session_id'][:8]}... ({s['n_findings']} findings, {s['created_at'][:16]})"
        session_options[label] = s["session_id"]

    selected_label = st.selectbox("Select Investigation Session", options=list(session_options.keys()))
    selected_session_id = session_options[selected_label]

    # Load findings
    if selected_session_id == current_state.session_id and current_state.findings:
        findings_to_display = current_state.findings
        is_current = True
    else:
        records = record_store.load_session(selected_session_id)
        findings_to_display = []
        is_current = False

    if is_current:
        st.info(f"Viewing active session: `{selected_session_id}` ({len(findings_to_display)} findings)")

        if not findings_to_display:
            st.warning("No investigations run in the current session yet.")
            return

        # Filter
        filter_type = st.radio("Filter by Type", ["All", "Conversational", "Discovery"], horizontal=True)
        filtered = [
            f for f in findings_to_display
            if filter_type == "All" or f.trigger.lower() == filter_type.lower()
        ]

        # Export section
        col_exp1, col_exp2 = st.columns(2)
        json_export = json.dumps([f.model_dump(mode="json") for f in filtered], indent=2, default=str)
        with col_exp1:
            st.download_button(
                "📥 Export Findings as JSON",
                data=json_export,
                file_name=f"datachef_investigation_{selected_session_id[:8]}.json",
                mime="application/json",
                width="stretch",
            )
        with col_exp2:
            md_export = _generate_markdown_report(filtered, current_state.dataset_profile.source_path if current_state.dataset_profile else "Dataset")
            st.download_button(
                "📄 Export Findings as Markdown Report",
                data=md_export,
                file_name=f"datachef_report_{selected_session_id[:8]}.md",
                mime="text/markdown",
                width="stretch",
            )

        st.markdown("---")
        for finding in filtered:
            with st.container(border=True):
                render_finding_card(finding)

    else:
        # Viewing saved on-disk records
        records = record_store.load_session(selected_session_id)
        if not records:
            st.warning(f"No records found on disk for session `{selected_session_id}`.")
            return

        st.success(f"Loaded {len(records)} findings from disk for session `{selected_session_id}`.")

        # Download raw records
        raw_ndjson = json.dumps(records, indent=2, default=str)
        st.download_button(
            "📥 Download Raw Session Records (JSON)",
            data=raw_ndjson,
            file_name=f"session_{selected_session_id}.json",
            mime="application/json",
        )

        st.markdown("---")
        for idx, rec in enumerate(records, 1):
            with st.expander(f"Finding #{idx}: {rec.get('question', 'Untitled')}", expanded=(idx == 1)):
                st.markdown(f"**Conclusion:** {rec.get('conclusion', 'N/A')}")
                if rec.get("interpretation"):
                    st.markdown(f"**Interpretation:** {rec['interpretation']}")

                tool_calls = rec.get("tool_calls", [])
                st.markdown(f"**Tools Invoked ({len(tool_calls)}):**")
                for tc in tool_calls:
                    st.code(
                        f"Tool: {tc.get('tool_name')}\n"
                        f"Arguments: {json.dumps(tc.get('arguments', {}))}\n"
                        f"Duration: {tc.get('execution_time_ms', 0)}ms\n"
                        f"Summary: {tc.get('result_summary', '')}",
                        language="yaml",
                    )


def _generate_markdown_report(findings: list, dataset_name: str) -> str:
    lines = [
        "# DataChef Investigation Report",
        f"> Dataset: **{dataset_name}** | Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "---",
        "",
    ]
    for idx, f in enumerate(findings, 1):
        lines.append(f"## Finding #{idx}: {f.question}")
        lines.append(f"- **Trigger:** `{f.trigger}`")
        lines.append(f"- **Tools Invocations:** {f.tool_call_count} | **Turns:** {f.turn_count}")
        lines.append("")
        lines.append("### Conclusion")
        lines.append(f"{f.conclusion}")
        lines.append("")
        if f.interpretation:
            lines.append("### Interpretation")
            lines.append(f"{f.interpretation}")
            lines.append("")
        if f.limitations:
            lines.append("### Limitations")
            for lim in f.limitations:
                lines.append(f"- {lim}")
            lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)
