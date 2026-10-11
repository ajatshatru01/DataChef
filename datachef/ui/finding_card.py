"""
FindingCard — renders a single Finding with metrics, charts, and auditable tool trace.
"""

from __future__ import annotations

import html

import plotly.io as pio
import streamlit as st

from datachef.context.models import Finding


def render_finding_card(finding: Finding, expanded: bool = True) -> None:
    """
    Render a structured finding card with rich styling, tabs/expanders,
    Plotly charts, and evidence separation.
    """
    card_title = html.escape(finding.question or "Dataset Investigation Finding")
    badge_color = "#3b82f6" if finding.trigger == "conversational" else "#10b981"
    trigger_badge = (
        f'<span style="background-color: {badge_color}22; color: {badge_color}; '
        f'padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;">'
        f'{finding.trigger}</span>'
    )

    limit_badge = ""
    if finding.hit_limit:
        limit_badge = (
            f'<span style="background-color: #f59e0b22; color: #f59e0b; '
            f'padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">'
            f'⚠️ Reached limit ({finding.limit_reason})</span>'
        )

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
            <div style="font-size: 1.15rem; font-weight: 600; color: #f8fafc;">{card_title}</div>
            <div>{trigger_badge} {limit_badge}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if finding.error:
        st.error(f"**Investigation encountered an error:** {finding.error}")
        return

    # Primary conclusion callout
    if finding.conclusion:
        st.info(f"**Conclusion:** {finding.conclusion}")
    if finding.unverified_numerical_claims:
        st.warning(
            "Potential unverified numerical claims: "
            + ", ".join(finding.unverified_numerical_claims)
        )

    # Tabs for Evidence & Visuals, Interpretation, Limitations, Audit Trail
    tab_evidence, tab_interp, tab_limits, tab_audit = st.tabs(
        ["📊 Evidence & Visuals", "💡 Interpretation", "⚠️ Limitations", "🔍 Audit Trace"]
    )

    with tab_evidence:
        if not finding.evidence:
            st.write("No direct deterministic evidence was logged for this finding.")
        else:
            for idx, ev in enumerate(finding.evidence, 1):
                col_meta, col_vis = st.columns([1, 2])
                with col_meta:
                    st.markdown(f"**Evidence #{idx}: `{ev.tool_name}`**")
                    args_str = ", ".join(f"`{k}`: {v}" for k, v in ev.arguments.items())
                    st.caption(f"Inputs: {args_str}")
                    with st.expander("Computed summary", expanded=False):
                        st.code(ev.result_summary, language="json")

                with col_vis:
                    if ev.chart_json:
                        try:
                            fig = pio.from_json(ev.chart_json)
                            st.plotly_chart(fig, width="stretch", key=f"chart_{finding.finding_id}_{idx}")
                        except Exception:
                            st.caption("Chart format could not be visualized directly.")
                    else:
                        st.caption("No chart generated for this tool result.")

    with tab_interp:
        if finding.interpretation:
            st.markdown(finding.interpretation)
        else:
            st.caption("No additional interpretation provided.")

    with tab_limits:
        if finding.limitations:
            for lim in finding.limitations:
                st.markdown(f"• {lim}")
        else:
            st.caption("No specific statistical or data limitations stated.")

    with tab_audit:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tool Invocations", finding.tool_call_count)
        c2.metric("Turns Taken", finding.turn_count)
        c3.metric("Model", finding.model_id.split("/")[-1] if finding.model_id else "N/A")
        c4.metric("Provider", finding.provider or "N/A")

        with st.expander("View Full Finding JSON Record", expanded=False):
            st.json(finding.model_dump(mode="json"))
