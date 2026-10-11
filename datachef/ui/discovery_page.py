"""
DiscoveryPage — autonomous dataset discovery and candidate prioritization.
"""

from __future__ import annotations

import streamlit as st

from datachef.analytics.discovery_engine import DiscoveryEngine
from datachef.persistence import session_store
from datachef.ui.finding_card import render_finding_card
from datachef.ui.progress_indicator import ProgressIndicator


def render_discovery_page() -> None:
    st.markdown("## 🚀 Autonomous Dataset Discovery")
    st.caption(
        "DataChef automatically scans the dataset with 7 deterministic statistical heuristics, "
        "ranks candidate findings, and conducts deep-dive investigations with supporting charts."
    )

    state = session_store.get_state()
    store = session_store.get_dataset_store()

    if not store.is_loaded() or state.dataset_profile is None:
        st.warning("⚠️ No dataset currently loaded. Please upload or load a sample dataset first.")
        if st.button("Go to Dataset Ingestion"):
            st.session_state["current_page"] = "Dataset Ingestion"
            st.rerun()
        return
    df = store.get_frame()

    # Check API key
    cfg_errors = session_store.get_config_errors()
    if cfg_errors:
        for err in cfg_errors:
            st.error(f"⚙️ **Configuration Error:** {err}")

    # Action bar
    col_run, col_candidates = st.columns([1, 2])

    with col_run:
        run_discovery = st.button("⚡ Run Autonomous Discovery", type="primary", width="stretch")

    # Heuristic candidate pre-check
    engine = DiscoveryEngine()
    candidates = engine.run(state.dataset_profile, df=df)

    with col_candidates:
        st.markdown(
            f"**Deterministic Candidate Pool:** `{len(candidates)}` candidate patterns identified by heuristic suite."
        )

    with st.expander(f"Inspect All {len(candidates)} Heuristic Candidates", expanded=False):
        for idx, cand in enumerate(candidates, 1):
            score_pct = int(cand.heuristic_score * 100)
            st.markdown(
                f"**#{idx} [{cand.check_type.upper()}]** (Heuristic Score: {score_pct}%) — {cand.description}"
            )
            if cand.involved_columns:
                st.caption(f"Involved columns: {', '.join(cand.involved_columns)}")

    # Execute discovery
    if run_discovery:
        progress = ProgressIndicator(label="Running autonomous dataset discovery...")
        try:
            agent = session_store.build_agent()
            findings = agent.discover(on_step=progress.on_step)
            progress.complete(label=f"Autonomous discovery finished! Found {len(findings)} key findings.")

            # Save each finding to persistence
            record_store = session_store.get_record_store()
            for f in findings:
                record_store.save(f, session_id=state.session_id)

            st.rerun()
        except Exception as exc:
            progress.error(label=f"Discovery failed: {exc}")
            st.error(f"Error during autonomous discovery: {exc}")

    # Render previous discovery findings
    discovery_findings = [f for f in state.findings if f.trigger == "discovery"]
    if discovery_findings:
        st.markdown("---")
        st.markdown(f"### 📑 Discovered Insights ({len(discovery_findings)})")
        for finding in discovery_findings:
            with st.container(border=True):
                render_finding_card(finding)
    elif not run_discovery:
        st.info("Click **'Run Autonomous Discovery'** above to let DataChef investigate interesting patterns in this dataset autonomously.")
