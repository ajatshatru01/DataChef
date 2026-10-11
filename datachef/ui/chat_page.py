"""
ChatPage — conversational dataset investigation interface.
"""

from __future__ import annotations

import html

import streamlit as st

from datachef import config
from datachef.persistence import session_store
from datachef.ui.finding_card import render_finding_card
from datachef.ui.progress_indicator import ProgressIndicator


def render_chat_page() -> None:
    st.markdown("## 💬 Conversational Dataset Investigation")
    st.caption("Ask questions in natural language. DataChef formulates plans, executes deterministic tools, and produces evidence-backed findings.")

    state = session_store.get_state()
    store = session_store.get_dataset_store()

    if not store.is_loaded() or state.dataset_profile is None:
        st.warning("⚠️ No dataset currently loaded. Please go to **Dataset Ingestion** to upload or load a sample dataset first.")
        if st.button("Go to Ingestion Page"):
            st.session_state["current_page"] = "Dataset Ingestion"
            st.rerun()
        return
    df = store.get_frame()

    # Check API key configuration
    cfg_errors = session_store.get_config_errors()
    if cfg_errors:
        for err in cfg_errors:
            st.error(f"⚙️ **Configuration Notice:** {err}")
        if config.PROVIDER == "ollama":
            st.info("Tip: Start Ollama with `ollama serve` and pull your model with `ollama pull " f"{config.OLLAMA_MODEL}`.")
        else:
            st.info("Tip: Set `HF_API_KEY` in your `.env` or set `DATACHEF_PROVIDER=ollama` to run 100% locally.")

    # Show dataset quick summary tag
    st.markdown(
        f"""
        <div style="background: rgba(30, 41, 59, 0.6); padding: 8px 16px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 1rem; display: flex; gap: 20px;">
            <span>📁 <b>{html.escape(state.dataset_profile.source_path)}</b></span>
            <span>📊 <b>{df.height:,}</b> rows</span>
            <span>🏷️ <b>{df.width}</b> columns</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sample query chips for instant exploration
    st.markdown("**Suggested questions to investigate:**")
    c1, c2, c3, c4 = st.columns(4)
    query_to_submit = None

    with c1:
        if st.button("🔍 Check missing patterns", width="stretch"):
            query_to_submit = "Which columns have missing values and what are their distributions?"
    with c2:
        if st.button("📈 Top correlations", width="stretch"):
            query_to_submit = "What are the strongest pairwise correlations among numeric columns?"
    with c3:
        if st.button("📊 Group comparisons", width="stretch"):
            num_cols = state.dataset_profile.numeric_columns()
            cat_cols = state.dataset_profile.categorical_columns()
            if num_cols and cat_cols:
                query_to_submit = f"Compare {num_cols[0].name} across different {cat_cols[0].name} groups."
            else:
                query_to_submit = "Compare values across available categorical groups."
    with c4:
        if st.button("🚨 Outlier detection", width="stretch"):
            num_cols = state.dataset_profile.numeric_columns()
            if num_cols:
                query_to_submit = f"Are there any statistical outliers in {num_cols[0].name}?"
            else:
                query_to_submit = "Detect outliers in numeric metrics."

    # Render previous findings in conversational order
    conversational_findings = [f for f in state.findings if f.trigger == "conversational"]
    if conversational_findings:
        st.markdown("### 📜 Investigation Findings")
        for finding in conversational_findings:
            with st.container(border=True):
                render_finding_card(finding)

    # Chat input box
    prompt = st.chat_input("Ask a question about this dataset (e.g. 'Is there a trend over time?')...")
    if query_to_submit:
        prompt = query_to_submit

    if prompt:
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            progress = ProgressIndicator(label=f"Investigating: {prompt}")
            try:
                agent = session_store.build_agent()
                finding = agent.investigate(prompt, on_step=progress.on_step)
                progress.complete(label=f"Completed: {prompt}")

                # Save record to persistence
                record_store = session_store.get_record_store()
                record_store.save(finding, session_id=state.session_id)

                with st.container(border=True):
                    render_finding_card(finding)
            except Exception as exc:
                progress.error(label=f"Investigation failed: {exc}")
                st.error(f"Error during investigation: {exc}")
