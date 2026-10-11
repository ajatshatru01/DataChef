"""
DataChef — Main Streamlit Application Entrypoint.

Tagline: Let your data tell its story.
"""

from __future__ import annotations

import html

import streamlit as st

from datachef import config
from datachef.persistence import session_store
from datachef.ui.chat_page import render_chat_page
from datachef.ui.discovery_page import render_discovery_page
from datachef.ui.history_page import render_history_page
from datachef.ui.upload_page import render_upload_page

# Set page configuration
st.set_page_config(
    page_title="DataChef — Autonomous Dataset Investigation",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for modern dark aesthetic and glassmorphism
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0b0f19;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

/* Card containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(17, 24, 39, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    backdrop-filter: blur(8px);
}

/* Metric tiles */
[data-testid="stMetric"] {
    background: rgba(31, 41, 55, 0.5);
    padding: 12px 16px;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

/* Primary buttons */
button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    transition: all 0.2s ease-in-out;
}
button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
}

/* Header badge styling */
.brand-title {
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px;
}
.brand-tagline {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-bottom: 1.5rem;
    font-style: italic;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main() -> None:
    session_store.initialize()
    state = session_store.get_state()
    store = session_store.get_dataset_store()
    df = store.get_frame() if store.is_loaded() else None

    # Sidebar Navigation & Diagnostics
    with st.sidebar:
        st.markdown('<p class="brand-title">🍳 DataChef</p>', unsafe_allow_html=True)
        st.markdown('<p class="brand-tagline">Let your data tell its story.</p>', unsafe_allow_html=True)

        # Navigation menu
        pages = [
            "Dataset Ingestion",
            "Chat Investigation",
            "Autonomous Discovery",
            "History & Audit",
        ]

        if "current_page" not in st.session_state:
            st.session_state["current_page"] = "Dataset Ingestion"

        page_choice = st.radio(
            "Navigation",
            options=pages,
            index=pages.index(st.session_state["current_page"]) if st.session_state["current_page"] in pages else 0,
            label_visibility="collapsed",
        )
        st.session_state["current_page"] = page_choice

        st.markdown("---")

        # Dataset Status Section
        st.markdown("##### 📁 Active Dataset")
        if df is not None and df.height > 0 and state.dataset_profile:
            st.success(f"**{html.escape(state.dataset_profile.source_path)}**")
            st.caption(f"📊 {df.height:,} rows · {df.width} cols")
        else:
            st.info("No dataset loaded yet.")

        # Open-Source Engine & Inference Status
        st.markdown("##### 🤖 Inference Engine")
        is_ollama = config.PROVIDER == "ollama"
        provider_label = "OLLAMA (Local)" if is_ollama else "HUGGING FACE (Cloud)"
        model_name = config.OLLAMA_MODEL if is_ollama else config.HF_MODEL_ID.split("/")[-1]
        st.caption(f"**Provider:** `{provider_label}`")
        st.caption(f"**Model:** `{model_name}`")

        cfg_errors = session_store.get_config_errors()
        if cfg_errors:
            for err in cfg_errors:
                st.warning(f"⚠️ {err}")
        else:
            status_text = "🟢 Ollama Connected" if is_ollama else "🟢 API Key Connected"
            st.success(status_text)

        st.markdown("---")
        if st.button("🔄 Reset Current Session", width="stretch"):
            session_store.reset_session()
            st.session_state["current_page"] = "Dataset Ingestion"
            st.rerun()

        st.caption("DataChef v0.1.0 · Open-Source Autonomous AI Agent")

    # Render selected page
    current = st.session_state["current_page"]
    if current == "Dataset Ingestion":
        render_upload_page()
    elif current == "Chat Investigation":
        render_chat_page()
    elif current == "Autonomous Discovery":
        render_discovery_page()
    elif current == "History & Audit":
        render_history_page()


if __name__ == "__main__":
    main()
