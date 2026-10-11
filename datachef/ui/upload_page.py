"""
UploadPage — dataset upload, validation, and deterministic profiling.
"""

from __future__ import annotations

import polars as pl
import streamlit as st

from datachef.context.models import DatasetProfile
from datachef.persistence import session_store


def render_upload_page() -> None:
    st.markdown("## 📁 Dataset Ingestion & Profiling")
    st.markdown(
        "Upload a **CSV** or **Parquet** file to start investigating. "
        "DataChef profiles your dataset locally using Polars."
    )

    # Privacy disclosure notice (as required by architecture section 8)
    st.info(
        "🛡️ **Local-First Privacy Guarantee:** Raw data rows are **never** transmitted "
        "over the network. Only schema information, high-level summary statistics, and concise "
        "tool execution results cross the boundary to the open-source inference model.",
        icon="🔒",
    )

    state = session_store.get_state()
    store = session_store.get_dataset_store()

    # Upload options
    col_upload, col_sample = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose a dataset file",
            type=["csv", "parquet"],
            help="Upload a tabular CSV or Parquet file (up to 200MB)",
        )

    with col_sample:
        st.markdown("**Or load a sample dataset:**")
        if st.button("📊 Load Sample: Ecommerce Orders", width="stretch"):
            _load_sample_ecommerce()
            st.rerun()
        if st.button("🚢 Load Sample: Titanic Passengers", width="stretch"):
            _load_sample_titanic()
            st.rerun()

    if uploaded_file is not None and (
        not store.is_loaded()
        or (
            state.dataset_profile
            and state.dataset_profile.source_path != uploaded_file.name
        )
    ):
        with st.spinner("Loading dataset into Polars engine..."):
            session_store.reset_session()
            df, _ = session_store.load_dataset(uploaded_file)
        with st.spinner("Running deterministic dataset profiler..."):
            session_store.profile_dataset(df, source_path=uploaded_file.name)
        st.success(f"Successfully loaded and profiled `{uploaded_file.name}`!")
        st.rerun()

    # If dataset is loaded, show profile overview
    store = session_store.get_dataset_store()
    state = session_store.get_state()
    df = store.get_frame() if store.is_loaded() else None
    profile = state.dataset_profile

    if df is not None and df.height > 0 and profile is not None:
        _render_profile_view(df, profile)


def _render_profile_view(df: pl.DataFrame, profile: DatasetProfile) -> None:
    st.markdown("---")
    st.markdown(f"### 📋 Profile Summary: `{profile.source_path}`")

    # Metric KPI tiles
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Rows", f"{profile.row_count:,}")
    m2.metric("Total Columns", profile.column_count)
    m3.metric("Duplicate Rows", f"{profile.duplicate_row_count:,}")
    total_nulls = sum(c.null_count for c in profile.columns)
    total_cells = profile.row_count * profile.column_count if profile.row_count > 0 else 1
    null_rate = (total_nulls / total_cells) * 100
    m4.metric("Overall Missingness", f"{null_rate:.1f}%")

    # Column Schema & Inferred Roles
    st.markdown("#### Column Roles & Statistics")
    col_data = []
    for c in profile.columns:
        col_data.append({
            "Column Name": c.name,
            "Polars Type": c.dtype,
            "Inferred Role": c.inferred_role.upper(),
            "Null Count": c.null_count,
            "Null %": f"{c.null_pct:.1f}%",
            "Unique Count": c.n_unique,
            "Sample Values": str(c.sample_values[:3]),
        })

    st.dataframe(col_data, width="stretch")

    # Data preview
    st.markdown("#### Dataset Preview (First 10 rows)")
    st.dataframe(df.head(10), width="stretch")

    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("💬 Start Chat Investigation →", type="primary", width="stretch"):
            st.session_state["current_page"] = "Chat Investigation"
            st.rerun()
    with c_btn2:
        if st.button("🚀 Run Autonomous Discovery →", width="stretch"):
            st.session_state["current_page"] = "Autonomous Discovery"
            st.rerun()


def _load_sample_ecommerce() -> None:
    """Generate a rich sample ecommerce dataset for instant demonstration."""
    import random
    from datetime import datetime, timedelta

    import numpy as np

    session_store.reset_session()
    n = 600
    np.random.seed(42)
    random.seed(42)

    categories = ["Electronics", "Home & Kitchen", "Apparel", "Books", "Beauty"]
    regions = ["North", "South", "East", "West", "Central"]
    segments = ["Consumer", "Corporate", "Small Business"]

    start_date = datetime(2025, 1, 1)
    dates = [(start_date + timedelta(days=int(i))).date() for i in np.random.randint(0, 180, n)]

    sales = np.round(np.random.exponential(scale=120, size=n) + 15, 2)
    # Add a few outliers
    sales[5] = 2850.0
    sales[12] = 3400.0

    quantities = np.random.randint(1, 10, n)
    discounts = np.round(np.random.choice([0.0, 0.05, 0.1, 0.15, 0.2, 0.3], size=n), 2)
    profits = np.round((sales * (1.0 - discounts) * 0.28) - np.random.normal(5, 2, n), 2)

    # Missing values in discounts and regions
    chosen_regions = [random.choice(regions) if random.random() > 0.05 else None for _ in range(n)]

    df = pl.DataFrame({
        "order_id": [f"ORD-{1000 + i}" for i in range(n)],
        "order_date": dates,
        "region": chosen_regions,
        "category": [random.choice(categories) for _ in range(n)],
        "segment": [random.choice(segments) for _ in range(n)],
        "sales": sales,
        "quantity": quantities,
        "discount": discounts,
        "profit": profits,
    })

    store = session_store.get_dataset_store()
    store.load_dataframe(df, source_path="sample_ecommerce_orders.csv")
    session_store.profile_dataset(df, source_path="sample_ecommerce_orders.csv")


def _load_sample_titanic() -> None:
    """Generate sample passenger dataset."""
    import random

    import numpy as np

    session_store.reset_session()
    n = 500
    np.random.seed(42)
    random.seed(42)

    classes = [1, 2, 3]
    genders = ["male", "female"]
    ports = ["Southampton", "Cherbourg", "Queenstown"]

    pclass = np.random.choice(classes, size=n, p=[0.24, 0.21, 0.55])
    sex = np.random.choice(genders, size=n, p=[0.64, 0.36])
    age = np.round(np.random.normal(29, 14, n), 1)
    age = [a if a > 1 else 1.0 for a in age]
    # some missing age
    age = [a if random.random() > 0.18 else None for a in age]

    # Survival probabilities influenced by gender and class
    survived = []
    for p, s in zip(pclass, sex, strict=True):
        prob = 0.75 if s == "female" else 0.20
        if p == 1:
            prob += 0.20
        elif p == 3:
            prob -= 0.10
        prob = max(0.05, min(0.95, prob))
        survived.append(1 if random.random() < prob else 0)

    fares = np.round(np.where(pclass == 1, np.random.exponential(60, n) + 30,
                    np.where(pclass == 2, np.random.exponential(20, n) + 12,
                             np.random.exponential(8, n) + 5)), 2)

    df = pl.DataFrame({
        "passenger_id": list(range(1, n + 1)),
        "survived": survived,
        "pclass": pclass.tolist(),
        "sex": sex.tolist(),
        "age": age,
        "fare": fares.tolist(),
        "embark_town": [random.choice(ports) if random.random() > 0.02 else None for _ in range(n)],
    })

    store = session_store.get_dataset_store()
    store.load_dataframe(df, source_path="sample_titanic_passengers.csv")
    session_store.profile_dataset(df, source_path="sample_titanic_passengers.csv")
