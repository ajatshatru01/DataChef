"""
ChartBuilder — constructs Plotly figure objects for all supported chart types.

Returns serialized Plotly figures (JSON strings) so they can be stored in
ToolCallRecords and rendered by the UI layer. Does not render charts itself.
"""

from __future__ import annotations

from typing import Any

import plotly.express as px
import plotly.graph_objects as go
import polars as pl

_PALETTE = px.colors.qualitative.Plotly
_BG = "#0f1117"
_PAPER_BG = "#1a1d2e"
_FONT_COLOR = "#e0e0e0"
_GRID_COLOR = "#2a2d3e"

_LAYOUT_DEFAULTS: dict[str, Any] = {
    "plot_bgcolor": _BG,
    "paper_bgcolor": _PAPER_BG,
    "font": {"color": _FONT_COLOR, "family": "Inter, system-ui, sans-serif"},
    "xaxis": {"gridcolor": _GRID_COLOR, "linecolor": _GRID_COLOR},
    "yaxis": {"gridcolor": _GRID_COLOR, "linecolor": _GRID_COLOR},
    "legend": {"bgcolor": "rgba(0,0,0,0)"},
    "margin": {"l": 60, "r": 30, "t": 60, "b": 60},
}


def _apply_defaults(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(title={"text": title, "font": {"size": 16}}, **_LAYOUT_DEFAULTS)
    return fig


class ChartBuilder:
    """Constructs serialized Plotly figures for DataChef findings."""

    # ------------------------------------------------------------------
    # Distribution charts
    # ------------------------------------------------------------------

    def histogram(
        self,
        df: pl.DataFrame,
        column: str,
        group_col: str | None = None,
        nbins: int = 40,
    ) -> str:
        """Return a histogram (optionally grouped / faceted) as JSON."""
        pdf = df.select(
            [column] + ([group_col] if group_col else [])
        ).drop_nulls().to_pandas()

        if group_col:
            fig = px.histogram(
                pdf,
                x=column,
                color=group_col,
                nbins=nbins,
                barmode="overlay",
                opacity=0.75,
                title=f"Distribution of {column} by {group_col}",
                color_discrete_sequence=_PALETTE,
            )
        else:
            fig = px.histogram(
                pdf,
                x=column,
                nbins=nbins,
                title=f"Distribution of {column}",
                color_discrete_sequence=[_PALETTE[0]],
            )
        _apply_defaults(fig, fig.layout.title.text)
        return fig.to_json()

    def box_plot(
        self,
        df: pl.DataFrame,
        value_col: str,
        group_col: str | None = None,
    ) -> str:
        """Return a box plot as JSON."""
        cols = [value_col] + ([group_col] if group_col else [])
        pdf = df.select(cols).drop_nulls().to_pandas()
        title = f"{value_col}" + (f" by {group_col}" if group_col else " — Box Plot")

        fig = px.box(
            pdf,
            x=group_col,
            y=value_col,
            color=group_col,
            title=title,
            color_discrete_sequence=_PALETTE,
            points="outliers",
        )
        _apply_defaults(fig, title)
        return fig.to_json()

    # ------------------------------------------------------------------
    # Relationship charts
    # ------------------------------------------------------------------

    def scatter(
        self,
        df: pl.DataFrame,
        x_col: str,
        y_col: str,
        color_col: str | None = None,
        sample_n: int = 2000,
    ) -> str:
        """Return a scatter plot (optionally colour-coded) as JSON."""
        cols = [x_col, y_col] + ([color_col] if color_col else [])
        pdf = df.select(cols).drop_nulls().to_pandas()
        if len(pdf) > sample_n:
            pdf = pdf.sample(n=sample_n, random_state=42)

        title = f"{x_col} vs {y_col}"
        fig = px.scatter(
            pdf,
            x=x_col,
            y=y_col,
            color=color_col,
            title=title,
            opacity=0.6,
            color_discrete_sequence=_PALETTE,
        )
        _apply_defaults(fig, title)
        return fig.to_json()

    def correlation_heatmap(self, matrix: dict[str, dict[str, float | None]]) -> str:
        """Return a correlation heatmap from a precomputed matrix as JSON."""
        cols = list(matrix.keys())
        z = [[matrix[r].get(c) if matrix[r].get(c) is not None else 0.0 for c in cols] for r in cols]
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=cols,
                y=cols,
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in z],
                texttemplate="%{text}",
            )
        )
        _apply_defaults(fig, "Correlation Matrix")
        fig.update_layout(
            xaxis={"tickangle": -45},
            height=max(400, len(cols) * 45),
        )
        return fig.to_json()

    # ------------------------------------------------------------------
    # Group comparison charts
    # ------------------------------------------------------------------

    def bar_group_means(
        self,
        group_stats: list[dict],
        value_col: str,
        group_col: str,
    ) -> str:
        """Return a bar chart of group means with error bars as JSON."""
        groups = [g["group"] for g in group_stats]
        means = [g["mean"] for g in group_stats]
        stds = [g.get("std") or 0 for g in group_stats]
        ns = [g["n"] for g in group_stats]

        fig = go.Figure(
            data=go.Bar(
                x=groups,
                y=means,
                error_y={"type": "data", "array": stds, "visible": True},
                marker_color=_PALETTE[0],
                text=[f"n={n}" for n in ns],
                textposition="outside",
            )
        )
        title = f"Mean {value_col} by {group_col}"
        fig.update_layout(
            xaxis_title=group_col,
            yaxis_title=f"Mean {value_col}",
        )
        _apply_defaults(fig, title)
        return fig.to_json()

    # ------------------------------------------------------------------
    # Time series
    # ------------------------------------------------------------------

    def line_trend(
        self,
        chart_data: dict,
        value_col: str,
        time_col: str,
    ) -> str:
        """Return a line chart with a rolling mean overlay as JSON."""
        times = chart_data["times"]
        values = chart_data["values"]
        rolling = chart_data.get("rolling_mean", [])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times,
            y=values,
            mode="lines",
            name=value_col,
            line={"color": _PALETTE[0], "width": 1.5},
            opacity=0.7,
        ))
        if rolling:
            fig.add_trace(go.Scatter(
                x=times,
                y=rolling,
                mode="lines",
                name="Rolling mean",
                line={"color": _PALETTE[1], "width": 2.5, "dash": "dot"},
            ))
        title = f"{value_col} over {time_col}"
        fig.update_layout(xaxis_title=time_col, yaxis_title=value_col)
        _apply_defaults(fig, title)
        return fig.to_json()

    # ------------------------------------------------------------------
    # Missing values
    # ------------------------------------------------------------------

    def missing_bar(self, missing_summary: dict) -> str:
        """Return a horizontal bar chart of missing value percentages."""
        cols_info = [c for c in missing_summary["columns"] if c["has_missing"]]
        if not cols_info:
            cols_info = missing_summary["columns"][:10]

        cols_info = sorted(cols_info, key=lambda x: -x["null_pct"])[:20]
        names = [c["column"] for c in cols_info]
        pcts = [c["null_pct"] for c in cols_info]
        counts = [c["null_count"] for c in cols_info]

        fig = go.Figure(go.Bar(
            x=pcts,
            y=names,
            orientation="h",
            marker_color=_PALETTE[3],
            text=[f"{p:.1f}% ({c:,})" for p, c in zip(pcts, counts, strict=True)],
            textposition="outside",
        ))
        _apply_defaults(fig, "Missing Values by Column")
        fig.update_layout(
            xaxis_title="% Missing",
            yaxis={"categoryorder": "total ascending"},
            height=max(300, len(names) * 30),
        )
        return fig.to_json()
