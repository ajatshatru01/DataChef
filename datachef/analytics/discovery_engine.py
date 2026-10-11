"""
DiscoveryEngine — generates ranked CandidateFinding objects from a DatasetProfile.

Runs a predefined suite of deterministic checks without calling the AI model.
The InvestigationAgent then prioritizes and investigates the top candidates.
"""

from __future__ import annotations

import math

import polars as pl

from datachef.analytics.group_comparator import GroupComparator
from datachef.analytics.trend_analyzer import TrendAnalyzer
from datachef.context.models import CandidateFinding, DatasetProfile


class DiscoveryEngine:
    """
    Orchestrates a predefined suite of statistical and data-quality checks
    against a DatasetProfile to produce a ranked list of candidate findings.

    No AI model calls are made here. This is entirely deterministic.
    """

    def __init__(self) -> None:
        self._group_comparator = GroupComparator()
        self._trend_analyzer = TrendAnalyzer()

    def run(
        self, profile: DatasetProfile, df: pl.DataFrame | None = None
    ) -> list[CandidateFinding]:
        """
        Run all checks and return a ranked list of CandidateFinding objects.

        The list is sorted by heuristic_score descending (most interesting first).
        """
        candidates: list[CandidateFinding] = []

        candidates.extend(self._check_missingness(profile))
        candidates.extend(self._check_outliers(profile))
        candidates.extend(self._check_correlations(profile))
        candidates.extend(self._check_group_differences(profile, df))
        candidates.extend(self._check_trends(profile, df))
        candidates.extend(self._check_duplicates(profile))
        candidates.extend(self._check_cardinality(profile))

        # Sort by heuristic score descending
        candidates.sort(key=lambda c: -c.heuristic_score)
        return candidates

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_missingness(
        self, profile: DatasetProfile
    ) -> list[CandidateFinding]:
        """Flag columns with notable missing value rates."""
        candidates: list[CandidateFinding] = []
        for col in profile.columns:
            if col.null_pct >= 5.0:  # at least 5% missing
                score = min(col.null_pct / 100.0, 1.0)
                candidates.append(CandidateFinding(
                    check_type="missingness",
                    description=(
                        f"Column '{col.name}' has {col.null_pct:.1f}% missing values "
                        f"({col.null_count:,} of {profile.row_count:,} rows)."
                    ),
                    involved_columns=[col.name],
                    heuristic_score=score,
                    suggested_tools=["get_missing_summary"],
                    context_data={
                        "column": col.name,
                        "null_pct": col.null_pct,
                        "null_count": col.null_count,
                    },
                ))
        return candidates

    def _check_outliers(
        self, profile: DatasetProfile
    ) -> list[CandidateFinding]:
        """Flag numeric columns with high outlier counts (IQR method)."""
        candidates: list[CandidateFinding] = []
        for col in profile.numeric_columns():
            if col.stats is None or not hasattr(col.stats, "n_outliers_iqr"):
                continue
            n_outliers = col.stats.n_outliers_iqr
            if n_outliers == 0:
                continue
            outlier_pct = n_outliers / max(profile.row_count, 1) * 100
            if outlier_pct < 0.5:  # less than 0.5% — not interesting enough
                continue
            score = min(outlier_pct / 10.0, 0.95)
            candidates.append(CandidateFinding(
                check_type="outliers",
                description=(
                    f"Column '{col.name}' contains {n_outliers:,} candidate outliers "
                    f"({outlier_pct:.1f}% of rows) by IQR method."
                ),
                involved_columns=[col.name],
                heuristic_score=score,
                suggested_tools=["detect_outliers", "get_column_stats"],
                context_data={
                    "column": col.name,
                    "n_outliers": n_outliers,
                    "outlier_pct": outlier_pct,
                },
            ))
        return candidates

    def _check_correlations(
        self, profile: DatasetProfile
    ) -> list[CandidateFinding]:
        """Flag strongly correlated variable pairs."""
        candidates: list[CandidateFinding] = []
        if not profile.correlation_matrix:
            return candidates

        seen: set[frozenset] = set()
        cols = list(profile.correlation_matrix.keys())

        for col_a in cols:
            for col_b in cols:
                if col_a == col_b:
                    continue
                pair = frozenset([col_a, col_b])
                if pair in seen:
                    continue
                seen.add(pair)
                r = profile.correlation_matrix[col_a].get(col_b)
                if r is None or not math.isfinite(r):
                    continue
                abs_r = abs(r)
                if abs_r < 0.5:
                    continue
                direction = "positive" if r > 0 else "negative"
                score = min(abs_r, 0.99)
                candidates.append(CandidateFinding(
                    check_type="correlation",
                    description=(
                        f"Strong {direction} correlation between '{col_a}' and '{col_b}' "
                        f"(Pearson r = {r:.3f})."
                    ),
                    involved_columns=[col_a, col_b],
                    heuristic_score=score,
                    suggested_tools=["get_correlation", "get_column_stats"],
                    context_data={"col_a": col_a, "col_b": col_b, "r": r},
                ))
        return candidates

    def _check_group_differences(
        self, profile: DatasetProfile, df: pl.DataFrame | None = None
    ) -> list[CandidateFinding]:
        """
        Identify (numeric, categorical) pairs likely to show group differences.

        When a DataFrame is available, compute the actual maximum pairwise
        Cohen's d and require statistical support. Profile-only callers retain
        a conservative candidate-generation fallback.
        """
        candidates: list[CandidateFinding] = []
        numeric_cols = profile.numeric_columns()
        cat_cols = profile.categorical_columns()

        for num_col in numeric_cols:
            if num_col.stats is None:
                continue
            std = getattr(num_col.stats, "std", None)
            mean = getattr(num_col.stats, "mean", None)
            if std is None or mean is None or mean == 0:
                continue
            cv = std / abs(mean)  # coefficient of variation
            if cv < 0.1:  # low variance — less interesting
                continue

            for cat_col in cat_cols:
                n_unique = cat_col.n_unique
                if not (2 <= n_unique <= 15):
                    continue
                context_data = {
                    "numeric_col": num_col.name,
                    "group_col": cat_col.name,
                    "n_groups": n_unique,
                    "cv": round(cv, 3),
                }
                description = (
                    f"Potential group differences in '{num_col.name}' "
                    f"across {n_unique} categories of '{cat_col.name}'."
                )
                score = min(cv * 0.35 + (n_unique / 15) * 0.15, 0.55)

                if df is not None:
                    result = self._group_comparator.compare(
                        df,
                        numeric_col=num_col.name,
                        group_col=cat_col.name,
                        run_kruskal=True,
                        top_groups=15,
                    )
                    effects = [
                        abs(e["cohens_d"])
                        for e in result.get("effect_sizes", [])
                        if e.get("cohens_d") is not None
                        and math.isfinite(e["cohens_d"])
                    ]
                    max_effect = max(effects, default=0.0)
                    kruskal = result.get("kruskal_wallis") or {}
                    p_value = kruskal.get("p_value")
                    significant = bool(
                        p_value is not None and p_value < 0.05
                    )
                    if max_effect < 0.5 or not significant:
                        continue
                    score = min(0.4 + max_effect / 3.0, 0.95)
                    context_data.update(
                        {"max_abs_cohens_d": round(max_effect, 4), "p_value": p_value}
                    )
                    description = (
                        f"'{num_col.name}' differs across '{cat_col.name}' groups "
                        f"(maximum |Cohen's d| = {max_effect:.3f}; "
                        f"Kruskal–Wallis p = {p_value:.4g})."
                    )
                candidates.append(CandidateFinding(
                    check_type="group_difference",
                    description=description,
                    involved_columns=[num_col.name, cat_col.name],
                    heuristic_score=score,
                    suggested_tools=["compare_groups", "get_group_distribution"],
                    context_data=context_data,
                ))

        # Limit to top 5 group-difference candidates to avoid explosion
        candidates.sort(key=lambda c: -c.heuristic_score)
        return candidates[:5]

    def _check_trends(
        self, profile: DatasetProfile, df: pl.DataFrame | None = None
    ) -> list[CandidateFinding]:
        """Flag (datetime, numeric) pairs as trend candidates."""
        candidates: list[CandidateFinding] = []
        datetime_cols = profile.datetime_columns()
        numeric_cols = profile.numeric_columns()

        if not datetime_cols:
            return candidates

        for dt_col in datetime_cols[:2]:  # max 2 datetime columns
            for num_col in numeric_cols[:4]:  # max 4 numeric columns
                score = 0.35
                description = (
                    f"Potential time trend in '{num_col.name}' "
                    f"indexed by '{dt_col.name}'."
                )
                context_data = {
                    "value_col": num_col.name,
                    "time_col": dt_col.name,
                }
                if df is not None:
                    result = self._trend_analyzer.monotonic_trend(
                        df, value_col=num_col.name, time_col=dt_col.name
                    )
                    trend = result.get("trend", {})
                    rho = trend.get("spearman_rho")
                    p_value = trend.get("p_value")
                    if (
                        rho is None
                        or p_value is None
                        or not math.isfinite(rho)
                        or abs(rho) < 0.3
                        or p_value >= 0.05
                    ):
                        continue
                    score = min(abs(rho), 0.95)
                    context_data.update({"spearman_rho": rho, "p_value": p_value})
                    description = (
                        f"'{num_col.name}' has a {trend.get('direction')} time trend "
                        f"by '{dt_col.name}' (Spearman rho = {rho:.3f}, "
                        f"p = {p_value:.4g})."
                    )
                candidates.append(CandidateFinding(
                    check_type="trend",
                    description=description,
                    involved_columns=[num_col.name, dt_col.name],
                    heuristic_score=score,
                    suggested_tools=["analyze_trend"],
                    context_data=context_data,
                ))
        return candidates

    def _check_duplicates(
        self, profile: DatasetProfile
    ) -> list[CandidateFinding]:
        """Flag datasets with substantial duplicate rows."""
        if profile.duplicate_row_count == 0:
            return []
        dup_pct = profile.duplicate_row_count / max(profile.row_count, 1) * 100
        if dup_pct < 1.0:
            return []
        score = min(dup_pct / 20.0, 0.9)
        return [CandidateFinding(
            check_type="duplicates",
            description=(
                f"Dataset contains {profile.duplicate_row_count:,} duplicate rows "
                f"({dup_pct:.1f}% of all rows)."
            ),
            involved_columns=[],
            heuristic_score=score,
            suggested_tools=["get_duplicate_summary"],
            context_data={
                "duplicate_row_count": profile.duplicate_row_count,
                "dup_pct": round(dup_pct, 2),
            },
        )]

    def _check_cardinality(
        self, profile: DatasetProfile
    ) -> list[CandidateFinding]:
        """Flag categorical columns with unexpectedly high or low cardinality."""
        candidates: list[CandidateFinding] = []
        for col in profile.categorical_columns():
            n_unique = col.n_unique
            n_rows = profile.row_count
            # Near-unique: >80% unique values in a categorical column
            if n_unique > 0.8 * n_rows and n_rows > 50:
                candidates.append(CandidateFinding(
                    check_type="high_cardinality",
                    description=(
                        f"Categorical column '{col.name}' has very high cardinality "
                        f"({n_unique:,} unique values in {n_rows:,} rows). "
                        "It may be an identifier column or free-text."
                    ),
                    involved_columns=[col.name],
                    heuristic_score=0.4,
                    suggested_tools=["get_value_counts"],
                    context_data={"column": col.name, "n_unique": n_unique},
                ))
            # Single value (effectively constant)
            elif n_unique == 1:
                candidates.append(CandidateFinding(
                    check_type="constant_column",
                    description=(
                        f"Column '{col.name}' has only one unique value — "
                        "it provides no discriminating information."
                    ),
                    involved_columns=[col.name],
                    heuristic_score=0.3,
                    suggested_tools=["get_value_counts"],
                    context_data={"column": col.name, "n_unique": 1},
                ))
        return candidates
