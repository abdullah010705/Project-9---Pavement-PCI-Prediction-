"""
eda.py — exploratory data analysis and data-quality analysis (owner: Aum)

This stage analyses the final modelling dataset before model training. It:
  * reports dataset size, road counts, column types and descriptive statistics
  * audits missing, duplicate, infinite and unusual/extreme numeric values
  * plots PCI distributions and PCI_NEXT relationships with key predictors
  * creates a correlation matrix and ranks numeric variables by relationship
    with future PCI (PCI_NEXT)
  * saves all figures to /plots and tabular/markdown outputs to /results

Run directly from the project root with:
    python -m src.eda
or as part of:
    python run_pipeline.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from src import schema_checks


# ---------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------
OVERVIEW_PATH = os.path.join(config.RESULTS_DIR, "eda_dataset_overview.csv")
DESCRIPTIVE_PATH = os.path.join(config.RESULTS_DIR, "eda_descriptive_statistics.csv")
MISSING_PATH = os.path.join(config.RESULTS_DIR, "eda_missing_values.csv")
OUTLIER_PATH = os.path.join(config.RESULTS_DIR, "eda_outlier_summary.csv")
CORRELATION_PATH = os.path.join(config.RESULTS_DIR, "eda_correlation_matrix.csv")
TARGET_CORRELATION_PATH = os.path.join(config.RESULTS_DIR, "eda_target_correlations.csv")
RUT_QUALITY_PATH = os.path.join(config.RESULTS_DIR, "eda_rut_missingness_quality.csv")
RUT_TRANSITION_PATH = os.path.join(config.RESULTS_DIR, "eda_rut_transition_quality.csv")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _boxplot_label_kwarg(labels):
    """matplotlib renamed boxplot's `labels` argument to `tick_labels` in 3.9 and
    removed the old name in 3.11. Pick whichever this install accepts, so the pipeline
    runs on both old and new matplotlib rather than forcing everyone onto one version.
    """
    from matplotlib import __version__ as _mpl_version
    major, minor = (int(x) for x in _mpl_version.split(".")[:2])
    key = "tick_labels" if (major, minor) >= (3, 9) else "labels"
    return {key: labels}


def _save_current_plot(filename: str) -> str:
    """Save the current matplotlib figure into the shared plots directory."""
    path = os.path.join(config.PLOTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def _available(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if c in df.columns]


def _scatter_with_trend(df: pd.DataFrame, x: str, y: str, filename: str,
                        xlabel: str | None = None, title: str | None = None) -> None:
    """Scatter plot plus a simple linear trend line for visual EDA."""
    data = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return

    plt.figure(figsize=(8, 5.5))
    plt.scatter(data[x], data[y], alpha=0.38, s=24)
    if data[x].nunique() > 1:
        slope, intercept = np.polyfit(data[x], data[y], 1)
        x_line = np.linspace(data[x].min(), data[x].max(), 200)
        plt.plot(x_line, slope * x_line + intercept, linewidth=2)
    plt.xlabel(xlabel or x)
    plt.ylabel("Future PCI (PCI_NEXT)")
    plt.title(title or f"Future PCI vs {x}")
    _save_current_plot(filename)


def _iqr_outlier_summary(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Flag potential extreme values using the conventional 1.5*IQR rule.

    IQR flags are diagnostic only: an extreme value is not automatically an
    error. Road, traffic and climate data can legitimately be skewed.
    """
    rows = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flagged = (s < lower) | (s > upper)

        rows.append({
            "variable": col,
            "count_non_missing": int(s.size),
            "min": float(s.min()),
            "q1": float(q1),
            "median": float(s.median()),
            "q3": float(q3),
            "max": float(s.max()),
            "iqr_lower_bound": float(lower),
            "iqr_upper_bound": float(upper),
            "iqr_flag_count": int(flagged.sum()),
            "iqr_flag_percent": float(flagged.mean() * 100),
        })

    return pd.DataFrame(rows).sort_values("iqr_flag_percent", ascending=False)


def _domain_quality_checks(df: pd.DataFrame) -> dict[str, int]:
    """Simple impossible-value checks for variables with clear physical bounds."""
    checks: dict[str, int] = {}

    if "PCI_SCORE" in df:
        checks["PCI_SCORE outside 0-100"] = int((~df["PCI_SCORE"].between(0, 100) & df["PCI_SCORE"].notna()).sum())
    if config.TARGET_COL in df:
        checks["PCI_NEXT outside 0-100"] = int((~df[config.TARGET_COL].between(0, 100) & df[config.TARGET_COL].notna()).sum())
    if "PAVEMENT_AGE_YEARS" in df:
        checks["negative pavement age"] = int((df["PAVEMENT_AGE_YEARS"] < 0).sum())
    if "CUM_ESAL_AT_SURVEY" in df:
        checks["negative cumulative ESAL"] = int((df["CUM_ESAL_AT_SURVEY"] < 0).sum())
    if "ESAL_SINCE_LAST_SURVEY" in df:
        checks["negative ESAL since last survey"] = int((df["ESAL_SINCE_LAST_SURVEY"] < 0).sum())
    if "PRECIPITATION" in df:
        checks["negative precipitation"] = int((df["PRECIPITATION"] < 0).sum())
    for col in ["AC_THICKNESS_MM", "BASE_THICKNESS_MM", "SUBBASE_THICKNESS_MM", "TOTAL_PAV_THICKNESS_MM"]:
        if col in df:
            checks[f"negative {col}"] = int((df[col] < 0).sum())

    return checks


# ---------------------------------------------------------------------
# EDA sections
# ---------------------------------------------------------------------
def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Save headline dataset information and variable types."""
    summary_rows = [
        {"metric": "observations", "value": len(df)},
        {"metric": "physical_roads", "value": df[config.PHYSICAL_ROAD_ID_COL].nunique()},
        {"metric": "pavement_lives", "value": df[config.ROAD_ID_COL].nunique()},
        {"metric": "pavement_segments", "value": df[config.SEGMENT_ID_COL].nunique()},
        {"metric": "variables", "value": df.shape[1]},
        {"metric": "train_rows", "value": int((df[config.SPLIT_COL] == config.TRAIN_VALUE).sum())},
        {"metric": "test_rows", "value": int((df[config.SPLIT_COL] == config.TEST_VALUE).sum())},
        {"metric": "duplicate_full_rows", "value": int(df.duplicated().sum())},
        {"metric": "duplicate_road_date_rows", "value": int(df.duplicated([config.ROAD_ID_COL, config.SURVEY_DATE_COL]).sum())},
    ]
    overview = pd.DataFrame(summary_rows)
    overview.to_csv(OVERVIEW_PATH, index=False)

    variable_types = pd.DataFrame({
        "variable": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "non_missing": df.notna().sum().values,
        "missing": df.isna().sum().values,
        "unique_values": df.nunique(dropna=True).values,
    })
    variable_types.to_csv(os.path.join(config.RESULTS_DIR, "eda_variable_types.csv"), index=False)
    return overview


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Save descriptive statistics for numeric modelling variables and target."""
    cols = _available(df, config.NUMERIC_FEATURE_COLS + [config.TARGET_COL])
    stats = df[cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    stats.index.name = "variable"
    stats.to_csv(DESCRIPTIVE_PATH)
    return stats


def missing_value_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Quantify remaining missingness and create a missingness figure."""
    missing = pd.DataFrame({
        "variable": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_percent": (df.isna().mean() * 100).values,
    }).sort_values(["missing_percent", "variable"], ascending=[False, True])
    missing.to_csv(MISSING_PATH, index=False)

    plot_data = missing[missing["missing_count"] > 0].copy()
    if not plot_data.empty:
        plt.figure(figsize=(10, max(5, 0.32 * len(plot_data))))
        y = np.arange(len(plot_data))
        plt.barh(y, plot_data["missing_percent"])
        plt.yticks(y, plot_data["variable"])
        plt.xlabel("Missing values (%)")
        plt.ylabel("")
        plt.title("Remaining Missing Values in the Modelling Dataset")
        _save_current_plot("eda_missing_values.png")

    return missing


def extreme_value_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Check potential outliers plus impossible values."""
    numeric_cols = _available(df, config.NUMERIC_FEATURE_COLS + [config.TARGET_COL])
    outliers = _iqr_outlier_summary(df, numeric_cols)
    outliers.to_csv(OUTLIER_PATH, index=False)

    top = outliers.head(15).sort_values("iqr_flag_percent", ascending=True)
    if not top.empty:
        plt.figure(figsize=(10, 7))
        y = np.arange(len(top))
        plt.barh(y, top["iqr_flag_percent"])
        plt.yticks(y, top["variable"])
        plt.xlabel("Observations flagged by 1.5×IQR rule (%)")
        plt.ylabel("")
        plt.title("Variables with the Most Potential Extreme Values")
        _save_current_plot("eda_outlier_flags.png")

    domain_checks = _domain_quality_checks(df)
    pd.DataFrame(
        [{"check": k, "flag_count": v} for k, v in domain_checks.items()]
    ).to_csv(os.path.join(config.RESULTS_DIR, "eda_domain_quality_checks.csv"), index=False)

    return outliers, domain_checks


def rut_missingness_target_audit(df: pd.DataFrame) -> dict[str, float | int]:
    """Audit whether missing rut-depth measurements are associated with PCI artifacts.

    This is a data-quality diagnostic rather than a modelling step. Because
    the PCI calculator uses rut depth when available, systematic PCI
    differences between rows with and without rut measurements can indicate
    that missingness is changing the scale of the generated PCI target.

    The final modelling table drops each pavement life's terminal survey
    (because it has no PCI_NEXT target), so next-survey rut status can only be
    inferred where the next survey also remains as a feature row.
    """
    needed = {
        "RUT_DEPTH_1_8IN", "PCI_SCORE", config.TARGET_COL,
        config.SEGMENT_ID_COL, "SURVEY_SEQ_NO", "PAVEMENT_AGE_YEARS",
    }
    if not needed.issubset(df.columns):
        return {}

    work = df.copy()
    work["RUT_DEPTH_MISSING"] = work["RUT_DEPTH_1_8IN"].isna()

    current_summary = (
        work.groupby("RUT_DEPTH_MISSING", dropna=False)
        .agg(
            rows=("PCI_SCORE", "size"),
            mean_pci=("PCI_SCORE", "mean"),
            median_pci=("PCI_SCORE", "median"),
            mean_age=("PAVEMENT_AGE_YEARS", "mean"),
            median_age=("PAVEMENT_AGE_YEARS", "median"),
        )
        .reset_index()
    )
    current_summary["rut_status"] = np.where(
        current_summary["RUT_DEPTH_MISSING"], "missing", "recorded"
    )
    current_summary.to_csv(RUT_QUALITY_PATH, index=False)

    recorded = work.loc[~work["RUT_DEPTH_MISSING"], "PCI_SCORE"].dropna()
    missing = work.loc[work["RUT_DEPTH_MISSING"], "PCI_SCORE"].dropna()
    if not recorded.empty and not missing.empty:
        plt.figure(figsize=(7.5, 5.5))
        plt.boxplot(
            [recorded, missing],
            showfliers=False,
            **_boxplot_label_kwarg(["Rut depth recorded", "Rut depth missing"]),
        )
        plt.ylabel("Current PCI (PCI_SCORE)")
        plt.title("PCI Distribution by Rut-Depth Availability")
        _save_current_plot("eda_pci_by_rut_missingness.png")

    ordered = work.sort_values([config.SEGMENT_ID_COL, "SURVEY_SEQ_NO"]).copy()
    grp = ordered.groupby(config.SEGMENT_ID_COL, sort=False)
    ordered["_NEXT_SEQ"] = grp["SURVEY_SEQ_NO"].shift(-1)
    ordered["_NEXT_RUT_DEPTH"] = grp["RUT_DEPTH_1_8IN"].shift(-1)
    ordered["_NEXT_ROW_AVAILABLE"] = ordered["_NEXT_SEQ"].eq(ordered["SURVEY_SEQ_NO"] + 1)
    transitions = ordered.loc[ordered["_NEXT_ROW_AVAILABLE"]].copy()
    transitions["NEXT_RUT_DEPTH_MISSING"] = transitions["_NEXT_RUT_DEPTH"].isna()
    transitions["PCI_CHANGE_TO_NEXT"] = transitions[config.TARGET_COL] - transitions["PCI_SCORE"]
    transitions["PCI_INCREASE_GT20"] = transitions["PCI_CHANGE_TO_NEXT"] > 20

    transition_summary = (
        transitions.groupby("NEXT_RUT_DEPTH_MISSING", dropna=False)
        .agg(
            transitions=("PCI_CHANGE_TO_NEXT", "size"),
            mean_pci_change=("PCI_CHANGE_TO_NEXT", "mean"),
            median_pci_change=("PCI_CHANGE_TO_NEXT", "median"),
            increases_gt20=("PCI_INCREASE_GT20", "sum"),
        )
        .reset_index()
    )
    transition_summary["next_rut_status"] = np.where(
        transition_summary["NEXT_RUT_DEPTH_MISSING"], "missing", "recorded"
    )
    transition_summary.to_csv(RUT_TRANSITION_PATH, index=False)

    if not transitions.empty:
        rec_change = transitions.loc[
            ~transitions["NEXT_RUT_DEPTH_MISSING"], "PCI_CHANGE_TO_NEXT"
        ].dropna()
        miss_change = transitions.loc[
            transitions["NEXT_RUT_DEPTH_MISSING"], "PCI_CHANGE_TO_NEXT"
        ].dropna()
        if not rec_change.empty and not miss_change.empty:
            plt.figure(figsize=(7.5, 5.5))
            plt.boxplot(
                [rec_change, miss_change],
                showfliers=False,
                **_boxplot_label_kwarg(["Next rut recorded", "Next rut missing"]),
            )
            plt.axhline(0, linewidth=1)
            plt.ylabel("PCI_NEXT − PCI_SCORE (points)")
            plt.title("Apparent PCI Change by Next-Survey Rut Availability")
            _save_current_plot("eda_pci_change_by_next_rut_missingness.png")

    current_recorded = current_summary.loc[
        ~current_summary["RUT_DEPTH_MISSING"], "mean_pci"
    ]
    current_missing = current_summary.loc[
        current_summary["RUT_DEPTH_MISSING"], "mean_pci"
    ]
    age_recorded = current_summary.loc[
        ~current_summary["RUT_DEPTH_MISSING"], "mean_age"
    ]
    age_missing = current_summary.loc[
        current_summary["RUT_DEPTH_MISSING"], "mean_age"
    ]

    n_suspicious = int(transitions["PCI_INCREASE_GT20"].sum())
    n_suspicious_next_missing = int(
        (transitions["PCI_INCREASE_GT20"] & transitions["NEXT_RUT_DEPTH_MISSING"]).sum()
    )

    return {
        "rut_missing_rows": int(work["RUT_DEPTH_MISSING"].sum()),
        "rut_missing_percent": float(work["RUT_DEPTH_MISSING"].mean() * 100),
        "mean_pci_rut_recorded": float(current_recorded.iloc[0]) if not current_recorded.empty else np.nan,
        "mean_pci_rut_missing": float(current_missing.iloc[0]) if not current_missing.empty else np.nan,
        "mean_age_rut_recorded": float(age_recorded.iloc[0]) if not age_recorded.empty else np.nan,
        "mean_age_rut_missing": float(age_missing.iloc[0]) if not age_missing.empty else np.nan,
        "auditable_transitions": int(len(transitions)),
        "mean_change_next_rut_recorded": float(
            transitions.loc[~transitions["NEXT_RUT_DEPTH_MISSING"], "PCI_CHANGE_TO_NEXT"].mean()
        ),
        "mean_change_next_rut_missing": float(
            transitions.loc[transitions["NEXT_RUT_DEPTH_MISSING"], "PCI_CHANGE_TO_NEXT"].mean()
        ),
        "pci_increases_gt20": n_suspicious,
        "pci_increases_gt20_next_rut_missing": n_suspicious_next_missing,
        "pci_increases_gt20_next_rut_missing_percent": (
            float(n_suspicious_next_missing / n_suspicious * 100) if n_suspicious else np.nan
        ),
    }


def pci_distribution_plots(df: pd.DataFrame) -> None:
    """Plot current and future PCI distributions."""
    for col, filename, label in [
        ("PCI_SCORE", "eda_pci_distribution.png", "Current PCI (PCI_SCORE)"),
        (config.TARGET_COL, "eda_pci_next_distribution.png", "Future PCI (PCI_NEXT)"),
    ]:
        if col not in df:
            continue
        values = df[col].dropna()
        plt.figure(figsize=(8, 5.5))
        plt.hist(values, bins=20, alpha=0.8)
        plt.axvline(values.mean(), linestyle="--", linewidth=1.5, label=f"Mean = {values.mean():.1f}")
        plt.axvline(values.median(), linestyle=":", linewidth=1.5, label=f"Median = {values.median():.1f}")
        plt.xlabel(label)
        plt.ylabel("Observations")
        plt.title(f"Distribution of {label}")
        plt.legend()
        _save_current_plot(filename)


def relationship_plots(df: pd.DataFrame) -> None:
    """Required future-PCI relationship plots for the agreed feature groups."""
    plot_specs = [
        ("PAVEMENT_AGE_YEARS", "eda_pci_vs_pavement_age.png", "Pavement age (years)", "Future PCI vs Pavement Age"),
        ("CUM_ESAL_AT_SURVEY", "eda_pci_vs_cumulative_esal.png", "Cumulative ESAL at survey", "Future PCI vs Cumulative Traffic Loading"),
        ("ESAL_SINCE_LAST_SURVEY", "eda_pci_vs_recent_esal.png", "ESAL since last survey", "Future PCI vs Recent Traffic Loading"),
        ("TEMP_AVG", "eda_pci_vs_temperature.png", "Average temperature", "Future PCI vs Average Temperature"),
        ("PRECIPITATION", "eda_pci_vs_precipitation.png", "Precipitation", "Future PCI vs Precipitation"),
        ("AC_THICKNESS_MM", "eda_pci_vs_ac_thickness.png", "AC thickness (mm)", "Future PCI vs Asphalt Thickness"),
        ("TOTAL_PAV_THICKNESS_MM", "eda_pci_vs_total_thickness.png", "Total pavement thickness (mm)", "Future PCI vs Total Pavement Thickness"),
        ("PCI_SCORE", "eda_pci_current_vs_future.png", "Current PCI (PCI_SCORE)", "Future PCI vs Current PCI"),
    ]
    for x, filename, xlabel, title in plot_specs:
        if x in df.columns and config.TARGET_COL in df.columns:
            _scatter_with_trend(df, x, config.TARGET_COL, filename, xlabel, title)


def correlation_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute numeric correlations and rank variables against PCI_NEXT.

    Pearson is used for the matrix because it is the conventional linear
    correlation summary. Spearman is included in the ranking to also reveal
    monotonic relationships that are not perfectly linear.
    """
    # Only genuine model inputs + target. Deliberately excludes identifiers,
    # dates, SPLIT and target-adjacent future-survey timing columns to avoid
    # presenting leakage variables as useful predictors.
    numeric_cols = _available(df, config.NUMERIC_FEATURE_COLS + [config.TARGET_COL])
    numeric = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    pearson = numeric.corr(method="pearson")
    pearson.to_csv(CORRELATION_PATH)

    spearman = numeric.corr(method="spearman")
    target_rank = pd.DataFrame({
        "variable": [c for c in numeric_cols if c != config.TARGET_COL],
        "pearson_r": [pearson.loc[c, config.TARGET_COL] for c in numeric_cols if c != config.TARGET_COL],
        "spearman_rho": [spearman.loc[c, config.TARGET_COL] for c in numeric_cols if c != config.TARGET_COL],
    })
    target_rank["abs_pearson_r"] = target_rank["pearson_r"].abs()
    target_rank["abs_spearman_rho"] = target_rank["spearman_rho"].abs()
    target_rank = target_rank.sort_values("abs_pearson_r", ascending=False).reset_index(drop=True)
    target_rank.to_csv(TARGET_CORRELATION_PATH, index=False)

    # A focused matrix is easier to read than plotting every numeric column.
    # Include the target plus the 14 strongest Pearson relationships.
    strongest = target_rank.head(14)["variable"].tolist()
    heatmap_cols = [config.TARGET_COL] + strongest
    heatmap_corr = numeric[heatmap_cols].corr(method="pearson")

    plt.figure(figsize=(12, 10))
    image = plt.imshow(heatmap_corr.values, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(image, label="Pearson correlation")
    plt.xticks(np.arange(len(heatmap_cols)), heatmap_cols, rotation=90)
    plt.yticks(np.arange(len(heatmap_cols)), heatmap_cols)
    for i in range(len(heatmap_cols)):
        for j in range(len(heatmap_cols)):
            plt.text(j, i, f"{heatmap_corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.title("Correlation Matrix — Future PCI and Strongest Numeric Relationships")
    _save_current_plot("eda_correlation_matrix.png")

    # Direct ranking plot makes variable selection evidence report-ready.
    top_rank = target_rank.head(12).sort_values("pearson_r")
    plt.figure(figsize=(9, 6.5))
    y = np.arange(len(top_rank))
    plt.barh(y, top_rank["pearson_r"])
    plt.yticks(y, top_rank["variable"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Pearson correlation with PCI_NEXT")
    plt.ylabel("")
    plt.title("Numeric Variables Most Related to Future PCI")
    _save_current_plot("eda_pci_next_correlations.png")

    return pearson, target_rank


def write_summary(df: pd.DataFrame, missing: pd.DataFrame,
                  outliers: pd.DataFrame, domain_checks: dict[str, int],
                  target_rank: pd.DataFrame,
                  rut_audit: dict[str, float | int]) -> str:
    """Write a concise, data-driven findings summary for the project report."""
    current = df["PCI_SCORE"].dropna()
    future = df[config.TARGET_COL].dropna()
    missing_nonzero = missing[missing["missing_count"] > 0]
    top_missing = missing_nonzero.head(8)
    strongest = target_rank.head(8)

    current_future_r = df[["PCI_SCORE", config.TARGET_COL]].corr().iloc[0, 1]
    future_change = future.mean() - current.mean()
    impossible_total = sum(domain_checks.values())

    lines = [
        "# EDA & Data-Quality Summary",
        "",
        "## Dataset overview",
        f"- The final modelling dataset contains **{len(df):,} observations**, "
        f"**{df[config.PHYSICAL_ROAD_ID_COL].nunique():,} physical roads**, "
        f"and **{df[config.ROAD_ID_COL].nunique():,} pavement lives**.",
        f"- The provided split contains **{(df[config.SPLIT_COL] == config.TRAIN_VALUE).sum():,} training rows** "
        f"and **{(df[config.SPLIT_COL] == config.TEST_VALUE).sum():,} test rows**.",
        f"- There are **{df.shape[1]} variables** in the final dataset and **{df.duplicated().sum()} exact duplicate rows**.",
        "",
        "## PCI distribution",
        f"- Current PCI (`PCI_SCORE`) has mean **{current.mean():.2f}**, median **{current.median():.2f}**, "
        f"and range **{current.min():.2f}–{current.max():.2f}**.",
        f"- Future PCI (`PCI_NEXT`) has mean **{future.mean():.2f}**, median **{future.median():.2f}**, "
        f"and range **{future.min():.2f}–{future.max():.2f}**.",
        f"- Mean future PCI is **{abs(future_change):.2f} points {'lower' if future_change < 0 else 'higher'}** than mean current PCI, "
        "which is consistent with condition changing between surveys rather than remaining constant.",
        "",
        "## Missing values and data quality",
        f"- `PCI_NEXT` has **{df[config.TARGET_COL].isna().sum()} missing values**, so every modelling row has a usable target.",
    ]

    if top_missing.empty:
        lines.append("- No missing values remain in the dataset.")
    else:
        lines.append("- Missingness is concentrated in a small set of variables. The largest gaps are:")
        for _, row in top_missing.iterrows():
            lines.append(f"  - `{row['variable']}`: **{int(row['missing_count']):,} ({row['missing_percent']:.1f}%)**")
        lines.append("- Some missing values are structurally expected: lagged fields are unavailable for the first survey in a pavement life, and patch timing is unavailable where no prior patch is recorded.")

    lines.extend([
        f"- Simple impossible-value checks found **{impossible_total}** values violating clear bounds "
        "(PCI outside 0–100, negative age/traffic/precipitation/thickness).",
        "- The IQR analysis flags statistical extremes for review, but these are not automatically treated as errors because traffic, climate and pavement-structure variables can legitimately be skewed.",
    ])

    if rut_audit:
        pci_gap = rut_audit["mean_pci_rut_missing"] - rut_audit["mean_pci_rut_recorded"]
        lines.extend([
            "",
            "## Important PCI target-quality warning",
            f"- `RUT_DEPTH_1_8IN` is missing in **{rut_audit['rut_missing_rows']:,} of {len(df):,} rows "
            f"({rut_audit['rut_missing_percent']:.1f}%)**.",
            f"- Rows with a recorded rut depth have mean current PCI **{rut_audit['mean_pci_rut_recorded']:.1f}**, "
            f"while rows with missing rut depth have mean current PCI **{rut_audit['mean_pci_rut_missing']:.1f}** "
            f"(a **{pci_gap:.1f}-point gap**), despite similar mean pavement ages "
            f"(**{rut_audit['mean_age_rut_recorded']:.1f}** vs **{rut_audit['mean_age_rut_missing']:.1f}** years).",
            f"- For the **{rut_audit['auditable_transitions']:,}** transitions where the next survey row is still present "
            "in the final modelling table, mean apparent PCI change is "
            f"**{rut_audit['mean_change_next_rut_recorded']:.1f} points** when the next rut measurement is recorded "
            f"versus **{rut_audit['mean_change_next_rut_missing']:+.1f} points** when it is missing.",
            f"- Of **{rut_audit['pci_increases_gt20']:,}** apparent PCI improvements greater than 20 points in those auditable "
            f"transitions, **{rut_audit['pci_increases_gt20_next_rut_missing']:,} "
            f"({rut_audit['pci_increases_gt20_next_rut_missing_percent']:.1f}%)** occur when the next survey's rut depth is missing.",
            "- This is a serious data-quality warning: the generated PCI appears systematically associated with rut-depth missingness. "
            "The team should resolve the upstream missing-rut handling and regenerate the modelling dataset before finalising model-accuracy claims.",
        ])

    lines.extend([
        "",
        "## Relationships with future PCI",
        f"- Current PCI is the strongest direct numeric relationship with future PCI: **Pearson r = {current_future_r:.3f}**.",
        "- The strongest numeric relationships with `PCI_NEXT` in this dataset are:",
    ])
    for _, row in strongest.iterrows():
        lines.append(
            f"  - `{row['variable']}`: Pearson **{row['pearson_r']:.3f}**, "
            f"Spearman **{row['spearman_rho']:.3f}**"
        )

    age_row = target_rank[target_rank["variable"] == "PAVEMENT_AGE_YEARS"]
    if not age_row.empty:
        age_r = age_row.iloc[0]["pearson_r"]
        lines.append(
            f"- Pavement age alone has a relatively {'weak' if abs(age_r) < 0.3 else 'moderate/strong'} linear relationship "
            f"with future PCI (**r = {age_r:.3f}**), so age should not be treated as a sufficient predictor by itself."
        )

    lines.extend([
        "- Correlation measures association, not causation. Weak Pearson correlation does not mean a feature is useless: tree-based models can exploit nonlinear relationships and interactions between traffic, climate, structure and current condition.",
        "",
        "## Modelling implication",
        "The EDA supports retaining the agreed multi-factor feature set rather than relying on pavement age alone. Current condition variables provide the clearest direct signal, while traffic, climate, pavement structure, maintenance and lagged-history variables may add nonlinear or interaction effects that are not fully captured by pairwise correlation.",
        "",
        "Generated automatically by `src/eda.py` from `data/model_dataset.csv`.",
    ])

    text = "\n".join(lines) + "\n"
    Path(config.EDA_SUMMARY_PATH).write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------
# Public pipeline entry point
# ---------------------------------------------------------------------
def run() -> None:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    print(f"[eda] Loading and validating: {config.MODEL_DATASET_PATH}")
    df = schema_checks.load_and_validate()

    overview = dataset_overview(df)
    descriptive_statistics(df)
    missing = missing_value_analysis(df)
    outliers, domain_checks = extreme_value_analysis(df)
    rut_audit = rut_missingness_target_audit(df)
    pci_distribution_plots(df)
    relationship_plots(df)
    _, target_rank = correlation_analysis(df)
    summary = write_summary(df, missing, outliers, domain_checks, target_rank, rut_audit)

    print("[eda] Dataset overview:")
    print(overview.to_string(index=False))
    print("\n[eda] Strongest numeric relationships with PCI_NEXT:")
    print(target_rank.head(10)[["variable", "pearson_r", "spearman_rho"]].to_string(index=False))
    print(f"\n[eda] Summary written to: {config.EDA_SUMMARY_PATH}")
    print(f"[eda] Tables written to:  {config.RESULTS_DIR}")
    print(f"[eda] Plots written to:   {config.PLOTS_DIR}")
    print("\n" + summary)


if __name__ == "__main__":
    run()
