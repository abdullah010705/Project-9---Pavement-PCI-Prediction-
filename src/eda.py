"""
EDA v2 — PCI + rutting exploratory data analysis and data-quality analysis
Owner: Aum

Purpose
-------
Characterise the two client-approved modelling datasets before model training:
  * distress-only PCI dataset (target: PCI_NEXT)
  * rutting dataset (target: RUT_NEXT)

The script produces:
  * dataset size / road counts / dtypes / descriptive statistics
  * missing-value and unusual/extreme-value checks
  * PCI and rut-depth distributions
  * target relationships with pavement age, ESAL, temperature,
    precipitation and pavement thickness
  * separate correlation matrices and target-correlation rankings
  * direct PCI-vs-rutting comparison on matched road-survey pairs
  * an automatically written report-ready EDA summary

Run from the project root:
    python -m src.eda

or directly:
    python src/eda.py

Optional custom paths:
    python src/eda.py --pci data/model_dataset_pci.csv \
                      --rut data/model_dataset_rut.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import spearmanr
except Exception:  # scipy is optional; pandas rank correlation is used as fallback
    spearmanr = None


# -----------------------------------------------------------------------------
# Project paths and schema
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent if THIS_FILE.parent.name == "src" else THIS_FILE.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "plots"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

PCI_TARGET = "PCI_NEXT"
RUT_TARGET = "RUT_NEXT"
PCI_CURRENT = "PCI_SCORE"
RUT_CURRENT = "RUT_DEPTH_1_8IN"
ROAD_COL = "PHYSICAL_ROAD_ID"
LIFE_COL = "PAVEMENT_SEGMENT_ID"
DATE_COL = "SURVEY_DATE"
SPLIT_COL = "SPLIT"

# Core predictors requested in the allocation plus useful lag/current-condition
# variables already engineered by the dataset-construction stage.
CANDIDATE_NUMERIC_FEATURES = [
    "PAVEMENT_AGE_YEARS",
    "CUM_ESAL_AT_SURVEY", "ESAL_SINCE_LAST_SURVEY",
    "PRECIPITATION", "EVAPORATION", "PRECIP_DAYS", "TEMP_AVG",
    "FREEZE_INDEX", "FREEZE_THAW", "SHORTWAVE_SURFACE_AVG",
    "AC_THICKNESS_MM", "BASE_THICKNESS_MM", "SUBBASE_THICKNESS_MM",
    "TOTAL_PAV_THICKNESS_MM", "NUM_AC_LAYERS", "NUM_LAYERS_TOTAL",
    "PRIOR_PATCH_COUNT", "DAYS_SINCE_LAST_PATCH",
    "PCI_SCORE", "RUT_DEPTH_1_8IN",
    "IRI_LEFT_WHEEL_PATH", "IRI_RIGHT_WHEEL_PATH", "MRI",
    "N_PRIOR_SURVEYS", "PCI_PREV", "PCI_CHANGE_SINCE_LAST",
    "RUT_PREV", "RUT_CHANGE_SINCE_LAST",
    "DAYS_SINCE_LAST_SURVEY", "YEARS_SINCE_LAST_SURVEY",
]

RELATIONSHIP_FEATURES = [
    ("PAVEMENT_AGE_YEARS", "Pavement age (years)", "pavement_age"),
    ("CUM_ESAL_AT_SURVEY", "Cumulative ESAL at survey", "cumulative_esal"),
    ("ESAL_SINCE_LAST_SURVEY", "ESAL since last survey", "recent_esal"),
    ("TEMP_AVG", "Average temperature", "temperature"),
    ("PRECIPITATION", "Precipitation", "precipitation"),
    ("AC_THICKNESS_MM", "AC thickness (mm)", "ac_thickness"),
    ("TOTAL_PAV_THICKNESS_MM", "Total pavement thickness (mm)", "total_thickness"),
]

EXPECTED_STRUCTURAL_MISSING = {
    "PCI_PREV", "PCI_CHANGE_SINCE_LAST", "RUT_PREV", "RUT_CHANGE_SINCE_LAST",
    "DAYS_SINCE_LAST_SURVEY", "YEARS_SINCE_LAST_SURVEY", "ESAL_SINCE_LAST_SURVEY",
}


def ensure_dirs(plots_dir: Path, results_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)


def resolve_input(explicit: str | None, filename: str) -> Path:
    """Resolve a dataset whether it sits in /data or the project root."""
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {p}")
        return p

    candidates = [DEFAULT_DATA_DIR / filename, PROJECT_ROOT / filename, Path.cwd() / filename]
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        f"Could not find {filename}. Looked in: " + ", ".join(str(p) for p in candidates)
    )


def validate_dataset(df: pd.DataFrame, name: str, target: str, current: str) -> None:
    required = {ROAD_COL, LIFE_COL, DATE_COL, target, current}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} dataset is missing required columns: {missing}")
    if df[target].isna().any():
        raise ValueError(f"{name} dataset contains missing target values in {target}")


def save_plot(fig: plt.Figure, plots_dir: Path, filename: str) -> Path:
    path = plots_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def numeric_features(df: pd.DataFrame, target: str) -> list[str]:
    cols = [c for c in CANDIDATE_NUMERIC_FEATURES if c in df.columns]
    if target in df.columns:
        cols.append(target)
    # Keep only columns pandas genuinely sees as numeric or can mostly coerce.
    out = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            out.append(c)
    return list(dict.fromkeys(out))


# -----------------------------------------------------------------------------
# Basic information, missingness and extremes
# -----------------------------------------------------------------------------
def dataset_overview(df: pd.DataFrame, name: str) -> pd.DataFrame:
    values = {
        "dataset": name,
        "observations": len(df),
        "physical_roads": df[ROAD_COL].nunique(),
        "pavement_lives": df[LIFE_COL].nunique(),
        "variables": df.shape[1],
        "duplicate_full_rows": int(df.duplicated().sum()),
        "duplicate_road_date_rows": int(df.duplicated([ROAD_COL, DATE_COL]).sum()),
    }
    if SPLIT_COL in df.columns:
        values["train_rows"] = int((df[SPLIT_COL] == "train").sum())
        values["test_rows"] = int((df[SPLIT_COL] == "test").sum())
    return pd.DataFrame([values])


def variable_types_table(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    return pd.DataFrame({
        "dataset": dataset_name,
        "variable": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "non_missing": df.notna().sum().values,
        "missing": df.isna().sum().values,
        "missing_percent": (df.isna().mean() * 100).values,
        "unique_values": df.nunique(dropna=True).values,
    })


def descriptive_table(df: pd.DataFrame, dataset_name: str, target: str) -> pd.DataFrame:
    cols = numeric_features(df, target)
    stats = df[cols].apply(pd.to_numeric, errors="coerce").describe(
        percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    ).T.reset_index(names="variable")
    stats.insert(0, "dataset", dataset_name)
    return stats


def missing_table(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    out = pd.DataFrame({
        "dataset": dataset_name,
        "variable": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_percent": (df.isna().mean() * 100).values,
    })
    out["interpretation"] = np.where(
        out["variable"].isin(EXPECTED_STRUCTURAL_MISSING),
        "partly expected for first/interval survey features",
        np.where(out["missing_count"] > 0, "review / source-data missingness", "complete"),
    )
    return out.sort_values(["missing_percent", "variable"], ascending=[False, True])


def iqr_outliers(df: pd.DataFrame, dataset_name: str, target: str) -> pd.DataFrame:
    rows = []
    for col in numeric_features(df, target):
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flagged = (s < lo) | (s > hi)
        rows.append({
            "dataset": dataset_name,
            "variable": col,
            "non_missing": len(s),
            "min": s.min(), "q1": q1, "median": s.median(), "q3": q3, "max": s.max(),
            "iqr_lower_bound": lo, "iqr_upper_bound": hi,
            "iqr_flag_count": int(flagged.sum()),
            "iqr_flag_percent": float(flagged.mean() * 100),
        })
    return pd.DataFrame(rows).sort_values("iqr_flag_percent", ascending=False)


def domain_checks(df: pd.DataFrame, dataset_name: str, target: str, current: str) -> pd.DataFrame:
    checks: list[dict] = []

    def add(label: str, mask: pd.Series) -> None:
        checks.append({"dataset": dataset_name, "check": label, "flag_count": int(mask.fillna(False).sum())})

    if current == PCI_CURRENT:
        add("PCI_SCORE outside 0-100", df[PCI_CURRENT].notna() & ~pd.to_numeric(df[PCI_CURRENT], errors="coerce").between(0, 100))
        add("PCI_NEXT outside 0-100", df[target].notna() & ~pd.to_numeric(df[target], errors="coerce").between(0, 100))
    if current == RUT_CURRENT:
        add("negative current rut depth", pd.to_numeric(df[current], errors="coerce") < 0)
        add("negative future rut depth", pd.to_numeric(df[target], errors="coerce") < 0)

    for col, label in [
        ("PAVEMENT_AGE_YEARS", "negative pavement age"),
        ("CUM_ESAL_AT_SURVEY", "negative cumulative ESAL"),
        ("ESAL_SINCE_LAST_SURVEY", "negative ESAL since last survey"),
        ("PRECIPITATION", "negative precipitation"),
        ("AC_THICKNESS_MM", "negative AC thickness"),
        ("BASE_THICKNESS_MM", "negative base thickness"),
        ("SUBBASE_THICKNESS_MM", "negative subbase thickness"),
        ("TOTAL_PAV_THICKNESS_MM", "negative total pavement thickness"),
    ]:
        if col in df.columns:
            add(label, pd.to_numeric(df[col], errors="coerce") < 0)
    return pd.DataFrame(checks)


# -----------------------------------------------------------------------------
# Plots and correlations
# -----------------------------------------------------------------------------
def distribution_plot(df: pd.DataFrame, current: str, target: str, label: str,
                      prefix: str, plots_dir: Path) -> None:
    for col, suffix, title in [
        (current, "current_distribution", f"Distribution of current {label}"),
        (target, "future_distribution", f"Distribution of future {label}"),
    ]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.hist(s, bins=30, edgecolor="black", linewidth=0.5)
        ax.axvline(s.mean(), linestyle="--", linewidth=1.5, label=f"Mean = {s.mean():.2f}")
        ax.axvline(s.median(), linestyle=":", linewidth=1.5, label=f"Median = {s.median():.2f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend()
        save_plot(fig, plots_dir, f"{prefix}_{suffix}.png")


def scatter_with_trend(df: pd.DataFrame, x: str, y: str, y_label: str,
                       x_label: str, filename: str, plots_dir: Path) -> None:
    data = df[[x, y]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(data[x], data[y], alpha=0.38, s=22)
    if data[x].nunique() > 1:
        slope, intercept = np.polyfit(data[x], data[y], 1)
        xx = np.linspace(data[x].min(), data[x].max(), 200)
        ax.plot(xx, slope * xx + intercept, linewidth=2)
    r = data[x].corr(data[y])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} vs {x_label} (Pearson r = {r:.3f})")
    save_plot(fig, plots_dir, filename)


def relationship_plots(df: pd.DataFrame, target: str, target_label: str,
                       prefix: str, plots_dir: Path) -> None:
    for col, x_label, slug in RELATIONSHIP_FEATURES:
        if col in df.columns:
            scatter_with_trend(
                df, col, target, target_label, x_label,
                f"{prefix}_vs_{slug}.png", plots_dir
            )


def correlation_outputs(df: pd.DataFrame, dataset_name: str, target: str,
                        prefix: str, plots_dir: Path, results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = numeric_features(df, target)
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(method="pearson")
    corr.to_csv(results_dir / f"{prefix}_correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title(f"{dataset_name} numeric correlation matrix")
    save_plot(fig, plots_dir, f"{prefix}_correlation_matrix.png")

    rows = []
    for col in cols:
        if col == target:
            continue
        pair = numeric[[col, target]].dropna()
        if len(pair) < 3 or pair[col].nunique() < 2:
            continue
        pearson = pair[col].corr(pair[target])
        if spearmanr is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sp = spearmanr(pair[col], pair[target], nan_policy="omit").statistic
        else:
            sp = pair[col].corr(pair[target], method="spearman")
        rows.append({
            "dataset": dataset_name,
            "variable": col,
            "n_pairwise": len(pair),
            "pearson_r": pearson,
            "spearman_rho": sp,
            "abs_pearson_r": abs(pearson) if pd.notna(pearson) else np.nan,
            "abs_spearman_rho": abs(sp) if pd.notna(sp) else np.nan,
        })
    ranking = pd.DataFrame(rows).sort_values("abs_pearson_r", ascending=False)
    ranking.to_csv(results_dir / f"{prefix}_target_correlations.csv", index=False)

    top = ranking.head(15).sort_values("pearson_r")
    if not top.empty:
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(top["variable"], top["pearson_r"])
        ax.axvline(0, linewidth=0.8)
        ax.set_xlabel(f"Pearson correlation with {target}")
        ax.set_title(f"Variables most related to {target}")
        save_plot(fig, plots_dir, f"{prefix}_target_correlations.png")
    return corr, ranking


def missing_plot(missing: pd.DataFrame, prefix: str, dataset_name: str, plots_dir: Path) -> None:
    p = missing[missing["missing_count"] > 0].head(20).sort_values("missing_percent")
    if p.empty:
        return
    fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(p))))
    ax.barh(p["variable"], p["missing_percent"])
    ax.set_xlabel("Missing values (%)")
    ax.set_title(f"{dataset_name}: variables with missing values")
    save_plot(fig, plots_dir, f"{prefix}_missing_values.png")


def outlier_plot(outliers: pd.DataFrame, prefix: str, dataset_name: str, plots_dir: Path) -> None:
    p = outliers.head(15).sort_values("iqr_flag_percent")
    if p.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(p["variable"], p["iqr_flag_percent"])
    ax.set_xlabel("Rows flagged by 1.5×IQR rule (%)")
    ax.set_title(f"{dataset_name}: potential extreme-value flags")
    save_plot(fig, plots_dir, f"{prefix}_outlier_flags.png")


# -----------------------------------------------------------------------------
# Direct PCI-vs-rutting comparison
# -----------------------------------------------------------------------------
def compare_pci_rut(pci: pd.DataFrame, rut: pd.DataFrame,
                    pci_rank: pd.DataFrame, rut_rank: pd.DataFrame,
                    plots_dir: Path, results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    keys = [ROAD_COL, DATE_COL]
    pci_cols = keys + [PCI_CURRENT, PCI_TARGET, "YEARS_TO_NEXT_SURVEY"]
    rut_cols = keys + [RUT_CURRENT, RUT_TARGET, "YEARS_TO_NEXT_SURVEY"]
    pci_m = pci[[c for c in pci_cols if c in pci.columns]].copy()
    rut_m = rut[[c for c in rut_cols if c in rut.columns]].copy()
    merged = pci_m.merge(rut_m, on=keys, how="inner", suffixes=("_PCI", "_RUT"))

    for c in [PCI_CURRENT, PCI_TARGET, RUT_CURRENT, RUT_TARGET]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    merged["PCI_CHANGE_TO_NEXT"] = merged[PCI_TARGET] - merged[PCI_CURRENT]
    merged["RUT_CHANGE_TO_NEXT"] = merged[RUT_TARGET] - merged[RUT_CURRENT]
    merged["PCI_DETERIORATED"] = merged["PCI_CHANGE_TO_NEXT"] < 0
    merged["RUT_DETERIORATED"] = merged["RUT_CHANGE_TO_NEXT"] > 0

    # Use PCI horizon if present; the paired target is the same next pavement-life survey
    horizon_col = "YEARS_TO_NEXT_SURVEY_PCI" if "YEARS_TO_NEXT_SURVEY_PCI" in merged.columns else "YEARS_TO_NEXT_SURVEY"
    horizon = pd.to_numeric(merged[horizon_col], errors="coerce") if horizon_col in merged.columns else pd.Series(np.nan, index=merged.index)
    good_horizon = horizon > 0
    merged["PCI_DETERIORATION_PER_YEAR"] = np.where(good_horizon, -merged["PCI_CHANGE_TO_NEXT"] / horizon, np.nan)
    merged["RUT_DETERIORATION_PER_YEAR"] = np.where(good_horizon, merged["RUT_CHANGE_TO_NEXT"] / horizon, np.nan)

    merged.to_csv(results_dir / "eda_pci_rut_matched_pairs.csv", index=False)

    current_corr = merged[[PCI_CURRENT, RUT_CURRENT]].dropna().corr().iloc[0, 1] if len(merged) else np.nan
    future_corr = merged[[PCI_TARGET, RUT_TARGET]].dropna().corr().iloc[0, 1] if len(merged) else np.nan
    change_corr = merged[["PCI_CHANGE_TO_NEXT", "RUT_CHANGE_TO_NEXT"]].dropna().corr().iloc[0, 1] if len(merged) else np.nan
    rate_corr = merged[["PCI_DETERIORATION_PER_YEAR", "RUT_DETERIORATION_PER_YEAR"]].dropna().corr().iloc[0, 1] if len(merged) else np.nan

    both = int((merged["PCI_DETERIORATED"] & merged["RUT_DETERIORATED"]).sum())
    pci_only = int((merged["PCI_DETERIORATED"] & ~merged["RUT_DETERIORATED"]).sum())
    rut_only = int((~merged["PCI_DETERIORATED"] & merged["RUT_DETERIORATED"]).sum())
    neither = int((~merged["PCI_DETERIORATED"] & ~merged["RUT_DETERIORATED"]).sum())

    comp = pd.DataFrame([
        {"metric": "matched road-survey current-to-next pairs", "value": len(merged)},
        {"metric": "current PCI vs current rut Pearson r", "value": current_corr},
        {"metric": "future PCI vs future rut Pearson r", "value": future_corr},
        {"metric": "raw change correlation (PCI change vs rut change)", "value": change_corr},
        {"metric": "deterioration-rate correlation per year", "value": rate_corr},
        {"metric": "both measures deteriorated", "value": both},
        {"metric": "PCI deteriorated only", "value": pci_only},
        {"metric": "rut deteriorated only", "value": rut_only},
        {"metric": "neither deteriorated", "value": neither},
    ])
    comp.to_csv(results_dir / "eda_pci_rut_comparison_summary.csv", index=False)

    if len(merged):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.scatter(merged[RUT_CURRENT], merged[PCI_CURRENT], alpha=0.4, s=22)
        ax.set_xlabel("Current rut depth (1/8 inch)")
        ax.set_ylabel("Current PCI")
        ax.set_title(f"Current PCI vs rut depth (r = {current_corr:.3f})")
        save_plot(fig, plots_dir, "eda_pci_vs_rut_current.png")

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.scatter(merged["RUT_DETERIORATION_PER_YEAR"], merged["PCI_DETERIORATION_PER_YEAR"], alpha=0.4, s=22)
        ax.axhline(0, linewidth=0.8)
        ax.axvline(0, linewidth=0.8)
        ax.set_xlabel("Rutting deterioration per year (increase in 1/8 inch/year)")
        ax.set_ylabel("PCI deterioration per year (PCI points/year)")
        ax.set_title(f"PCI vs rutting deterioration rate (r = {rate_corr:.3f})")
        save_plot(fig, plots_dir, "eda_pci_vs_rut_deterioration_rate.png")

    # Compare shared factor correlations side by side. Exclude the other condition measure
    # and lag measures that are target-specific so the comparison focuses on common drivers.
    p = pci_rank[["variable", "pearson_r", "abs_pearson_r"]].rename(
        columns={"pearson_r": "pci_target_r", "abs_pearson_r": "pci_abs_r"}
    )
    r = rut_rank[["variable", "pearson_r", "abs_pearson_r"]].rename(
        columns={"pearson_r": "rut_target_r", "abs_pearson_r": "rut_abs_r"}
    )
    factors = p.merge(r, on="variable", how="inner")
    excluded = {PCI_CURRENT, RUT_CURRENT, "PCI_PREV", "PCI_CHANGE_SINCE_LAST", "RUT_PREV", "RUT_CHANGE_SINCE_LAST"}
    factors = factors[~factors["variable"].isin(excluded)].copy()
    factors["same_sign"] = np.sign(factors["pci_target_r"]) == np.sign(factors["rut_target_r"])
    factors["mean_abs_r"] = (factors["pci_abs_r"] + factors["rut_abs_r"]) / 2
    factors = factors.sort_values("mean_abs_r", ascending=False)
    factors.to_csv(results_dir / "eda_shared_factor_comparison.csv", index=False)

    top_n = 10
    pci_top = set(factors.nlargest(top_n, "pci_abs_r")["variable"])
    rut_top = set(factors.nlargest(top_n, "rut_abs_r")["variable"])
    overlap = pci_top & rut_top
    metrics = {
        "matched_pairs": len(merged),
        "current_corr": current_corr,
        "future_corr": future_corr,
        "change_corr": change_corr,
        "rate_corr": rate_corr,
        "both_deteriorated": both,
        "pci_only": pci_only,
        "rut_only": rut_only,
        "neither": neither,
        "top10_shared_factor_overlap_count": len(overlap),
        "top10_shared_factor_overlap": sorted(overlap),
    }
    return merged, factors, metrics


# -----------------------------------------------------------------------------
# Summary writer
# -----------------------------------------------------------------------------
def top_corr_lines(rank: pd.DataFrame, n: int = 8) -> list[str]:
    lines = []
    for _, row in rank.head(n).iterrows():
        lines.append(
            f"- `{row['variable']}`: Pearson r = {row['pearson_r']:.3f}; "
            f"Spearman rho = {row['spearman_rho']:.3f} (n={int(row['n_pairwise'])})."
        )
    return lines


def write_summary(pci: pd.DataFrame, rut: pd.DataFrame,
                  pci_missing: pd.DataFrame, rut_missing: pd.DataFrame,
                  pci_domain: pd.DataFrame, rut_domain: pd.DataFrame,
                  pci_rank: pd.DataFrame, rut_rank: pd.DataFrame,
                  comp_metrics: dict, factors: pd.DataFrame,
                  results_dir: Path) -> str:
    pci_p = pd.to_numeric(pci[PCI_CURRENT], errors="coerce")
    pci_n = pd.to_numeric(pci[PCI_TARGET], errors="coerce")
    rut_p = pd.to_numeric(rut[RUT_CURRENT], errors="coerce")
    rut_n = pd.to_numeric(rut[RUT_TARGET], errors="coerce")

    pci_missing_nonzero = pci_missing[pci_missing["missing_count"] > 0]
    rut_missing_nonzero = rut_missing[rut_missing["missing_count"] > 0]
    impossible_total = int(pci_domain["flag_count"].sum() + rut_domain["flag_count"].sum())

    top_shared = factors.head(8)
    factor_lines = [
        f"- `{r.variable}`: r with PCI_NEXT = {r.pci_target_r:.3f}; r with RUT_NEXT = {r.rut_target_r:.3f}."
        for r in top_shared.itertuples()
    ]

    total_matched = max(comp_metrics.get("matched_pairs", 0), 1)
    both_pct = comp_metrics.get("both_deteriorated", 0) / total_matched * 100
    pci_only_pct = comp_metrics.get("pci_only", 0) / total_matched * 100
    rut_only_pct = comp_metrics.get("rut_only", 0) / total_matched * 100
    neither_pct = comp_metrics.get("neither", 0) / total_matched * 100

    text = f"""# EDA v2 — PCI and Rutting Data-Quality Summary

## 1. Dataset overview

The client-requested v2 rebuild now treats distress-only PCI and rut depth as separate condition measures.

- **PCI dataset:** {len(pci):,} current-to-next observations, {pci[ROAD_COL].nunique():,} physical roads, {pci[LIFE_COL].nunique():,} pavement lives, {pci.shape[1]} variables.
- **Rutting dataset:** {len(rut):,} current-to-next observations, {rut[ROAD_COL].nunique():,} physical roads, {rut[LIFE_COL].nunique():,} pavement lives, {rut.shape[1]} variables.
- Both modelling targets are complete: missing `{PCI_TARGET}` = {pci[PCI_TARGET].isna().sum()}, missing `{RUT_TARGET}` = {rut[RUT_TARGET].isna().sum()}.

## 2. Distribution check

The distress-only PCI distribution has one clear dominant high-condition peak rather than the separated two-group pattern seen when rutting was incorrectly merged into PCI. In the v2 PCI modelling table, current PCI has mean **{pci_p.mean():.2f}**, median **{pci_p.median():.2f}**, and range **{pci_p.min():.1f}–{pci_p.max():.1f}**. Future PCI has mean **{pci_n.mean():.2f}** and median **{pci_n.median():.2f}**. This supports the Review 1 conclusion that the old bimodality was a calculation artefact linked to rutting availability, not a true two-population PCI pattern.

Rutting is now analysed on its own scale. Current rut depth has mean **{rut_p.mean():.2f} eighths of an inch**, median **{rut_p.median():.2f}**, and range **{rut_p.min():.1f}–{rut_p.max():.1f}**. Future rut depth has mean **{rut_n.mean():.2f}** and median **{rut_n.median():.2f}**.

## 3. Missing values and data quality

- PCI dataset: {len(pci_missing_nonzero)} variables contain at least one missing value.
- Rutting dataset: {len(rut_missing_nonzero)} variables contain at least one missing value.
- `DAYS_SINCE_LAST_PATCH` is highly sparse in both datasets; this should be treated carefully because absence can mean no recorded prior patch rather than a random missing measurement.
- Lag/interval variables such as previous-condition fields and time since prior survey are structurally missing for first observations and are therefore not automatically data errors.
- The simple physical/domain checks found **{impossible_total}** impossible values across the checked PCI/rut bounds, negative age, negative ESAL, negative precipitation and negative thickness conditions.
- IQR flags are reported separately as potential extremes, not automatically removed: traffic, climate and thickness variables are legitimately skewed and should be reviewed rather than clipped without engineering justification.

## 4. Variables related to future PCI

{chr(10).join(top_corr_lines(pci_rank))}

These are univariate relationships only. A weak linear correlation does not mean a variable is useless in nonlinear or interaction-based models.

## 5. Variables related to future rut depth

{chr(10).join(top_corr_lines(rut_rank))}

## 6. Do PCI and rutting deteriorate together?

There are **{comp_metrics.get('matched_pairs', 0):,}** matched current-to-next road-survey pairs available in both modelling tables.

- Current PCI vs current rut depth: Pearson r = **{comp_metrics.get('current_corr', np.nan):.3f}**.
- Future PCI vs future rut depth: Pearson r = **{comp_metrics.get('future_corr', np.nan):.3f}**.
- PCI-change vs rut-change correlation: **{comp_metrics.get('change_corr', np.nan):.3f}**.
- Annualised deterioration-rate correlation: **{comp_metrics.get('rate_corr', np.nan):.3f}**.
- Both measures deteriorated on **{comp_metrics.get('both_deteriorated', 0)} ({both_pct:.1f}%)** matched pairs.
- PCI deteriorated while rutting did not on **{comp_metrics.get('pci_only', 0)} ({pci_only_pct:.1f}%)**.
- Rutting deteriorated while PCI did not on **{comp_metrics.get('rut_only', 0)} ({rut_only_pct:.1f}%)**.
- Neither deteriorated on **{comp_metrics.get('neither', 0)} ({neither_pct:.1f}%)**.

This directly tests the client's question. If the correlations are modest and there are many one-measure-only deterioration cases, PCI and rutting should not be treated as interchangeable condition measures — which supports modelling them separately.

## 7. Are they driven by the same factors?

Among the common numeric predictors, the overlap between the ten strongest absolute linear relationships for the two targets is **{comp_metrics.get('top10_shared_factor_overlap_count', 0)} of 10**. Shared variables in that overlap are: **{', '.join(comp_metrics.get('top10_shared_factor_overlap', [])) or 'none'}**.

Highest combined common-factor relationships:

{chr(10).join(factor_lines)}

The comparison should be interpreted as screening evidence rather than causal attribution. Model-based feature importance and SHAP should later be compared with these EDA relationships.

## 8. Modelling implications

1. **Keep PCI and rutting separate.** Their scales, available rows and deterioration behaviour are not identical.
2. **Current condition and condition-history variables should be retained as candidate predictors** where they are legitimately available at prediction time; they generally carry strong signal for next-survey condition.
3. **Age, traffic, climate and structure should remain in the candidate feature set** even where simple correlations are modest, because tree-based models can capture nonlinear thresholds and interactions.
4. **Do not silently delete highly sparse maintenance/history variables.** Their missingness has domain meaning and should be handled consistently in model preprocessing.
5. **Use the shared road-level train/test assignment.** EDA does not alter the split and therefore does not create leakage.
6. **Report MAE, RMSE and R² separately for PCI and rutting.** Cross-target error magnitudes are not directly comparable because the outcome scales differ.

## 9. Files produced

The script saves separate descriptive tables, missingness tables, outlier/domain-check tables, target-correlation rankings, two correlation matrices, matched PCI-rutting comparisons and all requested figures into `/results` and `/plots`.
"""
    out = results_dir / "eda_v2_summary.md"
    out.write_text(text, encoding="utf-8")
    return text


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def analyse_one(df: pd.DataFrame, dataset_name: str, target: str, current: str,
                label: str, prefix: str, plots_dir: Path, results_dir: Path):
    validate_dataset(df, dataset_name, target, current)

    overview = dataset_overview(df, dataset_name)
    types = variable_types_table(df, dataset_name)
    desc = descriptive_table(df, dataset_name, target)
    missing = missing_table(df, dataset_name)
    outliers = iqr_outliers(df, dataset_name, target)
    domain = domain_checks(df, dataset_name, target, current)

    overview.to_csv(results_dir / f"{prefix}_dataset_overview.csv", index=False)
    types.to_csv(results_dir / f"{prefix}_variable_types.csv", index=False)
    desc.to_csv(results_dir / f"{prefix}_descriptive_statistics.csv", index=False)
    missing.to_csv(results_dir / f"{prefix}_missing_values.csv", index=False)
    outliers.to_csv(results_dir / f"{prefix}_outlier_summary.csv", index=False)
    domain.to_csv(results_dir / f"{prefix}_domain_quality_checks.csv", index=False)

    distribution_plot(df, current, target, label, prefix, plots_dir)
    relationship_plots(df, target, f"Future {label}", prefix, plots_dir)
    missing_plot(missing, prefix, dataset_name, plots_dir)
    outlier_plot(outliers, prefix, dataset_name, plots_dir)
    corr, ranking = correlation_outputs(df, dataset_name, target, prefix, plots_dir, results_dir)

    return {
        "overview": overview,
        "types": types,
        "desc": desc,
        "missing": missing,
        "outliers": outliers,
        "domain": domain,
        "corr": corr,
        "ranking": ranking,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for separate PCI and rutting modelling datasets")
    parser.add_argument("--pci", default=None, help="Path to model_dataset_pci.csv")
    parser.add_argument("--rut", default=None, help="Path to model_dataset_rut.csv")
    parser.add_argument("--plots", default=str(DEFAULT_PLOTS_DIR), help="Plot output directory")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS_DIR), help="Results output directory")
    args = parser.parse_args()

    pci_path = resolve_input(args.pci, "model_dataset_pci.csv")
    rut_path = resolve_input(args.rut, "model_dataset_rut.csv")
    plots_dir = Path(args.plots).resolve()
    results_dir = Path(args.results).resolve()
    ensure_dirs(plots_dir, results_dir)

    print(f"[eda] Loading PCI dataset: {pci_path}")
    pci = pd.read_csv(pci_path, low_memory=False)
    print(f"[eda] Loading rutting dataset: {rut_path}")
    rut = pd.read_csv(rut_path, low_memory=False)

    pci_res = analyse_one(
        pci, "PCI distress-only dataset", PCI_TARGET, PCI_CURRENT,
        "PCI", "eda_pci", plots_dir, results_dir
    )
    rut_res = analyse_one(
        rut, "Rutting dataset", RUT_TARGET, RUT_CURRENT,
        "rut depth (1/8 inch)", "eda_rut", plots_dir, results_dir
    )

    _, factors, metrics = compare_pci_rut(
        pci, rut, pci_res["ranking"], rut_res["ranking"], plots_dir, results_dir
    )
    summary = write_summary(
        pci, rut,
        pci_res["missing"], rut_res["missing"],
        pci_res["domain"], rut_res["domain"],
        pci_res["ranking"], rut_res["ranking"],
        metrics, factors, results_dir
    )

    # Combined convenience files for reporting.
    pd.concat([pci_res["overview"], rut_res["overview"]], ignore_index=True).to_csv(
        results_dir / "eda_v2_dataset_overview.csv", index=False
    )
    pd.concat([pci_res["missing"], rut_res["missing"]], ignore_index=True).to_csv(
        results_dir / "eda_v2_missing_values.csv", index=False
    )
    pd.concat([pci_res["outliers"], rut_res["outliers"]], ignore_index=True).to_csv(
        results_dir / "eda_v2_outlier_summary.csv", index=False
    )
    pd.concat([pci_res["domain"], rut_res["domain"]], ignore_index=True).to_csv(
        results_dir / "eda_v2_domain_quality_checks.csv", index=False
    )

    print("\n[eda] COMPLETE")
    print(f"[eda] Plots:   {plots_dir}")
    print(f"[eda] Results: {results_dir}")
    print("[eda] Report summary: eda_v2_summary.md")
    print("\n" + summary.split("## 9. Files produced")[0][-1800:])


def run():
    """Pipeline entry point (added by Abdullah during integration).

    Aum's module is a standalone script with an argparse main(). The pipeline calls
    run(), so this wraps main() with the pipeline's configured paths. His analysis code
    is untouched.
    """
    import sys
    import config
    argv = sys.argv
    sys.argv = [argv[0],
                "--pci", config.MODEL_DATASET_PCI_PATH,
                "--rut", config.MODEL_DATASET_RUT_PATH,
                "--plots", config.PLOTS_DIR,
                "--results", config.RESULTS_DIR]
    try:
        main()
    finally:
        sys.argv = argv


if __name__ == "__main__":
    main()
