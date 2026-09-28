"""
config.py — single source of truth for paths, columns, and settings shared
across every member's script. Import this instead of hardcoding paths or
column names, so dataset construction (Rayyan), EDA (Aum), baseline/ML
(Luke), XGBoost+SHAP (Abdullah), and the pipeline (Ravtej) all agree on
inputs, targets, and train/test split.

=== v2 CHANGE (Project Group Allocations v2, client-directed) ===
PCI and rutting are now two separate condition measures with their own
datasets and models, instead of one merged score. There are two regression
targets (PCI_NEXT, RUT_NEXT) plus one classification target (NEEDS_REPAIR,
PCI dataset only). Everyone should now iterate `config.TARGETS` rather than
assuming a single `TARGET_COL` / `MODEL_DATASET_PATH` — those v1 names are
kept below as deprecated aliases (pointing at the PCI target) only so old
scratch scripts don't hard-crash; don't write new code against them.

Column lists below reflect the ACTUAL columns in data/model_dataset_pci.csv
(976 rows x 55 cols) and data/model_dataset_rut.csv (693 rows x 54 cols) as
delivered by Rayyan for v2. Update this file, not individual scripts, if
the schema changes.
"""
import os

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

# Raw input: PCI recomputed by pci_algorithm.py v2 (rutting removed from the
# CDV calc — PCI is now a distress-only score; see Decision Log #12).
RAW_DATA_PATH = os.path.join(DATA_DIR, "master_with_pci_3.csv")

# The two modelling datasets (replaces the single v1 model_dataset.csv).
MODEL_DATASET_PCI_PATH = os.path.join(DATA_DIR, "model_dataset_pci.csv")
MODEL_DATASET_RUT_PATH = os.path.join(DATA_DIR, "model_dataset_rut.csv")

# ONE train/test split, grouped by physical road, computed over the union of
# roads in either dataset, then applied to both by lookup (not two splits).
SPLIT_ASSIGNMENT_PATH = os.path.join(DATA_DIR, "split_assignment.csv")

for _dir in (DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR):
    os.makedirs(_dir, exist_ok=True)

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.2  # only used if a script needs to re-split; normally use SPLIT_COL below

# ---------------------------------------------------------------------
# Identifiers (from Rayyan's dataset construction — unchanged from v1)
# ---------------------------------------------------------------------
# One row = one road survey. ROAD_ID = one pavement life (resets at a
# rebuild). PHYSICAL_ROAD_ID = the physical location (does NOT reset at a
# rebuild) — this is the grouping key that must never leak across train/test.
ROAD_ID_COL = "ROAD_ID"
PHYSICAL_ROAD_ID_COL = "PHYSICAL_ROAD_ID"
SEGMENT_ID_COL = "PAVEMENT_SEGMENT_ID"
SURVEY_DATE_COL = "SURVEY_DATE"

# Train/test split is already assigned per row in both dataset files (and is
# also stored keyed by physical road in SPLIT_ASSIGNMENT_PATH, which is the
# source of truth src/schema_checks.py cross-checks both datasets against).
# Every member should read this column rather than re-splitting themselves.
SPLIT_COL = "SPLIT"
TRAIN_VALUE = "train"
TEST_VALUE = "test"

# ---------------------------------------------------------------------
# Feature groups shared by both datasets (identical feature engineering —
# only the target column and the row filter differ between PCI and RUT,
# per the client's instruction). Unchanged from v1.
# ---------------------------------------------------------------------
AGE_COLS = ["PAVEMENT_AGE_YEARS"]

TRAFFIC_COLS = ["CUM_ESAL_AT_SURVEY", "ESAL_SINCE_LAST_SURVEY"]

CLIMATE_COLS = [
    "PRECIPITATION", "EVAPORATION", "PRECIP_DAYS", "TEMP_AVG",
    "FREEZE_INDEX", "FREEZE_THAW", "SHORTWAVE_SURFACE_AVG",
]

STRUCTURE_NUMERIC_COLS = [
    "AC_THICKNESS_MM", "BASE_THICKNESS_MM", "SUBBASE_THICKNESS_MM",
    "TOTAL_PAV_THICKNESS_MM", "NUM_AC_LAYERS", "NUM_LAYERS_TOTAL",
]
STRUCTURE_CATEGORICAL_COLS = ["SURFACE_MATL", "BASE_MATL", "SUBGRADE_MATL"]

MAINTENANCE_COLS = ["PRIOR_PATCH_COUNT", "DAYS_SINCE_LAST_PATCH"]

# Both PCI_SCORE and RUT_DEPTH_1_8IN are present in BOTH datasets (Rayyan's
# build keeps the full current-condition block in each so the two tables can
# be lined up per road/survey, per the client's "unionise" note) — only the
# *target* (PCI_NEXT vs RUT_NEXT) and the row filter differ.
CURRENT_CONDITION_NUMERIC_COLS = [
    "PCI_SCORE", "RUT_DEPTH_1_8IN", "IRI_LEFT_WHEEL_PATH",
    "IRI_RIGHT_WHEEL_PATH", "MRI",
]
CURRENT_CONDITION_CATEGORICAL_COLS = ["PCI_RATING", "PCI_DOMINANT_DISTRESS"]

# Context columns kept for reference/filtering, not intended as model inputs.
CONTEXT_COLS = [
    "PAVEMENT_FAMILY", "PAVEMENT_FAMILY_EXP", "FUNC_CLASS_EXP",
    "LTPP_DIR_EXP", "LTPP_LANE", "GPS_SPS_EXP", "EXPERIMENT_NO_EXP",
]

# Identifier columns present in both dataset files (not features, not target).
IDENTIFIER_COLS = [
    ROAD_ID_COL, PHYSICAL_ROAD_ID_COL, SEGMENT_ID_COL,
    "STATE_CODE", "STATE_NAME", "SHRP_ID", "CONSTRUCTION_NO",
    SURVEY_DATE_COL, "SURVEY_YEAR", "SURVEY_SEQ_NO",
]

SHARED_NUMERIC_BASE_COLS = (
    AGE_COLS + TRAFFIC_COLS + CLIMATE_COLS + STRUCTURE_NUMERIC_COLS
    + MAINTENANCE_COLS + CURRENT_CONDITION_NUMERIC_COLS
)
SHARED_CATEGORICAL_COLS = STRUCTURE_CATEGORICAL_COLS + CURRENT_CONDITION_CATEGORICAL_COLS

# NOTE for Abdullah's feature-reduction task (v2 allocation, client flagged
# ~120 features as too many): once he delivers a reduced set with evidence,
# add it here as REDUCED_FEATURE_COLS and have modelling stages prefer it
# when present, rather than hand-editing the shared lists above.
# ---------------------------------------------------------------------
# Feature reduction (Abdullah, v2). Client flagged ~120 features as too many.
# Ranked by mean |SHAP| AND permutation importance, combined by average rank,
# then the feature count was swept under grouped CV on TRAINING ROWS ONLY.
# Chosen: smallest set within 2% of the best CV RMSE.
#
# Both targets scored slightly BETTER with 10 features than with the full set:
#   PCI  138 -> 10 features, CV RMSE 12.708 -> 12.418, R2 0.744 -> 0.756
#   RUT  123 -> 10 features, CV RMSE  1.511 ->  1.502, R2 0.863 -> 0.865
#
# These are SOURCE column names. The encoded names actually used by the models
# (one-hot levels, __isna flags) are in results/reduced_features_<key>.json,
# which src/models/xgboost_shap.py reads directly.
# ---------------------------------------------------------------------
REDUCED_FEATURE_COLS = {
    "pci": [
        "BASE_THICKNESS_MM",
        "CUM_ESAL_AT_SURVEY",
        "DAYS_SINCE_LAST_SURVEY",
        "IRI_LEFT_WHEEL_PATH",
        "PAVEMENT_AGE_YEARS",
        "PCI_CHANGE_SINCE_LAST",
        "PCI_RATING",
        "PCI_SCORE",
        "SUBBASE_THICKNESS_MM",
        "TEMP_AVG",
    ],
    "rut": [
        "BASE_THICKNESS_MM",
        "CUM_ESAL_AT_SURVEY",
        "EVAPORATION",
        "FREEZE_INDEX",
        "PAVEMENT_AGE_YEARS",
        "PCI_SCORE",
        "RUT_CHANGE_SINCE_LAST",
        "RUT_DEPTH_1_8IN",
        "RUT_PREV",
        "SHORTWAVE_SURFACE_AVG",
    ],
}

# ---------------------------------------------------------------------
# TARGETS — the two regression targets. Each entry carries everything a
# modelling stage needs for that target: which dataset file to load, the
# target column, the lag features specific to that target (PCI_PREV vs
# RUT_PREV — the only feature-list difference between the two datasets),
# and where that target's results/SHAP outputs should be written.
#
# Pattern for every modelling/EDA stage from here on:
#   for key, t in config.TARGETS.items():
#       df = schema_checks.load_and_validate(key)
#       ... train against t["target_col"], using t["all_feature_cols"] ...
#       results_io.append_results(t["results_metrics_path"], key, rows)
# ---------------------------------------------------------------------
TARGETS = {}

for _key, _label, _dataset_path, _target_col, _lag_target_prev, _lag_target_change in [
    ("pci", "PCI", MODEL_DATASET_PCI_PATH, "PCI_NEXT", "PCI_PREV", "PCI_CHANGE_SINCE_LAST"),
    ("rut", "RUT", MODEL_DATASET_RUT_PATH, "RUT_NEXT", "RUT_PREV", "RUT_CHANGE_SINCE_LAST"),
]:
    _lag_cols = [
        "N_PRIOR_SURVEYS", _lag_target_prev, _lag_target_change,
        "DAYS_SINCE_LAST_SURVEY", "YEARS_SINCE_LAST_SURVEY",
    ]
    _numeric_feature_cols = SHARED_NUMERIC_BASE_COLS + _lag_cols
    _all_feature_cols = _numeric_feature_cols + SHARED_CATEGORICAL_COLS
    _target_aux_cols = ["DAYS_TO_NEXT_SURVEY", "YEARS_TO_NEXT_SURVEY"]

    _required_columns = (
        IDENTIFIER_COLS + CONTEXT_COLS + _all_feature_cols
        + [_target_col] + _target_aux_cols + [SPLIT_COL]
    )
    if _key == "pci":
        _required_columns = _required_columns + ["NEEDS_REPAIR"]
    _required_columns = list(dict.fromkeys(_required_columns))

    TARGETS[_key] = {
        "key": _key,
        "label": _label,
        "dataset_path": _dataset_path,
        "target_col": _target_col,
        "target_aux_cols": _target_aux_cols,
        "lag_cols": _lag_cols,
        "numeric_feature_cols": _numeric_feature_cols,
        "categorical_feature_cols": SHARED_CATEGORICAL_COLS,
        "all_feature_cols": _all_feature_cols,
        "required_columns": _required_columns,
        "results_metrics_path": os.path.join(RESULTS_DIR, f"model_performance_{_key}.csv"),
        "predictions_path": os.path.join(RESULTS_DIR, f"predictions_{_key}.csv"),
        "shap_results_path": os.path.join(RESULTS_DIR, f"shap_values_{_key}.csv"),
        "comparison_plot_path": os.path.join(PLOTS_DIR, f"model_comparison_{_key}.png"),
    }

del _key, _label, _dataset_path, _target_col, _lag_target_prev, _lag_target_change
del _lag_cols, _numeric_feature_cols, _all_feature_cols, _target_aux_cols, _required_columns

# ---------------------------------------------------------------------
# CLASSIFIER — PCI-below-55 repair classifier (Abdullah, v2 new requirement).
# Client-confirmed threshold. Lives in the PCI dataset only (NEEDS_REPAIR is
# derived from PCI_NEXT; rutting has no repair-threshold concept).
# ---------------------------------------------------------------------
REPAIR_THRESHOLD = 55  # client: PCI_NEXT below this = road needs repair

CLASSIFIER = {
    "key": "repair",
    "label": "Needs-repair (PCI < 55)",
    "dataset_path": MODEL_DATASET_PCI_PATH,
    "target_col": "NEEDS_REPAIR",
    "all_feature_cols": TARGETS["pci"]["all_feature_cols"],
    "results_metrics_path": os.path.join(RESULTS_DIR, "classifier_performance.csv"),
    "predictions_path": os.path.join(RESULTS_DIR, "predictions_repair_classifier.csv"),
    "confusion_matrix_plot_path": os.path.join(PLOTS_DIR, "repair_classifier_confusion_matrix.png"),
}

# ---------------------------------------------------------------------
# EDA output paths (Aum runs EDA once per dataset in v2 — see eda.py)
# ---------------------------------------------------------------------
EDA_SUMMARY_PATH = os.path.join(RESULTS_DIR, "eda_summary.md")

# ---------------------------------------------------------------------
# Deprecated v1 aliases — kept only so nothing still importing the old
# single-target names hard-crashes on import. Do not use in new code; they
# point at the PCI target and ignore RUT entirely.
# ---------------------------------------------------------------------
MODEL_DATASET_PATH = MODEL_DATASET_PCI_PATH
TARGET_COL = TARGETS["pci"]["target_col"]
TARGET_AUX_COLS = TARGETS["pci"]["target_aux_cols"]
LAG_COLS = TARGETS["pci"]["lag_cols"]
NUMERIC_FEATURE_COLS = TARGETS["pci"]["numeric_feature_cols"]
CATEGORICAL_FEATURE_COLS = TARGETS["pci"]["categorical_feature_cols"]
ALL_FEATURE_COLS = TARGETS["pci"]["all_feature_cols"]
REQUIRED_COLUMNS = TARGETS["pci"]["required_columns"]
RESULTS_METRICS_PATH = TARGETS["pci"]["results_metrics_path"]
RESULTS_PREDICTIONS_PATH = TARGETS["pci"]["predictions_path"]
SHAP_RESULTS_PATH = TARGETS["pci"]["shap_results_path"]
