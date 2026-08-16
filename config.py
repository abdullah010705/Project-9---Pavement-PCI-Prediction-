"""
config.py — single source of truth for paths, columns, and settings shared
across every member's script. Import this instead of hardcoding paths or
column names, so dataset construction (Rayyan), EDA (Aum), baseline/ML
(Luke), XGBoost+SHAP (Abdullah), and the pipeline (Ravtej) all agree on
inputs, target, and train/test split.

Column list below reflects the ACTUAL columns in data/model_dataset.csv as
delivered by Rayyan (976 rows x 54 cols) — not the originally-planned list.
Update this file, not individual scripts, if the schema changes.
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

# Raw input (PCI already computed per survey — see pci_algorithm.py, not part
# of this pipeline) and the final modelling dataset built from it.
RAW_DATA_PATH = os.path.join(DATA_DIR, "master_with_pci_2.csv")
MODEL_DATASET_PATH = os.path.join(DATA_DIR, "model_dataset.csv")

# Where each stage should write its outputs. Use these, don't invent new
# paths in individual scripts, or results/plots end up scattered.
RESULTS_METRICS_PATH = os.path.join(RESULTS_DIR, "model_performance.csv")
RESULTS_PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "predictions.csv")
EDA_SUMMARY_PATH = os.path.join(RESULTS_DIR, "eda_summary.md")
SHAP_RESULTS_PATH = os.path.join(RESULTS_DIR, "shap_values.csv")

for _dir in (DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR):
    os.makedirs(_dir, exist_ok=True)

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.2  # only used if a script needs to re-split; normally use SPLIT_COL below

# ---------------------------------------------------------------------
# Identifiers (from Rayyan's dataset construction)
# ---------------------------------------------------------------------
# One row = one road survey. ROAD_ID = one pavement life (resets at a
# rebuild). PHYSICAL_ROAD_ID = the physical location (does NOT reset at a
# rebuild) — this is the grouping key that must never leak across train/test.
ROAD_ID_COL = "ROAD_ID"
PHYSICAL_ROAD_ID_COL = "PHYSICAL_ROAD_ID"
SEGMENT_ID_COL = "PAVEMENT_SEGMENT_ID"
SURVEY_DATE_COL = "SURVEY_DATE"

# Train/test split is already assigned per row by Rayyan's build script
# (GroupShuffleSplit, grouped by PHYSICAL_ROAD_ID, seeded with RANDOM_SEED).
# Every member should read this column rather than re-splitting themselves.
SPLIT_COL = "SPLIT"
TRAIN_VALUE = "train"
TEST_VALUE = "test"

# ---------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------
TARGET_COL = "PCI_NEXT"

# Secondary target-adjacent columns (not the label itself, but describe the
# forecast horizon — useful for analysis, must NOT be used as a model input).
TARGET_AUX_COLS = ["DAYS_TO_NEXT_SURVEY", "YEARS_TO_NEXT_SURVEY"]

# ---------------------------------------------------------------------
# Feature groups (the agreed model input variables)
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

CURRENT_CONDITION_NUMERIC_COLS = [
    "PCI_SCORE", "RUT_DEPTH_1_8IN", "IRI_LEFT_WHEEL_PATH",
    "IRI_RIGHT_WHEEL_PATH", "MRI",
]
CURRENT_CONDITION_CATEGORICAL_COLS = ["PCI_RATING", "PCI_DOMINANT_DISTRESS"]

# Lagged / cumulative features Rayyan engineered per pavement segment.
# NaN for the first survey of each pavement life — that's expected, not a bug.
LAG_COLS = [
    "N_PRIOR_SURVEYS", "PCI_PREV", "PCI_CHANGE_SINCE_LAST",
    "DAYS_SINCE_LAST_SURVEY", "YEARS_SINCE_LAST_SURVEY",
]

# Context columns kept for reference/filtering, not intended as model inputs
# (e.g. pavement family label, functional class, survey lane).
CONTEXT_COLS = [
    "PAVEMENT_FAMILY", "PAVEMENT_FAMILY_EXP", "FUNC_CLASS_EXP",
    "LTPP_DIR_EXP", "LTPP_LANE", "GPS_SPS_EXP", "EXPERIMENT_NO_EXP",
]

# ---------------------------------------------------------------------
# Convenience combinations
# ---------------------------------------------------------------------
NUMERIC_FEATURE_COLS = (
    AGE_COLS + TRAFFIC_COLS + CLIMATE_COLS + STRUCTURE_NUMERIC_COLS
    + MAINTENANCE_COLS + CURRENT_CONDITION_NUMERIC_COLS + LAG_COLS
)
CATEGORICAL_FEATURE_COLS = STRUCTURE_CATEGORICAL_COLS + CURRENT_CONDITION_CATEGORICAL_COLS

# Every agreed input variable, numeric + categorical. Categorical columns are
# left as raw text in model_dataset.csv — encode them before fitting a model
# that needs numeric input (e.g. pd.get_dummies, OrdinalEncoder).
ALL_FEATURE_COLS = NUMERIC_FEATURE_COLS + CATEGORICAL_FEATURE_COLS

# Identifier columns present in model_dataset.csv (not features, not target).
IDENTIFIER_COLS = [
    ROAD_ID_COL, PHYSICAL_ROAD_ID_COL, SEGMENT_ID_COL,
    "STATE_CODE", "STATE_NAME", "SHRP_ID", "CONSTRUCTION_NO",
    SURVEY_DATE_COL, "SURVEY_YEAR", "SURVEY_SEQ_NO",
]

# Full set of columns model_dataset.csv is expected to contain. Used by
# src/schema_checks.py to fail fast (before any modelling runs) if Rayyan's
# script or its output ever changes shape.
REQUIRED_COLUMNS = (
    IDENTIFIER_COLS + CONTEXT_COLS + ALL_FEATURE_COLS + LAG_COLS
    + [TARGET_COL] + TARGET_AUX_COLS + [SPLIT_COL]
)
# de-duplicate while preserving order (LAG_COLS appears in both
# NUMERIC_FEATURE_COLS and above for clarity — collapse to one check each)
REQUIRED_COLUMNS = list(dict.fromkeys(REQUIRED_COLUMNS))
