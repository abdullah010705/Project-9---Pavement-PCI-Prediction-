"""
schema_checks.py — validate model_dataset.csv against config.py before any
downstream stage (EDA, modelling, evaluation, explainability) runs.

Catches the failure mode where Rayyan's build script changes column names,
someone re-exports the dataset with a different tool, or a stage script was
written against a stale copy of the schema. Fails loudly and early instead
of letting a KeyError surface three stages later inside someone's notebook.
"""
import pandas as pd

import config


class SchemaError(Exception):
    """Raised when model_dataset.csv doesn't match the agreed schema."""


def validate_model_dataset(df: pd.DataFrame, strict: bool = True) -> list:
    """Check df against config.py's REQUIRED_COLUMNS / TARGET_COL / SPLIT_COL.

    Returns a list of human-readable warning strings for non-fatal issues
    (e.g. missing values in optional columns). Raises SchemaError for fatal
    issues (missing required columns, nulls in the target, bad split values,
    train/test leakage across physical roads).

    strict=False downgrades fatal checks to warnings too — useful for
    exploratory work, never for the pipeline itself.
    """
    warnings = []
    errors = []

    # 1. Every required column must be present.
    missing_cols = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(
            f"Missing {len(missing_cols)} required column(s): {missing_cols}"
        )

    # Stop here if columns are missing — everything below assumes they exist.
    if missing_cols:
        _raise_or_warn(errors, warnings, strict)
        return warnings

    # 2. Target column must have no nulls (rows without a future survey
    #    should already have been dropped by Rayyan's build script).
    n_null_target = df[config.TARGET_COL].isna().sum()
    if n_null_target > 0:
        errors.append(
            f"{n_null_target} row(s) have a null {config.TARGET_COL} "
            f"(rows with no future survey should have been dropped)."
        )

    # 3. SPLIT column must only contain the expected values.
    bad_split_values = set(df[config.SPLIT_COL].unique()) - {config.TRAIN_VALUE, config.TEST_VALUE}
    if bad_split_values:
        errors.append(
            f"{config.SPLIT_COL} contains unexpected values: {bad_split_values} "
            f"(expected only {config.TRAIN_VALUE!r}/{config.TEST_VALUE!r})."
        )

    # 4. No physical road should appear in both train and test (leak check).
    if not bad_split_values:
        train_roads = set(df.loc[df[config.SPLIT_COL] == config.TRAIN_VALUE, config.PHYSICAL_ROAD_ID_COL])
        test_roads = set(df.loc[df[config.SPLIT_COL] == config.TEST_VALUE, config.PHYSICAL_ROAD_ID_COL])
        overlap = train_roads & test_roads
        if overlap:
            errors.append(
                f"{len(overlap)} physical road(s) appear in BOTH train and test: "
                f"{sorted(overlap)[:10]}{'...' if len(overlap) > 10 else ''}"
            )

    # 5. Non-fatal: flag feature columns that are entirely null (usually a
    #    sign something upstream broke rather than genuine missingness).
    for col in config.ALL_FEATURE_COLS:
        if col in df.columns and df[col].isna().all():
            warnings.append(f"Feature column '{col}' is 100% null.")

    # 6. Non-fatal: row/group sanity numbers, useful when re-running after
    #    a change to the raw data or scope filter.
    warnings.append(
        f"{len(df)} rows | {df[config.ROAD_ID_COL].nunique()} pavement lives | "
        f"{df[config.PHYSICAL_ROAD_ID_COL].nunique()} physical roads | "
        f"train={int((df[config.SPLIT_COL] == config.TRAIN_VALUE).sum())} "
        f"test={int((df[config.SPLIT_COL] == config.TEST_VALUE).sum())}"
    )

    _raise_or_warn(errors, warnings, strict)
    return warnings


def _raise_or_warn(errors, warnings, strict):
    if errors and strict:
        raise SchemaError(
            "model_dataset.csv failed schema validation:\n  - " + "\n  - ".join(errors)
        )
    elif errors:
        warnings.extend(f"[DOWNGRADED FROM ERROR] {e}" for e in errors)


def load_and_validate(path: str = None, strict: bool = True) -> pd.DataFrame:
    """Convenience wrapper: load model_dataset.csv and validate it in one call."""
    path = path or config.MODEL_DATASET_PATH
    df = pd.read_csv(path, low_memory=False)
    warnings = validate_model_dataset(df, strict=strict)
    for w in warnings:
        print(f"[schema_checks] {w}")
    return df


if __name__ == "__main__":
    load_and_validate()
    print("[schema_checks] model_dataset.csv passed schema validation.")
