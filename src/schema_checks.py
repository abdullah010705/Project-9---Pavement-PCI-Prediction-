"""
schema_checks.py — validate model_dataset_pci.csv / model_dataset_rut.csv
against config.py before any downstream stage (EDA, modelling, evaluation,
explainability) runs.

Catches the failure mode where Rayyan's build script changes column names,
someone re-exports a dataset with a different tool, or a stage script was
written against a stale copy of the schema. Fails loudly and early instead
of letting a KeyError surface three stages later inside someone's notebook.

=== v2 CHANGE ===
There are now two dataset files sharing one train/test split
(config.SPLIT_ASSIGNMENT_PATH). Every check below is per-target (pass
"pci" or "rut", matching a key in config.TARGETS), plus one new
cross-dataset check — validate_shared_split() — that confirms both
dataset files and the standalone split_assignment.csv all agree on every
physical road's train/test assignment. Run that once after loading both
datasets, not per-target.
"""
import pandas as pd

import config


class SchemaError(Exception):
    """Raised when a dataset doesn't match the agreed schema."""


def validate_model_dataset(df: pd.DataFrame, target_key: str, strict: bool = True) -> list:
    """Check df against config.TARGETS[target_key]'s required columns / target / split.

    Returns a list of human-readable warning strings for non-fatal issues
    (e.g. missing values in optional columns). Raises SchemaError for fatal
    issues (missing required columns, nulls in the target, bad split values,
    train/test leakage across physical roads).

    strict=False downgrades fatal checks to warnings too — useful for
    exploratory work, never for the pipeline itself.
    """
    if target_key not in config.TARGETS:
        raise ValueError(f"Unknown target_key {target_key!r}; expected one of {list(config.TARGETS)}")
    target = config.TARGETS[target_key]

    warnings = []
    errors = []

    # 1. Every required column must be present.
    missing_cols = [c for c in target["required_columns"] if c not in df.columns]
    if missing_cols:
        errors.append(
            f"[{target['label']}] Missing {len(missing_cols)} required column(s): {missing_cols}"
        )

    # Stop here if columns are missing — everything below assumes they exist.
    if missing_cols:
        _raise_or_warn(errors, warnings, strict)
        return warnings

    # 2. Target column must have no nulls (rows without a future survey /
    #    without the relevant current reading should already have been
    #    dropped by Rayyan's build script).
    target_col = target["target_col"]
    n_null_target = df[target_col].isna().sum()
    if n_null_target > 0:
        errors.append(
            f"[{target['label']}] {n_null_target} row(s) have a null {target_col} "
            f"(rows with no future survey should have been dropped)."
        )

    # 3. SPLIT column must only contain the expected values.
    bad_split_values = set(df[config.SPLIT_COL].unique()) - {config.TRAIN_VALUE, config.TEST_VALUE}
    if bad_split_values:
        errors.append(
            f"[{target['label']}] {config.SPLIT_COL} contains unexpected values: {bad_split_values} "
            f"(expected only {config.TRAIN_VALUE!r}/{config.TEST_VALUE!r})."
        )

    # 4. No physical road should appear in both train and test (leak check).
    if not bad_split_values:
        train_roads = set(df.loc[df[config.SPLIT_COL] == config.TRAIN_VALUE, config.PHYSICAL_ROAD_ID_COL])
        test_roads = set(df.loc[df[config.SPLIT_COL] == config.TEST_VALUE, config.PHYSICAL_ROAD_ID_COL])
        overlap = train_roads & test_roads
        if overlap:
            errors.append(
                f"[{target['label']}] {len(overlap)} physical road(s) appear in BOTH train and test: "
                f"{sorted(overlap)[:10]}{'...' if len(overlap) > 10 else ''}"
            )

    # 5. Non-fatal: flag feature columns that are entirely null.
    for col in target["all_feature_cols"]:
        if col in df.columns and df[col].isna().all():
            warnings.append(f"[{target['label']}] Feature column '{col}' is 100% null.")

    # 6. PCI dataset only: NEEDS_REPAIR must be a clean 0/1/NA classification
    #    label consistent with the client's threshold (PCI_NEXT < REPAIR_THRESHOLD).
    if target_key == "pci" and "NEEDS_REPAIR" in df.columns:
        bad_values = set(df["NEEDS_REPAIR"].dropna().unique()) - {0, 1}
        if bad_values:
            errors.append(f"[PCI] NEEDS_REPAIR contains values other than 0/1: {bad_values}")
        mismatched = df.dropna(subset=["NEEDS_REPAIR", target_col])
        expected = (mismatched[target_col] < config.REPAIR_THRESHOLD).astype(int)
        n_mismatch = int((mismatched["NEEDS_REPAIR"].astype(int) != expected).sum())
        if n_mismatch:
            errors.append(
                f"[PCI] {n_mismatch} row(s) where NEEDS_REPAIR disagrees with "
                f"(PCI_NEXT < {config.REPAIR_THRESHOLD})."
            )

    # 7. Non-fatal: row/group sanity numbers.
    warnings.append(
        f"[{target['label']}] {len(df)} rows | {df[config.ROAD_ID_COL].nunique()} pavement lives | "
        f"{df[config.PHYSICAL_ROAD_ID_COL].nunique()} physical roads | "
        f"train={int((df[config.SPLIT_COL] == config.TRAIN_VALUE).sum())} "
        f"test={int((df[config.SPLIT_COL] == config.TEST_VALUE).sum())}"
    )

    _raise_or_warn(errors, warnings, strict)
    return warnings


def validate_shared_split(pci_df: pd.DataFrame, rut_df: pd.DataFrame, strict: bool = True) -> list:
    """v2 NEW — confirm the PCI dataset, RUT dataset, and split_assignment.csv
    all agree on every physical road's train/test assignment.

    This is the check behind the v2 "Finished when" line: one shared split,
    applied consistently, not two datasets that happen to agree by luck.
    """
    warnings = []
    errors = []

    try:
        split_assignment = pd.read_csv(config.SPLIT_ASSIGNMENT_PATH)
    except FileNotFoundError:
        errors.append(f"{config.SPLIT_ASSIGNMENT_PATH} not found — can't verify the shared split.")
        _raise_or_warn(errors, warnings, strict)
        return warnings

    lookup = split_assignment.set_index(config.PHYSICAL_ROAD_ID_COL)[config.SPLIT_COL]

    for name, df in [("PCI", pci_df), ("RUT", rut_df)]:
        roads = set(df[config.PHYSICAL_ROAD_ID_COL])
        missing_from_lookup = roads - set(lookup.index)
        if missing_from_lookup:
            errors.append(
                f"{name} dataset has {len(missing_from_lookup)} physical road(s) not present in "
                f"split_assignment.csv: {sorted(missing_from_lookup)[:10]}"
            )
            continue
        expected = df[config.PHYSICAL_ROAD_ID_COL].map(lookup)
        mismatch = int((df[config.SPLIT_COL].values != expected.values).sum())
        if mismatch:
            errors.append(
                f"{name} dataset: {mismatch} row(s) whose SPLIT column disagrees with "
                f"split_assignment.csv for the same physical road."
            )

    shared_roads = set(pci_df[config.PHYSICAL_ROAD_ID_COL]) & set(rut_df[config.PHYSICAL_ROAD_ID_COL])
    pci_assign = (pci_df[pci_df[config.PHYSICAL_ROAD_ID_COL].isin(shared_roads)]
                  .drop_duplicates(config.PHYSICAL_ROAD_ID_COL)
                  .set_index(config.PHYSICAL_ROAD_ID_COL)[config.SPLIT_COL])
    rut_assign = (rut_df[rut_df[config.PHYSICAL_ROAD_ID_COL].isin(shared_roads)]
                  .drop_duplicates(config.PHYSICAL_ROAD_ID_COL)
                  .set_index(config.PHYSICAL_ROAD_ID_COL)[config.SPLIT_COL])
    disagreement = int((pci_assign.reindex(sorted(shared_roads)) != rut_assign.reindex(sorted(shared_roads))).sum())
    if disagreement:
        errors.append(
            f"PCI and RUT datasets disagree on split assignment for {disagreement} of "
            f"{len(shared_roads)} shared physical road(s)."
        )
    else:
        warnings.append(
            f"Shared split verified: {len(shared_roads)} physical roads appear in both datasets "
            f"and all got the identical train/test assignment."
        )

    _raise_or_warn(errors, warnings, strict)
    return warnings


def _raise_or_warn(errors, warnings, strict):
    if errors and strict:
        raise SchemaError("Schema validation failed:\n  - " + "\n  - ".join(errors))
    elif errors:
        warnings.extend(f"[DOWNGRADED FROM ERROR] {e}" for e in errors)


def load_and_validate(target_key: str, path: str = None, strict: bool = True) -> pd.DataFrame:
    """Convenience wrapper: load one target's dataset and validate it in one call.

    target_key: "pci" or "rut" (a key in config.TARGETS).
    """
    target = config.TARGETS[target_key]
    path = path or target["dataset_path"]
    df = pd.read_csv(path, low_memory=False)
    warnings = validate_model_dataset(df, target_key, strict=strict)
    for w in warnings:
        print(f"[schema_checks] {w}")
    return df


def load_and_validate_all(strict: bool = True) -> dict:
    """Load + validate both datasets, then verify the shared split across them.
    Returns {"pci": df, "rut": df}.
    """
    dfs = {key: load_and_validate(key, strict=strict) for key in config.TARGETS}
    warnings = validate_shared_split(dfs["pci"], dfs["rut"], strict=strict)
    for w in warnings:
        print(f"[schema_checks] {w}")
    return dfs


if __name__ == "__main__":
    load_and_validate_all()
    print("[schema_checks] Both datasets passed schema validation; shared split verified.")
