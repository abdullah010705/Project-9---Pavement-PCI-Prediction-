"""
build_dataset.py — dataset construction stage (owner: Rayyan)

Turns master_with_pci_3.csv (PCI recomputed by pci_algorithm.py v2 —
rutting no longer contributes a deduct; distress-only score, see Decision
Log #12) into TWO leak-safe modelling datasets, sharing one train/test
split:
    model_dataset_pci.csv — target PCI_NEXT, rows with distress data
    model_dataset_rut.csv — target RUT_NEXT, rows with rut depth data
    split_assignment.csv  — ONE GroupShuffleSplit over the union of
                             physical roads in either dataset, applied to
                             both by lookup (not two separate splits)

=== v2 CHANGE ===
This replaces v1's build_model_dataset() (single dataset, single target).
This is Rayyan's build_model_datasets_v2.py, integrated into the shared
pipeline the same way v1's script was:
  - paths come from config.py instead of being hardcoded (the original
    script already used relative paths — INPUT_PATH="master_with_pci_3.csv",
    OUT_DIR="." — rather than v1's hardcoded absolute output path, so this
    integration is a smaller change than v1's was)
  - RANDOM_STATE / TEST_SIZE / REPAIR_THRESHOLD come from config.py instead
    of being redefined locally
  - wrapped in build_model_datasets() so run_pipeline.py can call it
    directly instead of shelling out to a standalone script
  - sklearn is imported inside build_model_datasets() rather than at module
    level, so `import src.data.build_dataset` doesn't require scikit-learn
    to be installed unless this function is actually called — the
    pipeline's default path uses the already-built dataset files and never
    touches this module (same lazy-import pattern as v1)

Everything verified/approved in v1 is reused unchanged: road ID logic,
chronological sort, rebuild/segment detection, lag feature logic (mirrored
for rutting), GPS-1/GPS-2 scope. Only the target split (PCI vs RUT) and the
two new datasets/columns are new in v2.

=== KNOWN OPEN RISK — see Decision Log #15 ===
Rayyan's v2 delivery notes he worked from a single copy of pci_algorithm.py
after two uploads of it overwrote each other, and flagged this needs
reconciling with Luke's copy before it's treated as the team's source of
truth (the client specifically asked "do this with whoever originally
wrote the calculator"). This build script assumes master_with_pci_3.csv
(produced by that pci_algorithm.py) is correct. If Luke's reconciliation
turns up drift, master_with_pci_3.csv needs regenerating and this script
re-run with --rebuild-dataset before any numbers downstream are final.

See data/README_model_dataset.md (v1 methodology) for how road identity,
rebuilds, and lag features are handled — unchanged in v2.
"""
import pandas as pd

import config


def build_model_datasets(raw_path: str = None) -> dict:
    """Builds both model_dataset_pci.csv and model_dataset_rut.csv plus the
    shared split_assignment.csv. Returns {"pci": df, "rut": df}."""
    from sklearn.model_selection import GroupShuffleSplit

    raw_path = raw_path or config.RAW_DATA_PATH

    # -------------------------------------------------------------
    # STEP 0: load + scope to GPS-1/GPS-2  (unchanged from v1)
    # -------------------------------------------------------------
    df = pd.read_csv(raw_path, low_memory=False)
    df["SURVEY_DATE_dt"] = pd.to_datetime(df["SURVEY_DATE"])
    print(f"Loaded: {df.shape}")

    gps12_labels = ["Asphalt Concrete on Unbound Granular Base", "Asphalt Concrete on Bound Base"]
    before_scope = len(df)
    df = df[(df["GPS_SPS_EXP"] == "General Pavement Studies") &
            (df["EXPERIMENT_NO_EXP"].isin(gps12_labels))].copy()
    print(f"Scope filter (GPS-1/GPS-2 only): {before_scope} -> {len(df)} rows, "
          f"{df.groupby(['STATE_CODE', 'SHRP_ID', 'CONSTRUCTION_NO']).ngroups} road segments")

    # -------------------------------------------------------------
    # STEP 1: unique road id + chronological sort  (unchanged from v1)
    # -------------------------------------------------------------
    df["ROAD_ID"] = (df["STATE_CODE"].astype(str) + "-" +
                      df["SHRP_ID"].astype(str) + "-" +
                      df["CONSTRUCTION_NO"].astype(str))
    df["PHYSICAL_ROAD_ID"] = df["STATE_CODE"].astype(str) + "-" + df["SHRP_ID"].astype(str)
    df = df.sort_values(["ROAD_ID", "SURVEY_DATE_dt"]).reset_index(drop=True)

    # -------------------------------------------------------------
    # STEP 2: rebuild/reconstruction segmentation  (unchanged from v1)
    # -------------------------------------------------------------
    df["_age_prev"] = df.groupby("ROAD_ID")["PAVEMENT_AGE_YEARS"].shift(1)
    df["_age_reset"] = (df["PAVEMENT_AGE_YEARS"] < df["_age_prev"] - 0.5).fillna(False)
    df["SEGMENT_NO"] = df.groupby("ROAD_ID")["_age_reset"].cumsum()
    print(f"Undocumented age-reset splits detected within a CONSTRUCTION_NO: {df['_age_reset'].sum()}")
    df["PAVEMENT_SEGMENT_ID"] = df["ROAD_ID"] + "-" + df["SEGMENT_NO"].astype(str)
    df = df.drop(columns=["_age_prev", "_age_reset"])

    # -------------------------------------------------------------
    # STEP 3: lagged / cumulative features (within each pavement segment)
    # PCI lag features kept as v1. RUT lag features NEW, mirroring PCI_PREV.
    # -------------------------------------------------------------
    seg = df.groupby("PAVEMENT_SEGMENT_ID")

    df["SURVEY_SEQ_NO"] = seg.cumcount() + 1
    df["N_PRIOR_SURVEYS"] = df["SURVEY_SEQ_NO"] - 1

    df["PCI_PREV"] = seg["PCI_SCORE"].shift(1)
    df["PCI_CHANGE_SINCE_LAST"] = df["PCI_SCORE"] - df["PCI_PREV"]

    df["RUT_PREV"] = seg["RUT_DEPTH_1_8IN"].shift(1)
    df["RUT_CHANGE_SINCE_LAST"] = df["RUT_DEPTH_1_8IN"] - df["RUT_PREV"]

    df["DAYS_SINCE_LAST_SURVEY"] = seg["SURVEY_DATE_dt"].diff().dt.days
    df["YEARS_SINCE_LAST_SURVEY"] = df["DAYS_SINCE_LAST_SURVEY"] / 365.25
    df["ESAL_SINCE_LAST_SURVEY"] = seg["CUM_ESAL_AT_SURVEY"].diff()

    # Targets (next-survey values), computed for both measures
    df["PCI_NEXT"] = seg["PCI_SCORE"].shift(-1)
    df["RUT_NEXT"] = seg["RUT_DEPTH_1_8IN"].shift(-1)
    df["DAYS_TO_NEXT_SURVEY"] = seg["SURVEY_DATE_dt"].diff(-1).dt.days.abs()
    df["YEARS_TO_NEXT_SURVEY"] = df["DAYS_TO_NEXT_SURVEY"] / 365.25

    # Repair-threshold label for the PCI dataset (client: PCI < REPAIR_THRESHOLD needs repair)
    df["NEEDS_REPAIR"] = (df["PCI_NEXT"] < config.REPAIR_THRESHOLD).astype("Int64")
    df.loc[df["PCI_NEXT"].isna(), "NEEDS_REPAIR"] = pd.NA

    # -------------------------------------------------------------
    # STEP 4: select agreed model input variables + engineered features.
    # Identical feature engineering for both datasets — only target/filter differ.
    # -------------------------------------------------------------
    identifier_cols = list(config.IDENTIFIER_COLS)
    context_cols = list(config.CONTEXT_COLS)
    shared_numeric = list(config.SHARED_NUMERIC_BASE_COLS)
    shared_categorical = list(config.SHARED_CATEGORICAL_COLS)

    shared_base_cols = identifier_cols + context_cols + shared_numeric + shared_categorical
    shared_base_cols = [c for c in shared_base_cols if c in df.columns]

    pci_lag_cols = config.TARGETS["pci"]["lag_cols"]
    rut_lag_cols = config.TARGETS["rut"]["lag_cols"]

    # -------------------------------------------------------------
    # STEP 5a: PCI dataset — rows with distress data (PCI_SCORE present)
    # -------------------------------------------------------------
    pci_cols = shared_base_cols + [c for c in pci_lag_cols if c in df.columns] + \
        ["PCI_NEXT", "NEEDS_REPAIR", "DAYS_TO_NEXT_SURVEY", "YEARS_TO_NEXT_SURVEY"]
    pci_cols = list(dict.fromkeys([c for c in pci_cols if c in df.columns]))

    pci_df = df[pci_cols].copy()
    before = len(pci_df)
    pci_df = pci_df[pci_df["PCI_SCORE"].notna()]
    pci_df = pci_df[pci_df["PCI_NEXT"].notna()]
    print(f"PCI dataset: dropped {before - len(pci_df)} rows with no distress data / no future survey "
          f"-> {len(pci_df)} rows remain.")

    # -------------------------------------------------------------
    # STEP 5b: RUT dataset — rows with rut depth data (RUT_DEPTH_1_8IN present)
    # -------------------------------------------------------------
    rut_cols = shared_base_cols + [c for c in rut_lag_cols if c in df.columns] + \
        ["RUT_NEXT", "DAYS_TO_NEXT_SURVEY", "YEARS_TO_NEXT_SURVEY"]
    rut_cols = list(dict.fromkeys([c for c in rut_cols if c in df.columns]))

    rut_df = df[rut_cols].copy()
    before = len(rut_df)
    rut_df = rut_df[rut_df["RUT_DEPTH_1_8IN"].notna()]
    rut_df = rut_df[rut_df["RUT_NEXT"].notna()]
    print(f"RUT dataset: dropped {before - len(rut_df)} rows with no rut depth data / no future survey "
          f"-> {len(rut_df)} rows remain.")

    # -------------------------------------------------------------
    # STEP 6: ONE train/test split, grouped by physical road, applied to both
    # -------------------------------------------------------------
    all_roads = pd.Index(sorted(set(pci_df["PHYSICAL_ROAD_ID"]) | set(rut_df["PHYSICAL_ROAD_ID"])))
    road_frame = pd.DataFrame({config.PHYSICAL_ROAD_ID_COL: all_roads})

    gss = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)
    train_idx, test_idx = next(gss.split(road_frame, groups=road_frame[config.PHYSICAL_ROAD_ID_COL]))
    road_frame[config.SPLIT_COL] = config.TRAIN_VALUE
    road_frame.loc[test_idx, config.SPLIT_COL] = config.TEST_VALUE

    split_lookup = road_frame.set_index(config.PHYSICAL_ROAD_ID_COL)[config.SPLIT_COL]
    pci_df[config.SPLIT_COL] = pci_df[config.PHYSICAL_ROAD_ID_COL].map(split_lookup)
    rut_df[config.SPLIT_COL] = rut_df[config.PHYSICAL_ROAD_ID_COL].map(split_lookup)

    # -------------------------------------------------------------
    # STEP 7: verify no physical road leaks train/test, in EITHER dataset,
    # and that the two datasets agree on every road they share.
    # -------------------------------------------------------------
    for name, d in [("PCI", pci_df), ("RUT", rut_df)]:
        overlap = set(d.loc[d[config.SPLIT_COL] == config.TRAIN_VALUE, config.PHYSICAL_ROAD_ID_COL]) & \
                  set(d.loc[d[config.SPLIT_COL] == config.TEST_VALUE, config.PHYSICAL_ROAD_ID_COL])
        assert not overlap, f"{name} dataset: road leaked across train/test split!"

    shared_roads = set(pci_df[config.PHYSICAL_ROAD_ID_COL]) & set(rut_df[config.PHYSICAL_ROAD_ID_COL])
    pci_assign = (pci_df[pci_df[config.PHYSICAL_ROAD_ID_COL].isin(shared_roads)]
                  .drop_duplicates(config.PHYSICAL_ROAD_ID_COL)
                  .set_index(config.PHYSICAL_ROAD_ID_COL)[config.SPLIT_COL])
    rut_assign = (rut_df[rut_df[config.PHYSICAL_ROAD_ID_COL].isin(shared_roads)]
                  .drop_duplicates(config.PHYSICAL_ROAD_ID_COL)
                  .set_index(config.PHYSICAL_ROAD_ID_COL)[config.SPLIT_COL])
    mismatch = (pci_assign.reindex(sorted(shared_roads)) != rut_assign.reindex(sorted(shared_roads))).sum()
    assert mismatch == 0, "PCI and RUT datasets disagree on split assignment for a shared road!"
    print(f"Verified: zero train/test leakage, zero split disagreement across "
          f"{len(shared_roads)} roads shared by both datasets.")

    print("\nPCI dataset split:")
    print(pci_df[config.SPLIT_COL].value_counts())
    print("\nRUT dataset split:")
    print(rut_df[config.SPLIT_COL].value_counts())

    # -------------------------------------------------------------
    # STEP 8: export
    # -------------------------------------------------------------
    pci_df = pci_df.sort_values(["ROAD_ID", "SURVEY_SEQ_NO"]).reset_index(drop=True)
    rut_df = rut_df.sort_values(["ROAD_ID", "SURVEY_SEQ_NO"]).reset_index(drop=True)

    pci_df.to_csv(config.TARGETS["pci"]["dataset_path"], index=False)
    rut_df.to_csv(config.TARGETS["rut"]["dataset_path"], index=False)
    road_frame.to_csv(config.SPLIT_ASSIGNMENT_PATH, index=False)

    print(f"\nSaved {config.TARGETS['pci']['dataset_path']}: {pci_df.shape[0]} rows x {pci_df.shape[1]} cols")
    print(f"Saved {config.TARGETS['rut']['dataset_path']}: {rut_df.shape[0]} rows x {rut_df.shape[1]} cols")
    print(f"Saved {config.SPLIT_ASSIGNMENT_PATH}: {road_frame.shape[0]} physical roads")

    return {"pci": pci_df, "rut": rut_df}


if __name__ == "__main__":
    build_model_datasets()
