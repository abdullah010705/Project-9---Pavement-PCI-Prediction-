"""
build_dataset.py — dataset construction stage (owner: Rayyan).

ASSEMBLED, NOT AUTHORED: this is Rayyan's build_model_dataset.py, wrapped in a
build_model_dataset() function and pointed at config.py paths instead of hardcoded
filenames, so run_pipeline.py can call it. The logic is unchanged — Rayyan should
review and take ownership.

Turns data/master_with_pci_2.csv into data/model_dataset.csv: a chronological,
leak-safe, current-conditions -> future-PCI modelling dataset, scoped to GPS-1/GPS-2.
"""
import os

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import config

RANDOM_STATE = config.RANDOM_SEED
TEST_SIZE = config.TEST_SIZE


def build_model_dataset():
    """Build model_dataset.csv from master_with_pci_2.csv. Returns the DataFrame."""
    # ---------------------------------------------------------------------
    # STEP 0: load + scope to GPS-1/GPS-2
    # ---------------------------------------------------------------------
    df = pd.read_csv(config.RAW_DATA_PATH, low_memory=False)
    df['SURVEY_DATE_dt'] = pd.to_datetime(df['SURVEY_DATE'])  # already ISO YYYY-MM-DD
    print(f"Loaded: {df.shape}")

    gps12_labels = ['Asphalt Concrete on Unbound Granular Base', 'Asphalt Concrete on Bound Base']
    before_scope = len(df)
    df = df[(df['GPS_SPS_EXP'] == 'General Pavement Studies') &
            (df['EXPERIMENT_NO_EXP'].isin(gps12_labels))].copy()
    print(f"Scope filter (GPS-1/GPS-2 only): {before_scope} -> {len(df)} rows, "
          f"{df.groupby(['STATE_CODE','SHRP_ID','CONSTRUCTION_NO']).ngroups} road segments")

    # ---------------------------------------------------------------------
    # STEP 1: unique road id + chronological sort
    # ---------------------------------------------------------------------
    df['ROAD_ID'] = (df['STATE_CODE'].astype(str) + '-' +
                      df['SHRP_ID'].astype(str) + '-' +
                      df['CONSTRUCTION_NO'].astype(str))
    df['PHYSICAL_ROAD_ID'] = df['STATE_CODE'].astype(str) + '-' + df['SHRP_ID'].astype(str)

    df = df.sort_values(['ROAD_ID', 'SURVEY_DATE_dt']).reset_index(drop=True)

    # ---------------------------------------------------------------------
    # STEP 2: rebuild/reconstruction segmentation
    # ---------------------------------------------------------------------
    # CONSTRUCTION_NO already gives one continuous pavement life per segment
    # (LTPP assigns a new one whenever a section is rebuilt/re-based). Also
    # check for undocumented resets (pavement age going backwards) within a
    # CONSTRUCTION_NO as a safety net.
    df['_age_prev'] = df.groupby('ROAD_ID')['PAVEMENT_AGE_YEARS'].shift(1)
    df['_age_reset'] = (df['PAVEMENT_AGE_YEARS'] < df['_age_prev'] - 0.5).fillna(False)
    df['SEGMENT_NO'] = df.groupby('ROAD_ID')['_age_reset'].cumsum()
    print(f"Undocumented age-reset splits detected within a CONSTRUCTION_NO: {df['_age_reset'].sum()}")

    df['PAVEMENT_SEGMENT_ID'] = df['ROAD_ID'] + '-' + df['SEGMENT_NO'].astype(str)
    df = df.drop(columns=['_age_prev', '_age_reset'])

    # ---------------------------------------------------------------------
    # STEP 3: lagged / cumulative features (within each pavement segment)
    # ---------------------------------------------------------------------
    seg = df.groupby('PAVEMENT_SEGMENT_ID')

    df['SURVEY_SEQ_NO'] = seg.cumcount() + 1
    df['N_PRIOR_SURVEYS'] = df['SURVEY_SEQ_NO'] - 1

    df['PCI_PREV'] = seg['PCI_SCORE'].shift(1)
    df['PCI_CHANGE_SINCE_LAST'] = df['PCI_SCORE'] - df['PCI_PREV']

    df['DAYS_SINCE_LAST_SURVEY'] = seg['SURVEY_DATE_dt'].diff().dt.days
    df['YEARS_SINCE_LAST_SURVEY'] = df['DAYS_SINCE_LAST_SURVEY'] / 365.25

    df['ESAL_SINCE_LAST_SURVEY'] = seg['CUM_ESAL_AT_SURVEY'].diff()

    df['PCI_NEXT'] = seg['PCI_SCORE'].shift(-1)
    df['DAYS_TO_NEXT_SURVEY'] = seg['SURVEY_DATE_dt'].diff(-1).dt.days.abs()
    df['YEARS_TO_NEXT_SURVEY'] = df['DAYS_TO_NEXT_SURVEY'] / 365.25

    # ---------------------------------------------------------------------
    # STEP 4: select agreed model input variables + engineered features
    # ---------------------------------------------------------------------
    identifier_cols = [
        'ROAD_ID', 'PHYSICAL_ROAD_ID', 'PAVEMENT_SEGMENT_ID',
        'STATE_CODE', 'STATE_NAME', 'SHRP_ID', 'CONSTRUCTION_NO',
        'SURVEY_DATE', 'SURVEY_YEAR', 'SURVEY_SEQ_NO',
    ]
    context_cols = [
        'PAVEMENT_FAMILY', 'PAVEMENT_FAMILY_EXP', 'FUNC_CLASS_EXP',
        'LTPP_DIR_EXP', 'LTPP_LANE', 'GPS_SPS_EXP', 'EXPERIMENT_NO_EXP',
    ]
    age_cols = ['PAVEMENT_AGE_YEARS']
    traffic_cols = ['CUM_ESAL_AT_SURVEY', 'ESAL_SINCE_LAST_SURVEY']
    climate_cols = ['PRECIPITATION', 'EVAPORATION', 'PRECIP_DAYS', 'TEMP_AVG',
                    'FREEZE_INDEX', 'FREEZE_THAW', 'SHORTWAVE_SURFACE_AVG']
    structure_cols = ['AC_THICKNESS_MM', 'BASE_THICKNESS_MM', 'SUBBASE_THICKNESS_MM',
                       'TOTAL_PAV_THICKNESS_MM', 'NUM_AC_LAYERS', 'NUM_LAYERS_TOTAL',
                       'SURFACE_MATL', 'BASE_MATL', 'SUBGRADE_MATL']
    maintenance_cols = ['PRIOR_PATCH_COUNT', 'DAYS_SINCE_LAST_PATCH', 'CN_CHANGE_REASON_EXP']
    current_condition_cols = ['PCI_SCORE', 'PCI_RATING', 'PCI_DOMINANT_DISTRESS',
                               'RUT_DEPTH_1_8IN', 'IRI_LEFT_WHEEL_PATH',
                               'IRI_RIGHT_WHEEL_PATH', 'MRI']
    lag_cols = ['N_PRIOR_SURVEYS', 'PCI_PREV', 'PCI_CHANGE_SINCE_LAST',
                'DAYS_SINCE_LAST_SURVEY', 'YEARS_SINCE_LAST_SURVEY']
    target_cols = ['PCI_NEXT', 'DAYS_TO_NEXT_SURVEY', 'YEARS_TO_NEXT_SURVEY']

    final_cols = (identifier_cols + context_cols + age_cols + traffic_cols +
                  climate_cols + structure_cols + maintenance_cols +
                  current_condition_cols + lag_cols + target_cols)
    final_cols = [c for c in final_cols if c in df.columns]
    model_df = df[final_cols].copy()

    # ---------------------------------------------------------------------
    # STEP 5: drop rows with no target / no usable current PCI
    # ---------------------------------------------------------------------
    before = len(model_df)
    model_df = model_df[model_df['PCI_NEXT'].notna()]
    model_df = model_df[model_df['PCI_SCORE'].notna()]
    print(f"Dropped {before - len(model_df)} rows with no future survey / no current PCI "
          f"-> {len(model_df)} rows remain.")

    # ---------------------------------------------------------------------
    # STEP 6: leak-safe train/test split (group by physical road)
    # ---------------------------------------------------------------------
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(model_df, groups=model_df['PHYSICAL_ROAD_ID']))
    model_df['SPLIT'] = 'train'
    model_df.iloc[test_idx, model_df.columns.get_loc('SPLIT')] = 'test'

    overlap = set(model_df.loc[model_df.SPLIT == 'train', 'PHYSICAL_ROAD_ID']) & \
              set(model_df.loc[model_df.SPLIT == 'test', 'PHYSICAL_ROAD_ID'])
    assert not overlap, "Road leaked across train/test split!"

    print(model_df['SPLIT'].value_counts())
    print(f"Unique physical roads - train: "
          f"{model_df.loc[model_df.SPLIT=='train','PHYSICAL_ROAD_ID'].nunique()}, "
          f"test: {model_df.loc[model_df.SPLIT=='test','PHYSICAL_ROAD_ID'].nunique()}")

    # ---------------------------------------------------------------------
    # STEP 7: export
    # ---------------------------------------------------------------------
    model_df = model_df.sort_values(['ROAD_ID', 'SURVEY_SEQ_NO']).reset_index(drop=True)
    model_df.to_csv(config.MODEL_DATASET_PATH, index=False)
    print(f"\nSaved model_dataset.csv: {model_df.shape[0]} rows x {model_df.shape[1]} cols")


    return model_df


def run():
    """Pipeline entry point."""
    return build_model_dataset()


if __name__ == "__main__":
    build_model_dataset()
