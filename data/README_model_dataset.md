# Temporal Model Dataset — README

**Output file:** `model_dataset.csv`
**Build script:** `build_model_dataset.py` (run: `python build_model_dataset.py`)
**Input:** `master_with_pci_2.csv` (PCI already calculated per survey by teammate)

**Rows:** 976 | **Physical roads:** 259 | **Pavement lives:** 327
**Scope:** GPS-1 and GPS-2 asphalt sections only

## What this dataset is

One row = one road survey, paired with the PCI recorded at that road's
*next* survey (`PCI_NEXT`). This is the dataset the ML team trains on to
predict future pavement condition from current condition + traffic +
climate + structure + maintenance history.

## How road identity and rebuilds are handled

- `PHYSICAL_ROAD_ID` = `STATE_CODE-SHRP_ID` — the physical location.
- `ROAD_ID` = `STATE_CODE-SHRP_ID-CONSTRUCTION_NO` — one **pavement
  life**. `CONSTRUCTION_NO` is LTPP's own marker for a structural
  rebuild/rehab event: whenever it increments, the pavement's layer
  structure changed. Using it as the grouping key means observations
  from before and after a rebuild are never treated as one continuous
  life — each construction era gets its own age-from-zero, its own
  survey sequence, and its own target column. Verified directly: e.g.
  road `16-1010` resets from 8.3 years of age (end of construction 1)
  back to 0.04 years (start of construction 2).
- 259 physical roads produce 327 pavement lives — some roads were
  rebuilt once or twice during the study period and correctly appear as
  2–3 separate lives.

**Known Excel artifact, fixed here:** the raw file has a
`PHYSICAL_ROAD_ID`-equivalent column corrupted by Excel auto-date
conversion (e.g. SHRP_ID `4-73` became `"Jan-73"`) for ~92 rows. This
script never touches that corrupted column — it builds
`PHYSICAL_ROAD_ID` fresh from `STATE_CODE`/`SHRP_ID` as plain text, so
it's clean throughout.

## Pipeline steps

1. **Scope filter** — restrict to `GPS_SPS_EXP == "General Pavement
   Studies"` and `EXPERIMENT_NO_EXP` in `{Asphalt Concrete on Unbound
   Granular Base, Asphalt Concrete on Bound Base}` (GPS-1/GPS-2). This
   drops SPS sections and other GPS experiment families, which are out
   of this project's agreed scope.
2. **Drop rows with no `PCI_SCORE`** — a survey with no PCI can't serve
   as a feature row (current condition) or a target row (next
   condition), so these are removed up front (6 rows).
3. **Sort** every road's observations by `SURVEY_DATE`.
4. **Lagged/cumulative features added**, computed per `ROAD_ID`:
   - `N_PRIOR_SURVEYS` — running count of prior surveys on this
     pavement life (0 for the first survey)
   - `PCI_PREV`, `PCI_CHANGE_SINCE_LAST`
   - `DAYS_SINCE_LAST_SURVEY`, `YEARS_SINCE_LAST_SURVEY`
   - `ESAL_SINCE_LAST_SURVEY` (traffic loading since the last survey)
5. **Target created:** `PCI_NEXT` = PCI at the next survey for that
   pavement life, plus `DAYS_TO_NEXT_SURVEY` / `YEARS_TO_NEXT_SURVEY`.
6. **Rows with no future survey dropped** — the last observation of
   each pavement life has no target (597 rows dropped here).
7. **Train/test split by physical road** (not by row, and not just by
   pavement life) — so the same physical location can never appear in
   both sets even across a rebuild. 80/20, seeded (`RANDOM_SEED = 42`)
   for reproducibility. Verified: zero physical roads appear in both
   splits.

## Model input variables included

- **Age:** `PAVEMENT_AGE_YEARS`
- **Traffic:** `CUM_ESAL_AT_SURVEY`, `ESAL_SINCE_LAST_SURVEY`
- **Climate:** `PRECIPITATION`, `EVAPORATION`, `PRECIP_DAYS`,
  `TEMP_AVG`, `FREEZE_INDEX`, `FREEZE_THAW`, `SHORTWAVE_SURFACE_AVG`
- **Structure:** `AC_THICKNESS_MM`, `BASE_THICKNESS_MM`,
  `SUBBASE_THICKNESS_MM`, `TOTAL_PAV_THICKNESS_MM`, `NUM_AC_LAYERS`,
  `NUM_LAYERS_TOTAL`, `SURFACE_MATL`, `BASE_MATL`, `SUBGRADE_MATL`
- **Maintenance:** `PRIOR_PATCH_COUNT`, `DAYS_SINCE_LAST_PATCH`
- **Current condition:** `PCI_SCORE`, `PCI_RATING`,
  `PCI_DOMINANT_DISTRESS`, plus `RUT_DEPTH_1_8IN`,
  `IRI_LEFT_WHEEL_PATH`, `IRI_RIGHT_WHEEL_PATH`, `MRI` for reference

## Quality checks performed

- Zero rows with a missing `PCI_NEXT` (by construction).
- Zero physical roads leak across train/test (checked directly).
- `PCI_SCORE` → `PCI_NEXT` correlates at ~0.70 — related but not
  identical, as expected for a forward-looking target.
- This script's output was diffed row-for-row against a hand-built
  version of this dataset: identical row count, identical pavement-life
  count, and identical feature values (differences only at
  floating-point noise level, ~1e-10).

## Known limitations (inherited from the source data, not introduced here)

- `CUM_ESAL_AT_SURVEY` missing in ~9% of rows, climate fields
  (`PRECIPITATION`, `EVAPORATION`, etc.) missing in ~7-8% — gaps in the
  upstream LTPP tables, not a pipeline bug.
- `DAYS_SINCE_LAST_PATCH` is missing for ~93% of rows — most sections
  simply have no recorded patch before that survey
  (`PRIOR_PATCH_COUNT == 0`).
- `PCI_PREV`, `PCI_CHANGE_SINCE_LAST`, `DAYS_SINCE_LAST_SURVEY`, etc.
  are NaN for exactly the first survey of each pavement life (no
  "previous" to reference) — expected, not an error.
- Categorical columns (`SURFACE_MATL`, `BASE_MATL`, `PCI_RATING`, etc.)
  are left as raw text; encoding them is the ML team's job.
