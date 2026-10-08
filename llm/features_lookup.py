"""
features_lookup.py — plain-English labels and display units for model features.

Unit policy (CLAUDE.md §3.4): a unit is only shown when it is certain.
  - thickness columns are mm in the source
  - rut columns are eighths of an inch in the source; converted to mm for display
  - IRI / MRI: LTPP standard is m/km, and the observed range (about 0.3–4.7) agrees
  - TEMP_AVG: range about −2 to 26 is only consistent with °C
  - other climate columns: unit not yet verified against the LTPP CLM tables, so
    `unit` is None and no unit is displayed
"""
from __future__ import annotations

RUT_EIGHTHS_TO_MM = 3.175

# column -> (label, display unit or None, decimals, scale applied for display)
FEATURES: dict[str, tuple[str, str | None, int, float]] = {
    "PCI_SCORE":              ("current PCI", None, 1, 1.0),
    "PCI_PREV":               ("PCI at the previous survey", None, 1, 1.0),
    "PCI_CHANGE_SINCE_LAST":  ("change in PCI since the previous survey", "points", 1, 1.0),
    "RUT_DEPTH_1_8IN":        ("current rut depth", "mm", 1, RUT_EIGHTHS_TO_MM),
    "RUT_PREV":               ("rut depth at the previous survey", "mm", 1, RUT_EIGHTHS_TO_MM),
    "RUT_CHANGE_SINCE_LAST":  ("change in rut depth since the previous survey", "mm", 1,
                               RUT_EIGHTHS_TO_MM),
    "DAYS_SINCE_LAST_SURVEY": ("days since the previous survey", "days", 0, 1.0),
    "PAVEMENT_AGE_YEARS":     ("pavement age", "years", 1, 1.0),
    "CUM_ESAL_AT_SURVEY":     ("cumulative traffic loading (ESALs)", None, 0, 1.0),
    "BASE_THICKNESS_MM":      ("base layer thickness", "mm", 0, 1.0),
    "SUBBASE_THICKNESS_MM":   ("subbase layer thickness", "mm", 0, 1.0),
    "IRI_LEFT_WHEEL_PATH":    ("roughness (IRI, left wheel path)", "m/km", 2, 1.0),
    "TEMP_AVG":               ("average annual temperature", "°C", 1, 1.0),
    "FREEZE_THAW":            ("freeze–thaw cycles (annual)", None, 0, 1.0),
    "FREEZE_INDEX":           ("freezing index", None, 0, 1.0),
    "EVAPORATION":            ("annual evaporation", None, 0, 1.0),
    "PRECIPITATION":          ("annual precipitation", None, 0, 1.0),
    "SHORTWAVE_SURFACE_AVG":  ("average solar radiation at the surface", None, 0, 1.0),
}

# One-hot encoded categoricals: prefix -> label
ONE_HOT_PREFIXES = {
    "PCI_RATING_": "current PCI rating",
    "PCI_DOMINANT_DISTRESS_": "dominant distress",
    "SURFACE_MATL_": "surface material",
    "BASE_MATL_": "base material",
    "SUBGRADE_MATL_": "subgrade material",
}


def describe(feature: str, value) -> dict:
    """Label and display value for one encoded feature at one row."""
    for prefix, label in ONE_HOT_PREFIXES.items():
        if feature.startswith(prefix):
            level = feature[len(prefix):]
            return {"label": f"{label} is {level}", "value": "yes" if value >= 0.5 else "no",
                    "unit": None}
    if feature.endswith("__isna"):
        base = feature[:-len("__isna")]
        base_label = FEATURES.get(base, (base,))[0]
        return {"label": f"{base_label} was not recorded", "value": "yes" if value >= 0.5
                else "no", "unit": None}
    label, unit, dp, scale = FEATURES.get(feature, (feature.replace("_", " ").lower(),
                                                    None, 2, 1.0))
    return {"label": label, "value": round(float(value) * scale, dp), "unit": unit}
