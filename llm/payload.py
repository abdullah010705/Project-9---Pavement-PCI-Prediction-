"""
payload.py — the fact payload for one road survey (Component E).

Every number an LLM brief may use is computed here, by code, from V0.0.3 data,
models and results files. The LLM receives `Payload.for_prompt()` and nothing
else. Rounding to display precision happens here, so the validator can match
figures in the output exactly.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from functools import lru_cache
from typing import Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from llm import v002
from llm.features_lookup import RUT_EIGHTHS_TO_MM, describe

PAYLOAD_VERSION = "1.0"
TOP_K_DRIVERS = 4

DISTRESS_NAMES = {"long_trans_cracking": "longitudinal and transverse cracking",
                  "alligator_cracking": "alligator (fatigue) cracking",
                  "raveling_weathering": "ravelling and weathering"}
SEVERITY_NAMES = {"L": "low", "M": "medium", "H": "high"}


def distress_label(code) -> Optional[str]:
    """'alligator_cracking (M)' -> 'alligator (fatigue) cracking, medium severity'."""
    if code is None or pd.isna(code):
        return None
    name, _, sev = str(code).partition(" (")
    label = DISTRESS_NAMES.get(name, name.replace("_", " "))
    sev = SEVERITY_NAMES.get(sev.rstrip(")"))
    return f"{label}, {sev} severity" if sev else label


RUT_SEVERITY_LABEL = {None: "below the low-severity threshold", "L": "low severity",
                      "M": "medium severity", "H": "high severity"}

# Action tiers are fixed in code. The LLM may explain a tier but never choose one.
# PROVISIONAL — rules to be confirmed with Luke / the client (llm-plan.md §9).
ACTION_TIERS = {
    "monitor": "Routine monitoring at the normal survey interval.",
    "preventive": "Consider preventive treatment and a closer inspection.",
    "corrective": "Programme a detailed inspection with a view to corrective treatment.",
    "investigate data": "Check the survey data before acting on this prediction.",
}
PREVENTIVE_PCI = 70  # below this predicted PCI (rating Fair or worse) → preventive


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class Identity(BaseModel):
    road_id: str
    physical_road_id: str
    survey_date: str  # ISO yyyy-mm-dd
    split: Literal["train", "test"]


class Condition(BaseModel):
    pci: float
    pci_rating: str
    dominant_distress: Optional[str]
    rut_depth_mm: Optional[float]
    rut_severity: Optional[str]
    iri_m_per_km: Optional[float]
    pavement_age_years: float
    years_to_next_survey: Optional[float]


class Driver(BaseModel):
    feature: str
    label: str
    value: float | str
    unit: Optional[str]
    effect: float  # contribution to this prediction, in the target's display unit
    direction: Literal["raised", "lowered"]
    value_was_filled: bool  # missing in the data; the model saw the training median


class Outlook(BaseModel):
    target: Literal["PCI", "rut depth"]
    unit: Optional[str]
    current: float
    predicted: float
    typical_error: float  # test RMSE, display unit
    range_low: float
    range_high: float
    cv_error: float  # grouped-CV RMSE over all rows, display unit
    model_average: float  # SHAP base value: the model's average prediction
    predicted_band: str  # PCI rating, or rut severity
    drivers: list[Driver]
    reliability_note: Optional[str]


class Repair(BaseModel):
    probability_pct: int
    flagged: bool
    decision_threshold_pct: int
    pci_repair_threshold: int
    test_precision_pct: int
    test_recall_pct: int
    test_repairs_caught: int
    test_repairs_total: int


class Action(BaseModel):
    tier: Literal["monitor", "preventive", "corrective", "investigate data"]
    description: str
    reasons: list[str]


class Hindsight(BaseModel):
    """Observed next-survey values. Evaluation only — never shown to the LLM."""
    pci_next_actual: Optional[float]
    rut_next_actual_mm: Optional[float]
    notes: list[str]


class Payload(BaseModel):
    payload_version: str
    generated_at: str
    identity: Identity
    condition: Condition
    pci: Optional[Outlook]
    rut: Optional[Outlook]
    repair: Optional[Repair]
    action: Action
    data_notes: list[str]
    hindsight: Hindsight

    def for_prompt(self) -> dict:
        """What the LLM is allowed to see, grouped by the output field it belongs to.

        Small local models follow a field-by-field layout far more reliably than a
        nested record. Hindsight and CV error are left out. model_average is the
        SHAP base value: driver effects are measured from it.
        """
        outlooks = [o for o in (self.pci, self.rut) if o is not None]
        return {
            "headline": {
                "road_id": self.identity.road_id,
                "predictions": [{"target": o.target, "predicted": o.predicted,
                                 "predicted_band": o.predicted_band} for o in outlooks],
                "action_tier": self.action.tier,
            },
            "condition_now": self.condition.model_dump(exclude={"years_to_next_survey"}),
            "outlook": [{"target": o.target, "unit": o.unit, "predicted": o.predicted,
                         "predicted_band": o.predicted_band, "typical_error": o.typical_error,
                         "range_low": o.range_low, "range_high": o.range_high,
                         "years_to_next_survey": self.condition.years_to_next_survey,
                         "model_average": o.model_average}
                        for o in outlooks],
            "drivers": [{"target": o.target, **d.model_dump(exclude={"feature"}),
                         "effect_unit": "PCI points" if o.target == "PCI" else o.unit}
                        for o in outlooks for d in o.drivers],
            "repair": None if self.repair is None else self.repair.model_dump(),
            "action": self.action.model_dump(),
            "caveats": self.caveats(),
        }

    def caveats(self) -> list[str]:
        notes = list(self.data_notes)
        notes += [o.reliability_note for o in (self.pci, self.rut) if o and o.reliability_note]
        notes.append("The drivers describe how the model reached its estimate; they are not "
                     "evidence of why the pavement deteriorates.")
        return notes


# ---------------------------------------------------------------------------
# Row lookup
# ---------------------------------------------------------------------------
def _parse_date(s) -> date:
    if isinstance(s, (date, datetime)):
        return s if isinstance(s, date) else s.date()
    s = str(s).strip()
    if len(s) >= 8 and s[4] == "-":  # ISO
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    return datetime.strptime(s, "%d/%m/%Y").date()  # dataset format d/m/yyyy


@lru_cache(maxsize=None)
def _row_index(target_key: str) -> dict:
    df = v002.dataset(target_key)
    keys = zip(df[v002.config.ROAD_ID_COL].astype(str),
               df[v002.config.SURVEY_DATE_COL].map(_parse_date))
    return {k: i for i, k in enumerate(keys)}


@lru_cache(maxsize=None)
def _raw_errors() -> dict:
    """(STATE_CODE, SHRP_ID, CONSTRUCTION_NO, date) -> PCI_ERROR, for rows with an error."""
    raw = v002.raw_master()
    raw = raw[raw.PCI_ERROR.notna()]
    return {(int(r.STATE_CODE), str(r.SHRP_ID), int(r.CONSTRUCTION_NO),
             _parse_date(r.SURVEY_DATE)): str(r.PCI_ERROR) for r in raw.itertuples()}


def find_row(target_key: str, road_id: str, survey_date) -> Optional[int]:
    return _row_index(target_key).get((str(road_id), _parse_date(survey_date)))


def list_surveys(split: Optional[str] = None) -> pd.DataFrame:
    """Every (ROAD_ID, SURVEY_DATE) available, with which targets cover it."""
    frames = []
    for key in v002.TARGET_KEYS:
        df = v002.dataset(key)
        f = df[[v002.config.ROAD_ID_COL, v002.config.SURVEY_DATE_COL,
                v002.config.SPLIT_COL, "PCI_SCORE"]].copy()
        f["target"] = key
        frames.append(f)
    out = pd.concat(frames)
    if split:
        out = out[out[v002.config.SPLIT_COL] == split]
    out["SURVEY_DATE_ISO"] = out[v002.config.SURVEY_DATE_COL].map(
        lambda s: _parse_date(s).isoformat())
    return (out.groupby([v002.config.ROAD_ID_COL, "SURVEY_DATE_ISO"])
               .agg(split=(v002.config.SPLIT_COL, "first"),
                    pci=("PCI_SCORE", "first"),
                    targets=("target", lambda s: "+".join(sorted(s))))
               .reset_index())


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _num(x) -> Optional[float]:
    return None if x is None or pd.isna(x) else float(x)


def _drivers(target_key: str, i: int, scale: float, dp: int) -> tuple[list[Driver], float]:
    X = v002.feature_matrix(target_key)
    raw = v002.dataset(target_key)
    contrib = v002.predict(f"xgboost_{target_key}", X.iloc[[i]], contribs=True)[0]
    base = float(contrib[-1]) * scale
    order = np.argsort(-np.abs(contrib[:-1]))[:TOP_K_DRIVERS]
    out = []
    for j in order:
        feat = X.columns[j]
        d = describe(feat, X.iloc[i, j])
        filled = feat in raw.columns and pd.isna(raw.at[i, feat])
        effect = round(float(contrib[j]) * scale, dp)
        if effect == 0:
            continue
        out.append(Driver(feature=feat, label=d["label"], value=d["value"], unit=d["unit"],
                          effect=abs(effect), direction="raised" if effect > 0 else "lowered",
                          value_was_filled=bool(filled)))
    return out, base


def _outlook(target_key: str, i: int) -> Outlook:
    row = v002.dataset(target_key).iloc[i]
    X = v002.feature_matrix(target_key)
    pred_raw = float(v002.predict(f"xgboost_{target_key}", X.iloc[[i]])[0])
    err = v002.model_error(target_key)

    if target_key == "pci":
        scale, dp, unit, lo, hi = 1.0, 1, None, 0.0, 100.0
        current = float(row["PCI_SCORE"])
        pred = min(max(pred_raw, lo), hi)
        band = v002.rating_from_pci(pred)
        note = None  # PCI vs baseline claims held back until CLAUDE.md §6.1 is settled
    else:
        scale, dp, unit, lo, hi = RUT_EIGHTHS_TO_MM, 1, "mm", 0.0, None
        current = float(row["RUT_DEPTH_1_8IN"])
        pred = max(pred_raw, lo)
        band = RUT_SEVERITY_LABEL[v002.classify_rut_severity(pred)]
        note = None
        if err["bootstrap_verdict"] != "XGBoost better":
            note = ("On held-out data this rutting model was not reliably more accurate "
                    "than simply assuming rut depth stays as it is now.")

    drivers, base = _drivers(target_key, i, scale, dp)
    pred_d, rmse_d = pred * scale, err["test_rmse"] * scale
    low = max(pred_d - rmse_d, lo)
    high = pred_d + rmse_d if hi is None else min(pred_d + rmse_d, hi)
    return Outlook(
        target="PCI" if target_key == "pci" else "rut depth", unit=unit,
        current=round(current * scale, dp), predicted=round(pred_d, dp),
        typical_error=round(rmse_d, dp), range_low=round(low, dp), range_high=round(high, dp),
        cv_error=round(err["cv_rmse"] * scale, dp), model_average=round(base, dp),
        predicted_band=band, drivers=drivers, reliability_note=note)


def _repair(i: int) -> Repair:
    X = v002.feature_matrix("pci")
    p = float(v002.predict("repair_classifier", X.iloc[[i]])[0])
    s = v002.classifier_summary()
    t = s["test"]
    return Repair(
        probability_pct=int(round(100 * p)), flagged=p >= 0.5, decision_threshold_pct=50,
        pci_repair_threshold=int(s["threshold_pci"]),
        test_precision_pct=int(round(100 * t["precision"])),
        test_recall_pct=int(round(100 * t["recall"])),
        test_repairs_caught=int(t["tp"]), test_repairs_total=int(t["tp"] + t["fn"]))


def _pci_error(row) -> Optional[str]:
    return _raw_errors().get((int(row["STATE_CODE"]), str(row["SHRP_ID"]),
                              int(row["CONSTRUCTION_NO"]), _parse_date(row["SURVEY_DATE"])))


def _action(pci: Optional[Outlook], rut: Optional[Outlook], repair: Optional[Repair],
            data_problem: bool) -> Action:
    reasons = []
    if data_problem:
        return Action(tier="investigate data", description=ACTION_TIERS["investigate data"],
                      reasons=["the PCI calculation for this survey reported an error"])
    tier = "monitor"
    if repair and repair.flagged:
        tier = "corrective"
        reasons.append("the repair classifier flags this survey")
    if pci and pci.predicted < (repair.pci_repair_threshold if repair else 55):
        tier = "corrective"
        reasons.append("predicted PCI is below the repair threshold")
    if rut and rut.predicted_band == "high severity":
        tier = "corrective"
        reasons.append("predicted rutting is in the high-severity band")
    if tier == "monitor":
        if pci and pci.predicted < PREVENTIVE_PCI:
            tier = "preventive"
            reasons.append("predicted PCI is in the Fair band or below")
        if rut and rut.predicted_band == "medium severity":
            tier = "preventive"
            reasons.append("predicted rutting is in the medium-severity band")
    if tier == "monitor":
        reasons.append("no prediction crosses a treatment trigger")
    return Action(tier=tier, description=ACTION_TIERS[tier], reasons=reasons)


def build_payload(road_id: str, survey_date) -> Payload:
    rows = {k: find_row(k, road_id, survey_date) for k in v002.TARGET_KEYS}
    if all(v is None for v in rows.values()):
        raise KeyError(f"no survey for ROAD_ID={road_id} on {survey_date}")

    src_key = "pci" if rows["pci"] is not None else "rut"
    row = v002.dataset(src_key).iloc[rows[src_key]]

    pci = _outlook("pci", rows["pci"]) if rows["pci"] is not None else None
    rut = _outlook("rut", rows["rut"]) if rows["rut"] is not None else None
    repair = _repair(rows["pci"]) if rows["pci"] is not None else None

    notes = []
    if pd.isna(row.get("DAYS_SINCE_LAST_SURVEY")):
        notes.append("This is the first survey of this pavement life, so change-since-last-"
                     "survey inputs were not available and were filled with typical values.")
    if pci is None:
        notes.append("No PCI prediction: this survey is not in the PCI modelling dataset.")
    if rut is None:
        notes.append("No rutting prediction: this survey is not in the rutting dataset.")
    pci_err = _pci_error(row)
    if pci_err:
        notes.append(f"The PCI calculation for this survey reported an error: {pci_err}.")

    rut_now = _num(row.get("RUT_DEPTH_1_8IN"))
    condition = Condition(
        pci=round(float(row["PCI_SCORE"]), 1), pci_rating=str(row["PCI_RATING"]),
        dominant_distress=distress_label(row.get("PCI_DOMINANT_DISTRESS")),
        rut_depth_mm=None if rut_now is None else round(rut_now * RUT_EIGHTHS_TO_MM, 1),
        rut_severity=None if rut_now is None
        else RUT_SEVERITY_LABEL[v002.classify_rut_severity(rut_now)],
        iri_m_per_km=None if _num(row.get("IRI_LEFT_WHEEL_PATH")) is None
        else round(float(row["IRI_LEFT_WHEEL_PATH"]), 2),
        pavement_age_years=round(float(row["PAVEMENT_AGE_YEARS"]), 1),
        years_to_next_survey=None if _num(row.get("YEARS_TO_NEXT_SURVEY")) is None
        else round(float(row["YEARS_TO_NEXT_SURVEY"]), 1))

    # Hindsight (evaluation only)
    h_notes = []
    pci_next = (_num(v002.dataset("pci").at[rows["pci"], "PCI_NEXT"])
                if rows["pci"] is not None else None)
    rut_next = (_num(v002.dataset("rut").at[rows["rut"], "RUT_NEXT"])
                if rows["rut"] is not None else None)
    if pci_next is not None and pci_next <= 10 and condition.pci >= 70:
        h_notes.append("Observed next PCI is implausibly low given current condition: "
                       "possible unrecorded rebuild or PCI calculation failure. "
                       "Treat as a possible data artefact, not a confirmed failure.")

    return Payload(
        payload_version=PAYLOAD_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        identity=Identity(road_id=str(row[v002.config.ROAD_ID_COL]),
                          physical_road_id=str(row[v002.config.PHYSICAL_ROAD_ID_COL]),
                          survey_date=_parse_date(row[v002.config.SURVEY_DATE_COL]).isoformat(),
                          split=row[v002.config.SPLIT_COL]),
        condition=condition, pci=pci, rut=rut, repair=repair,
        action=_action(pci, rut, repair, data_problem=pci_err is not None),
        data_notes=notes,
        hindsight=Hindsight(
            pci_next_actual=None if pci_next is None else round(pci_next, 1),
            rut_next_actual_mm=None if rut_next is None
            else round(rut_next * RUT_EIGHTHS_TO_MM, 1),
            notes=h_notes))
