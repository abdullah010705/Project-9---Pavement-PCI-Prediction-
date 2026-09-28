"""
ASTM D6433-20 Pavement Condition Index (PCI) Calculator
==========================================================
Adapted for LTPP-style flexible (AC) pavement per-sample-unit distress
records (HPMS16 / MEPDG / GATOR_CRACK / BLK_CRACK / ... field schema).

Reads a CSV of records (e.g. master.csv) and writes a copy with PCI,
rating, and diagnostic columns appended - one row in, one row out.

----------------------------------------------------------------------
WHY THIS FILE LOOKS THE WAY IT DOES (read before using in production)
----------------------------------------------------------------------
ASTM D6433 computes PCI in four steps for each sample unit:

  1. Convert each distress-type/severity quantity to a % DENSITY
     (quantity / sample-unit area x 100).
  2. Look up a DEDUCT VALUE (DV) for each density from a curve unique
     to that distress type and severity (Appendix X3, asphalt).
  3. Combine the individual deduct values into a single CORRECTED
     DEDUCT VALUE (CDV) using an iterative q/"allowable deducts"
     procedure (Section 9.5) and a CORRECTION CURVE (also Appendix X3).
  4. PCI = 100 - max(CDV), optionally refined by the multi-severity
     correction in 9.6.2.

Steps 1, 3 and 4 are procedural/arithmetic and are implemented here
EXACTLY as specified in the standard (verified against the uploaded
ASTM D6433-20 text, Sections 9.1-9.6).

Step 2 (and the correction curve in step 3) are defined in the
standard as ~20 hand-drawn GRAPHS (Appendix X3), not equations or
tables. The uploaded PDF excerpt stopped partway through Appendix X1
and did not include Appendix X3, and the full standard is
copyrighted ASTM content that could not be retrieved as machine-
readable curve data. DEDUCT_CURVE_PARAMS below is therefore a
clearly-labelled, EDITABLE, order-of-magnitude APPROXIMATION of the
shape of those curves (a saturating exponential per distress/
severity), not a digitization of the official figures.

==>  For compliance-grade PCI values, replace DEDUCT_CURVE_PARAMS /
     CORRECTION_CURVE_PARAMS with points digitized from the actual
     ASTM D6433 Appendix X3 figures (or an equivalent licensed
     source such as MicroPAVER). The rest of the algorithm
     (sections 9.5-9.6, the CDV iteration, multi-severity
     correction, and PCI = 100 - max CDV) does not change.
----------------------------------------------------------------------
"""

from __future__ import annotations
import argparse
import csv
import math
import json
import sys
from typing import Optional


# ======================================================================
# 1. DISTRESS CATALOGUE  (ASTM D6433 Appendix X1 - asphalt distresses)
# ======================================================================
# key -> (display name, quantity type)
AC_DISTRESS_TYPES = {
    "alligator_cracking":       ("Alligator (Fatigue) Cracking", "area"),
    "bleeding":                 ("Bleeding", "area"),
    "block_cracking":           ("Block Cracking", "area"),
    "bumps_sags":                ("Bumps and Sags", "count"),
    "corrugation":               ("Corrugation", "area"),
    "depression":                 ("Depression", "area"),
    "edge_cracking":             ("Edge Cracking", "length"),
    "joint_reflective_cracking": ("Joint Reflective Cracking", "length"),
    "lane_shoulder_dropoff":     ("Lane/Shoulder Drop Off", "length"),
    "long_trans_cracking":       ("Longitudinal & Transverse Cracking", "length"),
    "patching":                  ("Patching and Utility Cut Patching", "area"),
    "polished_aggregate":        ("Polished Aggregate", "area"),
    "potholes":                  ("Potholes", "count"),
    "railroad_crossing":         ("Railroad Crossing", "area"),
    "rutting":                   ("Rutting", "area"),
    "shoving":                   ("Shoving", "area"),
    "slippage_cracking":         ("Slippage Cracking", "area"),
    "swell":                     ("Swell", "area"),
    "raveling_weathering":       ("Weathering and Raveling", "area"),
}

SEVERITIES = ("L", "M", "H")


# ======================================================================
# 2. DEDUCT CURVE APPROXIMATION  (see module docstring - REPLACE ME)
# ======================================================================
DEDUCT_CURVE_PARAMS = {
    "alligator_cracking":       {"L": (35, 0.05), "M": (60, 0.06), "H": (85, 0.08)},
    "block_cracking":           {"L": (25, 0.05), "M": (45, 0.06), "H": (65, 0.07)},
    "edge_cracking":            {"L": (15, 0.08), "M": (30, 0.09), "H": (45, 0.10)},
    "long_trans_cracking":      {"L": (12, 0.10), "M": (24, 0.10), "H": (40, 0.12)},
    "patching":                 {"L": (14, 0.08), "M": (28, 0.09), "H": (40, 0.10)},
    "potholes":                 {"L": (20, 0.15), "M": (45, 0.18), "H": (75, 0.20)},
    "rutting":                  {"L": (15, 0.20), "M": (35, 0.22), "H": (60, 0.25)},
    "shoving":                  {"L": (18, 0.10), "M": (32, 0.11), "H": (48, 0.12)},
    "bleeding":                 {"L": (10, 0.10), "M": (18, 0.10), "H": (25, 0.10)},
    "polished_aggregate":       {"L": (8,  0.10), "M": (12, 0.10), "H": (15, 0.10)},
    "raveling_weathering":      {"L": (10, 0.05), "M": (20, 0.06), "H": (35, 0.08)},
    "bumps_sags":                {"L": (10, 0.10), "M": (20, 0.10), "H": (30, 0.10)},
    "corrugation":               {"L": (10, 0.10), "M": (20, 0.10), "H": (30, 0.10)},
    "depression":                {"L": (8,  0.08), "M": (16, 0.08), "H": (28, 0.08)},
    "joint_reflective_cracking": {"L": (12, 0.10), "M": (24, 0.10), "H": (40, 0.12)},
    "lane_shoulder_dropoff":     {"L": (12, 0.08), "M": (24, 0.09), "H": (36, 0.10)},
    "railroad_crossing":         {"L": (15, 0.10), "M": (30, 0.10), "H": (45, 0.10)},
    "slippage_cracking":         {"L": (15, 0.10), "M": (28, 0.10), "H": (40, 0.10)},
    "swell":                     {"L": (10, 0.08), "M": (20, 0.08), "H": (32, 0.08)},
}


def deduct_value(distress_key: str, severity: str, density_pct: float) -> float:
    """Look up the deduct value for one distress/severity/density.

    Replace this body with a real interpolation over digitized ASTM
    Appendix X3 points (see `load_curve_points_from_csv` below) for a
    standards-compliant result.
    """
    if density_pct <= 0:
        return 0.0
    density_pct = min(density_pct, 100.0)
    dmax, k = DEDUCT_CURVE_PARAMS[distress_key][severity]
    return round(dmax * (1 - math.exp(-k * density_pct)), 2)


def load_curve_points_from_csv(path: str) -> dict:
    """Optional: load exact digitized curves to override the approximation.

    Expected CSV columns: distress_key,severity,density_pct,deduct_value
    """
    pts: dict = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            d, s = row["distress_key"], row["severity"]
            pts.setdefault(d, {}).setdefault(s, []).append(
                (float(row["density_pct"]), float(row["deduct_value"]))
            )
    for d in pts:
        for s in pts[d]:
            pts[d][s].sort()
    return pts


# ======================================================================
# 3. CORRECTION CURVE APPROXIMATION (Section 9.5.5.3 - see docstring)
# ======================================================================
def corrected_deduct_value(total_deduct_value: float, q: int) -> float:
    """Approximate ASTM correction-curve family: CDV(TDV, q).

    Exact behaviour required by the standard and preserved here:
      * q == 1  ->  CDV == TDV  (identity; no correction applies, 9.5.1)
      * CDV increases monotonically with TDV
      * CDV decreases monotonically with q for a fixed TDV
    The specific curvature is an approximation (see module docstring).
    """
    if q <= 1:
        return round(total_deduct_value, 2)
    cdv = total_deduct_value / (1 + 0.15 * math.log(q))
    return round(min(cdv, 100.0), 2)


# ======================================================================
# 4. SECTION 9.5 - MAXIMUM CDV ITERATION  (exact per ASTM text)
# ======================================================================
def max_allowable_deducts(hdv: float) -> float:
    """Eq 4: m = 1 + (9/98)(100 - HDV), capped at 10."""
    return min(1 + (9.0 / 98.0) * (100 - hdv), 10.0)


def compute_max_cdv(individual_deducts: list[float]) -> dict:
    """Implements ASTM D6433 9.5.1 - 9.5.5 exactly."""
    deducts = sorted([d for d in individual_deducts if d > 0], reverse=True)

    if len([d for d in deducts if d > 2]) <= 1:
        total = round(sum(deducts), 2)
        return {"max_cdv": total, "trace": [{"note": "9.5.1 shortcut", "cdv": total}]}

    working = deducts[:]
    cdv_candidates = []
    trace = []
    while True:
        hdv = working[0]
        m = max_allowable_deducts(hdv)
        m_floor = int(math.floor(m))
        frac = m - m_floor

        kept = working[:m_floor] if m_floor <= len(working) else working[:]
        if m_floor < len(working) and frac > 0:
            kept = working[:m_floor] + [round(working[m_floor] * frac, 2)]

        total_deduct_value = round(sum(kept), 2)
        q = len([d for d in kept if d > 2])
        cdv = corrected_deduct_value(total_deduct_value, q)
        cdv_candidates.append(cdv)
        trace.append({
            "highest_deduct": hdv, "m": round(m, 2), "kept_deducts": kept,
            "total_deduct_value": total_deduct_value, "q": q, "cdv": cdv,
        })

        idxs_gt2 = [i for i, d in enumerate(working) if d > 2]
        if len(idxs_gt2) <= 1:
            break
        working[idxs_gt2[-1]] = 2.0
        working.sort(reverse=True)

    max_cdv = max(cdv_candidates)
    return {"max_cdv": max_cdv, "trace": trace}


# ======================================================================
# 5. SECTION 9.6.2 - MULTI-SEVERITY CORRECTION (optional refinement)
# ======================================================================
def best_pci_over_severity_pools(distress_key: str, densities: dict,
                                   other_deducts: list[float]) -> tuple[float, dict]:
    """9.6.2: pool lower severities into the highest one if that yields
    a higher PCI; implements the general N-severity pooling procedure.
    """
    present = {s: d for s, d in densities.items() if d and d > 0}
    if len(present) <= 1:
        deducts = [deduct_value(distress_key, s, d) for s, d in present.items()]
        result = compute_max_cdv(deducts + other_deducts)
        return round(100 - result["max_cdv"], 2), present

    order = ["L", "M", "H"]
    candidates = [dict(present)]

    keys_present = [s for s in order if s in present]
    for cut in range(1, len(keys_present)):
        pooled = dict(present)
        lower = keys_present[:cut]
        target = keys_present[cut]
        pooled[target] = sum(pooled.pop(s) for s in lower) + pooled[target]
        candidates.append(pooled)

    best_pci = -1.0
    best_combo = present
    for combo in candidates:
        deducts = [deduct_value(distress_key, s, d) for s, d in combo.items()]
        result = compute_max_cdv(deducts + other_deducts)
        pci = round(100 - result["max_cdv"], 2)
        if pci > best_pci:
            best_pci, best_combo = pci, combo

    return best_pci, best_combo


# ======================================================================
# 6. FIELD MAPPING  -  LTPP schema  ->  ASTM distress quantities
# ======================================================================
NON_PCI_REFERENCE_FIELDS = [
    "HPMS16_CRACKING_PERCENT_AC", "MEPDG_CRACKING_PERCENT_AC",
    "MEPDG_TRANS_CRACK_LENGTH_AC", "MEPDG_LONG_CRACK_LENGTH_AC",
    "ME_PERCENT_WHEEL_PATH_CRACK", "WP_LENGTH_CRACKED",
    "TRANS_CRACK_L_GT183", "LONGIGATOR_CRACKING", "BLOCKVERSE_CRACKING",
    "OTHER", "PUMPING_NO", "PUMPING_L",
    "IRI_LEFT_WHEEL_PATH", "IRI_RIGHT_WHEEL_PATH", "MRI",
]


def _get(rec: dict, key: str) -> float:
    v = rec.get(key)
    if v is None:
        return 0.0
    if isinstance(v, str) and v.strip() == "":
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def map_record_to_quantities(rec: dict) -> dict:
    """Convert one LTPP-style row into {distress_key: {"L":q,"M":q,"H":q}}
    raw quantities (area m^2 / length m / count), per Section 9.1.
    """
    q: dict = {k: {"L": 0.0, "M": 0.0, "H": 0.0} for k in AC_DISTRESS_TYPES}

    for sev, suf in zip(SEVERITIES, ("_L", "_M", "_H")):
        q["alligator_cracking"][sev] = _get(rec, f"GATOR_CRACK_A{suf}")
        q["block_cracking"][sev] = _get(rec, f"BLK_CRACK_A{suf}")
        q["edge_cracking"][sev] = _get(rec, f"EDGE_CRACK_L{suf}")
        q["patching"][sev] = _get(rec, f"PATCH_A{suf}")
        q["potholes"][sev] = _get(rec, f"POTHOLES_NO{suf}")

        q["long_trans_cracking"][sev] = (
            _get(rec, f"LONG_CRACK_WP_L{suf}")
            + _get(rec, f"LONG_CRACK_NWP_L{suf}")
            + _get(rec, f"LONG_CRACK_WP_SEAL_L{suf}")
            + _get(rec, f"LONG_CRACK_NWP_SEAL_L{suf}")
            + _get(rec, f"TRANS_CRACK_L{suf}")
            + _get(rec, f"TRANS_CRACK_SEAL_L{suf}")
        )

    # single-value fields (no L/M/H breakdown in this schema) - assigned
    # to Medium severity by default; override if you have a real split.
    q["shoving"]["M"] = _get(rec, "SHOVING_A")
    q["bleeding"]["M"] = _get(rec, "BLEEDING")
    q["polished_aggregate"]["M"] = _get(rec, "POLISH_AGG_A")
    q["raveling_weathering"]["M"] = _get(rec, "RAVELING")

    # --- v2 CHANGE (Sept 2026) ---------------------------------------
    # Client decision: PCI and rutting are now two separate condition
    # measurements, each with its own dataset and model. Rutting no
    # longer contributes a deduct here — PCI becomes a distress-only
    # score. RUT_DEPTH_1_8IN is still read elsewhere (it feeds the
    # separate rutting dataset/model), it just no longer enters this
    # PCI calculation. classify_rut_severity() is kept below since the
    # rutting dataset build re-uses it for severity labelling.
    # -------------------------------------------------------------------

    return q


def classify_rut_severity(rut_1_8in: float) -> Optional[str]:
    """L: 0.25-0.5 in (2-<4 eighths); M: 0.5-1.0 in (4-<8 eighths);
    H: >1.0 in (>=8 eighths)."""
    if rut_1_8in is None or rut_1_8in < 2:
        return None
    if rut_1_8in < 4:
        return "L"
    if rut_1_8in < 8:
        return "M"
    return "H"


# ======================================================================
# 7. TOP-LEVEL DRIVER
# ======================================================================
DEFAULT_LTPP_SECTION_LENGTH_M = 152.4  # standard LTPP test-section length (500 ft)


def sample_unit_area_m2(rec: dict, section_length_m: float = DEFAULT_LTPP_SECTION_LENGTH_M) -> float:
    width = _get(rec, "SURVEY_WIDTH")
    if width <= 0:
        raise ValueError("SURVEY_WIDTH missing or zero - cannot compute sample-unit area")
    return width * section_length_m


def compute_pci(rec: dict, section_length_m: float = DEFAULT_LTPP_SECTION_LENGTH_M,
                 apply_multi_severity_correction: bool = True) -> dict:
    """End-to-end ASTM D6433 Section 9 PCI calculation for one AC
    sample-unit record.
    """
    notes = []
    area = sample_unit_area_m2(rec, section_length_m)
    qty = map_record_to_quantities(rec)

    densities: dict = {}
    deducts: dict = {}
    individual_deducts: list[float] = []

    for dkey, sev_qty in qty.items():
        densities[dkey] = {}
        for sev, val in sev_qty.items():
            if val == "FULL":
                dens = 100.0
            else:
                dens = round((val / area) * 100.0, 3) if val else 0.0
            densities[dkey][sev] = dens
            if dens > 0:
                dv = deduct_value(dkey, sev, dens)
                deducts.setdefault(dkey, {})[sev] = dv
                individual_deducts.append(dv)

    if apply_multi_severity_correction:
        for dkey, sev_map in list(densities.items()):
            present = {s: d for s, d in sev_map.items() if d > 0}
            if len(present) > 1:
                others = [
                    dv for k, sevs in deducts.items() if k != dkey
                    for dv in sevs.values()
                ]
                _, best_combo = best_pci_over_severity_pools(dkey, present, others)
                if best_combo != present:
                    notes.append(
                        f"9.6.2 correction applied to '{dkey}': pooled {present} -> {best_combo}"
                    )
                    deducts[dkey] = {
                        s: deduct_value(dkey, s, d) for s, d in best_combo.items()
                    }
                    individual_deducts = [
                        dv for sevs in deducts.values() for dv in sevs.values()
                    ]

    if not individual_deducts:
        return {
            "pci": 100.0, "rating": rating_from_pci(100.0), "area_m2": round(area, 1),
            "densities": densities, "deduct_values": deducts,
            "max_cdv_trace": [], "notes": notes + ["No distress recorded - PCI = 100"],
        }

    cdv_result = compute_max_cdv(individual_deducts)
    pci = round(max(0.0, 100 - cdv_result["max_cdv"]), 1)

    return {
        "pci": pci,
        "rating": rating_from_pci(pci),
        "area_m2": round(area, 1),
        "densities": densities,
        "deduct_values": deducts,
        "max_cdv_trace": cdv_result["trace"],
        "notes": notes,
    }


def rating_from_pci(pci: float) -> str:
    """Fig. 1 rating scale."""
    bands = [
        (100, 85, "Good"), (85, 70, "Satisfactory"), (70, 55, "Fair"),
        (55, 40, "Poor"), (40, 25, "Very Poor"), (25, 10, "Serious"),
        (10, 0, "Failed"),
    ]
    for hi, lo, name in bands:
        if lo <= pci <= hi:
            return name
    return "Failed"


# ======================================================================
# 8. CSV BATCH PROCESSING  -  reads master.csv, writes PCI per row
# ======================================================================
def dominant_distress(result: dict) -> str:
    """Convenience column: the single distress/severity contributing the
    largest individual deduct value for this row (quick triage aid)."""
    best_key, best_dv = "", -1.0
    for dkey, sevs in result["deduct_values"].items():
        for sev, dv in sevs.items():
            if dv > best_dv:
                best_key, best_dv, best_sev = dkey, dv, sev
    return f"{best_key} ({best_sev})" if best_dv > 0 else ""


def process_csv(input_path: str, output_path: str,
                 section_length_m: float = DEFAULT_LTPP_SECTION_LENGTH_M,
                 apply_multi_severity_correction: bool = True,
                 progress_every: int = 1000) -> dict:
    """Reads every row of `input_path`, computes PCI, and writes
    `output_path` = original columns + PCI_SCORE, PCI_RATING,
    SAMPLE_UNIT_AREA_M2, PCI_DOMINANT_DISTRESS, PCI_ERROR.

    Rows that fail (e.g. missing/zero SURVEY_WIDTH) get blank PCI
    columns and an explanation in PCI_ERROR rather than stopping the
    whole run.
    """
    added_cols = ["PCI_SCORE", "PCI_RATING", "SAMPLE_UNIT_AREA_M2",
                  "PCI_DOMINANT_DISTRESS", "PCI_ERROR"]

    n_ok, n_err = 0, 0
    with open(input_path, newline="", encoding="utf-8-sig") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames + added_cols
        with open(output_path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            for i, row in enumerate(reader, start=1):
                out_row = dict(row)
                try:
                    result = compute_pci(row, section_length_m,
                                          apply_multi_severity_correction)
                    out_row["PCI_SCORE"] = result["pci"]
                    out_row["PCI_RATING"] = result["rating"]
                    out_row["SAMPLE_UNIT_AREA_M2"] = result["area_m2"]
                    out_row["PCI_DOMINANT_DISTRESS"] = dominant_distress(result)
                    out_row["PCI_ERROR"] = ""
                    n_ok += 1
                except Exception as e:
                    out_row["PCI_SCORE"] = ""
                    out_row["PCI_RATING"] = ""
                    out_row["SAMPLE_UNIT_AREA_M2"] = ""
                    out_row["PCI_DOMINANT_DISTRESS"] = ""
                    out_row["PCI_ERROR"] = str(e)
                    n_err += 1
                writer.writerow(out_row)

                if progress_every and i % progress_every == 0:
                    print(f"  ...{i} rows processed", file=sys.stderr)

    return {"rows_processed": n_ok + n_err, "rows_ok": n_ok, "rows_failed": n_err,
            "output_path": output_path}


# ======================================================================
# 9. CLI
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute ASTM D6433-style PCI for every row of a LTPP-schema CSV."
    )
    parser.add_argument("input_csv", nargs="?", default="master.csv",
                         help="Path to input CSV (default: master.csv)")
    parser.add_argument("output_csv", nargs="?", default="master_with_pci.csv",
                         help="Path to write output CSV (default: master_with_pci.csv)")
    parser.add_argument("--section-length-m", type=float, default=DEFAULT_LTPP_SECTION_LENGTH_M,
                         help="Assumed sample-unit length in metres (default: 152.4, i.e. 500 ft)")
    parser.add_argument("--no-multi-severity-correction", action="store_true",
                         help="Skip the ASTM 9.6.2 multi-severity pooling refinement (faster)")
    args = parser.parse_args()

    print(f"Reading {args.input_csv} ...")
    summary = process_csv(
        args.input_csv, args.output_csv,
        section_length_m=args.section_length_m,
        apply_multi_severity_correction=not args.no_multi_severity_correction,
    )
    print(f"Done. {summary['rows_ok']} rows scored, {summary['rows_failed']} failed "
          f"(see PCI_ERROR column). Output written to {summary['output_path']}")
