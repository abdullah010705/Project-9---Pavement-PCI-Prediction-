"""
validate.py — checks every brief before it is accepted.

Errors (brief rejected):
  - a number in the text that is not in the payload (after allowed rounding)
  - a required fact missing (completeness)
  - a driver's direction or the repair flag contradicting the payload (meaning)
  - causal or over-confident wording about SHAP drivers
  - imperial units
  - claiming the rutting model beats persistence when the bootstrap says otherwise
Warnings (recorded, brief kept):
  - US spellings (the project writes Australian English)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from llm.payload import Payload
from llm.schema import BriefText

_NUM = re.compile(r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+|-?\d+)(\.\d+)?(\s*million)?", re.I)

BANNED = [
    (r"\bwill (?:definitely |certainly )?fail\b", "states the road will fail"),
    (r"\bbecause (?:of )?(?:its|the) (?:current|base|subbase|pavement|traffic)",
     "causal wording about a model driver"),
    (r"\b(?:caus(?:e|es|ed|ing))\b", "causal wording ('cause')"),
    (r"\bguarantee", "over-confident wording"),
    (r"\bcertain(?:ly)?\b", "over-confident wording"),
    (r"\binch(?:es)?\b|\bfeet\b|\bfoot\b|\bmiles?\b|°F", "imperial units"),
]
RUT_OVERCLAIM = (r"\b(?:more accurate|better|outperform\w*) than (?:simply )?"
                 r"(?:persistence|assuming)", "claims rutting model beats persistence")

US_SPELLINGS = {
    "analyze": "analyse", "color": "colour", "behavior": "behaviour",
    "modeling": "modelling", "program ": "programme ", "center": "centre",
    "prioritize": "prioritise", "utilize": "utilise",
}


@dataclass
class Result:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    numbers_checked: int = 0
    numbers_unsupported: list[str] = field(default_factory=list)


def _payload_numbers(obj, out: set[float]) -> set[float]:
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _payload_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _payload_numbers(v, out)
    elif isinstance(obj, str):
        for m in _NUM.finditer(obj):
            out.add(_to_float(m))
    return out


def _to_float(m: re.Match) -> float:
    v = float((m.group(1) + (m.group(2) or "")).replace(",", ""))
    return v * 1e6 if m.group(3) else v


def allowed_numbers(p: Payload) -> set[float]:
    nums = _payload_numbers(p.for_prompt(), set())
    y, mth, d = p.identity.survey_date.split("-")
    nums |= {float(y), float(mth), float(d)}
    # Counts the text may legitimately state
    nums |= {float(len(o.drivers)) for o in (p.pci, p.rut) if o}
    return nums


def _supported(x: float, allowed: set[float]) -> bool:
    """x matches a payload value rounded to 0–2 dp (sign ignored: "lowered by 3.1").

    Large counts may also be written in millions, e.g. 4,580,441 as "4.6 million".
    """
    x = abs(x)
    for v in allowed:
        v = abs(v)
        for k in (0, 1, 2):
            if abs(round(v, k) - x) < 1e-9:
                return True
            if v >= 1e5 and abs(round(v / 1e6, k) * 1e6 - x) < 1e-3:
                return True
    return False


def brief_text(b: BriefText) -> str:
    return "\n".join([b.headline, b.condition_now, b.outlook, *b.drivers, b.repair,
                      b.action, *b.caveats])


def _mentions(text: str, value: float) -> bool:
    return any(_supported(_to_float(m), {value}) for m in _NUM.finditer(text))


_UP = re.compile(r"\b(?:rais(?:ed|es|ing)|increas(?:ed|es|ing)|push(?:ed|es)? up)\b", re.I)
_DOWN = re.compile(r"\b(?:lower(?:ed|s|ing)|reduc(?:ed|es|ing)|decreas(?:ed|es|ing)|"
                   r"pull(?:ed|s)? down)\b", re.I)
_NOT_FLAGGED = re.compile(r"\b(?:does not|doesn't|did not|didn't) flag|\bnot flagged\b", re.I)


def meaning(b: BriefText, p: Payload) -> list[str]:
    """Checks the numbers cannot catch: the direction of each driver and whether the
    repair classifier flags the road. (Numbers are matched ignoring sign, so
    "lowered by 2.5" passes the grounding check even when the payload says raised.)
    """
    errs = []
    drivers = [d for o in (p.pci, p.rut) if o for d in o.drivers]
    for i, (d, s) in enumerate(zip(drivers, b.drivers), start=1):
        want, other = (_UP, _DOWN) if d.direction == "raised" else (_DOWN, _UP)
        if other.search(s) and not want.search(s):
            errs.append(f"driver {i} ({d.label}) says the opposite direction: payload "
                        f"says {d.direction}")
    if p.repair is not None and b.repair.strip():
        says_not = bool(_NOT_FLAGGED.search(b.repair))
        if says_not == p.repair.flagged:
            errs.append("repair text contradicts the classifier: payload flagged="
                        f"{p.repair.flagged}")
    return errs


def completeness(b: BriefText, p: Payload) -> list[str]:
    """Facts the brief must contain. Grounding alone passes a brief that omits them."""
    errs = []
    for o in (p.pci, p.rut):
        if o is None:
            continue
        for name in ("predicted", "typical_error", "range_low", "range_high"):
            if not _mentions(b.outlook, getattr(o, name)):
                errs.append(f"outlook omits {o.target} {name} ({getattr(o, name):g})")
    n_drivers = sum(len(o.drivers) for o in (p.pci, p.rut) if o)
    if len(b.drivers) != n_drivers:
        errs.append(f"expected {n_drivers} driver sentences, got {len(b.drivers)}")
    if p.repair is None and b.repair.strip():
        errs.append("repair text given but payload has no repair result")
    if p.repair is not None:
        r = p.repair
        if not _mentions(b.repair, r.probability_pct):
            errs.append(f"repair text omits the probability ({r.probability_pct}%)")
        if not (_mentions(b.repair, r.test_repairs_caught)
                and _mentions(b.repair, r.test_repairs_total)):
            errs.append(f"repair text omits the test record ({r.test_repairs_caught} of "
                        f"{r.test_repairs_total})")
    n_notes = len(p.caveats())
    if len(b.caveats) < n_notes:
        errs.append(f"caveats list {len(b.caveats)} items; payload has {n_notes} notes")
    for name in ("headline", "condition_now", "outlook", "action"):
        if not getattr(b, name).strip():
            errs.append(f"{name} is empty")
    text = brief_text(b)
    if re.search(r"[{}\[\]]|\"\s*:", text):
        errs.append("JSON pasted into the text")
    leaked = sorted(set(re.findall(r"\b[a-z]+_[a-z_]+\b", text)))
    if leaked:
        errs.append(f"field names in the text: {', '.join(leaked)}")
    return errs


def validate(b: BriefText, p: Payload) -> Result:
    text = brief_text(b)
    r = Result(ok=True)

    allowed = allowed_numbers(p)
    for m in _NUM.finditer(text):
        r.numbers_checked += 1
        x = _to_float(m)
        if not _supported(x, allowed):
            r.numbers_unsupported.append(m.group(0).strip())
    if r.numbers_unsupported:
        r.errors.append(f"numbers not in payload: {', '.join(r.numbers_unsupported)}")

    for pattern, why in BANNED:
        if re.search(pattern, text, re.I):
            r.errors.append(why)
    if p.rut and p.rut.reliability_note:
        for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
            if (re.search(RUT_OVERCLAIM[0], sentence, re.I)
                    and not re.search(r"\bnot\b|\bno\b|n't\b", sentence, re.I)):
                r.errors.append(RUT_OVERCLAIM[1])
                break

    r.errors += completeness(b, p)
    r.errors += meaning(b, p)

    low = text.lower()
    for us, au in US_SPELLINGS.items():
        if us in low:
            r.warnings.append(f"US spelling '{us.strip()}' (use '{au.strip()}')")

    r.ok = not r.errors
    return r
