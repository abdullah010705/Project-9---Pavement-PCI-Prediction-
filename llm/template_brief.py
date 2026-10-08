"""
template_brief.py — deterministic, non-LLM brief built from the same payload.

Two roles (llm-plan.md §4, §7): the fallback when LLM output fails validation, and
the baseline the LLM output is evaluated against.
"""
from __future__ import annotations

from llm.payload import Driver, Outlook, Payload
from llm.schema import BriefText


def _fmt(v: float, unit: str | None) -> str:
    if isinstance(v, float) and abs(v) >= 10_000:
        s = f"{v:,.0f}"
    else:
        s = f"{v:g}"
    return f"{s} {unit}" if unit else s


def _driver_sentence(d: Driver, o: Outlook) -> str:
    unit = "PCI points" if o.target == "PCI" else (o.unit or "")
    value = d.value if isinstance(d.value, str) else _fmt(d.value, d.unit)
    filled = " (not recorded; the model used a typical value)" if d.value_was_filled else ""
    return (f"{d.label[0].upper()}{d.label[1:]} ({value}){filled} {d.direction} the model's "
            f"{o.target} estimate by {_fmt(d.effect, unit)}.")


def _outlook_sentence(o: Outlook, years: float | None) -> str:
    when = f"about {years:g} years later" if years is not None else "the next survey"
    unit = f" {o.unit}" if o.unit else ""
    return (f"The model predicts {o.target} of {o.predicted:g}{unit} at the next survey "
            f"({when}), rated {o.predicted_band}. Given its typical error of "
            f"{o.typical_error:g}{unit}, the likely range is {o.range_low:g}–"
            f"{o.range_high:g}{unit}. The driver effects below are measured from the "
            f"model's average {o.target} prediction of {o.model_average:g}{unit}.")


def build(p: Payload) -> BriefText:
    c = p.condition
    now = f"Current PCI is {c.pci:g} ({c.pci_rating})"
    if c.dominant_distress:
        now += f", with {c.dominant_distress} the dominant distress"
    now += "."
    if c.rut_depth_mm is not None:
        now += f" Rut depth is {c.rut_depth_mm:g} mm ({c.rut_severity})."
    now += f" The pavement is {c.pavement_age_years:g} years old."

    outlooks = [o for o in (p.pci, p.rut) if o is not None]
    outlook = " ".join(_outlook_sentence(o, c.years_to_next_survey) for o in outlooks)
    drivers = [_driver_sentence(d, o) for o in outlooks for d in o.drivers]

    repair = ""
    if p.repair:
        r = p.repair
        verdict = "flags" if r.flagged else "does not flag"
        repair = (f"The repair classifier {verdict} this road (estimated probability "
                  f"{r.probability_pct}% that PCI falls below {r.pci_repair_threshold}). On "
                  f"test data it caught {r.test_repairs_caught} of {r.test_repairs_total} "
                  f"roads that needed repair, with precision {r.test_precision_pct}%.")

    headline_o = p.pci or p.rut
    headline = (f"Road {p.identity.road_id}: {headline_o.target} predicted to be "
                f"{headline_o.predicted_band} at the next survey; recommended action is "
                f"{p.action.tier}.")

    caveats = p.caveats()

    return BriefText(headline=headline, condition_now=now, outlook=outlook, drivers=drivers,
                     repair=repair,
                     action=f"{p.action.description} Reason: {'; '.join(p.action.reasons)}.",
                     caveats=caveats)
