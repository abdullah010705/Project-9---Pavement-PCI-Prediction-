"""
schema.py — the structure every brief (LLM or template) is written into.

The LLM is constrained to this JSON schema through Ollama's `format` parameter,
so its output is a set of checkable fields, not free prose.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class BriefText(BaseModel):
    headline: str = Field(description="One sentence summarising the outlook for this road.")
    condition_now: str = Field(description="One or two sentences on the current condition.")
    outlook: str = Field(description="Predicted condition at the next survey, with the "
                                     "typical-error range.")
    drivers: list[str] = Field(description="One short sentence per driver in the payload, "
                                           "phrased as effects on the model's prediction.")
    repair: str = Field(description="The repair classifier result and how reliable it is. "
                                    "Empty string if there is no repair result.")
    action: str = Field(description="Explain the recommended action tier given in the "
                                    "payload. Do not choose a different tier.")
    caveats: list[str] = Field(description="Limitations the reader must know, including "
                                           "every data note and reliability note.")


def render_markdown(brief: BriefText, payload) -> str:
    """Render a brief. Identity and the action tier come from the payload, not the text."""
    ident = payload.identity
    lines = [
        f"## Condition brief — road {ident.road_id}, survey {ident.survey_date}",
        "",
        f"**{brief.headline}**",
        "",
        f"**Now.** {brief.condition_now}",
        "",
        f"**Outlook.** {brief.outlook}",
        "",
        "**What drove the model's estimate**",
        *[f"- {d}" for d in brief.drivers],
        "",
    ]
    if brief.repair:
        lines += [f"**Repair flag.** {brief.repair}", ""]
    lines += [f"**Recommended action: {payload.action.tier}.** {brief.action}", ""]
    if brief.caveats:
        lines += ["**Caveats**", *[f"- {c}" for c in brief.caveats], ""]
    return "\n".join(lines)
