"""Tests for the LLM layer that do not need Ollama running.

Run from the pipeline folder:  python -m pytest tests -q   (or ./run_llm.sh test)
"""
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore", category=UserWarning)

from llm import template_brief, v002  # noqa: E402
from llm.payload import build_payload, list_surveys  # noqa: E402
from llm.schema import BriefText  # noqa: E402
from llm.validate import validate  # noqa: E402


# --- model loading reproduces the numbers in results/ ------------------------------
@pytest.mark.parametrize("key", ["pci", "rut"])
def test_regressor_reproduces_reported_test_rmse(key):
    df = v002.dataset(key)
    X = v002.feature_matrix(key)
    test = df[v002.config.SPLIT_COL] == v002.config.TEST_VALUE
    pred = v002.predict(f"xgboost_{key}", X[test])
    y = df.loc[test, v002.config.TARGETS[key]["target_col"]].values
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    assert rmse == pytest.approx(v002.model_error(key)["test_rmse"], abs=5e-4)


def test_classifier_reproduces_reported_confusion_matrix():
    df = v002.dataset("pci")
    X = v002.feature_matrix("pci")
    test = df[v002.config.SPLIT_COL] == v002.config.TEST_VALUE
    flag = v002.predict("repair_classifier", X[test]) >= 0.5
    y = df.loc[test, "NEEDS_REPAIR"].astype(int).values == 1
    s = v002.classifier_summary()["test"]
    assert (flag & y).sum() == s["tp"] and (flag & ~y).sum() == s["fp"]
    assert (~flag & y).sum() == s["fn"] and (~flag & ~y).sum() == s["tn"]


@pytest.mark.parametrize("key", ["pci", "rut"])
def test_shap_contributions_sum_to_prediction(key):
    X = v002.feature_matrix(key).iloc[:20]
    c = v002.predict(f"xgboost_{key}", X, contribs=True)
    assert np.allclose(c.sum(axis=1), v002.predict(f"xgboost_{key}", X), atol=1e-3)


# --- payload -------------------------------------------------------------------------
@pytest.fixture(scope="module")
def payload():
    return build_payload("1-4126-1", "1993-03-30")


def test_payload_accepts_both_date_formats(payload):
    assert build_payload("1-4126-1", "30/3/1993").pci.predicted == payload.pci.predicted


def test_hindsight_is_not_shown_to_llm(payload):
    view = str(payload.for_prompt())
    assert "hindsight" not in view and "pci_next_actual" not in view
    assert str(payload.hindsight.pci_next_actual) not in view


def test_range_is_prediction_plus_minus_rmse_clipped(payload):
    o = payload.pci
    assert o.range_low == pytest.approx(max(o.predicted - o.typical_error, 0), abs=0.11)
    assert o.range_high == pytest.approx(min(o.predicted + o.typical_error, 100), abs=0.11)


def test_unknown_survey_raises():
    with pytest.raises(KeyError):
        build_payload("1-4126-1", "1900-01-01")


# --- validator -----------------------------------------------------------------------
def _template(p):
    return template_brief.build(p)


def test_template_brief_valid_for_every_test_survey():
    surveys = list_surveys("test")
    failures = []
    for r in surveys.itertuples():
        p = build_payload(r.ROAD_ID, r.SURVEY_DATE_ISO)
        res = validate(_template(p), p)
        if not res.ok:
            failures.append((r.ROAD_ID, r.SURVEY_DATE_ISO, res.errors))
    assert not failures, failures[:5]


def _edit(p, **changes) -> BriefText:
    return _template(p).model_copy(update=changes)


def test_rejects_invented_number(payload):
    b = _edit(payload, headline="PCI is expected to drop by 37 points.")
    assert not validate(b, payload).ok


def test_accepts_rounded_payload_number(payload):
    b = _edit(payload, headline=f"PCI is predicted to be about "
                                f"{round(payload.pci.predicted)} at the next survey.")
    assert validate(b, payload).ok


def test_rejects_causal_wording(payload):
    b = _edit(payload, headline="The traffic loading will cause the road to fail.")
    res = validate(b, payload)
    assert not res.ok and any("caus" in e or "fail" in e for e in res.errors)


def test_rejects_json_and_field_names(payload):
    b = _edit(payload, outlook=payload.pci.model_dump_json())
    assert not validate(b, payload).ok


def test_rejects_missing_prediction(payload):
    b = _edit(payload, outlook="The road should stay in good condition.")
    assert not validate(b, payload).ok


def test_rejects_imperial_units(payload):
    b = _edit(payload, condition_now="Rut depth is 0 inches.")
    assert not validate(b, payload).ok


def test_rut_overclaim_rejected_but_negation_allowed():
    s = list_surveys("test")
    row = s[s.targets == "pci+rut"].iloc[0]
    p = build_payload(row.ROAD_ID, row.SURVEY_DATE_ISO)
    assert p.rut.reliability_note
    bad = _edit(p, headline="The rutting model is more accurate than persistence.")
    assert not validate(bad, p).ok
    assert validate(_template(p), p).ok  # the template's negated caveat is fine


# --- prompts ------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["brief_v1", "brief_v2"])
def test_prompt_renders_and_ends_with_this_payload(name, payload):
    from llm.brief import _render_prompt
    messages, sha = _render_prompt(payload, name)
    assert messages[0]["role"] == "system" and messages[-1]["role"] == "user"
    assert payload.identity.road_id in messages[-1]["content"]
    assert "hindsight" not in messages[-1]["content"] and len(sha) == 12


def test_v2_checklists_add_no_new_numbers(payload):
    from llm.brief import _view_v2
    from llm.validate import _payload_numbers
    base = _payload_numbers(payload.for_prompt(), set())
    assert _payload_numbers(_view_v2(payload), set()) <= base


def test_prompt_examples_are_training_roads_and_valid():
    import json
    from llm.brief import PROMPTS, prompt_files
    for name in PROMPTS:
        with open(prompt_files(name)[1], encoding="utf-8") as f:
            ex = json.load(f)
        for e in ex if isinstance(ex, list) else [ex]:
            # Example files carry no survey date (v1 is frozen), so match the survey
            # by its current condition block.
            road = e["payload"]["headline"]["road_id"]
            candidates = (build_payload(r.ROAD_ID, r.SURVEY_DATE_ISO)
                          for r in list_surveys().itertuples() if r.ROAD_ID == road)
            p = next(c for c in candidates
                     if c.for_prompt()["condition_now"] == e["payload"]["condition_now"])
            assert p.identity.split == "train"
            assert validate(BriefText(**e["brief"]), p).ok


def test_rejects_flipped_driver_direction(payload):
    b = _template(payload)
    d = payload.pci.drivers[0]
    flipped = "lowered" if d.direction == "raised" else "raised"
    drivers = [b.drivers[0].replace(d.direction, flipped)] + b.drivers[1:]
    res = validate(b.model_copy(update={"drivers": drivers}), payload)
    assert not res.ok and any("opposite direction" in e for e in res.errors)


def test_rejects_wrong_repair_flag_wording(payload):
    assert not payload.repair.flagged
    r = payload.repair
    text = (f"The repair classifier flags this road, with an estimated {r.probability_pct}% "
            f"probability. On test data it caught {r.test_repairs_caught} of "
            f"{r.test_repairs_total} roads that needed repair.")
    res = validate(_edit(payload, repair=text), payload)
    assert not res.ok and any("contradicts the classifier" in e for e in res.errors)
