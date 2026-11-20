"""Unit tests for every node of the intake workflow, plus end-to-end runs on the
synthetic cases. Runs offline (no API key): slot fill uses the rule-based stand-in."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import run_intake as w

HERE = Path(__file__).parent


def case(name):
    return json.loads((HERE / "synthetic" / f"{name}.json").read_text())


# N2 Condition tag
@pytest.mark.parametrize("text, expected", [
    ("I gained weight and my ankles are swollen", "HF"),
    ("My COPD inhaler is not helping", "COPD"),
    ("Hi, just checking in", "unclear"),
])
def test_condition_tag(text, expected):
    assert w.tag_condition(text) == expected


# Pydantic schema check
def test_pydantic_accepts_valid_slots():
    assert w.validate_slots('{"chest_pain": false, "weight_gain_lb": 4}', "HF")["weight_gain_lb"] == 4.0


def test_pydantic_rejects_wrong_type():
    with pytest.raises(ValidationError):
        w.validate_slots('{"mmrc_grade": "very bad"}', "COPD")


# N3 Slot fill (offline stand-in)
def test_slot_fill_reads_weight_and_days():
    slots = w.fill_slots("I gained 4 pounds since yesterday", [], "HF", w.load_checklist("HF"))
    assert slots["weight_gain_lb"] == 4.0 and slots["weight_gain_days"] == 1


def test_slot_fill_leaves_unsure_answers_missing():
    msg = "I am not sure if I have swelling."
    slots = w.fill_slots(msg, w.flag_symptoms(msg), "HF", w.load_checklist("HF"))
    assert slots["leg_swelling"] is None


# N4 Completeness gate
def test_gate_lists_missing_and_asks_one_question():
    checklist = w.load_checklist("HF")
    missing = w.missing_slots({"chest_pain": False}, checklist)
    assert "chest_pain" not in missing and len(missing) == 5
    assert w.next_question(missing, checklist) == checklist["slots"][missing[0]]["question"]


# N5a HF red flags
@pytest.mark.parametrize("record, fired", [
    ({"weight_gain_lb": 3, "weight_gain_days": 1}, True),
    ({"weight_gain_lb": 5, "weight_gain_days": 7}, True),
    ({"weight_gain_lb": 2, "weight_gain_days": 1}, False),
    ({"weight_gain_lb": 4, "weight_gain_days": 7}, False),
])
def test_hf_weight_threshold(record, fired):
    assert ("rapid_weight_gain" in w.hf_red_flags(record)) == fired


# N5b COPD red flags
def test_copd_sputum_needs_fever_too():
    assert w.copd_red_flags({"sputum_change": True, "fever": False}) == []
    assert "sputum_change_with_fever" in w.copd_red_flags({"sputum_change": True, "fever": True})


# N6 Understatement flag
def test_hedge_flag_marks_but_needs_a_red_flag():
    assert w.understatement_flag("just a little swollen", ["new_or_worse_swelling"], []) is True
    assert w.understatement_flag("just a little swollen", [], []) is False


# N7 Route
def test_route_has_only_two_branches():
    assert w.route(["chest_pain"]) == "nurse_now"
    assert w.route([]) == "coordinator_same_day"


# Presidio (export only)
def test_presidio_redacts_name_and_phone():
    out = w.deidentify("My name is Maria Lopez, call me at 212-555-0147.")
    assert "Maria Lopez" not in out and "212-555-0147" not in out


# End to end on synthetic cases
@pytest.mark.parametrize("name, expected_route", [
    ("hf_direct", "nurse_now"),
    ("hf_understated", "nurse_now"),
    ("hf_routine", "coordinator_same_day"),
    ("copd_direct", "nurse_now"),
    ("copd_understated", "nurse_now"),
    ("copd_no_reply", "coordinator_queue"),
    ("unclear", "coordinator_queue"),
])
def test_end_to_end_route(name, expected_route):
    assert w.run_case(case(name))["route"] == expected_route


@pytest.mark.parametrize("pair", ["hf", "copd"])
def test_matched_pair_same_flags_hedge_marked(pair):
    direct, under = w.run_case(case(f"{pair}_direct")), w.run_case(case(f"{pair}_understated"))
    assert direct["flags"] == under["flags"]
    assert under["hedge_flag"] is True and direct["hedge_flag"] is False


def test_loop_asks_follow_up_questions():
    result = w.run_case(case("hf_routine"))
    assert result["turns_used"] == 3
    assert sum(1 for who, _ in result["transcript"] if who == "agent") == 2


# Known limitation (carried from the ConText tests): "No improvement in my
# shortness of breath" is read as a denial, so the red flag is missed.
@pytest.mark.xfail(strict=True, reason="ConText reads 'no improvement in' as negation")
def test_no_improvement_still_flags_breathless_at_rest():
    msg = "No improvement in my shortness of breath at night."
    slots = w.fill_slots(msg, w.flag_symptoms(msg), "HF", w.load_checklist("HF"))
    assert slots["breathless_at_rest"] is True
