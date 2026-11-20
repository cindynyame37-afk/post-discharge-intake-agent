"""Adaptive checklist intake for heart failure (HF) and COPD post-discharge messages.

Rough draft (Session 11). Synthetic data only. One command runs one case end to end:

    python run_intake.py synthetic/hf_direct.json

Each section below is one node of the wire diagram (docs/wire_diagram.png).
Atomic protocols: medspaCy ConText, Pydantic, Microsoft Presidio.
Disposable components: everything else (my code, unit-tested, no upkeep promise).

Slot fill (N3) runs an offline rule-based stand-in by default, so GitHub Actions
needs no API key. Set USE_LLM=1 and GEMINI_API_KEY to use Gemini instead.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ValidationError

HERE = Path(__file__).parent
MAX_TURNS = 3  # first message + up to two follow-up replies


# ---------------------------------------------------------------------------
# Data assets: checklists (read from data/, sit on edges, not in nodes)
# ---------------------------------------------------------------------------

def load_checklist(condition):
    name = {"HF": "checklist_hf.json", "COPD": "checklist_copd.json"}[condition]
    return json.loads((HERE / "data" / name).read_text())


# ---------------------------------------------------------------------------
# ATOMIC PROTOCOL: medspaCy ConText symptom flags
# Input: str. Output: list[dict], one per symptom mention.
# ---------------------------------------------------------------------------

SYMPTOM_TERMS = {
    "chest pain": "chest_pain", "chest tightness": "chest_pain",
    "shortness of breath": "breathless", "short of breath": "breathless",
    "breathless": "breathless", "winded": "breathless",
    "swelling": "leg_swelling", "swollen": "leg_swelling",
    "fever": "fever", "chills": "fever",
    "phlegm": "sputum_change", "sputum": "sputum_change", "mucus": "sputum_change",
}

_nlp = None


def get_nlp():
    """Build the medspaCy pipeline once and reuse it."""
    global _nlp
    if _nlp is None:
        from loguru import logger  # PyRuSH (inside medspaCy) logs every token at DEBUG
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
        import medspacy
        from medspacy.ner import TargetRule
        _nlp = medspacy.load()
        matcher = _nlp.get_pipe("medspacy_target_matcher")
        matcher.add([TargetRule(term, label) for term, label in SYMPTOM_TERMS.items()])
    return _nlp


def flag_symptoms(message):
    if not isinstance(message, str):
        raise TypeError("message must be a string")
    doc = get_nlp()(message)
    out = []
    for ent in doc.ents:
        out.append({
            "text": ent.text.lower(),
            "label": ent.label_,
            "sentence": ent.sent.text.lower(),
            "is_negated": ent._.is_negated,
            "is_uncertain": ent._.is_uncertain,
            "is_historical": ent._.is_historical,
            "is_hypothetical": ent._.is_hypothetical,
            "is_family": ent._.is_family,
        })
    return out


# ---------------------------------------------------------------------------
# N2 Condition tag (disposable). Input: str. Output: "HF", "COPD" or "unclear".
# ---------------------------------------------------------------------------

HF_WORDS = ["heart failure", "water pill", "lasix", "furosemide", "weight", "ankles", "swelling", "swollen"]
COPD_WORDS = ["copd", "inhaler", "puffer", "phlegm", "sputum", "mucus", "oxygen", "nebulizer"]


def tag_condition(text):
    t = text.lower()
    hf = sum(w in t for w in HF_WORDS)
    copd = sum(w in t for w in COPD_WORDS)
    if hf > copd:
        return "HF"
    if copd > hf:
        return "COPD"
    return "unclear"  # tie or no keywords: a person decides


# ---------------------------------------------------------------------------
# ATOMIC PROTOCOL: Pydantic schema check on the slot record
# ---------------------------------------------------------------------------

class HFSlots(BaseModel):
    weight_gain_lb: Optional[float] = None
    weight_gain_days: Optional[int] = None
    breathless_at_rest: Optional[bool] = None
    leg_swelling: Optional[bool] = None
    chest_pain: Optional[bool] = None
    taking_water_pill: Optional[bool] = None


class COPDSlots(BaseModel):
    mmrc_grade: Optional[int] = None
    breathless_at_rest: Optional[bool] = None
    sputum_change: Optional[bool] = None
    fever: Optional[bool] = None
    rescue_inhaler_per_day: Optional[int] = None
    chest_pain: Optional[bool] = None


SCHEMAS = {"HF": HFSlots, "COPD": COPDSlots}


def validate_slots(raw_json, condition):
    """str (JSON) -> dict of slots. Raises ValidationError if the JSON does not fit."""
    return SCHEMAS[condition].model_validate_json(raw_json).model_dump()


# ---------------------------------------------------------------------------
# N3 Slot fill (disposable). Input: text + ConText flags + checklist.
# Output: JSON string, checked by Pydantic before use.
# ---------------------------------------------------------------------------

REST_WORDS = ["at rest", "sitting", "lying", "in bed", "at night", "resting", "doing nothing"]
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}


def _num(s):
    return float(NUMBER_WORDS.get(s, s))


def fill_slots_offline(text, mentions, condition):
    """Rule-based stand-in for the LLM. Leaves a slot as None when unsure."""
    t = text.lower()
    slots = {}

    # Yes/no symptom slots come from ConText: present -> True, denied -> False,
    # uncertain/hypothetical/family/historical -> leave missing so the gate asks.
    for m in mentions:
        if m["is_uncertain"] or m["is_hypothetical"] or m["is_family"] or m["is_historical"]:
            continue
        value = not m["is_negated"]
        if m["label"] == "breathless":
            if any(w in m["sentence"] for w in REST_WORDS):
                slots["breathless_at_rest"] = value
            elif m["is_negated"]:
                slots["breathless_at_rest"] = False
        else:
            slots[m["label"]] = value

    # Weight change (HF): "gained 4 pounds", "up 3 lbs", "weight is the same"
    w = re.search(r"(gained|gain|up|put on)\D{0,15}?(\d+(?:\.\d+)?|one|two|three|four|five|six|seven)\s*(?:lb|lbs|pound)", t)
    if w:
        slots["weight_gain_lb"] = _num(w.group(2))
    elif re.search(r"weight (is|has been|stayed) (the same|stable|steady)|no weight gain|haven't gained", t):
        slots["weight_gain_lb"] = 0.0
        slots["weight_gain_days"] = 1
    d = re.search(r"(?:in|over|since)\s+(?:the\s+(?:last|past)\s+)?(\d+|one|two|three|four|five|six|seven)\s+days?", t)
    if d:
        slots["weight_gain_days"] = int(_num(d.group(1)))
    elif re.search(r"since yesterday|overnight|in a day|today", t) and "weight_gain_lb" in slots:
        slots.setdefault("weight_gain_days", 1)
    elif re.search(r"this week|in a week|over the week", t) and "weight_gain_lb" in slots:
        slots.setdefault("weight_gain_days", 7)

    # Water pill (HF)
    if re.search(r"water pill|lasix|furosemide|diuretic", t):
        stopped = re.search(r"ran out|stopped|missed|skipp|forgot|not taking|haven't taken", t)
        slots["taking_water_pill"] = not stopped

    # mMRC grade (COPD): a bare digit 0-4 as a reply, or "grade 3"
    g = re.search(r"(?:grade|number|it's|its|i'm a|i am a)?\s*\b([0-4])\b(?!\s*(?:times|puffs|days|lb|pounds))", t)
    if condition == "COPD" and g:
        slots["mmrc_grade"] = int(g.group(1))

    # Rescue inhaler uses per day (COPD)
    r = re.search(r"(\d+|one|two|three|four|five|six|seven)\s+(?:times|puffs)", t)
    if r and re.search(r"inhaler|puffer|albuterol|rescue", t):
        slots["rescue_inhaler_per_day"] = int(_num(r.group(1)))

    keep = SCHEMAS[condition].model_fields.keys()
    return json.dumps({k: v for k, v in slots.items() if k in keep})


def fill_slots_llm(text, condition, checklist):
    """Gemini, temperature 0, JSON output in the checklist schema.
    Needs GEMINI_API_KEY in the environment. Never put the key in the code or repo."""
    from google import genai  # library: google-genai
    client = genai.Client()
    prompt = (
        "You fill a post-discharge intake checklist from a patient's message.\n"
        "Return JSON with only these keys. Use null for anything the message does not clearly say.\n"
        "Do not guess. A hedged or unsure answer is null.\n\n"
        f"Checklist: {json.dumps({k: v['question'] for k, v in checklist['slots'].items()})}\n\n"
        f"Message: {text}"
    )
    resp = client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", "gemini-flash-latest"),
        contents=prompt,
        config={"temperature": 0, "response_mime_type": "application/json",
                "response_schema": SCHEMAS[condition]},
    )
    return resp.text


def fill_slots(text, mentions, condition, checklist):
    if os.environ.get("USE_LLM") == "1":
        raw = fill_slots_llm(text, condition, checklist)
    else:
        raw = fill_slots_offline(text, mentions, condition)
    return validate_slots(raw, condition)  # Pydantic check on either path


# ---------------------------------------------------------------------------
# N4 Completeness gate (disposable decision node)
# Input: slot record + checklist. Output: list of missing required slots.
# ---------------------------------------------------------------------------

def missing_slots(record, checklist):
    return [k for k, v in checklist["slots"].items() if v["required"] and record.get(k) is None]


def next_question(missing, checklist):
    return checklist["slots"][missing[0]]["question"] if missing else None


# ---------------------------------------------------------------------------
# N5a HF red-flag rules (AHA warning signs) and N5b COPD red-flag rules (mine)
# Input: slot record. Output: list[str] of fired flags.
# ---------------------------------------------------------------------------

def hf_red_flags(r):
    flags = []
    gain, days = r.get("weight_gain_lb"), r.get("weight_gain_days")
    if gain is not None and days is not None:
        if (days <= 1 and gain >= 3) or (days <= 7 and gain >= 5):
            flags.append("rapid_weight_gain")
    if r.get("breathless_at_rest"):
        flags.append("breathless_at_rest")
    if r.get("leg_swelling"):
        flags.append("new_or_worse_swelling")
    if r.get("chest_pain"):
        flags.append("chest_pain")
    if r.get("taking_water_pill") is False:
        flags.append("not_taking_water_pill")
    return flags


def copd_red_flags(r):
    # Provisional thresholds from my follow-up calls, to be replaced by a cited GOLD page.
    flags = []
    if r.get("breathless_at_rest"):
        flags.append("breathless_at_rest")
    if (r.get("mmrc_grade") or 0) >= 4:
        flags.append("mmrc_grade_4")
    if r.get("sputum_change") and r.get("fever"):
        flags.append("sputum_change_with_fever")
    if (r.get("rescue_inhaler_per_day") or 0) >= 4:
        flags.append("rescue_inhaler_4_plus_per_day")
    if r.get("chest_pain"):
        flags.append("chest_pain")
    return flags


# ---------------------------------------------------------------------------
# N6 Understatement flag (disposable). Marks the record; never re-routes.
# ---------------------------------------------------------------------------

HEDGE_WORDS = ["a little", "a bit", "kind of", "kinda", "sort of", "not too bad", "probably nothing",
               "just", "slightly", "manageable", "nothing major", "don't want to bother", "i guess"]


def understatement_flag(all_text, flags, mentions):
    t = all_text.lower()
    hedged = any(h in t for h in HEDGE_WORDS) or any(m["is_uncertain"] for m in mentions)
    return bool(hedged and flags)  # hedged wording on a message that still fired a red flag


# ---------------------------------------------------------------------------
# N7 Route (disposable decision node). Any red flag -> nurse_now. No other branch.
# ---------------------------------------------------------------------------

def route(flags):
    return "nurse_now" if flags else "coordinator_same_day"


# ---------------------------------------------------------------------------
# ATOMIC PROTOCOL: Microsoft Presidio, only when a transcript leaves the care team
# ---------------------------------------------------------------------------

_analyzer = None


def deidentify(text):
    global _analyzer
    import logging
    logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)  # hide setup chatter
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine
    if _analyzer is None:
        nlp_engine = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }).create_engine()
        _analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
    results = _analyzer.analyze(text=text, language="en")
    return AnonymizerEngine().anonymize(text=text, analyzer_results=results).text


# ---------------------------------------------------------------------------
# Minimal glue: run one synthetic case through the loop
# ---------------------------------------------------------------------------

def run_case(case):
    """case = {"case_id": str, "turns": [first message, reply 1, reply 2]}.
    Replies are the synthetic patient's answers, used in order."""
    turns = case["turns"]
    transcript = [("patient", turns[0])]
    condition = tag_condition(turns[0])
    if condition == "unclear":
        return {"case_id": case["case_id"], "condition": "unclear", "route": "coordinator_queue",
                "reason": "condition unclear", "slots": {}, "flags": [], "hedge_flag": False,
                "turns_used": 1, "transcript": transcript}

    checklist = load_checklist(condition)
    record, all_mentions, text_so_far = {}, [], ""
    for i in range(MAX_TURNS):
        message = turns[i] if i < len(turns) else ""
        if i > 0:
            transcript.append(("patient", message))
        text_so_far += " " + message
        mentions = flag_symptoms(message)
        all_mentions += mentions
        new = fill_slots(message, mentions, condition, checklist)
        record.update({k: v for k, v in new.items() if v is not None})
        missing = missing_slots(record, checklist)
        if not missing:
            break
        if i < MAX_TURNS - 1:
            transcript.append(("agent", next_question(missing, checklist)))

    flags = hf_red_flags(record) if condition == "HF" else copd_red_flags(record)
    missing = missing_slots(record, checklist)
    out = {
        "case_id": case["case_id"], "condition": condition, "slots": record, "flags": flags,
        "hedge_flag": understatement_flag(text_so_far, flags, all_mentions),
        "turns_used": len([t for t in transcript if t[0] == "patient"]), "transcript": transcript,
    }
    if missing and not flags:
        out["route"], out["reason"] = "coordinator_queue", f"3 turns, still missing {missing}"
    else:
        out["route"] = route(flags)
        if missing:
            out["reason"] = f"red flag fired; still missing {missing}"
    return out


def write_audit_log(result, path):
    """Route record + de-identified turns. Raw text stays out of the log."""
    entry = {k: v for k, v in result.items() if k != "transcript"}
    entry["transcript"] = [(who, deidentify(text)) for who, text in result["transcript"]]
    path.parent.mkdir(exist_ok=True)
    log = json.loads(path.read_text()) if path.exists() else []
    log.append(entry)
    path.write_text(json.dumps(log, indent=2))


def main(argv):
    if len(argv) < 2:
        print("usage: python run_intake.py synthetic/<case>.json [more cases]")
        return 1
    for p in argv[1:]:
        result = run_case(json.loads(Path(p).read_text()))
        write_audit_log(result, HERE / "outputs" / "audit_log.json")
        print(json.dumps({k: v for k, v in result.items() if k != "transcript"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
