"""
Protocol: medspaCy (ConText component)
Citable Sources:
    - Eyre H, Chapman AB, Peterson KS, et al. Launching into clinical space with
      medspaCy: a new clinical text processing toolkit in Python.
      AMIA Annu Symp Proc 2021:438-447. https://github.com/medspacy/medspacy
    - Harkema H, Dowling JN, Thornblade T, Chapman WW. Context: An algorithm for
      determining negation, experiencer, and temporal status from clinical reports.
      J Biomed Inform 2009;42(5):839-851.
Input -> Output Contract:
    - Input datatype: str (one patient message, English, free text)
    - Output datatype: list[dict], one dict per symptom mention, with keys:
        - text (str): matched symptom text span
        - label (str): entity label ("SYMPTOM")
        - is_negated (bool): True if symptom mention is negated
        - is_uncertain (bool): True if symptom mention is uncertain/possible
        - is_historical (bool): True if symptom mention is historical/past
        - is_hypothetical (bool): True if symptom mention is conditional/hypothetical
        - is_family (bool): True if symptom mention pertains to a family member
      Returns [] if no symptom terms are found.
Package Versions:
    - medspacy==1.3.1
    - spacy==3.8.2
    - pytest==8.3.4
"""

import medspacy
from medspacy.ner import TargetRule
import pytest


@pytest.fixture(scope="session")
def nlp():
    """Initializes the medspaCy pipeline once per session to stay within CI budgets."""
    pipeline = medspacy.load()
    target_matcher = (
        pipeline.get_pipe("medspacy_target_matcher")
        if "medspacy_target_matcher" in pipeline.pipe_names
        else pipeline.get_pipe("target_matcher")
    )
    checklist_terms = [
        "chest pain",
        "shortness of breath",
        "swelling",
        "cough",
        "fever",
        "weight gain",
        "dizziness",
        "wheezing",
    ]
    target_rules = [TargetRule(term, "SYMPTOM") for term in checklist_terms]
    target_matcher.add(target_rules)
    return pipeline


@pytest.fixture(scope="session")
def run_protocol(nlp):
    """Executes the protocol directly without adding intermediate type checks."""
    def _run(text):
        doc = nlp(text)
        return [
            {
                "text": ent.text,
                "label": ent.label_,
                "is_negated": bool(ent._.is_negated),
                "is_uncertain": bool(ent._.is_uncertain),
                "is_historical": bool(ent._.is_historical),
                "is_hypothetical": bool(ent._.is_hypothetical),
                "is_family": bool(ent._.is_family),
            }
            for ent in doc.ents
            if ent.label_ == "SYMPTOM"
        ]

    return _run


# ============================================================================
# Typical Inputs
# ============================================================================

def test_denied_symptom_is_negated(run_protocol):
    text = "I don't have any chest pain today."
    expected = [
        {
            "text": "chest pain",
            "label": "SYMPTOM",
            "is_negated": True,
            "is_uncertain": False,
            "is_historical": False,
            "is_hypothetical": False,
            "is_family": False,
        }
    ]
    assert run_protocol(text) == expected


def test_reported_symptom_is_not_negated(run_protocol):
    text = "My ankles have more swelling since yesterday."
    expected = [
        {
            "text": "swelling",
            "label": "SYMPTOM",
            "is_negated": False,
            "is_uncertain": False,
            "is_historical": False,
            "is_hypothetical": False,
            "is_family": False,
        }
    ]
    assert run_protocol(text) == expected


def test_family_member_is_not_the_patient(run_protocol):
    text = "My brother has a cough."
    expected = [
        {
            "text": "cough",
            "label": "SYMPTOM",
            "is_negated": False,
            "is_uncertain": False,
            "is_historical": False,
            "is_hypothetical": False,
            "is_family": True,
        }
    ]
    assert run_protocol(text) == expected


def test_negation_stops_at_but(run_protocol):
    text = "No fever but my cough is worse."
    expected = [
        {
            "text": "fever",
            "label": "SYMPTOM",
            "is_negated": True,
            "is_uncertain": False,
            "is_historical": False,
            "is_hypothetical": False,
            "is_family": False,
        },
        {
            "text": "cough",
            "label": "SYMPTOM",
            "is_negated": False,
            "is_uncertain": False,
            "is_historical": False,
            "is_hypothetical": False,
            "is_family": False,
        },
    ]
    assert run_protocol(text) == expected


def test_uncertain_symptom_is_uncertain(run_protocol):
    text = "Possible shortness of breath when I walk up stairs."
    expected = [
        {
            "text": "shortness of breath",
            "label": "SYMPTOM",
            "is_negated": False,
            "is_uncertain": True,
            "is_historical": False,
            "is_hypothetical": False,
            "is_family": False,
        }
    ]
    assert run_protocol(text) == expected


def test_historical_symptom_is_historical(run_protocol):
    text = "I have a history of dizziness."
    expected = [
        {
            "text": "dizziness",
            "label": "SYMPTOM",
            "is_negated": False,
            "is_uncertain": False,
            "is_historical": True,
            "is_hypothetical": False,
            "is_family": False,
        }
    ]
    assert run_protocol(text) == expected


def test_hypothetical_symptom_is_hypothetical(run_protocol):
    text = "The clinic said to message if fever develops."
    expected = [
        {
            "text": "fever",
            "label": "SYMPTOM",
            "is_negated": False,
            "is_uncertain": False,
            "is_historical": False,
            "is_hypothetical": True,
            "is_family": False,
        }
    ]
    assert run_protocol(text) == expected


def test_sentence_boundary_isolation(run_protocol):
    text = "No fever. I have noticed weight gain."
    expected = [
        {
            "text": "fever",
            "label": "SYMPTOM",
            "is_negated": True,
            "is_uncertain": False,
            "is_historical": False,
            "is_hypothetical": False,
            "is_family": False,
        },
        {
            "text": "weight gain",
            "label": "SYMPTOM",
            "is_negated": False,
            "is_uncertain": False,
            "is_historical": False,
            "is_hypothetical": False,
            "is_family": False,
        },
    ]
    assert run_protocol(text) == expected


# ============================================================================
# Edge Cases
# ============================================================================

def test_empty_message_returns_empty_list(run_protocol):
    text = ""
    assert run_protocol(text) == []


def test_no_symptoms_found(run_protocol):
    text = "I slept well and had breakfast at 8am."
    assert run_protocol(text) == []


def test_case_insensitivity_matching(run_protocol):
    text = "I HAVE SEVERE CHEST PAIN."
    expected = [
        {
            "text": "CHEST PAIN",
            "label": "SYMPTOM",
            "is_negated": False,
            "is_uncertain": False,
            "is_historical": False,
            "is_hypothetical": False,
            "is_family": False,
        }
    ]
    assert run_protocol(text) == expected


def test_very_long_message(run_protocol):
    text = "My ankles have swelling. " * 500
    results = run_protocol(text)
    assert len(results) == 500
    assert all(r["text"] == "swelling" and not r["is_negated"] for r in results)


# ============================================================================
# Edge Cases: Known Limitations (xfail strict=True)
# ============================================================================

@pytest.mark.xfail(strict=True, reason="Known limitation: 'not sure if' read as negation")
def test_not_sure_is_uncertain(run_protocol):
    text = "I'm not sure if the swelling is worse."
    expected = [
        {
            "text": "swelling",
            "label": "SYMPTOM",
            "is_negated": False,
            "is_uncertain": True,
            "is_historical": False,
            "is_hypothetical": False,
            "is_family": False,
        }
    ]
    assert run_protocol(text) == expected


@pytest.mark.xfail(strict=True, reason="Known limitation: 'no improvement in' falsely triggers negation")
def test_no_improvement_is_not_negated(run_protocol):
    text = "No improvement in my shortness of breath."
    expected = [
        {
            "text": "shortness of breath",
            "label": "SYMPTOM",
            "is_negated": False,
            "is_uncertain": False,
            "is_historical": False,
            "is_hypothetical": False,
            "is_family": False,
        }
    ]
    assert run_protocol(text) == expected


@pytest.mark.xfail(strict=True, reason="Known limitation: 'not only' falsely triggers negation")
def test_not_only_is_not_negated(run_protocol):
    text = "Not only chest pain but also swelling."
    expected = [
        {
            "text": "chest pain",
            "label": "SYMPTOM",
            "is_negated": False,
            "is_uncertain": False,
            "is_historical": False,
            "is_hypothetical": False,
            "is_family": False,
        },
        {
            "text": "swelling",
            "label": "SYMPTOM",
            "is_negated": False,
            "is_uncertain": False,
            "is_historical": False,
            "is_hypothetical": False,
            "is_family": False,
        },
    ]
    assert run_protocol(text) == expected


@pytest.mark.xfail(
    strict=True,
    reason="Known limitation: post-target question followed by 'No' does not attach negation across sentence boundary",
)
def test_question_followed_by_no_is_negated(run_protocol):
    text = "Chest pain? No."
    expected = [
        {
            "text": "Chest pain",
            "label": "SYMPTOM",
            "is_negated": True,
            "is_uncertain": False,
            "is_historical": False,
            "is_hypothetical": False,
            "is_family": False,
        }
    ]
    assert run_protocol(text) == expected


@pytest.mark.xfail(
    strict=True,
    reason="Known limitation: modal 'might have' not matched as uncertainty modifier in default rules",
)
def test_might_have_is_uncertain(run_protocol):
    text = "I might have some swelling."
    expected = [
        {
            "text": "swelling",
            "label": "SYMPTOM",
            "is_negated": False,
            "is_uncertain": True,
            "is_historical": False,
            "is_hypothetical": False,
            "is_family": False,
        }
    ]
    assert run_protocol(text) == expected


@pytest.mark.xfail(
    strict=True,
    reason="Known limitation: lay phrasing 'short of breath' not covered by exact TargetRule('shortness of breath')",
)
def test_lay_wording_is_found(run_protocol):
    text = "I'm a little short of breath, nothing major."
    expected = [
        {
            "text": "short of breath",
            "label": "SYMPTOM",
            "is_negated": False,
            "is_uncertain": False,
            "is_historical": False,
            "is_hypothetical": False,
            "is_family": False,
        }
    ]
    assert run_protocol(text) == expected


# ============================================================================
# Invalid Inputs
# ============================================================================

def test_none_raises_value_error(run_protocol):
    with pytest.raises(ValueError):
        run_protocol(None)


def test_number_raises_value_error(run_protocol):
    with pytest.raises(ValueError):
        run_protocol(98.6)