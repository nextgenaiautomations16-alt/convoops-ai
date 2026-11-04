"""Deterministic intent classifier tests."""

import pytest

from app.services.intent_classifier import (
    CLASSIFICATION_METHOD,
    Intent,
    IntentResult,
    classify_intent,
    detect_medical_topic,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello, I would like to book an appointment.", Intent.APPOINTMENT_BOOKING),
        ("Can I book a visit for next week?", Intent.APPOINTMENT_BOOKING),
        ("I want to see a doctor", Intent.APPOINTMENT_BOOKING),
        ("I'd like to make an appointment", Intent.APPOINTMENT_BOOKING),
        ("How much does a visit cost?", Intent.PRICING),
        ("What are your prices?", Intent.PRICING),
        ("Is there a fee for the first consultation?", Intent.PRICING),
        ("Can I reschedule to Friday?", Intent.RESCHEDULE),
        ("I need to move my appointment to next week", Intent.RESCHEDULE),
        ("Could we rescheduling it please", Intent.RESCHEDULE),
        ("I need to cancel my appointment tomorrow", Intent.CANCELLATION),
        ("Please cancel it", Intent.CANCELLATION),
        ("What are your opening hours?", Intent.GENERAL_QUESTION),
        ("Do you offer virtual visits?", Intent.GENERAL_QUESTION),
        ("Where are you located?", Intent.GENERAL_QUESTION),
        ("I want to talk to a real person", Intent.HUMAN_HANDOFF),
        ("Can I speak with a representative?", Intent.HUMAN_HANDOFF),
        ("let me talk to a human please", Intent.HUMAN_HANDOFF),
        ("asdf qwerty", Intent.UNKNOWN),
        ("Hi", Intent.UNKNOWN),
        ("", Intent.UNKNOWN),
    ],
)
def test_classifies_expected_intent(text: str, expected: Intent) -> None:
    assert classify_intent(text).intent is expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Can I reschedule or cancel?", Intent.CANCELLATION),  # tie -> priority order
        ("I want to talk to a person about cancelling", Intent.HUMAN_HANDOFF),
        ("How much is an appointment?", Intent.PRICING),  # pricing outranks the weak 'appointment' match
        ("Do I need insurance to book an appointment?", Intent.APPOINTMENT_BOOKING),  # strong beats weak
        ("I want to cancel my appointment", Intent.CANCELLATION),
        ("reschedule my appointment", Intent.RESCHEDULE),
    ],
)
def test_overlapping_intents_resolve_predictably(text: str, expected: Intent) -> None:
    assert classify_intent(text).intent is expected


def test_result_has_intent_confidence_and_explicit_method() -> None:
    result = classify_intent("I would like to book an appointment")
    assert isinstance(result, IntentResult)
    assert result.intent is Intent.APPOINTMENT_BOOKING
    assert result.method == CLASSIFICATION_METHOD == "deterministic_rules_v1"  # clearly NOT an LLM
    assert "llm" not in result.method.lower()
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.parametrize(
    "text, low, high",
    [
        ("book an appointment", 0.85, 0.95),  # strong match
        ("Is anyone available?", 0.65, 0.75),  # weak match
        ("Hmm, is that so?", 0.45, 0.55),  # question mark only
        ("asdf qwerty", 0.1, 0.3),  # unknown
    ],
)
def test_confidence_reflects_match_strength(text: str, low: float, high: float) -> None:
    assert low <= classify_intent(text).confidence <= high


def test_classification_is_deterministic_and_case_insensitive() -> None:
    assert classify_intent("CANCEL my VISIT") == classify_intent("cancel my visit")
    assert classify_intent("how much") == classify_intent("how much")


def test_unknown_text_with_a_question_mark_is_a_low_confidence_general_question() -> None:
    result = classify_intent("Hmm, is that so?")
    assert result.intent is Intent.GENERAL_QUESTION
    assert result.confidence < 0.6


def test_classifier_supports_exactly_the_documented_intents() -> None:
    assert {i.value for i in Intent} == {
        "appointment_booking", "pricing", "reschedule", "cancellation",
        "general_question", "human_handoff", "unknown",
    }  # fmt: skip


@pytest.mark.parametrize(
    "text",
    [
        "I have chest pain, what should I do?",
        "What dose of ibuprofen should I take?",
        "Can you diagnose this rash?",
        "I feel dizzy and have a fever",
        "Is this medication safe during pregnancy?",
    ],
)
def test_medical_topics_are_flagged(text: str) -> None:
    assert detect_medical_topic(text) is True


@pytest.mark.parametrize(
    "text", ["I would like to book an appointment", "How much does a visit cost?", "What are your hours?", ""]
)
def test_non_medical_text_is_not_flagged(text: str) -> None:
    assert detect_medical_topic(text) is False
