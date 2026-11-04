"""Deterministic keyword/regex intent classifier.

THIS IS NOT AN LLM. It matches word patterns, so it cannot understand negation
("don't cancel") or context. Confidence values are fixed heuristic scores, not
probabilities. It is kept separate from the provider layer so it can later be
replaced or supplemented by an LLM classifier without touching the pipeline.
"""

import re
from dataclasses import dataclass
from enum import Enum


class Intent(str, Enum):
    APPOINTMENT_BOOKING = "appointment_booking"
    PRICING = "pricing"
    RESCHEDULE = "reschedule"
    CANCELLATION = "cancellation"
    GENERAL_QUESTION = "general_question"
    HUMAN_HANDOFF = "human_handoff"
    UNKNOWN = "unknown"


CLASSIFICATION_METHOD = "deterministic_rules_v1"

STRONG_SCORE = 0.90  # a clear, intent-specific phrase matched
WEAK_SCORE = 0.70  # only a loosely related word matched
QUESTION_ONLY_SCORE = 0.50  # contains '?' but matched nothing else
UNKNOWN_SCORE = 0.20


@dataclass(frozen=True)
class IntentResult:
    intent: Intent
    confidence: float
    method: str = CLASSIFICATION_METHOD


def _compile(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


_PEOPLE = r"(?:human|person|people|someone|somebody|agent|representative|rep|staff|receptionist|nurse|doctor|operator|manager|team)"

# (intent, strong patterns, weak patterns), in tie-break priority order.
_RULES: tuple[tuple[Intent, tuple[re.Pattern[str], ...], tuple[re.Pattern[str], ...]], ...] = (
    (
        Intent.HUMAN_HANDOFF,
        _compile(
            rf"\b(?:speak|talk|chat|connect)\s+(?:to|with)\s+(?:(?:a|an|the|your)\s+)?(?:(?:real|live)\s+)?{_PEOPLE}\b",
            r"\b(?:human|live|real)\s+(?:agent|representative|person|being)\b",
        ),
        _compile(r"\b(?:agent|representative|human|operator|receptionist)\b", r"\bcustomer service\b", r"\bcall (?:me|back)\b"),
    ),
    (
        Intent.CANCELLATION,
        _compile(r"\bcancel\w*"),
        _compile(r"\bcall off\b", r"\bno longer (?:need|want)\b", r"\bnot (?:going|able) to (?:come|attend)\b"),
    ),
    (
        Intent.RESCHEDULE,
        _compile(
            r"\bre-?schedul\w*",
            r"\b(?:move|change|push|shift|postpone|delay)\s+(?:back\s+|up\s+)?(?:my|the|our|this)\s+(?:appointment|visit|booking|session|consultation)\b",
        ),
        _compile(
            r"\b(?:another|different|new|later|earlier)\s+(?:time|day|date|slot)\b",
            r"\b(?:can'?t|cannot|won'?t be able to)\s+(?:make|come|attend)\b",
        ),
    ),
    (
        Intent.PRICING,
        _compile(r"\bhow much\b", r"\b(?:price|prices|pricing|cost|costs|fee|fees|charge|charges)\b"),
        _compile(r"\b(?:insurance|copay|co-pay|payment|billing|coverage|covered|afford\w*|expensive|cheap|self-pay)\b", r"\bpay\b"),
    ),
    (
        Intent.APPOINTMENT_BOOKING,
        _compile(
            r"\b(?:book|booking)\b",
            r"\b(?:make|schedule|set up|arrange|request|need|want)\s+(?:an?\s+)?(?:appointment|visit|consultation|session)\b",
            r"\bsee\s+(?:(?:a|an|the|my)\s+)?(?:doctor|provider|clinician|physician|nurse|therapist)\b",
        ),
        _compile(r"\b(?:appointments?|availability|available|slots?|consultations?)\b", r"\bnew patient\b"),
    ),
    (
        Intent.GENERAL_QUESTION,
        _compile(
            r"\b(?:opening|office|business|clinic)\s+hours\b",
            r"\bwhat (?:are|is) your (?:hours|address|location|services)\b",
            r"\bdo you (?:offer|accept|provide|have|treat)\b",
            r"\b(?:where are you|parking)\b",
        ),
        _compile(
            r"\b(?:hours|open|located|location|address|services?|telehealth|virtual|online)\b",
            r"\bwhat (?:is|are|do|does)\b",
            r"\bhow (?:do|does|can|long)\b",
            r"\b(?:when|where|who)\b",
        ),
    ),
)

_PRIORITY = {rule[0]: index for index, rule in enumerate(_RULES)}

_MEDICAL = re.compile(
    r"\b(?:symptoms?|pain|painful|fever|cough|rash|bleeding|headache|migraine|nausea|dizzy|dizziness|"
    r"dosage|dose|medications?|medicine|prescri\w+|diagnos\w+|treatments?|side effects?|infection|allerg\w+|"
    r"pregnan\w+|blood pressure|chest|breath\w*|injur\w+|swelling|swollen|antibiotics?|overdose|suicid\w*|"
    r"emergency|unconscious|stroke|heart attack|vomit\w*|diabet\w+|anxiety|depress\w+)\b",
    re.IGNORECASE,
)


def classify_intent(text: str) -> IntentResult:
    """Return the best-matching intent for `text`.

    The highest score wins; ties go to the intent listed first in `_RULES`
    (human handoff, cancellation, reschedule, pricing, booking, general question).
    """
    best_intent: Intent | None = None
    best_score = 0.0
    for intent, strong, weak in _RULES:
        score = 0.0
        if any(p.search(text) for p in strong):
            score = STRONG_SCORE
        elif any(p.search(text) for p in weak):
            score = WEAK_SCORE
        elif intent is Intent.GENERAL_QUESTION and "?" in text:
            score = QUESTION_ONLY_SCORE
        if score > best_score or (score == best_score and score > 0 and _PRIORITY[intent] < _PRIORITY[best_intent]):  # type: ignore[index]
            best_intent, best_score = intent, score
    if best_intent is None:
        return IntentResult(Intent.UNKNOWN, UNKNOWN_SCORE)
    return IntentResult(best_intent, best_score)


def detect_medical_topic(text: str) -> bool:
    """True if the text mentions symptoms, medication, diagnosis or similar.

    A deterministic safety flag passed to the provider so replies point to qualified
    staff instead of giving medical advice. A keyword check can miss phrasings.
    """
    return _MEDICAL.search(text) is not None
