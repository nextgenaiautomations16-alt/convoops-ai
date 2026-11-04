"""Offline, deterministic provider for tests and demos. Needs no API key or network.

This is NOT an LLM. It selects a fixed template by intent. Replies are written for
the fictional BrightPath Telehealth demo: no prices, no availability, no medical
advice, and no claim that a staff member has actually been notified.
"""

from app.providers.base import LLMRequest, LLMResponse
from app.providers.safety import SAFETY_MODEL, SAFETY_PROVIDER, medical_safe_reply
from app.services.intent_classifier import Intent

_TEMPLATES: dict[str, str] = {
    Intent.APPOINTMENT_BOOKING.value: (
        "Thanks for contacting {business}. I can help you start an appointment request. "
        "Please tell me your preferred days and times and the clinic team will confirm what is available. "
        "I can't see or confirm real appointment times in this demo."
    ),
    Intent.PRICING.value: (
        "Thanks for asking about pricing at {business}. Fees depend on the type of visit, and I don't have "
        "verified pricing to share in this demo. Please ask the clinic staff for current fees."
    ),
    Intent.RESCHEDULE.value: (
        "I can help you ask to reschedule your visit with {business}. Please share your current appointment "
        "date and a few times that would work better. Clinic staff will confirm any change; "
        "I can't change appointments myself in this demo."
    ),
    Intent.CANCELLATION.value: (
        "I'm sorry to hear you need to cancel your visit with {business}. Please confirm the date and time of the "
        "appointment you want to cancel. Clinic staff will confirm the cancellation; "
        "I can't cancel appointments myself in this demo."
    ),
    Intent.GENERAL_QUESTION.value: (
        "Thanks for your question about {business}. I don't have verified clinic details such as hours or "
        "services in this demo. Please contact the clinic staff for specifics, or tell me more about what you need."
    ),
    Intent.HUMAN_HANDOFF.value: (
        "Your request to speak with a person at {business} has been recorded in this conversation so a staff "
        "member can follow up. This demo does not notify anyone in real time, so if it is urgent please "
        "contact the clinic directly."
    ),
    Intent.UNKNOWN.value: (
        "Thanks for your message to {business}. I'm not sure I understood it. Could you tell me a bit more "
        "about what you need, such as booking, rescheduling, cancelling, pricing, or speaking with a staff member?"
    ),
}


class MockLLMProvider:
    """Deterministic provider: the same request always produces the same reply."""

    name = "mock"
    model = "mock-deterministic-v1"

    def generate(self, request: LLMRequest) -> LLMResponse:
        if request.medical_topic:
            return LLMResponse(
                content=medical_safe_reply(request.business_name),
                provider=SAFETY_PROVIDER,
                model=SAFETY_MODEL,
            )

        template = _TEMPLATES.get(request.intent, _TEMPLATES["unknown"])
        # str.replace (not str.format) so a business name containing braces cannot break the template
        content = template.replace("{business}", request.business_name)
        return LLMResponse(content=content, provider=self.name, model=self.model)