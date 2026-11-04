"""Deterministic safe replies shared by every provider.

A provider must NEVER send a medical question to a language model. When
`request.medical_topic` is true it returns `medical_safe_reply(...)` instead. The reply is
labelled with SAFETY_PROVIDER / SAFETY_MODEL so stored messages show that no model wrote it.
"""

SAFETY_PROVIDER = "safety_guard"
SAFETY_MODEL = "deterministic-medical-v1"

_MEDICAL_SAFE_REPLY = (
    "I'm not able to give medical advice, diagnoses or treatment recommendations. "
    "Please contact the {business} staff or another qualified clinician about this. "
    "If you think this could be an emergency, call your local emergency number right away."
)


def medical_safe_reply(business_name: str) -> str:
    """The fixed reply for medical questions. str.replace (not format) so braces in a name are harmless."""
    return _MEDICAL_SAFE_REPLY.replace("{business}", business_name)
