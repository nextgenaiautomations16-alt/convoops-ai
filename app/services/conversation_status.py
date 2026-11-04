"""The single place that defines which conversation status changes are allowed."""

from app.models.conversation import Conversation, ConversationStatus
from app.core.exceptions import ConflictError

S = ConversationStatus

ALLOWED_TRANSITIONS: dict[ConversationStatus, frozenset[ConversationStatus]] = {
    S.OPEN: frozenset({S.AWAITING_RESPONSE, S.RESOLVED, S.CLOSED}),
    S.AWAITING_RESPONSE: frozenset({S.OPEN, S.RESOLVED, S.CLOSED}),
    S.RESOLVED: frozenset({S.OPEN, S.CLOSED}),  # a customer message may reopen a resolved conversation
    S.CLOSED: frozenset(),  # final: closed conversations are never reopened
}


class InvalidStatusTransitionError(ConflictError):
    code = "invalid_status_transition"


def ensure_transition_allowed(current: ConversationStatus, target: ConversationStatus) -> None:
    """Raise InvalidStatusTransitionError unless current -> target is allowed (same status is a no-op)."""
    if target is current:
        return
    if target not in ALLOWED_TRANSITIONS[current]:
        raise InvalidStatusTransitionError(
            f"Cannot change conversation status from '{current.value}' to '{target.value}'."
        )


def apply_status(conversation: Conversation, target: ConversationStatus) -> None:
    """Validate and set the status. Does not commit."""
    ensure_transition_allowed(conversation.status, target)
    conversation.status = target
