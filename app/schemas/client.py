"""Pydantic request/response schemas for clients."""

import re
import uuid
from datetime import datetime
from functools import lru_cache
from zoneinfo import available_timezones

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.client import ConfigurationStatus

SLUG_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_NON_NULLABLE_UPDATE_FIELDS = (
    "business_name",
    "industry",
    "timezone",
    "business_description",
    "configuration_status",
)


@lru_cache
def _valid_timezones() -> frozenset[str]:
    return frozenset(available_timezones())


def _check_timezone(value: str | None) -> str | None:
    if value is not None and value not in _valid_timezones():
        raise ValueError("must be a valid IANA timezone name, e.g. 'America/New_York'")
    return value


def _check_email(value: str | None) -> str | None:
    if value is None:
        return None
    if not _EMAIL_RE.match(value):
        raise ValueError("must be a valid email address")
    return value.lower()


class ClientCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    business_name: str = Field(min_length=1, max_length=200)
    slug: str = Field(
        min_length=2,
        max_length=64,
        pattern=SLUG_PATTERN,
        description="Lowercase letters, digits and single hyphens, e.g. 'sunrise-dental'.",
    )
    industry: str = Field(min_length=1, max_length=100)
    timezone: str = Field(max_length=64, description="IANA timezone, e.g. 'America/Chicago'.")
    business_description: str = Field(min_length=1, max_length=2000)
    contact_email: str | None = Field(default=None, max_length=254)
    configuration_status: ConfigurationStatus = ConfigurationStatus.DRAFT

    @field_validator("timezone")
    @classmethod
    def _timezone(cls, value: str) -> str:
        return _check_timezone(value)  # type: ignore[return-value]

    @field_validator("contact_email")
    @classmethod
    def _email(cls, value: str | None) -> str | None:
        return _check_email(value)


class ClientUpdate(BaseModel):
    """Partial update. Only fields present in the request body are changed.

    The slug is intentionally not updatable: it is the stable external
    identifier that later phases (webhooks, knowledge bases) will reference.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    business_name: str | None = Field(default=None, min_length=1, max_length=200)
    industry: str | None = Field(default=None, min_length=1, max_length=100)
    timezone: str | None = Field(default=None, max_length=64)
    business_description: str | None = Field(default=None, min_length=1, max_length=2000)
    contact_email: str | None = Field(default=None, max_length=254)  # null clears it
    configuration_status: ConfigurationStatus | None = None

    @field_validator("timezone")
    @classmethod
    def _timezone(cls, value: str | None) -> str | None:
        return _check_timezone(value)

    @field_validator("contact_email")
    @classmethod
    def _email(cls, value: str | None) -> str | None:
        return _check_email(value)

    @model_validator(mode="after")
    def _check_payload(self) -> "ClientUpdate":
        if not self.model_fields_set:
            raise ValueError("at least one field must be provided")
        for name in _NON_NULLABLE_UPDATE_FIELDS:
            if name in self.model_fields_set and getattr(self, name) is None:
                raise ValueError(f"'{name}' cannot be null")
        return self


class ClientRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    business_name: str
    slug: str
    industry: str
    timezone: str
    business_description: str
    contact_email: str | None
    configuration_status: ConfigurationStatus
    created_at: datetime
    updated_at: datetime


class ClientList(BaseModel):
    items: list[ClientRead]
    total: int
    limit: int
    offset: int


class ErrorDetail(BaseModel):
    field: str
    message: str
    type: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[ErrorDetail] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
