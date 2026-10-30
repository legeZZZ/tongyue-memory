"""AML Cycle 2 Add / Search bodies.

Search accepts ``options`` and ignores them. Choice text is not evidence and
is not an answer key.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: str
    timestamp: int | None = None

    @field_validator("role")
    @classmethod
    def role_present(cls, value: str) -> str:
        if not isinstance(value, str) or value == "":
            raise ValueError("role is required")
        return value

    @field_validator("content")
    @classmethod
    def content_present(cls, value: str) -> str:
        if not isinstance(value, str) or value.strip() == "":
            raise ValueError("content must be a non-empty string")
        return value


class AddRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request_id: str
    messages: list[Message]
    user_id: str
    session_id: str

    @field_validator("request_id", "user_id", "session_id")
    @classmethod
    def id_present(cls, value: str) -> str:
        if not isinstance(value, str) or value == "":
            raise ValueError("identifier must be a non-empty string")
        return value

    @field_validator("messages")
    @classmethod
    def messages_present(cls, value: list[Message]) -> list[Message]:
        if not value:
            raise ValueError("messages must be a non-empty array")
        return value


class AddResponse(BaseModel):
    success: bool
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str
    user_id: str
    top_k: int
    options: list[str] | None = None

    @field_validator("query", "user_id")
    @classmethod
    def text_present(cls, value: str) -> str:
        if not isinstance(value, str) or value.strip() == "":
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("top_k")
    @classmethod
    def top_k_ok(cls, value: int) -> int:
        if value < 1:
            raise ValueError("top_k must be >= 1")
        if value > 1000:
            raise ValueError("top_k must be <= 1000")
        return value


class SearchHit(BaseModel):
    id: str
    content: str
    score: float
    created_at: str


class SearchResponse(BaseModel):
    data: list[SearchHit] = Field(default_factory=list)
