"""Domain exceptions. Each carries a stable machine-readable ``code`` and an HTTP status."""

from __future__ import annotations

from typing import Any


class LegalBotError(Exception):
    code = "INTERNAL_ERROR"
    status_code = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            body["details"] = self.details
        return body


# --- Upload / parsing -------------------------------------------------------
class UnsupportedFileTypeError(LegalBotError):
    code = "UNSUPPORTED_FILE_TYPE"
    status_code = 415


class FileTooLargeError(LegalBotError):
    code = "FILE_TOO_LARGE"
    status_code = 413


class PasswordRequiredError(LegalBotError):
    code = "PASSWORD_REQUIRED"
    status_code = 422


class InvalidPasswordError(LegalBotError):
    code = "INVALID_PASSWORD"
    status_code = 422


class DocumentUnreadableError(LegalBotError):
    code = "DOCUMENT_UNREADABLE"
    status_code = 422


class TooManyPagesError(LegalBotError):
    code = "TOO_MANY_PAGES"
    status_code = 422


# --- Document lifecycle -----------------------------------------------------
class DocumentNotFoundError(LegalBotError):
    code = "DOCUMENT_NOT_FOUND"
    status_code = 404


class DocumentNotReadyError(LegalBotError):
    code = "DOCUMENT_NOT_READY"
    status_code = 409


# --- LLM --------------------------------------------------------------------
class ModelUnavailableError(LegalBotError):
    code = "MODEL_UNAVAILABLE"
    status_code = 503


class PromptTooLongError(LegalBotError):
    code = "PROMPT_TOO_LONG"
    status_code = 422


class LLMGenerationError(LegalBotError):
    code = "GENERATION_FAILED"
    status_code = 500
