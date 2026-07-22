"""Defensive redaction for logs and error surfaces."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any

REDACTED = "[REDACTED]"

_EXACT_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "credentials",
    "password",
    "secret",
    "client_secret",
    "access_token",
    "refresh_token",
    "id_token",
    "patient_name",
    "patient_id",
    "patient_birth_date",
    "patient_address",
    "accession_number",
    "medical_record_number",
    "referring_physician",
    "operator_name",
}

_PHI_KEY_FRAGMENTS = (
    "patient_",
    "patientname",
    "patientid",
    "birth_date",
    "birthdate",
    "accession",
    "physician",
    "medical_record",
    "telephone",
)

_TEXT_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(
        r"(?i)\b(api[_-]?key|authorization|access[_-]?token|refresh[_-]?token|"
        r"client[_-]?secret|password|patient[_-]?(?:name|id|birth[_-]?date)|"
        r"accession[_-]?number)\b\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^,;\r\n]+)"
    ),
    re.compile(
        r"(?i)\b(PatientName|PatientID|PatientBirthDate|AccessionNumber)\b"
        r"(?:\s+[A-Z]{2})?\s*[:=]\s*[^,;\r\n]+"
    ),
    re.compile(r"(?i)(\(\s*0010\s*,\s*[0-9a-f]{4}\s*\))[^\r\n]*"),
    re.compile(
        r"(?i)(?:\b[A-Z]:[\\/]|(?<!\w)/)[^,;\r\n]*?"
        r"\.(?:dcm|dicom|nii(?:\.gz)?|mha|mhd)\b"
    ),
)


def _normalize_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")


def is_sensitive_key(key: object) -> bool:
    normalized = _normalize_key(key)
    if normalized in _EXACT_SENSITIVE_KEYS:
        return True
    if normalized.endswith(("_api_key", "_secret", "_password", "_credential")):
        return True
    if normalized.endswith("_token") and not normalized.endswith("_tokens"):
        return True
    return any(fragment in normalized for fragment in _PHI_KEY_FRAGMENTS)


def redact_text(value: object) -> str:
    """Remove common credential and PHI assignments from free-form text."""

    text = str(value)
    text = _TEXT_PATTERNS[0].sub("Bearer " + REDACTED, text)
    text = _TEXT_PATTERNS[1].sub(REDACTED, text)

    def replace_assignment(match: re.Match[str]) -> str:
        return f"{match.group(1)}={REDACTED}"

    text = _TEXT_PATTERNS[2].sub(replace_assignment, text)
    text = _TEXT_PATTERNS[3].sub(replace_assignment, text)
    text = _TEXT_PATTERNS[4].sub(
        lambda match: f"{match.group(1)} {REDACTED}",
        text,
    )
    return _TEXT_PATTERNS[5].sub(f"<medical-file-path:{REDACTED}>", text)


def redact(value: Any) -> Any:
    """Return a recursively redacted, log-safe representation."""

    if isinstance(value, Mapping):
        return {
            str(key): REDACTED if is_sensitive_key(key) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, set):
        return {redact(item) for item in value}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<binary:{len(value)} bytes>"
    if isinstance(value, str):
        return redact_text(value)
    return value


class RedactingFilter(logging.Filter):
    """Render then sanitize a log record before any handler sees it.

    Tracebacks are collapsed to an exception type with redacted details because
    request payloads, local file names, or credentials may otherwise appear in
    stack locals or exception text.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        if record.exc_info:
            error_type, _, _ = record.exc_info
            if error_type is not None:
                message += f" | {error_type.__name__}: {REDACTED}"
        record.msg = redact_text(message)
        record.args = ()
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True


def install_redaction(logger: logging.Logger) -> RedactingFilter:
    """Install one redaction filter on a logger and all existing handlers."""

    redactor = next(
        (existing for existing in logger.filters if isinstance(existing, RedactingFilter)),
        None,
    )
    if redactor is None:
        redactor = RedactingFilter()
        logger.addFilter(redactor)
    for handler in logger.handlers:
        if redactor not in handler.filters:
            handler.addFilter(redactor)
    return redactor
