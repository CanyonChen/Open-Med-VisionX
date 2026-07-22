"""Runtime services shared by GUI, loaders, algorithms, and plugins."""

from .credentials import (
    CredentialReference,
    CredentialResolutionError,
    CredentialResolver,
)
from .privacy import (
    REDACTED,
    RedactingFilter,
    install_redaction,
    redact,
    redact_text,
)
from .state import AtomicSessionState, SessionConflictError, SessionSnapshot
from .tasks import (
    BackgroundTask,
    CancellationToken,
    TaskContext,
    TaskProgress,
    TaskRunner,
    TaskStatus,
)

__all__ = [
    "AtomicSessionState",
    "BackgroundTask",
    "CancellationToken",
    "CredentialReference",
    "CredentialResolutionError",
    "CredentialResolver",
    "REDACTED",
    "RedactingFilter",
    "SessionConflictError",
    "SessionSnapshot",
    "TaskContext",
    "TaskProgress",
    "TaskRunner",
    "TaskStatus",
    "install_redaction",
    "redact",
    "redact_text",
]
