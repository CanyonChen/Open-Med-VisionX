"""Exceptions shared by the model-inference extension layer.

The inference package deliberately has no dependency on a model runtime.  Its
errors therefore live in a small standalone module that can be imported by a
viewer installation that has no ONNX, PyTorch, CUDA, or plugin dependencies.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One manifest or plugin-contract validation problem."""

    path: str
    message: str
    value: Any = None

    def __str__(self) -> str:
        suffix = "" if self.value is None else f" (got {self.value!r})"
        return f"{self.path}: {self.message}{suffix}"


class InferenceError(Exception):
    """Base class for public inference-layer failures."""


class InferenceCancelledError(InferenceError):
    """Raised when a model request is cancelled before producing a result."""


class ManifestError(InferenceError):
    """Base class for manifest loading and validation failures."""


class ManifestNotFoundError(ManifestError, FileNotFoundError):
    """Raised when a requested ``manifest.yaml`` does not exist."""


class ManifestDependencyError(ManifestError, ImportError):
    """Raised when YAML support is requested but PyYAML is unavailable."""


class ManifestFormatError(ManifestError, ValueError):
    """Raised when a manifest cannot be decoded as one safe YAML document."""


class ManifestValidationError(ManifestError, ValueError):
    """Raised when a decoded manifest violates the stable schema."""

    def __init__(self, issues: ValidationIssue | Iterable[ValidationIssue]):
        normalized = (issues,) if isinstance(issues, ValidationIssue) else tuple(issues)
        if not normalized:
            normalized = (ValidationIssue("manifest", "validation failed"),)
        self.issues = normalized
        details = "\n".join(f"  - {issue}" for issue in normalized)
        super().__init__(f"Model manifest validation failed:\n{details}")


class SecurityBoundaryError(InferenceError, ValueError):
    """Base class for rejected unsafe inference-layer operations."""


class RemoteReferenceError(SecurityBoundaryError):
    """Raised when a manifest refers to a remote model or weight."""


class PathBoundaryError(SecurityBoundaryError):
    """Raised when an adapter entry point escapes its plugin directory."""


class UntrustedPluginError(SecurityBoundaryError, PermissionError):
    """Raised when Python adapter execution lacks explicit user consent."""


class PluginContractError(InferenceError, TypeError):
    """Raised when a plugin violates the stable ``ModelPlugin`` contract."""


class PluginNotLoadedError(InferenceError, RuntimeError):
    """Raised when prediction is attempted before a plugin is loaded."""
