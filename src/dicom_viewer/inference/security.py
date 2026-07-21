"""Security boundaries for externally supplied model manifests and adapters.

These helpers validate intent and paths.  They do not claim to be an operating
system sandbox; the runtime/orchestrator must still launch Python adapters in an
isolated subprocess with the declared environment and platform restrictions.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from .enums import RuntimeKind
from .errors import (
    ManifestValidationError,
    PathBoundaryError,
    RemoteReferenceError,
    SecurityBoundaryError,
    UntrustedPluginError,
    ValidationIssue,
)

if TYPE_CHECKING:
    from .manifest import ModelManifest


PYTHON_ADAPTER_WARNING = (
    "Python model adapters execute third-party code. Review the adapter and its "
    "licenses, then run it only in the declared isolated subprocess environment."
)

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_PYTHON_OBJECT_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")


@dataclass(frozen=True, slots=True)
class InferenceSecurityPolicy:
    """Non-relaxable platform invariants for local inference plugins."""

    allow_downloads: bool = False
    allow_weight_copy: bool = False
    allow_dependency_install: bool = False
    require_python_subprocess: bool = True

    def __post_init__(self) -> None:
        if self.allow_downloads:
            raise SecurityBoundaryError("automatic model/weight downloads are forbidden")
        if self.allow_weight_copy:
            raise SecurityBoundaryError("copying weights into the project is forbidden")
        if self.allow_dependency_install:
            raise SecurityBoundaryError("automatic plugin dependency installation is forbidden")
        if not self.require_python_subprocess:
            raise SecurityBoundaryError("Python adapters must run in a separate subprocess")


DEFAULT_SECURITY_POLICY = InferenceSecurityPolicy()


def assert_local_reference(reference: str, *, field: str = "path") -> None:
    """Reject URLs, UNC shares, and other non-local model references."""

    if not isinstance(reference, str) or not reference.strip():
        raise SecurityBoundaryError(f"{field} must be a non-empty local path")
    value = reference.strip()
    if "\x00" in value:
        raise SecurityBoundaryError(f"{field} must not contain NUL characters")
    if value.startswith(("\\\\", "//")):
        raise RemoteReferenceError(f"{field} must not use a UNC/network path: {value!r}")
    # urllib treats C:\model.onnx as scheme "c"; exempt an actual drive prefix.
    if not _WINDOWS_DRIVE_RE.match(value):
        parsed = urlsplit(value)
        if parsed.scheme or parsed.netloc:
            raise RemoteReferenceError(
                f"{field} must be a native local path, not a URL/URI: {value!r}"
            )


def resolve_local_reference(
    reference: str,
    plugin_root: str | Path,
    *,
    field: str = "path",
    require_within_root: bool = False,
) -> Path:
    """Resolve a local reference without opening, copying, or downloading it."""

    assert_local_reference(reference, field=field)
    root = Path(plugin_root).expanduser().resolve(strict=False)
    candidate = Path(reference).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=False)
    if require_within_root:
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise PathBoundaryError(f"{field} escapes plugin root {root}: {reference!r}") from exc
    return resolved


def validate_manifest_references(
    manifest: ModelManifest,
    plugin_root: str | Path,
) -> None:
    """Validate reference boundaries without requiring any file to exist."""

    root = Path(plugin_root).expanduser().resolve(strict=False)
    issues: list[ValidationIssue] = []
    try:
        resolve_local_reference(
            manifest.entrypoint.path,
            root,
            field="entrypoint.path",
            require_within_root=True,
        )
    except SecurityBoundaryError as exc:
        issues.append(ValidationIssue("entrypoint.path", str(exc)))

    python_env = manifest.runtime.python
    if python_env is not None:
        if python_env.python_executable is not None:
            try:
                assert_local_reference(
                    python_env.python_executable,
                    field="runtime.python.python_executable",
                )
            except SecurityBoundaryError as exc:
                issues.append(ValidationIssue("runtime.python.python_executable", str(exc)))
        if python_env.requirements_file is not None:
            try:
                resolve_local_reference(
                    python_env.requirements_file,
                    root,
                    field="runtime.python.requirements_file",
                    require_within_root=True,
                )
            except SecurityBoundaryError as exc:
                issues.append(ValidationIssue("runtime.python.requirements_file", str(exc)))
    for index, weight in enumerate(manifest.weights):
        try:
            assert_local_reference(weight.path, field=f"weights[{index}].path")
        except SecurityBoundaryError as exc:
            issues.append(ValidationIssue(f"weights[{index}].path", str(exc)))
    if issues:
        raise ManifestValidationError(issues)


def validate_manifest_files(
    manifest: ModelManifest,
    plugin_root: str | Path,
    *,
    verify_hashes: bool = False,
) -> None:
    """Validate declared local files without loading a model or adapter.

    Adapter entrypoints and plugin requirement files must remain inside the
    plugin directory.  Weight files may live elsewhere on the user's machine;
    they are referenced in place and are never copied into the project.
    """

    root = Path(plugin_root).expanduser().resolve(strict=False)
    validate_manifest_references(manifest, root)
    issues: list[ValidationIssue] = []
    if not root.exists() or not root.is_dir():
        issues.append(ValidationIssue("plugin_root", "must be an existing directory", str(root)))
    try:
        entrypoint = resolve_local_reference(
            manifest.entrypoint.path,
            root,
            field="entrypoint.path",
            require_within_root=True,
        )
        if not entrypoint.exists():
            issues.append(
                ValidationIssue("entrypoint.path", "file does not exist", str(entrypoint))
            )
        elif not entrypoint.is_file():
            issues.append(
                ValidationIssue("entrypoint.path", "must reference a regular file", str(entrypoint))
            )
        if manifest.runtime.kind is RuntimeKind.PYTHON_ADAPTER:
            if entrypoint.suffix.lower() != ".py":
                issues.append(
                    ValidationIssue(
                        "entrypoint.path",
                        "Python adapter entrypoint must be a .py file",
                    )
                )
            object_name = manifest.entrypoint.object_name or ""
            if not _PYTHON_OBJECT_RE.fullmatch(object_name):
                issues.append(
                    ValidationIssue(
                        "entrypoint.object",
                        "must be a dotted Python identifier",
                        object_name,
                    )
                )
    except SecurityBoundaryError as exc:
        issues.append(ValidationIssue("entrypoint.path", str(exc)))

    python_env = manifest.runtime.python
    if python_env is not None and python_env.requirements_file is not None:
        try:
            requirements = resolve_local_reference(
                python_env.requirements_file,
                root,
                field="runtime.python.requirements_file",
                require_within_root=True,
            )
            if not requirements.is_file():
                issues.append(
                    ValidationIssue(
                        "runtime.python.requirements_file",
                        "file does not exist",
                        str(requirements),
                    )
                )
        except SecurityBoundaryError as exc:
            issues.append(ValidationIssue("runtime.python.requirements_file", str(exc)))

    for index, weight in enumerate(manifest.weights):
        path = f"weights[{index}].path"
        try:
            resolved = resolve_local_reference(weight.path, root, field=path)
        except SecurityBoundaryError as exc:
            issues.append(ValidationIssue(path, str(exc)))
            continue
        if not resolved.exists():
            if weight.required:
                issues.append(
                    ValidationIssue(
                        path,
                        "required weight file does not exist",
                        str(resolved),
                    )
                )
            continue
        if not resolved.is_file():
            issues.append(ValidationIssue(path, "must reference a regular file", str(resolved)))
            continue
        stat_size = resolved.stat().st_size
        if weight.size_bytes is not None and stat_size != weight.size_bytes:
            issues.append(
                ValidationIssue(
                    f"weights[{index}].size_bytes",
                    f"declared {weight.size_bytes} bytes but file has {stat_size} bytes",
                )
            )
        if verify_hashes and weight.sha256 is not None:
            actual = sha256_file(resolved)
            if actual.lower() != weight.sha256.lower():
                issues.append(
                    ValidationIssue(
                        f"weights[{index}].sha256",
                        "SHA-256 digest does not match the local file",
                        actual,
                    )
                )
    if issues:
        raise ManifestValidationError(issues)


def require_python_adapter_consent(
    manifest: ModelManifest,
    *,
    user_consented: bool,
    subprocess_isolated: bool,
) -> None:
    """Enforce the explicit-warning and subprocess boundary before execution."""

    if manifest.runtime.kind is not RuntimeKind.PYTHON_ADAPTER:
        return
    if not user_consented:
        raise UntrustedPluginError(f"Explicit user consent is required. {PYTHON_ADAPTER_WARNING}")
    if not subprocess_isolated:
        raise UntrustedPluginError("Python adapters may not execute in the GUI process")


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Read-only integrity verification for a user-selected local weight file."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
