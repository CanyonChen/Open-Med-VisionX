"""Stable runtime-neutral contract implemented by external model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .enums import DeviceKind, ValidationSeverity, VisualizationKind
from .errors import InferenceCancelledError, ManifestValidationError, PluginContractError
from .manifest import CapabilitySpec, ModelManifest
from .results import InferenceResult
from .security import (
    DEFAULT_SECURITY_POLICY,
    InferenceSecurityPolicy,
    require_python_adapter_consent,
    validate_manifest_files,
)

if TYPE_CHECKING:
    from dicom_viewer.domain.transforms import TransformRecord


@dataclass(frozen=True, slots=True, kw_only=True)
class InferenceRequest:
    """Named model inputs plus explicit, non-secret prediction parameters."""

    inputs: Mapping[str, Any]
    request_id: str | None = None
    prompts: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    transform_records: Mapping[str, TransformRecord] = field(default_factory=dict)
    cancellation_check: Callable[[], bool] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ValueError("inference request must contain at least one named input")
        for name in self.inputs:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("inference input names must be non-empty strings")
        object.__setattr__(self, "inputs", dict(self.inputs))
        object.__setattr__(self, "prompts", dict(self.prompts))
        object.__setattr__(self, "parameters", dict(self.parameters))
        object.__setattr__(self, "transform_records", dict(self.transform_records))
        if self.cancellation_check is not None and not callable(self.cancellation_check):
            raise TypeError("cancellation_check must be callable")

    def raise_if_cancelled(self) -> None:
        """Raise a stable inference-layer error when cancellation was requested."""

        if self.cancellation_check is not None and self.cancellation_check():
            raise InferenceCancelledError("Model inference was cancelled.")


@dataclass(frozen=True, slots=True, kw_only=True)
class PluginValidationContext:
    plugin_root: Path
    verify_weight_hashes: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "plugin_root", Path(self.plugin_root).expanduser())


@dataclass(frozen=True, slots=True, kw_only=True)
class PluginValidationItem:
    severity: ValidationSeverity
    code: str
    message: str
    path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.severity, ValidationSeverity):
            object.__setattr__(self, "severity", ValidationSeverity.coerce(self.severity))
        if not self.code.strip() or not self.message.strip():
            raise ValueError("validation item code and message must not be empty")


@dataclass(frozen=True, slots=True, kw_only=True)
class PluginValidationReport:
    items: tuple[PluginValidationItem, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not any(item.severity is ValidationSeverity.ERROR for item in self.items)

    @property
    def errors(self) -> tuple[PluginValidationItem, ...]:
        return tuple(item for item in self.items if item.severity is ValidationSeverity.ERROR)

    @property
    def warnings(self) -> tuple[PluginValidationItem, ...]:
        return tuple(item for item in self.items if item.severity is ValidationSeverity.WARNING)

    def raise_for_errors(self) -> None:
        if self.is_valid:
            return
        details = "\n".join(
            f"  - {item.code}: {item.message}" + ("" if item.path is None else f" [{item.path}]")
            for item in self.errors
        )
        raise PluginContractError(f"Plugin validation failed:\n{details}")


@dataclass(frozen=True, slots=True, kw_only=True)
class PluginLoadContext:
    """Explicit authority and environment passed to ``ModelPlugin.load``."""

    plugin_root: Path
    device: DeviceKind = DeviceKind.AUTO
    user_consented_python_code: bool = False
    subprocess_isolated: bool = False
    runtime_options: Mapping[str, Any] = field(default_factory=dict)
    security_policy: InferenceSecurityPolicy = DEFAULT_SECURITY_POLICY

    def __post_init__(self) -> None:
        object.__setattr__(self, "plugin_root", Path(self.plugin_root).expanduser())
        if not isinstance(self.device, DeviceKind):
            object.__setattr__(self, "device", DeviceKind.coerce(self.device))
        if not isinstance(self.security_policy, InferenceSecurityPolicy):
            raise TypeError("security_policy must be an InferenceSecurityPolicy")
        object.__setattr__(self, "runtime_options", dict(self.runtime_options))


@dataclass(frozen=True, slots=True, kw_only=True)
class VisualizationContext:
    """Read-only source objects and display preferences supplied by the GUI."""

    sources: Mapping[str, Any] = field(default_factory=dict)
    preferences: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", dict(self.sources))
        object.__setattr__(self, "preferences", dict(self.preferences))


@dataclass(frozen=True, slots=True, kw_only=True)
class VisualizationArtifact:
    """A declarative visualization payload; rendering remains a UI concern."""

    kind: VisualizationKind
    title: str
    payload: Any
    coordinate_system: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, VisualizationKind):
            object.__setattr__(self, "kind", VisualizationKind.coerce(self.kind))
        if not self.title.strip():
            raise ValueError("visualization title must not be empty")
        object.__setattr__(self, "metadata", dict(self.metadata))


class ModelPlugin(ABC):
    """Stable extension point for ONNX, TorchScript, and Python adapters.

    A host discovers and validates a manifest before obtaining an implementation
    of this interface.  Manifest loading itself never imports the implementation.
    Python implementations must be hosted behind an isolated subprocess proxy.
    """

    @abstractmethod
    def describe(self) -> ModelManifest:
        """Return the immutable validated model manifest."""

    @abstractmethod
    def validate(self, context: PluginValidationContext | None = None) -> PluginValidationReport:
        """Validate the adapter contract and, optionally, its local files."""

    @abstractmethod
    def capabilities(self) -> CapabilitySpec:
        """Return capability flags; normally those declared in the manifest."""

    @abstractmethod
    def load(self, context: PluginLoadContext) -> None:
        """Load existing local model resources without download or installation."""

    @abstractmethod
    def predict(self, request: InferenceRequest) -> InferenceResult:
        """Run one inference request and return a typed result."""

    @abstractmethod
    def visualize(
        self,
        result: InferenceResult,
        context: VisualizationContext | None = None,
    ) -> Sequence[VisualizationArtifact]:
        """Describe visualizations without mutating GUI state."""

    def close(self) -> None:
        """Release runtime resources.  Stateless plugins need not override it."""

        return None


class ManifestBackedModelPlugin(ModelPlugin):
    """Convenience base implementing manifest-backed description and validation."""

    def __init__(self, manifest: ModelManifest):
        if not isinstance(manifest, ModelManifest):
            raise TypeError("manifest must be a validated ModelManifest")
        self._manifest = manifest

    def describe(self) -> ModelManifest:
        return self._manifest

    def capabilities(self) -> CapabilitySpec:
        return self._manifest.capabilities

    def validate(self, context: PluginValidationContext | None = None) -> PluginValidationReport:
        if context is None:
            return PluginValidationReport()
        try:
            validate_manifest_files(
                self._manifest,
                context.plugin_root,
                verify_hashes=context.verify_weight_hashes,
            )
        except ManifestValidationError as exc:
            return PluginValidationReport(
                items=tuple(
                    PluginValidationItem(
                        severity=ValidationSeverity.ERROR,
                        code="manifest.file",
                        message=issue.message,
                        path=issue.path,
                    )
                    for issue in exc.issues
                )
            )
        return PluginValidationReport()

    def authorize_load(self, context: PluginLoadContext) -> None:
        """Call at the start of a subclass ``load`` implementation."""

        require_python_adapter_consent(
            self._manifest,
            user_consented=context.user_consented_python_code,
            subprocess_isolated=context.subprocess_isolated,
        )
        # This also proves callers did not construct a relaxed policy.
        if context.security_policy != DEFAULT_SECURITY_POLICY:
            raise PluginContractError("load context violates the platform security policy")

    def validate_result(self, result: InferenceResult) -> None:
        """Call before returning a result from ``predict``."""

        validate_prediction_result(self._manifest, result)


def validate_prediction_result(manifest: ModelManifest, result: InferenceResult) -> None:
    """Validate task and provenance invariants at the plugin boundary."""

    if not isinstance(result, InferenceResult):
        raise PluginContractError("predict() must return an InferenceResult")
    if result.task not in manifest.tasks:
        declared = ", ".join(task.value for task in manifest.tasks)
        raise PluginContractError(
            f"result task {result.task.value!r} is not declared by manifest ({declared})"
        )
    if result.provenance.model_name != manifest.name:
        raise PluginContractError("result provenance model_name does not match manifest")
    if result.provenance.model_version != manifest.version:
        raise PluginContractError("result provenance model_version does not match manifest")
    if result.provenance.runtime is not manifest.runtime.kind:
        raise PluginContractError("result provenance runtime does not match manifest")


def assert_model_plugin(value: Any) -> ModelPlugin:
    """Return a plugin after a cheap contract check, otherwise raise clearly."""

    if not isinstance(value, ModelPlugin):
        raise PluginContractError("adapter object must implement ModelPlugin")
    manifest = value.describe()
    if not isinstance(manifest, ModelManifest):
        raise PluginContractError("describe() must return ModelManifest")
    capabilities = value.capabilities()
    if not isinstance(capabilities, CapabilitySpec):
        raise PluginContractError("capabilities() must return CapabilitySpec")
    return value
