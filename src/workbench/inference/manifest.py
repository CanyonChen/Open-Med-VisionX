"""Strict, dependency-light ``manifest.yaml`` model for external model plugins.

Only YAML parsing requires PyYAML, and that dependency is imported lazily by
``load_manifest``.  Importing :mod:`workbench.inference` never imports an AI
runtime or executes adapter code.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._schema import (
    check_keys,
    expect_mapping,
    expect_sequence,
    fail,
    parse_bool,
    parse_enum,
    parse_int,
    parse_json_value,
    parse_number,
    parse_number_tuple,
    parse_optional_string,
    parse_shape,
    parse_string,
    parse_string_tuple,
)
from .enums import (
    ActivationKind,
    CoordinateSystem,
    DeviceKind,
    Dimensionality,
    InputSemantic,
    InterpolationMode,
    Modality,
    OutputSemantic,
    RuntimeKind,
    Task,
    TensorDType,
    UncertaintyKind,
)
from .errors import (
    ManifestDependencyError,
    ManifestFormatError,
    ManifestNotFoundError,
    ManifestValidationError,
)
from .preprocessing import Preprocessing2DSpec

MANIFEST_FILENAME = "manifest.yaml"
SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0"})
DEFAULT_MAX_MANIFEST_BYTES = 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True, slots=True)
class LicenseSpec:
    """Separate licenses for adapter code, model definition, and weights."""

    code: str
    model: str
    weights: str
    notice: str | None = None
    urls: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.model.strip() or not self.weights.strip():
            raise ValueError("code, model, and weights licenses must all be declared")
        object.__setattr__(self, "urls", dict(self.urls))

    @classmethod
    def from_mapping(cls, value: Any, path: str = "license") -> LicenseSpec:
        data = expect_mapping(value, path)
        check_keys(
            data,
            path=path,
            allowed={"code", "model", "weights", "notice", "urls"},
            required={"code", "model", "weights"},
        )
        urls_data = expect_mapping(data.get("urls", {}), f"{path}.urls")
        urls = {
            parse_string(key, f"{path}.urls.key"): parse_string(url, f"{path}.urls.{key}")
            for key, url in urls_data.items()
        }
        return cls(
            code=parse_string(data["code"], f"{path}.code"),
            model=parse_string(data["model"], f"{path}.model"),
            weights=parse_string(data["weights"], f"{path}.weights"),
            notice=parse_optional_string(data.get("notice"), f"{path}.notice"),
            urls=urls,
        )

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "model": self.model,
            "weights": self.weights,
        }
        if self.notice is not None:
            result["notice"] = self.notice
        if self.urls:
            result["urls"] = dict(self.urls)
        return result


@dataclass(frozen=True, slots=True)
class ModelSourceSpec:
    """Provenance that disambiguates similarly named model families."""

    name: str
    organization: str | None = None
    repository: str | None = None
    publication: str | None = None
    model_id: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("source name must not be empty")

    @classmethod
    def from_mapping(cls, value: Any, path: str = "source") -> ModelSourceSpec:
        data = expect_mapping(value, path)
        check_keys(
            data,
            path=path,
            allowed={"name", "organization", "repository", "publication", "model_id"},
            required={"name"},
        )
        return cls(
            name=parse_string(data["name"], f"{path}.name"),
            organization=parse_optional_string(data.get("organization"), f"{path}.organization"),
            repository=parse_optional_string(data.get("repository"), f"{path}.repository"),
            publication=parse_optional_string(data.get("publication"), f"{path}.publication"),
            model_id=parse_optional_string(data.get("model_id"), f"{path}.model_id"),
        )

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {"name": self.name}
        for key in ("organization", "repository", "publication", "model_id"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result


@dataclass(frozen=True, slots=True)
class PythonEnvironmentSpec:
    """User-selected environment for an isolated Python adapter process."""

    conda_environment: str | None = None
    python_executable: str | None = None
    requirements_file: str | None = None
    subprocess: bool = True

    def __post_init__(self) -> None:
        if not self.conda_environment and not self.python_executable:
            raise ValueError("a Python adapter must declare conda_environment or python_executable")
        if not self.subprocess:
            raise ValueError("Python adapters must run in a separate subprocess")

    @classmethod
    def from_mapping(cls, value: Any, path: str = "runtime.python") -> PythonEnvironmentSpec:
        data = expect_mapping(value, path)
        check_keys(
            data,
            path=path,
            allowed={
                "conda_environment",
                "python_executable",
                "requirements_file",
                "subprocess",
            },
        )
        try:
            return cls(
                conda_environment=parse_optional_string(
                    data.get("conda_environment"), f"{path}.conda_environment"
                ),
                python_executable=parse_optional_string(
                    data.get("python_executable"), f"{path}.python_executable"
                ),
                requirements_file=parse_optional_string(
                    data.get("requirements_file"), f"{path}.requirements_file"
                ),
                subprocess=parse_bool(data.get("subprocess", True), f"{path}.subprocess"),
            )
        except ManifestValidationError:
            raise
        except ValueError as exc:
            fail(path, str(exc))

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {"subprocess": True}
        if self.conda_environment is not None:
            result["conda_environment"] = self.conda_environment
        if self.python_executable is not None:
            result["python_executable"] = self.python_executable
        if self.requirements_file is not None:
            result["requirements_file"] = self.requirements_file
        return result


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    kind: RuntimeKind
    device: DeviceKind = DeviceKind.AUTO
    provider: str | None = None
    python: PythonEnvironmentSpec | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind is RuntimeKind.PYTHON_ADAPTER and self.python is None:
            raise ValueError("python-adapter runtime requires a python environment")
        if self.kind is not RuntimeKind.PYTHON_ADAPTER and self.python is not None:
            raise ValueError("only python-adapter runtime may declare a python environment")
        object.__setattr__(self, "options", dict(self.options))

    @classmethod
    def from_mapping(cls, value: Any, path: str = "runtime") -> RuntimeSpec:
        data = expect_mapping(value, path)
        check_keys(
            data,
            path=path,
            allowed={"kind", "device", "provider", "python", "options"},
            required={"kind"},
        )
        python_value = data.get("python")
        python = (
            None
            if python_value is None
            else PythonEnvironmentSpec.from_mapping(python_value, f"{path}.python")
        )
        options = parse_json_value(data.get("options", {}), f"{path}.options")
        try:
            return cls(
                kind=parse_enum(RuntimeKind, data["kind"], f"{path}.kind"),
                device=parse_enum(DeviceKind, data.get("device", "auto"), f"{path}.device"),
                provider=parse_optional_string(data.get("provider"), f"{path}.provider"),
                python=python,
                options=options,
            )
        except ManifestValidationError:
            raise
        except ValueError as exc:
            fail(path, str(exc))

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind.value, "device": self.device.value}
        if self.provider is not None:
            result["provider"] = self.provider
        if self.python is not None:
            result["python"] = self.python.to_mapping()
        if self.options:
            result["options"] = dict(self.options)
        return result


@dataclass(frozen=True, slots=True)
class EntrypointSpec:
    """A local model file or a local Python adapter object."""

    path: str
    object_name: str | None = None

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ValueError("entrypoint path must not be empty")

    @classmethod
    def from_mapping(cls, value: Any, path: str = "entrypoint") -> EntrypointSpec:
        data = expect_mapping(value, path)
        check_keys(
            data,
            path=path,
            allowed={"path", "object"},
            required={"path"},
        )
        return cls(
            path=parse_string(data["path"], f"{path}.path"),
            object_name=parse_optional_string(data.get("object"), f"{path}.object"),
        )

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {"path": self.path}
        if self.object_name is not None:
            result["object"] = self.object_name
        return result


@dataclass(frozen=True, slots=True)
class WeightSpec:
    """A reference to an existing local weight file; never a download source."""

    name: str
    path: str
    format: str
    sha256: str | None = None
    size_bytes: int | None = None
    required: bool = True
    license: str | None = None

    def __post_init__(self) -> None:
        if self.sha256 is not None and not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")

    @classmethod
    def from_mapping(cls, value: Any, path: str = "weights[]") -> WeightSpec:
        data = expect_mapping(value, path)
        check_keys(
            data,
            path=path,
            allowed={"name", "path", "format", "sha256", "size_bytes", "required", "license"},
            required={"name", "path", "format"},
        )
        sha256 = parse_optional_string(data.get("sha256"), f"{path}.sha256")
        if sha256 is not None and not _SHA256_RE.fullmatch(sha256):
            fail(f"{path}.sha256", "must contain exactly 64 hexadecimal characters", sha256)
        size_value = data.get("size_bytes")
        return cls(
            name=parse_string(data["name"], f"{path}.name"),
            path=parse_string(data["path"], f"{path}.path"),
            format=parse_string(data["format"], f"{path}.format"),
            sha256=sha256.lower() if sha256 is not None else None,
            size_bytes=(
                None
                if size_value is None
                else parse_int(size_value, f"{path}.size_bytes", minimum=0)
            ),
            required=parse_bool(data.get("required", True), f"{path}.required"),
            license=parse_optional_string(data.get("license"), f"{path}.license"),
        )

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "path": self.path,
            "format": self.format,
            "required": self.required,
        }
        if self.sha256 is not None:
            result["sha256"] = self.sha256
        if self.size_bytes is not None:
            result["size_bytes"] = self.size_bytes
        if self.license is not None:
            result["license"] = self.license
        return result


@dataclass(frozen=True, slots=True)
class SpacingSpec:
    """Physical-spacing requirement for an input, without inventing defaults."""

    required: bool
    values: tuple[float, ...] | None = None
    unit: str = "mm"
    tolerance: float | None = None

    def __post_init__(self) -> None:
        if self.values is not None and any(item <= 0 for item in self.values):
            raise ValueError("spacing values must be greater than zero")
        if self.tolerance is not None and self.tolerance < 0:
            raise ValueError("spacing tolerance must be non-negative")

    @classmethod
    def from_mapping(cls, value: Any, path: str = "spacing") -> SpacingSpec:
        data = expect_mapping(value, path)
        check_keys(
            data,
            path=path,
            allowed={"required", "values", "unit", "tolerance"},
            required={"required"},
        )
        values_raw = data.get("values")
        values = (
            None
            if values_raw is None
            else parse_number_tuple(values_raw, f"{path}.values", minimum_length=2, positive=True)
        )
        tolerance_raw = data.get("tolerance")
        return cls(
            required=parse_bool(data["required"], f"{path}.required"),
            values=values,
            unit=parse_string(data.get("unit", "mm"), f"{path}.unit"),
            tolerance=(
                None
                if tolerance_raw is None
                else parse_number(tolerance_raw, f"{path}.tolerance", minimum=0.0)
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {"required": self.required, "unit": self.unit}
        if self.values is not None:
            result["values"] = list(self.values)
        if self.tolerance is not None:
            result["tolerance"] = self.tolerance
        return result


@dataclass(frozen=True, slots=True)
class InputSpec:
    name: str
    semantic: InputSemantic
    modalities: tuple[Modality, ...]
    dimensionality: Dimensionality
    shape: tuple[int | str | None, ...]
    spacing: SpacingSpec
    preprocessing: Preprocessing2DSpec | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.modalities:
            raise ValueError("input modalities must not be empty")
        requires_2d = self.dimensionality in {
            Dimensionality.TWO_D,
            Dimensionality.TWO_POINT_FIVE_D,
        } and self.semantic in {InputSemantic.IMAGE, InputSemantic.IMAGE_SEQUENCE}
        if requires_2d and self.preprocessing is None:
            raise ValueError("2D and 2.5D image inputs require preprocessing")

    @classmethod
    def from_mapping(cls, value: Any, path: str = "inputs[]") -> InputSpec:
        data = expect_mapping(value, path)
        check_keys(
            data,
            path=path,
            allowed={
                "name",
                "semantic",
                "modalities",
                "dimensionality",
                "shape",
                "spacing",
                "preprocessing",
                "description",
            },
            required={"name", "semantic", "modalities", "dimensionality", "shape", "spacing"},
        )
        modality_items = expect_sequence(data["modalities"], f"{path}.modalities")
        modalities = tuple(
            parse_enum(Modality, item, f"{path}.modalities[{index}]")
            for index, item in enumerate(modality_items)
        )
        if not modalities:
            fail(f"{path}.modalities", "must contain at least one modality")
        preprocessing_value = data.get("preprocessing")
        preprocessing = (
            None
            if preprocessing_value is None
            else Preprocessing2DSpec.from_mapping(preprocessing_value, f"{path}.preprocessing")
        )
        try:
            return cls(
                name=parse_string(data["name"], f"{path}.name"),
                semantic=parse_enum(InputSemantic, data["semantic"], f"{path}.semantic"),
                modalities=modalities,
                dimensionality=parse_enum(
                    Dimensionality, data["dimensionality"], f"{path}.dimensionality"
                ),
                shape=parse_shape(data["shape"], f"{path}.shape"),
                spacing=SpacingSpec.from_mapping(data["spacing"], f"{path}.spacing"),
                preprocessing=preprocessing,
                description=parse_optional_string(data.get("description"), f"{path}.description"),
            )
        except ManifestValidationError:
            raise
        except ValueError as exc:
            fail(path, str(exc))

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "semantic": self.semantic.value,
            "modalities": [item.value for item in self.modalities],
            "dimensionality": self.dimensionality.value,
            "shape": list(self.shape),
            "spacing": self.spacing.to_mapping(),
        }
        if self.preprocessing is not None:
            result["preprocessing"] = self.preprocessing.to_mapping()
        if self.description is not None:
            result["description"] = self.description
        return result


@dataclass(frozen=True, slots=True)
class PostprocessingSpec:
    activation: ActivationKind
    threshold: float | None
    nms_iou_threshold: float | None
    interpolation: InterpolationMode | None
    discrete_labels: bool
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("threshold", "nms_iou_threshold"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        object.__setattr__(self, "parameters", dict(self.parameters))

    @classmethod
    def from_mapping(cls, value: Any, path: str = "postprocessing") -> PostprocessingSpec:
        data = expect_mapping(value, path)
        required = {
            "activation",
            "threshold",
            "nms_iou_threshold",
            "interpolation",
            "discrete_labels",
        }
        check_keys(
            data,
            path=path,
            allowed=required | {"parameters"},
            required=required,
        )
        threshold_raw = data["threshold"]
        nms_raw = data["nms_iou_threshold"]
        interpolation_raw = data["interpolation"]
        return cls(
            activation=parse_enum(ActivationKind, data["activation"], f"{path}.activation"),
            threshold=(
                None
                if threshold_raw is None
                else parse_number(threshold_raw, f"{path}.threshold", minimum=0.0, maximum=1.0)
            ),
            nms_iou_threshold=(
                None
                if nms_raw is None
                else parse_number(nms_raw, f"{path}.nms_iou_threshold", minimum=0.0, maximum=1.0)
            ),
            interpolation=(
                None
                if interpolation_raw is None
                else parse_enum(InterpolationMode, interpolation_raw, f"{path}.interpolation")
            ),
            discrete_labels=parse_bool(data["discrete_labels"], f"{path}.discrete_labels"),
            parameters=parse_json_value(data.get("parameters", {}), f"{path}.parameters"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "activation": self.activation.value,
            "threshold": self.threshold,
            "nms_iou_threshold": self.nms_iou_threshold,
            "interpolation": None if self.interpolation is None else self.interpolation.value,
            "discrete_labels": self.discrete_labels,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class UncertaintySpec:
    kind: UncertaintyKind
    description: str
    calibrated: bool
    units: str | None = None

    @classmethod
    def from_mapping(cls, value: Any, path: str = "uncertainty") -> UncertaintySpec:
        data = expect_mapping(value, path)
        check_keys(
            data,
            path=path,
            allowed={"kind", "description", "calibrated", "units"},
            required={"kind", "description", "calibrated"},
        )
        return cls(
            kind=parse_enum(UncertaintyKind, data["kind"], f"{path}.kind"),
            description=parse_string(data["description"], f"{path}.description", allow_empty=True),
            calibrated=parse_bool(data["calibrated"], f"{path}.calibrated"),
            units=parse_optional_string(data.get("units"), f"{path}.units"),
        )

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind.value,
            "description": self.description,
            "calibrated": self.calibrated,
        }
        if self.units is not None:
            result["units"] = self.units
        return result


@dataclass(frozen=True, slots=True)
class OutputSpec:
    name: str
    semantic: OutputSemantic
    dtype: TensorDType
    shape: tuple[int | str | None, ...]
    coordinate_system: CoordinateSystem
    labels: Mapping[str, str]
    postprocessing: PostprocessingSpec
    uncertainty: UncertaintySpec
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", dict(self.labels))
        spatial = {
            OutputSemantic.MASKS,
            OutputSemantic.BOXES,
            OutputSemantic.KEYPOINTS,
            OutputSemantic.ATTENTION_MAPS,
            OutputSemantic.VECTOR_FIELDS,
            OutputSemantic.AFFINE_TRANSFORMS,
            OutputSemantic.ANOMALY_MAPS,
            OutputSemantic.TRACKS,
        }
        if self.semantic in spatial and self.coordinate_system is CoordinateSystem.NOT_APPLICABLE:
            raise ValueError(f"{self.semantic.value} output requires a coordinate_system")
        if (
            self.semantic is OutputSemantic.MASKS
            and self.postprocessing.discrete_labels
            and self.postprocessing.interpolation is not InterpolationMode.NEAREST
        ):
            raise ValueError(
                "discrete mask outputs must use nearest interpolation for inverse mapping"
            )

    @classmethod
    def from_mapping(cls, value: Any, path: str = "outputs[]") -> OutputSpec:
        data = expect_mapping(value, path)
        required = {
            "name",
            "semantic",
            "dtype",
            "shape",
            "coordinate_system",
            "labels",
            "postprocessing",
            "uncertainty",
        }
        check_keys(
            data,
            path=path,
            allowed=required | {"description"},
            required=required,
        )
        label_data = expect_mapping(data["labels"], f"{path}.labels")
        labels: dict[str, str] = {}
        for key, label in label_data.items():
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                fail(f"{path}.labels", "label keys must be strings or integers", key)
            labels[str(key)] = parse_string(label, f"{path}.labels.{key}")
        try:
            return cls(
                name=parse_string(data["name"], f"{path}.name"),
                semantic=parse_enum(OutputSemantic, data["semantic"], f"{path}.semantic"),
                dtype=parse_enum(TensorDType, data["dtype"], f"{path}.dtype"),
                shape=parse_shape(data["shape"], f"{path}.shape"),
                coordinate_system=parse_enum(
                    CoordinateSystem, data["coordinate_system"], f"{path}.coordinate_system"
                ),
                labels=labels,
                postprocessing=PostprocessingSpec.from_mapping(
                    data["postprocessing"], f"{path}.postprocessing"
                ),
                uncertainty=UncertaintySpec.from_mapping(
                    data["uncertainty"], f"{path}.uncertainty"
                ),
                description=parse_optional_string(data.get("description"), f"{path}.description"),
            )
        except ManifestValidationError:
            raise
        except ValueError as exc:
            fail(path, str(exc))

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "semantic": self.semantic.value,
            "dtype": self.dtype.value,
            "shape": list(self.shape),
            "coordinate_system": self.coordinate_system.value,
            "labels": dict(self.labels),
            "postprocessing": self.postprocessing.to_mapping(),
            "uncertainty": self.uncertainty.to_mapping(),
        }
        if self.description is not None:
            result["description"] = self.description
        return result


@dataclass(frozen=True, slots=True)
class CapabilitySpec:
    prompts: bool
    intermediate_features: bool
    attention: bool
    embeddings: bool
    multiscale_outputs: bool
    vector_fields: bool
    sampling_trajectory: bool
    multimodal_text: bool
    uncertainty: bool

    @classmethod
    def from_mapping(cls, value: Any, path: str = "capabilities") -> CapabilitySpec:
        data = expect_mapping(value, path)
        fields = {
            "prompts",
            "intermediate_features",
            "attention",
            "embeddings",
            "multiscale_outputs",
            "vector_fields",
            "sampling_trajectory",
            "multimodal_text",
            "uncertainty",
        }
        check_keys(data, path=path, allowed=fields, required=fields)
        return cls(**{name: parse_bool(data[name], f"{path}.{name}") for name in fields})

    def to_mapping(self) -> dict[str, bool]:
        return {
            "prompts": self.prompts,
            "intermediate_features": self.intermediate_features,
            "attention": self.attention,
            "embeddings": self.embeddings,
            "multiscale_outputs": self.multiscale_outputs,
            "vector_fields": self.vector_fields,
            "sampling_trajectory": self.sampling_trajectory,
            "multimodal_text": self.multimodal_text,
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Validated stable description returned by ``ModelPlugin.describe``."""

    schema_version: str
    name: str
    version: str
    family: str
    source: ModelSourceSpec
    description: str
    tasks: tuple[Task, ...]
    subtasks: tuple[str, ...]
    license: LicenseSpec
    runtime: RuntimeSpec
    entrypoint: EntrypointSpec
    weights: tuple[WeightSpec, ...]
    inputs: tuple[InputSpec, ...]
    outputs: tuple[OutputSpec, ...]
    capabilities: CapabilitySpec
    authors: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            supported = ", ".join(sorted(SUPPORTED_SCHEMA_VERSIONS))
            raise ValueError(
                f"unsupported schema_version {self.schema_version!r}; supported: {supported}"
            )
        if not self.tasks:
            raise ValueError("at least one task is required")
        if not self.weights:
            raise ValueError("at least one local weight reference is required")
        if not self.inputs:
            raise ValueError("at least one input is required")
        if not self.outputs:
            raise ValueError("at least one output is required")
        self._require_unique("tasks", [task.value for task in self.tasks])
        self._require_unique("weights", [weight.name for weight in self.weights])
        self._require_unique("inputs", [item.name for item in self.inputs])
        self._require_unique("outputs", [item.name for item in self.outputs])
        if self.runtime.kind is RuntimeKind.PYTHON_ADAPTER and not self.entrypoint.object_name:
            raise ValueError("python-adapter entrypoint must declare an object")
        if self.runtime.kind is not RuntimeKind.PYTHON_ADAPTER and self.entrypoint.object_name:
            raise ValueError("ONNX/TorchScript entrypoint must not declare a Python object")
        object.__setattr__(self, "extensions", dict(self.extensions))
        # Validate reference syntax only.  No file is opened, copied, imported, or downloaded.
        from .security import assert_local_reference

        assert_local_reference(self.entrypoint.path, field="entrypoint.path")
        for index, weight in enumerate(self.weights):
            assert_local_reference(weight.path, field=f"weights[{index}].path")

    @staticmethod
    def _require_unique(label: str, values: list[str]) -> None:
        if len(values) != len(set(values)):
            raise ValueError(f"{label} entries must be unique")

    @classmethod
    def from_mapping(cls, value: Any, path: str = "manifest") -> ModelManifest:
        data = expect_mapping(value, path)
        required = {
            "schema_version",
            "name",
            "version",
            "family",
            "source",
            "description",
            "tasks",
            "subtasks",
            "license",
            "runtime",
            "entrypoint",
            "weights",
            "inputs",
            "outputs",
            "capabilities",
        }
        check_keys(
            data,
            path=path,
            allowed=required | {"authors", "references", "extensions"},
            required=required,
        )
        schema_version = parse_string(data["schema_version"], f"{path}.schema_version")
        if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            supported = ", ".join(sorted(SUPPORTED_SCHEMA_VERSIONS))
            fail(
                f"{path}.schema_version",
                f"unsupported schema version; supported: {supported}",
                schema_version,
            )
        task_items = expect_sequence(data["tasks"], f"{path}.tasks")
        tasks = tuple(
            parse_enum(Task, item, f"{path}.tasks[{index}]")
            for index, item in enumerate(task_items)
        )
        weights_items = expect_sequence(data["weights"], f"{path}.weights")
        inputs_items = expect_sequence(data["inputs"], f"{path}.inputs")
        outputs_items = expect_sequence(data["outputs"], f"{path}.outputs")
        try:
            return cls(
                schema_version=schema_version,
                name=parse_string(data["name"], f"{path}.name"),
                version=parse_string(data["version"], f"{path}.version"),
                family=parse_string(data["family"], f"{path}.family"),
                source=ModelSourceSpec.from_mapping(data["source"], f"{path}.source"),
                description=parse_string(data["description"], f"{path}.description"),
                tasks=tasks,
                subtasks=parse_string_tuple(data["subtasks"], f"{path}.subtasks", unique=True),
                license=LicenseSpec.from_mapping(data["license"], f"{path}.license"),
                runtime=RuntimeSpec.from_mapping(data["runtime"], f"{path}.runtime"),
                entrypoint=EntrypointSpec.from_mapping(data["entrypoint"], f"{path}.entrypoint"),
                weights=tuple(
                    WeightSpec.from_mapping(item, f"{path}.weights[{index}]")
                    for index, item in enumerate(weights_items)
                ),
                inputs=tuple(
                    InputSpec.from_mapping(item, f"{path}.inputs[{index}]")
                    for index, item in enumerate(inputs_items)
                ),
                outputs=tuple(
                    OutputSpec.from_mapping(item, f"{path}.outputs[{index}]")
                    for index, item in enumerate(outputs_items)
                ),
                capabilities=CapabilitySpec.from_mapping(
                    data["capabilities"], f"{path}.capabilities"
                ),
                authors=parse_string_tuple(data.get("authors", []), f"{path}.authors", unique=True),
                references=parse_string_tuple(
                    data.get("references", []), f"{path}.references", unique=True
                ),
                extensions=parse_json_value(data.get("extensions", {}), f"{path}.extensions"),
            )
        except ManifestValidationError:
            raise
        except ValueError as exc:
            fail(path, str(exc))

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "family": self.family,
            "source": self.source.to_mapping(),
            "description": self.description,
            "tasks": [task.value for task in self.tasks],
            "subtasks": list(self.subtasks),
            "license": self.license.to_mapping(),
            "runtime": self.runtime.to_mapping(),
            "entrypoint": self.entrypoint.to_mapping(),
            "weights": [weight.to_mapping() for weight in self.weights],
            "inputs": [item.to_mapping() for item in self.inputs],
            "outputs": [item.to_mapping() for item in self.outputs],
            "capabilities": self.capabilities.to_mapping(),
        }
        if self.authors:
            result["authors"] = list(self.authors)
        if self.references:
            result["references"] = list(self.references)
        if self.extensions:
            result["extensions"] = dict(self.extensions)
        return result


def manifest_path(plugin_root: str | Path) -> Path:
    """Return the one supported manifest location for a plugin directory."""

    return Path(plugin_root).expanduser() / MANIFEST_FILENAME


def load_plugin_manifest(
    plugin_root: str | Path,
    *,
    validate_files: bool = False,
    max_bytes: int = DEFAULT_MAX_MANIFEST_BYTES,
) -> ModelManifest:
    """Load ``<plugin_root>/manifest.yaml`` without importing plugin code."""

    root = Path(plugin_root).expanduser()
    return load_manifest(
        root / MANIFEST_FILENAME,
        plugin_root=root,
        validate_files=validate_files,
        max_bytes=max_bytes,
    )


def load_manifest(
    path: str | Path,
    *,
    plugin_root: str | Path | None = None,
    validate_files: bool = False,
    max_bytes: int = DEFAULT_MAX_MANIFEST_BYTES,
) -> ModelManifest:
    """Safely decode and validate one model manifest.

    The loader uses PyYAML's safe loader with duplicate-key rejection and a
    strict size limit.  It does not resolve network resources, import adapters,
    install packages, copy weights, or instantiate a model runtime.
    """

    manifest_file = Path(path).expanduser()
    if manifest_file.is_dir():
        manifest_file = manifest_file / MANIFEST_FILENAME
    if not manifest_file.exists():
        raise ManifestNotFoundError(f"Model manifest not found: {manifest_file}")
    if not manifest_file.is_file():
        raise ManifestFormatError(f"Model manifest is not a regular file: {manifest_file}")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be greater than zero")
    size = manifest_file.stat().st_size
    if size > max_bytes:
        raise ManifestFormatError(
            f"Model manifest exceeds the {max_bytes}-byte safety limit: {size} bytes"
        )
    try:
        raw = manifest_file.read_bytes()
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ManifestFormatError(
            f"Model manifest must be valid UTF-8: {manifest_file}: {exc}"
        ) from exc
    data = _safe_yaml_load(text, source=manifest_file)
    if data is None:
        raise ManifestFormatError(f"Model manifest is empty: {manifest_file}")
    manifest = ModelManifest.from_mapping(data)
    from .security import validate_manifest_references

    root = Path(plugin_root).expanduser() if plugin_root is not None else manifest_file.parent
    validate_manifest_references(manifest, root)
    if validate_files:
        from .security import validate_manifest_files

        validate_manifest_files(manifest, root)
    return manifest


def _safe_yaml_load(text: str, *, source: Path) -> Any:
    try:
        import yaml
        from yaml.constructor import ConstructorError
    except ImportError as exc:
        raise ManifestDependencyError(
            "Loading manifest.yaml requires PyYAML; install the base YAML dependency"
        ) from exc

    class UniqueKeySafeLoader(yaml.SafeLoader):
        pass

    def construct_unique_mapping(loader: Any, node: Any, deep: bool = False) -> dict[Any, Any]:
        loader.flatten_mapping(node)
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            try:
                duplicate = key in result
            except TypeError as exc:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable mapping key",
                    key_node.start_mark,
                ) from exc
            if duplicate:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeySafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_unique_mapping,
    )
    loader = UniqueKeySafeLoader(text)
    try:
        return loader.get_single_data()
    except yaml.YAMLError as exc:
        raise ManifestFormatError(f"Invalid safe YAML in {source}: {exc}") from exc
    finally:
        loader.dispose()
