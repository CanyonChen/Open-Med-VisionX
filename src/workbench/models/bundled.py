"""Integrity-gated access to the three reviewed, offline model bundles.

The runtime never calls :func:`torch.load`. The MONAI payload is an official
TorchScript archive; the DIVal and DeepInverse payloads are numeric NPZ tensor
sets loaded with ``allow_pickle=False`` into trusted, versioned source graphs.
"""

from __future__ import annotations

import hashlib
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal
from urllib.parse import urlsplit

import numpy as np

from ._safe_npz import ArraySpec, load_safe_npz

BUNDLED_MODEL_IDS = (
    "deepinv-mri-modl",
    "dival-lodopab-fbpunet",
    "monai-brats-segmentation",
)
MODEL_BUDGET_BYTES = 25 * 1024 * 1024
_RECORD_NAME = "bundle.yaml"
_MAX_RECORD_BYTES = 128 * 1024
_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_ALLOWED_TOP_LEVEL_KEYS = {
    "schema_version",
    "bundle_id",
    "bundle_version",
    "display_name",
    "display_name_zh",
    "description",
    "source",
    "converted_artifact",
    "golden_reference",
    "licenses",
    "task_contract",
    "resources",
    "intended_use",
    "training_domain_and_limitations",
    "golden_tests",
}
_REQUIRED_TOP_LEVEL_KEYS = _ALLOWED_TOP_LEVEL_KEYS
_EXPECTED_FORMATS = {
    "deepinv-mri-modl": "numpy-npz-state-dict",
    "dival-lodopab-fbpunet": "numpy-npz-state-dict",
    "monai-brats-segmentation": "torchscript",
}
_SOURCE_KEYS = {
    "official_repository",
    "immutable_revision_or_release",
    "publication",
    "original_artifact_url",
    "original_size_bytes",
    "original_sha256",
}
_ARTIFACT_KEYS = {
    "repository_path",
    "format",
    "size_bytes",
    "sha256",
    "converter_script",
    "environment_lock",
}
_GOLDEN_KEYS = {
    "repository_path",
    "format",
    "size_bytes",
    "sha256",
    "input_keys",
    "output_keys",
    "rtol",
    "atol",
}
_LICENSE_KEYS = {
    "code",
    "weights",
    "training_data_terms",
    "redistribution_reviewed",
    "reviewed_by",
    "reviewed_on",
    "evidence_and_notice_paths",
}
_TASK_KEYS = {
    "task",
    "modality_and_channel_order",
    "dimensionality",
    "input_semantics",
    "spacing_orientation_intensity",
    "labels_and_output_semantics",
    "preprocessing_postprocessing",
}
_RESOURCE_KEYS = {
    "cpu_supported",
    "gpu_backend",
    "tested_ram_vram",
    "expected_runtime_cpu_gpu",
}


class BundleError(RuntimeError):
    """Base class for reviewed bundle failures."""


class BundleIntegrityError(BundleError):
    """A bundle record or artifact failed its release integrity gate."""


class BundleDependencyError(BundleError):
    """The optional PyTorch runtime is not available."""


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BundleIntegrityError(f"{path} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise BundleIntegrityError(f"{path} keys must be strings")
    return dict(value)


def _require_exact_keys(value: Any, path: str, expected: set[str]) -> dict[str, Any]:
    mapping = _require_mapping(value, path)
    missing = expected - mapping.keys()
    extra = mapping.keys() - expected
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"unexpected {sorted(extra)}")
        raise BundleIntegrityError(f"invalid keys in {path}: {', '.join(details)}")
    return mapping


def _require_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BundleIntegrityError(f"{path} must be a non-empty string")
    return value.strip()


def _require_positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BundleIntegrityError(f"{path} must be a positive integer")
    return value


def _require_nonnegative_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise BundleIntegrityError(f"{path} must be a non-negative number")
    return float(value)


def _require_string_list(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise BundleIntegrityError(f"{path} must be a non-empty list")
    return tuple(_require_text(item, f"{path}[{index}]") for index, item in enumerate(value))


def _require_sha256(value: Any, path: str) -> str:
    digest = _require_text(value, path).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise BundleIntegrityError(f"{path} must be a 64-character hex digest")
    return digest


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _models_root() -> Path:
    return Path(__file__).resolve().parents[1] / "resources" / "model_bundles"


def _safe_child(root: Path, relative: str, *, label: str) -> Path:
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise BundleIntegrityError(f"{label} must stay inside its bundle directory")
    candidate = (root / candidate_relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise BundleIntegrityError(f"{label} escapes its bundle directory") from exc
    return candidate


@dataclass(frozen=True, slots=True)
class ModelBundleRecord:
    """Machine-validated provenance and task contract for one bundled model."""

    schema_version: int
    bundle_id: str
    bundle_version: int
    display_name: str
    display_name_zh: str
    description: Any
    source: Any
    converted_artifact: Any
    golden_reference: Any
    licenses: Any
    task_contract: Any
    resources: Any
    intended_use: str
    training_domain_and_limitations: str
    golden_tests: tuple[str, ...]
    record_path: Path = field(repr=False, compare=False)
    verified: bool = False

    @property
    def id(self) -> str:
        return self.bundle_id

    @property
    def description_en(self) -> str:
        return _require_text(self.description.get("en"), "description.en")

    @property
    def description_zh(self) -> str:
        return _require_text(self.description.get("zh-CN"), "description.zh-CN")

    @property
    def manifest_path(self) -> Path:
        return self.record_path

    @property
    def artifact_path(self) -> Path:
        relative = _require_text(
            self.converted_artifact.get("repository_path"),
            "converted_artifact.repository_path",
        )
        return _safe_child(self.record_path.parent, relative, label="artifact path")

    @property
    def artifact_sha256(self) -> str:
        return _require_text(
            self.converted_artifact.get("sha256"),
            "converted_artifact.sha256",
        ).lower()

    @property
    def artifact_size_bytes(self) -> int:
        return _require_positive_int(
            self.converted_artifact.get("size_bytes"),
            "converted_artifact.size_bytes",
        )

    @property
    def artifact_format(self) -> str:
        return _require_text(self.converted_artifact.get("format"), "converted_artifact.format")

    @property
    def golden_path(self) -> Path:
        relative = _require_text(
            self.golden_reference.get("repository_path"),
            "golden_reference.repository_path",
        )
        return _safe_child(self.record_path.parent, relative, label="golden reference path")

    @property
    def golden_sha256(self) -> str:
        return _require_sha256(self.golden_reference.get("sha256"), "golden_reference.sha256")

    @property
    def golden_size_bytes(self) -> int:
        return _require_positive_int(
            self.golden_reference.get("size_bytes"),
            "golden_reference.size_bytes",
        )

    @property
    def weights_paths(self) -> tuple[Path, ...]:
        return (self.artifact_path,)

    @property
    def task(self) -> str:
        return _require_text(self.task_contract.get("task"), "task_contract.task")

    @property
    def modalities(self) -> tuple[str, ...]:
        return tuple(self.task_contract.get("modality_and_channel_order", ()))

    @property
    def dimensionality(self) -> str:
        return _require_text(
            self.task_contract.get("dimensionality"),
            "task_contract.dimensionality",
        )

    @property
    def runtime(self) -> str:
        return _require_text(self.resources.get("gpu_backend"), "resources.gpu_backend")

    @property
    def license(self) -> str:
        return _require_text(self.licenses.get("weights"), "licenses.weights")

    @property
    def source_url(self) -> str:
        return _require_text(self.source.get("official_repository"), "source.official_repository")

    @property
    def limitations(self) -> str:
        return self.training_domain_and_limitations


def _parse_record(data: Any, path: Path, expected_id: str) -> ModelBundleRecord:
    mapping = _require_mapping(data, str(path))
    missing = _REQUIRED_TOP_LEVEL_KEYS - mapping.keys()
    extra = mapping.keys() - _ALLOWED_TOP_LEVEL_KEYS
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"unexpected {sorted(extra)}")
        raise BundleIntegrityError(f"invalid keys in {path}: {', '.join(details)}")

    schema_version = _require_positive_int(mapping["schema_version"], "schema_version")
    if schema_version != 2:
        raise BundleIntegrityError(f"unsupported bundle schema version: {schema_version}")
    bundle_id = _require_text(mapping["bundle_id"], "bundle_id")
    if bundle_id != expected_id:
        raise BundleIntegrityError(
            f"bundle directory {expected_id!r} declares mismatched id {bundle_id!r}"
        )

    description = _require_exact_keys(mapping["description"], "description", {"en", "zh-CN"})
    _require_text(description["en"], "description.en")
    _require_text(description["zh-CN"], "description.zh-CN")
    source = _require_exact_keys(mapping["source"], "source", _SOURCE_KEYS)
    artifact = _require_exact_keys(
        mapping["converted_artifact"], "converted_artifact", _ARTIFACT_KEYS
    )
    golden_reference = _require_exact_keys(
        mapping["golden_reference"], "golden_reference", _GOLDEN_KEYS
    )
    licenses = _require_exact_keys(mapping["licenses"], "licenses", _LICENSE_KEYS)
    task_contract = _require_exact_keys(mapping["task_contract"], "task_contract", _TASK_KEYS)
    resources = _require_exact_keys(mapping["resources"], "resources", _RESOURCE_KEYS)

    for key in (
        "official_repository",
        "immutable_revision_or_release",
        "publication",
        "original_artifact_url",
    ):
        _require_text(source.get(key), f"source.{key}")
    revision = _require_text(
        source.get("immutable_revision_or_release"),
        "source.immutable_revision_or_release",
    )
    if not _GIT_REVISION.fullmatch(revision):
        raise BundleIntegrityError("source revision must be a full lowercase 40-character SHA")
    for key in ("official_repository", "publication", "original_artifact_url"):
        parsed = urlsplit(_require_text(source.get(key), f"source.{key}"))
        if parsed.scheme != "https" or not parsed.hostname:
            raise BundleIntegrityError(f"source.{key} must be an absolute HTTPS URL")
    _require_positive_int(source.get("original_size_bytes"), "source.original_size_bytes")
    _require_sha256(source.get("original_sha256"), "source.original_sha256")

    for key in ("repository_path", "format", "sha256", "converter_script", "environment_lock"):
        _require_text(artifact.get(key), f"converted_artifact.{key}")
    expected_format = _EXPECTED_FORMATS[bundle_id]
    if artifact["format"] != expected_format:
        raise BundleIntegrityError(
            f"{bundle_id} format must be {expected_format!r}, got {artifact['format']!r}"
        )
    _require_positive_int(artifact.get("size_bytes"), "converted_artifact.size_bytes")
    _require_sha256(artifact.get("sha256"), "converted_artifact.sha256")

    if _require_text(golden_reference.get("format"), "golden_reference.format") != "numpy-npz":
        raise BundleIntegrityError("golden_reference.format must be 'numpy-npz'")
    _require_text(golden_reference.get("repository_path"), "golden_reference.repository_path")
    _require_positive_int(golden_reference.get("size_bytes"), "golden_reference.size_bytes")
    _require_sha256(golden_reference.get("sha256"), "golden_reference.sha256")
    _require_string_list(golden_reference.get("input_keys"), "golden_reference.input_keys")
    _require_string_list(golden_reference.get("output_keys"), "golden_reference.output_keys")
    _require_nonnegative_float(golden_reference.get("rtol"), "golden_reference.rtol")
    _require_nonnegative_float(golden_reference.get("atol"), "golden_reference.atol")

    for key in ("code", "weights", "training_data_terms", "reviewed_by", "reviewed_on"):
        _require_text(licenses.get(key), f"licenses.{key}")
    try:
        review_date = date.fromisoformat(str(licenses["reviewed_on"]))
    except ValueError as exc:
        raise BundleIntegrityError("licenses.reviewed_on must be an ISO date") from exc
    if review_date > date.today():
        raise BundleIntegrityError("licenses.reviewed_on cannot be in the future")
    if licenses.get("redistribution_reviewed") is not True:
        raise BundleIntegrityError("licenses.redistribution_reviewed must be true")
    evidence = _require_string_list(
        licenses.get("evidence_and_notice_paths"),
        "licenses.evidence_and_notice_paths",
    )
    for index, relative in enumerate(evidence):
        evidence_path = _safe_child(path.parent, relative, label=f"license evidence {index}")
        if not evidence_path.is_file():
            raise BundleIntegrityError(f"missing license evidence: {evidence_path}")

    _require_string_list(
        task_contract.get("modality_and_channel_order"),
        "task_contract.modality_and_channel_order",
    )
    for key in _TASK_KEYS - {"modality_and_channel_order"}:
        _require_text(task_contract.get(key), f"task_contract.{key}")
    if resources.get("cpu_supported") is not True:
        raise BundleIntegrityError("resources.cpu_supported must be true")
    for key in ("gpu_backend", "tested_ram_vram", "expected_runtime_cpu_gpu"):
        _require_text(resources.get(key), f"resources.{key}")

    intended_use = _require_text(mapping["intended_use"], "intended_use")
    if intended_use != "education-and-research-only":
        raise BundleIntegrityError(
            "bundled models must declare intended_use: education-and-research-only"
        )
    record = ModelBundleRecord(
        schema_version=schema_version,
        bundle_id=bundle_id,
        bundle_version=_require_positive_int(mapping["bundle_version"], "bundle_version"),
        display_name=_require_text(mapping["display_name"], "display_name"),
        display_name_zh=_require_text(mapping["display_name_zh"], "display_name_zh"),
        description=_freeze(description),
        source=_freeze(source),
        converted_artifact=_freeze(artifact),
        golden_reference=_freeze(golden_reference),
        licenses=_freeze(licenses),
        task_contract=_freeze(task_contract),
        resources=_freeze(resources),
        intended_use=intended_use,
        training_domain_and_limitations=_require_text(
            mapping["training_domain_and_limitations"],
            "training_domain_and_limitations",
        ),
        golden_tests=_require_string_list(mapping["golden_tests"], "golden_tests"),
        record_path=path,
    )
    _ = record.artifact_path
    _ = record.golden_path
    return record


def load_bundle_record(bundle_id: str) -> ModelBundleRecord:
    """Load one allow-listed record with safe YAML and strict keys."""

    if bundle_id not in BUNDLED_MODEL_IDS:
        raise BundleIntegrityError(f"unknown bundled model id: {bundle_id!r}")
    path = _models_root() / bundle_id / _RECORD_NAME
    if not path.is_file():
        raise BundleIntegrityError(f"missing bundle record: {path}")
    if path.stat().st_size > _MAX_RECORD_BYTES:
        raise BundleIntegrityError(
            f"bundle record exceeds the {_MAX_RECORD_BYTES}-byte safety limit: {path}"
        )
    try:
        import yaml
        from yaml.constructor import ConstructorError
    except ImportError as exc:  # pragma: no cover - required base dependency
        raise BundleDependencyError("PyYAML is required to read model bundle records") from exc

    class UniqueKeySafeLoader(yaml.SafeLoader):
        pass

    def construct_unique_mapping(
        loader: Any,
        node: Any,
        deep: bool = False,
    ) -> dict[Any, Any]:
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
    try:
        loader = UniqueKeySafeLoader(path.read_text(encoding="utf-8"))
        try:
            data = loader.get_single_data()
        finally:
            loader.dispose()
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise BundleIntegrityError(f"cannot read safe YAML bundle record {path}: {exc}") from exc
    return _parse_record(data, path, bundle_id)


def _verify_payload(path: Path, expected_size: int, expected_hash: str, label: str) -> None:
    if not path.is_file():
        raise BundleIntegrityError(f"missing {label}: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise BundleIntegrityError(
            f"{label} size mismatch: expected {expected_size}, got {actual_size}"
        )
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise BundleIntegrityError(
            f"{label} SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )


def verify_bundle(bundle_id: str) -> ModelBundleRecord:
    """Verify the record, model payload, and deterministic golden payload."""

    record = load_bundle_record(bundle_id)
    _verify_payload(
        record.artifact_path,
        record.artifact_size_bytes,
        record.artifact_sha256,
        f"{bundle_id} model artifact",
    )
    _verify_payload(
        record.golden_path,
        record.golden_size_bytes,
        record.golden_sha256,
        f"{bundle_id} golden reference",
    )
    return replace(record, verified=True)


def verify_all_bundles() -> tuple[ModelBundleRecord, ...]:
    """Apply the release allow-list, integrity checks, and 25 MiB budget."""

    root = _models_root()
    actual_directories = {path.name for path in root.iterdir() if path.is_dir()}
    expected_directories = set(BUNDLED_MODEL_IDS)
    if actual_directories != expected_directories:
        raise BundleIntegrityError(
            "bundle directory allow-list mismatch: "
            f"expected {sorted(expected_directories)}, got {sorted(actual_directories)}"
        )
    records = tuple(verify_bundle(bundle_id) for bundle_id in BUNDLED_MODEL_IDS)
    registered_paths = {
        path for record in records for path in (record.artifact_path, record.golden_path)
    }
    unregistered = {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".ts", ".pt", ".pth", ".onnx", ".npz", ".npy"}
    } - registered_paths
    if unregistered:
        paths = ", ".join(sorted(str(path) for path in unregistered))
        raise BundleIntegrityError(f"unregistered model artifact(s): {paths}")
    total_size = sum(record.artifact_size_bytes + record.golden_size_bytes for record in records)
    if total_size > MODEL_BUDGET_BYTES:
        raise BundleIntegrityError(
            f"bundled models use {total_size} bytes, over the {MODEL_BUDGET_BYTES}-byte budget"
        )
    return records


def list_bundled_models(*, verify: bool = True) -> tuple[ModelBundleRecord, ...]:
    """Return the fixed offline catalog, with integrity status by default."""

    if verify:
        return verify_all_bundles()
    return tuple(load_bundle_record(bundle_id) for bundle_id in BUNDLED_MODEL_IDS)


def _import_torch() -> Any:
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise BundleDependencyError(
            "PyTorch is required for bundled models. Install the documented GPU or CPU runtime."
        ) from exc
    return torch


def _load_module(record: ModelBundleRecord, device: str) -> Any:
    torch = _import_torch()
    try:
        if record.artifact_format == "torchscript":
            return torch.jit.load(str(record.artifact_path), map_location=device).eval()
        if record.artifact_format == "numpy-npz-state-dict":
            from ._adapters import load_npz_model

            return load_npz_model(record.bundle_id, record.artifact_path, device)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise BundleIntegrityError(f"cannot load reviewed model {record.bundle_id}: {exc}") from exc
    raise BundleIntegrityError(
        f"unsupported reviewed format for {record.bundle_id}: {record.artifact_format}"
    )


def _inference_context(torch: Any, device: str) -> Any:
    if not device.startswith("cuda"):
        return nullcontext()
    return torch.backends.cudnn.flags(
        enabled=True,
        benchmark=False,
        deterministic=True,
        allow_tf32=False,
    )


@dataclass(frozen=True, slots=True)
class GoldenValidationResult:
    """Result of executing a reviewed payload against its packaged reference."""

    bundle_id: str
    device: str
    elapsed_seconds: float
    maximum_absolute_error: float
    maximum_relative_error: float
    fallback_reason: str | None


@dataclass(slots=True)
class LoadedBundle:
    """A verified model module with visible device/fallback provenance."""

    record: ModelBundleRecord
    module: Any = field(repr=False)
    requested_device: str
    device: str
    fallback_reason: str | None = None

    @staticmethod
    def _model_tensor(torch: Any, value: Any, device: str) -> Any:
        """Own NumPy input storage before exposing it to PyTorch.

        Teaching-case arrays are intentionally read-only.  ``torch.as_tensor``
        warns for that storage because writes through the tensor would be
        undefined, even though inference itself is non-mutating.  A private,
        writable C-contiguous copy keeps that safety boundary explicit and
        leaves the caller's array unchanged.
        """

        if isinstance(value, np.ndarray):
            owned = np.array(value, dtype=np.float32, order="C", copy=True)
            return torch.from_numpy(owned).to(device=device, dtype=torch.float32)
        return torch.as_tensor(value).to(device=device, dtype=torch.float32)

    def run(self, *inputs: Any) -> tuple[tuple[Any, ...], float]:
        """Run tensors on the selected device and return CPU tensors + elapsed seconds.

        With ``requested_device='auto'``, a CUDA runtime failure retries the exact
        same reviewed model on CPU and records the visible fallback reason.
        """

        torch = _import_torch()
        tensors = tuple(self._model_tensor(torch, value, self.device) for value in inputs)
        started = time.perf_counter()
        try:
            with _inference_context(torch, self.device), torch.inference_mode():
                raw = self.module(*tensors)
        except RuntimeError as exc:
            if self.requested_device != "auto" or not self.device.startswith("cuda"):
                raise
            self.fallback_reason = f"CUDA execution failed; retried on CPU: {exc}"
            self.device = "cpu"
            self.module = _load_module(self.record, "cpu")
            cpu_inputs = tuple(tensor.detach().cpu() for tensor in tensors)
            with torch.inference_mode():
                raw = self.module(*cpu_inputs)
        elapsed = time.perf_counter() - started
        outputs = raw if isinstance(raw, (tuple, list)) else (raw,)
        return tuple(output.detach().cpu() for output in outputs), elapsed


def load_bundled_model(
    bundle_id: str,
    device: Literal["auto", "cpu", "cuda"] = "auto",
) -> LoadedBundle:
    """Verify and load one fixed model, preferring CUDA when requested as auto."""

    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    record = verify_bundle(bundle_id)
    torch = _import_torch()
    fallback_reason: str | None = None
    if device == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
        if resolved == "cpu":
            fallback_reason = "CUDA is unavailable; using the reviewed CPU path."
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise BundleDependencyError("CUDA was requested, but PyTorch reports it unavailable")
        resolved = "cuda"
    else:
        resolved = "cpu"
    try:
        module = _load_module(record, resolved)
    except BundleIntegrityError as exc:
        if device != "auto" or resolved != "cuda":
            raise
        resolved = "cpu"
        fallback_reason = f"CUDA load failed; using the reviewed CPU path: {exc}"
        module = _load_module(record, "cpu")
    return LoadedBundle(
        record=record,
        module=module,
        requested_device=device,
        device=resolved,
        fallback_reason=fallback_reason,
    )


def _golden_contract(
    bundle_id: str,
) -> tuple[dict[str, ArraySpec], tuple[str, ...], tuple[str, ...]]:
    if bundle_id == "deepinv-mri-modl":
        shape = (1, 2, 32, 32)
        specs = {
            "input.kspace": ArraySpec(shape),
            "input.mask": ArraySpec(shape),
            "output.reconstruction": ArraySpec(shape),
        }
        return specs, ("input.kspace", "input.mask"), ("output.reconstruction",)
    if bundle_id == "dival-lodopab-fbpunet":
        shape = (1, 1, 362, 362)
        specs = {
            "input.fbp": ArraySpec(shape),
            "output.reconstruction": ArraySpec(shape),
        }
        return specs, ("input.fbp",), ("output.reconstruction",)
    if bundle_id == "monai-brats-segmentation":
        specs = {
            "input.modalities": ArraySpec((1, 4, 32, 32, 32)),
            "output.logits": ArraySpec((1, 3, 32, 32, 32)),
        }
        return specs, ("input.modalities",), ("output.logits",)
    raise BundleIntegrityError(f"no golden contract for {bundle_id!r}")


def validate_bundle_golden(
    bundle_id: str,
    device: Literal["auto", "cpu", "cuda"] = "auto",
) -> GoldenValidationResult:
    """Execute one bundled model and compare it with the packaged golden output."""

    loaded = load_bundled_model(bundle_id, device=device)
    specs, input_keys, output_keys = _golden_contract(bundle_id)
    declared_inputs = tuple(loaded.record.golden_reference["input_keys"])
    declared_outputs = tuple(loaded.record.golden_reference["output_keys"])
    if declared_inputs != input_keys or declared_outputs != output_keys:
        raise BundleIntegrityError(f"{bundle_id} golden key order does not match its adapter")
    arrays = load_safe_npz(loaded.record.golden_path, specs)
    outputs, elapsed = loaded.run(*(arrays[name] for name in input_keys))
    if len(outputs) != len(output_keys):
        raise BundleIntegrityError(
            f"{bundle_id} produced {len(outputs)} outputs, expected {len(output_keys)}"
        )

    rtol = float(loaded.record.golden_reference["rtol"])
    atol = float(loaded.record.golden_reference["atol"])
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    for output, key in zip(outputs, output_keys, strict=True):
        observed = output.detach().cpu().numpy()
        expected = arrays[key]
        if observed.shape != expected.shape or not np.isfinite(observed).all():
            raise BundleIntegrityError(f"{bundle_id} produced an invalid golden output for {key}")
        difference = np.abs(observed.astype(np.float64) - expected.astype(np.float64))
        maximum_absolute_error = max(maximum_absolute_error, float(difference.max()))
        denominator = np.maximum(
            np.abs(expected.astype(np.float64)),
            max(atol, float(np.finfo(np.float32).eps)),
        )
        maximum_relative_error = max(
            maximum_relative_error,
            float((difference / denominator).max()),
        )
        if not np.allclose(observed, expected, rtol=rtol, atol=atol, equal_nan=False):
            raise BundleIntegrityError(
                f"{bundle_id} golden mismatch for {key}: max abs error "
                f"{maximum_absolute_error:.6g}, tolerance rtol={rtol}, atol={atol}"
            )
    return GoldenValidationResult(
        bundle_id=bundle_id,
        device=loaded.device,
        elapsed_seconds=elapsed,
        maximum_absolute_error=maximum_absolute_error,
        maximum_relative_error=maximum_relative_error,
        fallback_reason=loaded.fallback_reason,
    )
