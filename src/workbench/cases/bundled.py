"""Integrity-gated access to reviewed, non-pickle teaching data assets."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

BUNDLED_TEACHING_CASE_IDS = ("lodopab-ct-test-03456",)
TEACHING_CASE_BUDGET_BYTES = 2 * 1024 * 1024
_RECORD_NAME = "case.yaml"
_CASE_DIRECTORIES = {"lodopab-ct-test-03456": "lodopab-ct"}
_ALLOWED_RECORD_KEYS = {
    "schema_version",
    "case_id",
    "case_version",
    "display_name",
    "dataset",
    "artifact",
    "arrays",
    "source_members",
    "sample",
    "license",
    "privacy",
    "intended_use",
    "limitations",
    "converter_script",
}
_NPZ_ENTRIES = {
    "fbp.npy",
    "ground_truth.npy",
    "metadata_json.npy",
    "observation.npy",
}


class TeachingCaseIntegrityError(RuntimeError):
    """A bundled teaching case failed its immutable release contract."""


def _resources_root() -> Path:
    return Path(__file__).resolve().parents[1] / "resources" / "teaching_cases"


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TeachingCaseIntegrityError(f"{label} must be a string-keyed mapping")
    return dict(value)


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TeachingCaseIntegrityError(f"{label} must be non-empty text")
    return value.strip()


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TeachingCaseIntegrityError(f"{label} must be a positive integer")
    return value


def _require_sha256(value: Any, label: str) -> str:
    digest = _require_text(value, label).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise TeachingCaseIntegrityError(f"{label} must be 64 hexadecimal characters")
    return digest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_child(root: Path, relative: str, *, label: str) -> Path:
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise TeachingCaseIntegrityError(f"{label} must stay inside its case directory")
    candidate = (root / candidate_relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise TeachingCaseIntegrityError(f"{label} escapes its case directory") from exc
    return candidate


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class TeachingCaseRecord:
    schema_version: int
    case_id: str
    case_version: int
    display_name: str
    dataset: Any
    artifact: Any
    arrays: Any
    source_members: Any
    sample: Any
    license: Any
    privacy: Any
    intended_use: str
    limitations: str
    converter_script: str
    record_path: Path

    @property
    def artifact_path(self) -> Path:
        relative = _require_text(self.artifact.get("repository_path"), "artifact path")
        return _safe_child(self.record_path.parent, relative, label="artifact path")

    @property
    def artifact_size_bytes(self) -> int:
        return _require_positive_int(self.artifact.get("size_bytes"), "artifact size")

    @property
    def artifact_sha256(self) -> str:
        return _require_text(self.artifact.get("sha256"), "artifact SHA-256").lower()


@dataclass(frozen=True, slots=True)
class LodopabTeachingCase:
    record: TeachingCaseRecord
    observation: np.ndarray
    fbp: np.ndarray
    ground_truth: np.ndarray
    metadata: Any


def _parse_record(data: Any, path: Path, expected_id: str) -> TeachingCaseRecord:
    mapping = _require_mapping(data, str(path))
    missing = _ALLOWED_RECORD_KEYS - mapping.keys()
    extra = mapping.keys() - _ALLOWED_RECORD_KEYS
    if missing or extra:
        raise TeachingCaseIntegrityError(
            f"invalid case record keys; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if mapping["schema_version"] != 1:
        raise TeachingCaseIntegrityError("unsupported teaching-case schema version")
    case_id = _require_text(mapping["case_id"], "case_id")
    if case_id != expected_id:
        raise TeachingCaseIntegrityError("case directory and record ID do not match")
    artifact = _require_mapping(mapping["artifact"], "artifact")
    if artifact.get("format") != "safe-npz":
        raise TeachingCaseIntegrityError("teaching cases accept only the reviewed safe NPZ format")
    _require_sha256(artifact.get("sha256"), "artifact.sha256")
    allowed_entries = artifact.get("allowed_entries")
    if not isinstance(allowed_entries, list) or set(allowed_entries) != _NPZ_ENTRIES:
        raise TeachingCaseIntegrityError("artifact allowed-entry list does not match the contract")
    license_data = _require_mapping(mapping["license"], "license")
    if license_data.get("identifier") != "CC-BY-4.0":
        raise TeachingCaseIntegrityError("LoDoPaB teaching data must retain CC-BY-4.0")
    if license_data.get("redistribution_reviewed") is not True:
        raise TeachingCaseIntegrityError("teaching-case redistribution review is missing")
    evidence_paths = license_data.get("evidence_paths")
    if not isinstance(evidence_paths, list) or not evidence_paths:
        raise TeachingCaseIntegrityError("license evidence paths are required")
    for index, relative in enumerate(evidence_paths):
        evidence = _safe_child(
            path.parent,
            _require_text(relative, f"license evidence {index}"),
            label=f"license evidence {index}",
        )
        if not evidence.is_file():
            raise TeachingCaseIntegrityError(f"missing license evidence: {evidence}")
    arrays = _require_mapping(mapping["arrays"], "arrays")
    if set(arrays) != {"observation", "fbp", "ground_truth"}:
        raise TeachingCaseIntegrityError("teaching-case array contract changed")
    expected_shapes = {
        "observation": [1000, 513],
        "fbp": [362, 362],
        "ground_truth": [362, 362],
    }
    for name, expected_shape in expected_shapes.items():
        specification = _require_mapping(arrays[name], f"arrays.{name}")
        if specification.get("shape") != expected_shape:
            raise TeachingCaseIntegrityError(f"arrays.{name}.shape changed")
        if specification.get("dtype") != "float32":
            raise TeachingCaseIntegrityError(f"arrays.{name}.dtype must be float32")
        _require_sha256(specification.get("sha256"), f"arrays.{name}.sha256")
        _require_text(specification.get("semantics"), f"arrays.{name}.semantics")
    sample = _require_mapping(mapping["sample"], "sample")
    if (
        sample.get("split") != "test"
        or sample.get("index") != 3456
        or sample.get("source_member_row") != 0
    ):
        raise TeachingCaseIntegrityError("reviewed LoDoPaB sample selection changed")
    privacy = _require_mapping(mapping["privacy"], "privacy")
    if any(
        privacy.get(key) is not False
        for key in (
            "contains_patient_identifiers",
            "contains_dicom_metadata",
            "contains_source_paths",
        )
    ):
        raise TeachingCaseIntegrityError("bundled teaching cases cannot retain private metadata")
    return TeachingCaseRecord(
        schema_version=1,
        case_id=case_id,
        case_version=_require_positive_int(mapping["case_version"], "case_version"),
        display_name=_require_text(mapping["display_name"], "display_name"),
        dataset=_freeze(_require_mapping(mapping["dataset"], "dataset")),
        artifact=_freeze(artifact),
        arrays=_freeze(arrays),
        source_members=_freeze(_require_mapping(mapping["source_members"], "source_members")),
        sample=_freeze(sample),
        license=_freeze(license_data),
        privacy=_freeze(privacy),
        intended_use=_require_text(mapping["intended_use"], "intended_use"),
        limitations=_require_text(mapping["limitations"], "limitations"),
        converter_script=_require_text(mapping["converter_script"], "converter_script"),
        record_path=path,
    )


def _load_record(case_id: str) -> TeachingCaseRecord:
    if case_id not in BUNDLED_TEACHING_CASE_IDS:
        raise TeachingCaseIntegrityError(f"unknown bundled teaching case: {case_id!r}")
    path = _resources_root() / _CASE_DIRECTORIES[case_id] / _RECORD_NAME
    if not path.is_file():
        raise TeachingCaseIntegrityError(f"missing teaching-case record: {path}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - base dependency
        raise TeachingCaseIntegrityError("PyYAML is required for teaching-case records") from exc
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise TeachingCaseIntegrityError(f"cannot read teaching-case record: {exc}") from exc
    return _parse_record(data, path, case_id)


def verify_teaching_case(case_id: str) -> TeachingCaseRecord:
    record = _load_record(case_id)
    artifact = record.artifact_path
    if not artifact.is_file():
        raise TeachingCaseIntegrityError(f"missing teaching-case artifact: {artifact}")
    if artifact.stat().st_size != record.artifact_size_bytes:
        raise TeachingCaseIntegrityError("teaching-case artifact size does not match its record")
    if _sha256(artifact) != record.artifact_sha256:
        raise TeachingCaseIntegrityError("teaching-case artifact SHA-256 mismatch")
    try:
        with zipfile.ZipFile(artifact) as archive:
            names = {item.filename for item in archive.infolist()}
            if names != _NPZ_ENTRIES:
                raise TeachingCaseIntegrityError("teaching-case NPZ has unexpected entries")
            for item in archive.infolist():
                if item.is_dir() or item.file_size > 8 * 1024 * 1024:
                    raise TeachingCaseIntegrityError("teaching-case NPZ entry exceeds its limit")
                if item.file_size and item.file_size / max(item.compress_size, 1) > 100:
                    raise TeachingCaseIntegrityError(
                        "teaching-case NPZ compression ratio is unsafe"
                    )
    except zipfile.BadZipFile as exc:
        raise TeachingCaseIntegrityError("teaching-case artifact is not a valid NPZ") from exc
    return record


def verify_bundled_teaching_cases() -> tuple[TeachingCaseRecord, ...]:
    records = tuple(verify_teaching_case(case_id) for case_id in BUNDLED_TEACHING_CASE_IDS)
    total_bytes = sum(record.artifact_size_bytes for record in records)
    if total_bytes > TEACHING_CASE_BUDGET_BYTES:
        raise TeachingCaseIntegrityError("bundled teaching cases exceed the 2 MiB budget")
    return records


def _readonly_float32(value: np.ndarray, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.dtype("float32"):
        raise TeachingCaseIntegrityError(f"{label} shape or dtype does not match its record")
    if not np.all(np.isfinite(array)):
        raise TeachingCaseIntegrityError(f"{label} contains NaN or infinity")
    result = np.array(array, dtype=np.float32, copy=True)
    result.setflags(write=False)
    return result


def _array_sha256(value: np.ndarray) -> str:
    little_endian = np.asarray(value, dtype="<f4")
    return hashlib.sha256(little_endian.tobytes(order="C")).hexdigest()


def load_lodopab_case() -> LodopabTeachingCase:
    """Load the reviewed LoDoPaB teaching arrays without enabling pickle."""

    record = verify_teaching_case("lodopab-ct-test-03456")
    try:
        with np.load(record.artifact_path, allow_pickle=False) as archive:
            observation = _readonly_float32(
                archive["observation"],
                (1000, 513),
                "LoDoPaB observation",
            )
            fbp = _readonly_float32(
                archive["fbp"],
                (362, 362),
                "LoDoPaB DIVal FBP input",
            )
            ground_truth = _readonly_float32(
                archive["ground_truth"],
                (362, 362),
                "LoDoPaB ground truth",
            )
            metadata_array = np.asarray(archive["metadata_json"])
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise TeachingCaseIntegrityError(f"cannot decode teaching-case NPZ: {exc}") from exc
    if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1:
        raise TeachingCaseIntegrityError(
            "teaching-case metadata must be a one-dimensional uint8 array"
        )
    try:
        metadata = json.loads(metadata_array.tobytes().decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TeachingCaseIntegrityError("teaching-case metadata is not canonical JSON") from exc
    if not isinstance(metadata, dict) or metadata.get("case_id") != record.case_id:
        raise TeachingCaseIntegrityError("embedded teaching-case metadata ID does not match")
    if metadata.get("sample_index") != 3456 or metadata.get("source_member_row") != 0:
        raise TeachingCaseIntegrityError("embedded teaching-case sample selection changed")
    for name, array in (
        ("observation", observation),
        ("fbp", fbp),
        ("ground_truth", ground_truth),
    ):
        expected_hash = _require_sha256(record.arrays[name].get("sha256"), f"arrays.{name}.sha256")
        if _array_sha256(array) != expected_hash:
            raise TeachingCaseIntegrityError(f"{name} derived-array SHA-256 mismatch")
    return LodopabTeachingCase(
        record=record,
        observation=observation,
        fbp=fbp,
        ground_truth=ground_truth,
        metadata=_freeze(metadata),
    )
