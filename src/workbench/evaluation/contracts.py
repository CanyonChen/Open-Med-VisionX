"""Anonymous dataset manifests and leakage-safe split contracts.

The manifest intentionally stores content hashes instead of filesystem paths.
``group_id`` is the unit used for splitting (normally one person or study
group), so every sample from that group must stay in exactly one split.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np
import yaml

from ..errors import ValidationError

JsonScalar: TypeAlias = str | int | float | bool | None
FrozenJson: TypeAlias = JsonScalar | tuple["FrozenJson", ...] | Mapping[str, "FrozenJson"]

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_SPLITS = frozenset({"train", "validation", "test"})
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_SENSITIVE_KEY_FRAGMENTS = (
    "accession",
    "address",
    "birth",
    "dicomtag",
    "email",
    "filename",
    "filepath",
    "medicalrecord",
    "metadata",
    "mrn",
    "patient",
    "personname",
    "phone",
    "sourcepath",
)
_SENSITIVE_KEYS = frozenset(
    {
        "dateofbirth",
        "dob",
        "fullname",
        "givenname",
        "name",
        "path",
        "personid",
        "surname",
    }
)


def _normalized_key(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _is_sensitive_key(value: object) -> bool:
    compact = _normalized_key(value)
    return (
        compact in _SENSITIVE_KEYS
        or compact.endswith(("localpath", "sourcepath"))
        or any(fragment in compact for fragment in _SENSITIVE_KEY_FRAGMENTS)
    )


def _looks_like_local_path(value: str) -> bool:
    candidate = value.strip()
    lowered = candidate.lower()
    if lowered.startswith(("file:", "./", "../", ".\\", "..\\", "~/", "~\\")):
        return True
    if candidate.startswith(("/", "\\\\")):
        return True
    if re.match(r"^[A-Za-z]:[\\/]", candidate):
        return True
    # Relative paths carrying a common data or image suffix are also excluded.
    return bool(
        ("/" in candidate or "\\" in candidate)
        and re.search(
            r"\.(?:dcm|nii(?:\.gz)?|npy|npz|png|jpe?g|tiff?|json|ya?ml|csv)$",
            lowered,
        )
    )


def _safe_identifier(value: object, *, field_name: str) -> str:
    text = str(value).strip()
    if not _IDENTIFIER.fullmatch(text) or _looks_like_local_path(text):
        raise ValidationError(
            f"{field_name} must be an opaque identifier without names or filesystem paths."
        )
    return text


def _safe_text(value: object, *, field_name: str, maximum: int = 256) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be text.")
    text = value.strip()
    if not text or len(text) > maximum or _looks_like_local_path(text):
        raise ValidationError(
            f"{field_name} must be non-empty safe text without a filesystem path."
        )
    return text


def _freeze_safe_json(value: Any, *, field_name: str, depth: int = 0) -> FrozenJson:
    if depth > 8:
        raise ValidationError(f"{field_name} is nested too deeply.")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValidationError(f"{field_name} must contain only finite numbers.")
        return value
    if isinstance(value, str):
        if len(value) > 4_096 or _looks_like_local_path(value):
            raise ValidationError(f"{field_name} contains a path or oversized text value.")
        return value
    if isinstance(value, Mapping):
        if len(value) > 256:
            raise ValidationError(f"{field_name} contains too many fields.")
        frozen: dict[str, FrozenJson] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str) or not raw_key.strip() or len(raw_key) > 128:
                raise ValidationError(f"{field_name} contains an invalid field name.")
            key = raw_key.strip()
            if _is_sensitive_key(key):
                raise ValidationError(f"{field_name}.{key} may contain identity or a path.")
            frozen[key] = _freeze_safe_json(
                item,
                field_name=f"{field_name}.{key}",
                depth=depth + 1,
            )
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        if len(value) > 10_000:
            raise ValidationError(f"{field_name} contains too many values.")
        return tuple(
            _freeze_safe_json(item, field_name=field_name, depth=depth + 1) for item in value
        )
    raise ValidationError(f"{field_name} contains unsupported type {type(value).__name__}.")


def thaw_json(value: FrozenJson) -> JsonScalar | list[Any] | dict[str, Any]:
    """Return ordinary JSON containers from an immutable contract value."""

    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class DatasetSample:
    """One deidentified sample referenced by content hash.

    ``group_id`` must itself be pseudonymous. It is intentionally named
    generically because the split unit may be a person, study, or acquisition
    group depending on the task.
    """

    sample_id: str
    group_id: str
    split: str
    artifact_sha256: str
    modality: str
    labels: Mapping[str, FrozenJson] = field(default_factory=dict)
    site: str | None = None
    scanner: str | None = None
    deidentified: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sample_id", _safe_identifier(self.sample_id, field_name="sample_id")
        )
        object.__setattr__(self, "group_id", _safe_identifier(self.group_id, field_name="group_id"))
        normalized_split = str(self.split).strip().lower()
        if normalized_split not in _ALLOWED_SPLITS:
            raise ValidationError("split must be train, validation, or test.")
        object.__setattr__(self, "split", normalized_split)
        digest = str(self.artifact_sha256).strip().lower()
        if not _SHA256.fullmatch(digest):
            raise ValidationError("artifact_sha256 must be a lowercase SHA-256 digest.")
        object.__setattr__(self, "artifact_sha256", digest)
        object.__setattr__(
            self, "modality", _safe_text(self.modality, field_name="modality", maximum=64)
        )
        if self.site is not None:
            object.__setattr__(self, "site", _safe_identifier(self.site, field_name="site"))
        if self.scanner is not None:
            object.__setattr__(
                self, "scanner", _safe_identifier(self.scanner, field_name="scanner")
            )
        if self.deidentified is not True:
            raise ValidationError("Dataset samples must be explicitly marked deidentified.")
        frozen_labels = _freeze_safe_json(self.labels, field_name="labels")
        if not isinstance(frozen_labels, Mapping):  # defensive: declared type is a mapping
            raise ValidationError("labels must be a mapping.")
        object.__setattr__(self, "labels", frozen_labels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "split": self.split,
            "artifact_sha256": self.artifact_sha256,
            "modality": self.modality,
            "labels": thaw_json(self.labels),
            "site": self.site,
            "scanner": self.scanner,
            "deidentified": True,
        }


@dataclass(frozen=True, slots=True)
class SplitValidationReport:
    sample_counts: Mapping[str, int]
    group_counts: Mapping[str, int]
    total_samples: int
    total_groups: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_counts", MappingProxyType(dict(self.sample_counts)))
        object.__setattr__(self, "group_counts", MappingProxyType(dict(self.group_counts)))


def validate_group_splits(samples: Sequence[DatasetSample]) -> SplitValidationReport:
    """Validate uniqueness and ensure no group crosses a dataset split."""

    if not samples:
        raise ValidationError("A dataset manifest must contain at least one sample.")
    seen_samples: set[str] = set()
    group_to_split: dict[str, str] = {}
    sample_counts: Counter[str] = Counter()
    groups_by_split: dict[str, set[str]] = {split: set() for split in _ALLOWED_SPLITS}
    for sample in samples:
        if not isinstance(sample, DatasetSample):
            raise ValidationError("Dataset manifests accept DatasetSample entries only.")
        if sample.sample_id in seen_samples:
            raise ValidationError(f"Duplicate sample_id {sample.sample_id!r}.")
        seen_samples.add(sample.sample_id)
        previous = group_to_split.setdefault(sample.group_id, sample.split)
        if previous != sample.split:
            raise ValidationError(
                f"Group {sample.group_id!r} leaks across {previous!r} and {sample.split!r}."
            )
        sample_counts[sample.split] += 1
        groups_by_split[sample.split].add(sample.group_id)
    return SplitValidationReport(
        sample_counts={split: sample_counts[split] for split in sorted(_ALLOWED_SPLITS)},
        group_counts={split: len(groups_by_split[split]) for split in sorted(_ALLOWED_SPLITS)},
        total_samples=len(samples),
        total_groups=len(group_to_split),
    )


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Versioned, path-free dataset description suitable for safe export."""

    dataset_id: str
    dataset_version: str
    task: str
    license_id: str
    samples: tuple[DatasetSample, ...]
    label_schema: Mapping[str, FrozenJson] = field(default_factory=dict)
    provenance: Mapping[str, FrozenJson] = field(default_factory=dict)
    schema_version: int = 1
    deidentified: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValidationError("Only dataset manifest schema_version 1 is supported.")
        if self.deidentified is not True:
            raise ValidationError("Dataset manifests must be explicitly marked deidentified.")
        object.__setattr__(
            self, "dataset_id", _safe_identifier(self.dataset_id, field_name="dataset_id")
        )
        object.__setattr__(
            self,
            "dataset_version",
            _safe_identifier(self.dataset_version, field_name="dataset_version"),
        )
        object.__setattr__(self, "task", _safe_identifier(self.task, field_name="task"))
        object.__setattr__(
            self,
            "license_id",
            _safe_text(self.license_id, field_name="license_id", maximum=128),
        )
        normalized_samples = tuple(self.samples)
        validate_group_splits(normalized_samples)
        object.__setattr__(self, "samples", normalized_samples)
        for name in ("label_schema", "provenance"):
            frozen = _freeze_safe_json(getattr(self, name), field_name=name)
            if not isinstance(frozen, Mapping):
                raise ValidationError(f"{name} must be a mapping.")
            object.__setattr__(self, name, frozen)

    @property
    def split_report(self) -> SplitValidationReport:
        return validate_group_splits(self.samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "openmedvisionx-dataset-manifest/v1",
            "schema_version": 1,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "task": self.task,
            "license_id": self.license_id,
            "deidentified": True,
            "label_schema": thaw_json(self.label_schema),
            "provenance": thaw_json(self.provenance),
            "samples": [sample.to_dict() for sample in self.samples],
        }


def dataset_manifest_from_mapping(payload: Mapping[str, Any]) -> DatasetManifest:
    """Parse one strict v1 mapping without retaining its source location."""

    if not isinstance(payload, Mapping):
        raise ValidationError("A dataset manifest document must contain one mapping.")
    allowed = {
        "schema",
        "schema_version",
        "dataset_id",
        "dataset_version",
        "task",
        "license_id",
        "deidentified",
        "label_schema",
        "provenance",
        "samples",
    }
    unknown = sorted(str(key) for key in payload if key not in allowed)
    if unknown:
        raise ValidationError(f"Dataset manifest contains unknown fields: {unknown}.")
    if payload.get("schema", "openmedvisionx-dataset-manifest/v1") != (
        "openmedvisionx-dataset-manifest/v1"
    ):
        raise ValidationError("Unsupported dataset manifest schema identifier.")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, Sequence) or isinstance(raw_samples, (str, bytes, bytearray)):
        raise ValidationError("Dataset manifest samples must be a sequence.")
    samples: list[DatasetSample] = []
    sample_fields = {
        "sample_id",
        "group_id",
        "split",
        "artifact_sha256",
        "modality",
        "labels",
        "site",
        "scanner",
        "deidentified",
    }
    for index, raw_sample in enumerate(raw_samples):
        if not isinstance(raw_sample, Mapping):
            raise ValidationError(f"samples[{index}] must be a mapping.")
        unknown_sample = sorted(str(key) for key in raw_sample if key not in sample_fields)
        if unknown_sample:
            raise ValidationError(f"samples[{index}] contains unknown fields: {unknown_sample}.")
        try:
            samples.append(
                DatasetSample(
                    sample_id=raw_sample["sample_id"],
                    group_id=raw_sample["group_id"],
                    split=raw_sample["split"],
                    artifact_sha256=raw_sample["artifact_sha256"],
                    modality=raw_sample["modality"],
                    labels=raw_sample.get("labels", {}),
                    site=raw_sample.get("site"),
                    scanner=raw_sample.get("scanner"),
                    deidentified=raw_sample.get("deidentified", False),
                )
            )
        except KeyError as exc:
            raise ValidationError(
                f"samples[{index}] is missing required field {exc.args[0]!r}."
            ) from exc
    try:
        return DatasetManifest(
            dataset_id=payload["dataset_id"],
            dataset_version=payload["dataset_version"],
            task=payload["task"],
            license_id=payload["license_id"],
            samples=tuple(samples),
            label_schema=payload.get("label_schema", {}),
            provenance=payload.get("provenance", {}),
            schema_version=payload.get("schema_version", 1),
            deidentified=payload.get("deidentified", False),
        )
    except KeyError as exc:
        raise ValidationError(
            f"Dataset manifest is missing required field {exc.args[0]!r}."
        ) from exc


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    """Load bounded JSON/YAML and return a path-free immutable manifest."""

    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise ValidationError("The selected dataset manifest is unavailable.") from exc
    if size <= 0 or size > _MAX_MANIFEST_BYTES:
        raise ValidationError("Dataset manifest size must be between 1 byte and 4 MiB.")
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValidationError("Dataset manifest must be readable UTF-8 text.") from exc
    suffix = source.suffix.lower()
    try:
        if suffix == ".json":
            payload = json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(text)
        else:
            raise ValidationError("Dataset manifest filename must end in .json, .yaml, or .yml.")
    except ValidationError:
        raise
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValidationError("Dataset manifest syntax is invalid.") from exc
    return dataset_manifest_from_mapping(payload)
