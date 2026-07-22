"""Immutable experiment records and atomic, path-free exports."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..errors import ResourceLimitError, ValidationError
from .contracts import (
    FrozenJson,
    _freeze_safe_json,
    _is_sensitive_key,
    _looks_like_local_path,
    _safe_identifier,
    thaw_json,
)

_MAX_RECORD_BYTES = 4 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MEDIA_TYPE = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")


def _safe_short_text(value: object, *, field_name: str, maximum: int = 512) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be text.")
    text = value.strip()
    if not text or len(text) > maximum or _looks_like_local_path(text):
        raise ValidationError(f"{field_name} must be safe text without a filesystem path.")
    return text


def _freeze_mapping(value: Mapping[str, Any], *, field_name: str) -> Mapping[str, FrozenJson]:
    frozen = _freeze_safe_json(value, field_name=field_name)
    if not isinstance(frozen, Mapping):  # defensive: declared input is a mapping
        raise ValidationError(f"{field_name} must be a mapping.")
    return frozen


def _validate_numeric_tree(value: FrozenJson, *, field_name: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_numeric_tree(item, field_name=f"{field_name}.{key}")
        return
    if isinstance(value, tuple):
        for item in value:
            _validate_numeric_tree(item, field_name=field_name)
        return
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field_name} must contain numeric values or null only.")


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """A path-free reference to an immutable input or output artifact."""

    artifact_id: str
    sha256: str
    kind: str
    media_type: str
    shape: tuple[int, ...] = ()
    coordinate_system: str | None = None
    label_schema: Mapping[str, FrozenJson] = field(default_factory=dict)
    deidentified: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_id",
            _safe_identifier(self.artifact_id, field_name="artifact_id"),
        )
        digest = str(self.sha256).strip().lower()
        if not _SHA256.fullmatch(digest):
            raise ValidationError("sha256 must be a lowercase SHA-256 digest.")
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "kind", _safe_identifier(self.kind, field_name="kind"))
        normalized_media_type = str(self.media_type).strip().lower()
        if not _MEDIA_TYPE.fullmatch(normalized_media_type):
            raise ValidationError("media_type must be a simple lowercase MIME type.")
        object.__setattr__(self, "media_type", normalized_media_type)
        normalized_shape: list[int] = []
        if len(self.shape) > 5:
            raise ValidationError("shape may contain at most five dimensions.")
        for dimension in self.shape:
            if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)):
                raise ValidationError("shape dimensions must be positive integers.")
            integer = int(dimension)
            if integer <= 0:
                raise ValidationError("shape dimensions must be positive integers.")
            normalized_shape.append(integer)
        object.__setattr__(self, "shape", tuple(normalized_shape))
        if self.coordinate_system is not None:
            object.__setattr__(
                self,
                "coordinate_system",
                _safe_identifier(self.coordinate_system, field_name="coordinate_system"),
            )
        object.__setattr__(
            self,
            "label_schema",
            _freeze_mapping(self.label_schema, field_name="label_schema"),
        )
        if self.deidentified is not True:
            raise ValidationError("Artifact references must be explicitly marked deidentified.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "sha256": self.sha256,
            "kind": self.kind,
            "media_type": self.media_type,
            "shape": list(self.shape),
            "coordinate_system": self.coordinate_system,
            "label_schema": thaw_json(self.label_schema),
            "deidentified": True,
        }


@dataclass(frozen=True, slots=True)
class TransformStep:
    """One explicit preprocessing or postprocessing step."""

    name: str
    version: str
    parameters: Mapping[str, FrozenJson] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _safe_identifier(self.name, field_name="transform.name"))
        object.__setattr__(
            self,
            "version",
            _safe_identifier(self.version, field_name="transform.version"),
        )
        object.__setattr__(
            self,
            "parameters",
            _freeze_mapping(self.parameters, field_name="transform.parameters"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "parameters": thaw_json(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """A versioned, immutable and pixel-free experiment description."""

    record_id: str
    created_at: datetime
    application_version: str
    code_revision: str
    task: str
    model_id: str
    model_version: str
    inputs: tuple[ArtifactReference, ...]
    transforms: tuple[TransformStep, ...]
    outputs: tuple[ArtifactReference, ...]
    metrics: Mapping[str, FrozenJson]
    parameters: Mapping[str, FrozenJson] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    duration_ms: float | None = None
    dataset_manifest_id: str | None = None
    schema_version: int = 1
    contains_phi: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValidationError("Only experiment record schema_version 1 is supported.")
        if self.contains_phi is not False:
            raise ValidationError(
                "Experiment records containing PHI cannot be created or exported."
            )
        for name in (
            "record_id",
            "application_version",
            "code_revision",
            "task",
            "model_id",
            "model_version",
        ):
            object.__setattr__(self, name, _safe_identifier(getattr(self, name), field_name=name))
        if not isinstance(self.created_at, datetime) or self.created_at.tzinfo is None:
            raise ValidationError("created_at must be a timezone-aware datetime.")
        timestamp = self.created_at.astimezone(timezone.utc)
        object.__setattr__(self, "created_at", timestamp)
        normalized_inputs = tuple(self.inputs)
        normalized_transforms = tuple(self.transforms)
        normalized_outputs = tuple(self.outputs)
        if not normalized_inputs:
            raise ValidationError("An experiment record requires at least one input artifact.")
        if any(not isinstance(item, ArtifactReference) for item in normalized_inputs):
            raise ValidationError("inputs must contain ArtifactReference values only.")
        if any(not isinstance(item, TransformStep) for item in normalized_transforms):
            raise ValidationError("transforms must contain TransformStep values only.")
        if any(not isinstance(item, ArtifactReference) for item in normalized_outputs):
            raise ValidationError("outputs must contain ArtifactReference values only.")
        artifact_ids = [item.artifact_id for item in (*normalized_inputs, *normalized_outputs)]
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValidationError("Input and output artifact_id values must be unique.")
        object.__setattr__(self, "inputs", normalized_inputs)
        object.__setattr__(self, "transforms", normalized_transforms)
        object.__setattr__(self, "outputs", normalized_outputs)
        frozen_metrics = _freeze_mapping(self.metrics, field_name="metrics")
        _validate_numeric_tree(frozen_metrics, field_name="metrics")
        object.__setattr__(self, "metrics", frozen_metrics)
        object.__setattr__(
            self,
            "parameters",
            _freeze_mapping(self.parameters, field_name="parameters"),
        )
        if len(self.warnings) > 128:
            raise ValidationError("warnings may contain at most 128 entries.")
        object.__setattr__(
            self,
            "warnings",
            tuple(_safe_short_text(item, field_name="warning") for item in self.warnings),
        )
        if self.duration_ms is not None:
            try:
                duration = float(self.duration_ms)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValidationError("duration_ms must be a non-negative finite number.") from exc
            if not np.isfinite(duration) or duration < 0.0:
                raise ValidationError("duration_ms must be a non-negative finite number.")
            object.__setattr__(self, "duration_ms", duration)
        if self.dataset_manifest_id is not None:
            object.__setattr__(
                self,
                "dataset_manifest_id",
                _safe_identifier(self.dataset_manifest_id, field_name="dataset_manifest_id"),
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible representation."""

        return {
            "schema": "openmedvisionx-experiment-record/v1",
            "schema_version": 1,
            "record_id": self.record_id,
            "created_at": self.created_at.isoformat().replace("+00:00", "Z"),
            "application_version": self.application_version,
            "code_revision": self.code_revision,
            "dataset_manifest_id": self.dataset_manifest_id,
            "task": self.task,
            "model": {"id": self.model_id, "version": self.model_version},
            "inputs": [item.to_dict() for item in self.inputs],
            "transforms": [item.to_dict() for item in self.transforms],
            "parameters": thaw_json(self.parameters),
            "outputs": [item.to_dict() for item in self.outputs],
            "metrics": thaw_json(self.metrics),
            "warnings": list(self.warnings),
            "duration_ms": self.duration_ms,
            "contains_phi": False,
        }


def _validate_export_payload(value: Any, *, field_name: str = "record", depth: int = 0) -> None:
    if depth > 12:
        raise ValidationError("Export payload is nested too deeply.")
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise ValidationError("Export field names must be text.")
            # ``name`` is a controlled TransformStep field. Dynamic mappings
            # have already rejected it during construction.
            if _is_sensitive_key(raw_key) and raw_key != "name":
                raise ValidationError(
                    f"Export field {field_name}.{raw_key} may contain PHI or a path."
                )
            _validate_export_payload(item, field_name=f"{field_name}.{raw_key}", depth=depth + 1)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        for item in value:
            _validate_export_payload(item, field_name=field_name, depth=depth + 1)
        return
    if isinstance(value, str) and _looks_like_local_path(value):
        raise ValidationError(f"Export field {field_name} contains a filesystem path.")
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValidationError(f"Export field {field_name} contains an unsupported value.")
    if isinstance(value, float) and not np.isfinite(value):
        raise ValidationError(f"Export field {field_name} contains a non-finite number.")


def _encode_record(record: ExperimentRecord, suffix: str) -> bytes:
    payload = record.to_dict()
    _validate_export_payload(payload)
    if suffix == ".json":
        encoded_text = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    else:
        encoded_text = yaml.safe_dump(
            payload,
            allow_unicode=True,
            sort_keys=True,
            default_flow_style=False,
        )
    encoded = (encoded_text.rstrip() + "\n").encode("utf-8")
    if len(encoded) > _MAX_RECORD_BYTES:
        raise ResourceLimitError("Experiment record export exceeds the 4 MiB safety limit.")
    return encoded


def export_experiment_record(
    path: str | Path,
    record: ExperimentRecord,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically export a validated record as JSON or safe YAML.

    With the default ``overwrite=False``, a hard-link commit makes the new
    target appear in one filesystem operation and cannot replace an existing
    file. ``overwrite=True`` uses ``os.replace`` for an explicit atomic update.
    """

    if not isinstance(record, ExperimentRecord):
        raise ValidationError("Only validated ExperimentRecord values can be exported.")
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix not in {".json", ".yaml", ".yml"}:
        raise ValidationError("Experiment records must use .json, .yaml, or .yml.")
    if not target.parent.is_dir():
        raise ValidationError("Select an existing local output directory.")
    encoded = _encode_record(record, suffix)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            try:
                os.link(temporary, target)
            except FileExistsError as exc:
                raise ValidationError(
                    "Output already exists; enable overwrite only after explicit confirmation."
                ) from exc
            except OSError as exc:
                raise ValidationError(
                    "The selected filesystem does not support an atomic no-overwrite commit."
                ) from exc
            temporary.unlink()
    except BaseException:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise
    return target
