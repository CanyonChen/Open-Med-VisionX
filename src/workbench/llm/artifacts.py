"""Provider-neutral contracts for structured multimodal LLM artifacts.

This module deliberately contains no transport or provider SDK integration.  A
provider adapter must first authenticate a response, normalize it into one of
the payload classes below, and retain the exact normalized bytes.  The public
contracts then validate those bytes before an artifact can be described as a
new derived layer.

Raw prompts, local paths, arbitrary download URLs, and patient metadata have no
field in these contracts.  Requests bind to opaque input references and to the
digest of an already reviewed :class:`~workbench.llm.types.TransferPlan`.
"""

from __future__ import annotations

import binascii
import gzip
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from io import BytesIO
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np
from PIL import Image

from ..domain.images import IntensitySemantics
from ..errors import MissingDependencyError, ValidationError

MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
MAX_DECODED_BYTES = 128 * 1024 * 1024
MAX_SPATIAL_ELEMENTS = 33_554_432

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}$")
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+/-]{0,191}$")
_LANGUAGE_RE = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")
_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_METADATA_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SAFE_PNG_CHUNKS = {b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tRNS"}
_SENSITIVE_KEY_TOKENS = {
    "accession",
    "address",
    "birth",
    "credential",
    "filename",
    "institution",
    "localpath",
    "medicalrecord",
    "mrn",
    "name",
    "patient",
    "path",
    "physician",
    "prompt",
    "secret",
    "token",
    "url",
}


class LLMTaskKind(str, Enum):
    """Controlled task names shared by every provider adapter."""

    RECONSTRUCT = "reconstruct"
    RESTORE = "restore"
    ENHANCE = "enhance"
    GENERATE = "generate"
    SEGMENT = "segment"
    CLASSIFY = "classify"
    LABELS = "labels"


class LLMInputKind(str, Enum):
    RENDERED_SLICE = "rendered_slice"
    IMAGE_2D = "image_2d"
    VOLUME_3D = "volume_3d"
    VOLUME_4D = "volume_4d"
    KSPACE = "kspace"
    SINOGRAM = "sinogram"


class LLMArtifactType(str, Enum):
    """The seven intentionally disjoint response artifact types."""

    TEXT = "text"
    CLASS_SCORES = "class_scores"
    LABELS = "labels"
    MASK_2D = "mask_2d"
    MASK_3D = "mask_3d"
    RECONSTRUCTED_IMAGE = "reconstructed_image"
    RECONSTRUCTED_VOLUME = "reconstructed_volume"


class ArtifactValidationStatus(str, Enum):
    """Semantic review state; byte-level validation always occurs on construction."""

    UNVERIFIED = "unverified"
    USER_CONFIRMED = "user_confirmed"
    REJECTED = "rejected"


class ScoreSemantics(str, Enum):
    RAW = "raw"
    LOGIT = "logit"
    PROBABILITY = "probability"
    RISK = "risk"


class CalibrationMethod(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    UNCALIBRATED = "uncalibrated"
    TEMPERATURE_SCALING = "temperature_scaling"
    PLATT_SCALING = "platt_scaling"
    ISOTONIC = "isotonic"
    PROVIDER_DECLARED = "provider_declared"


class MaskValueSemantics(str, Enum):
    DISCRETE = "discrete"
    BINARY = "binary"
    FRACTIONAL = "fractional"


class DataConsistencyMethod(str, Enum):
    HARD_PROJECTION = "hard_projection"
    SOFT_PENALTY = "soft_penalty"
    ITERATIVE_UPDATE = "iterative_update"


class DerivedLayerKind(str, Enum):
    SEGMENTATION = "segmentation"
    VOLUME = "volume"


class ArtifactReviewDecision(str, Enum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValidationError(f"{field_name} must be a datetime.")
    if value.tzinfo is None:
        raise ValidationError(f"{field_name} must include a timezone.")
    return value.astimezone(timezone.utc)


def _sha256(value: str, field_name: str) -> str:
    normalized = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValidationError(f"{field_name} must be a lowercase SHA-256 digest.")
    return normalized


def _opaque_id(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not _OPAQUE_ID_RE.fullmatch(normalized):
        raise ValidationError(
            f"{field_name} must be an opaque identifier without paths, URLs, or control text."
        )
    return normalized


def _model_id(value: str) -> str:
    """Accept registry-style model IDs while rejecting URL and traversal syntax."""

    normalized = str(value).strip()
    if (
        not _MODEL_ID_RE.fullmatch(normalized)
        or normalized.startswith("/")
        or normalized.endswith("/")
        or "//" in normalized
        or ".." in normalized
        or "://" in normalized
        or "\\" in normalized
    ):
        raise ValidationError(
            "model_id must be an opaque identifier, not a filesystem path or URL."
        )
    return normalized


def _short_text(value: str, field_name: str, *, maximum: int = 256) -> str:
    normalized = str(value).strip()
    if (
        not normalized
        or len(normalized) > maximum
        or any(character in normalized for character in ("\0", "\r"))
    ):
        raise ValidationError(f"{field_name} must be non-empty safe text up to {maximum} chars.")
    return normalized


def _language(value: str) -> str:
    normalized = str(value).strip()
    if not _LANGUAGE_RE.fullmatch(normalized):
        raise ValidationError("language must be a valid BCP-47-style language tag.")
    return normalized


def _shape(
    value: Sequence[int], field_name: str, *, dimensions: tuple[int, ...]
) -> tuple[int, ...]:
    try:
        normalized = tuple(int(item) for item in value)
    except (TypeError, ValueError):
        raise ValidationError(f"{field_name} must contain integer dimensions.") from None
    if len(normalized) not in dimensions or any(item <= 0 for item in normalized):
        expected = ", ".join(str(item) for item in dimensions)
        raise ValidationError(f"{field_name} must contain {expected} positive dimension(s).")
    if math.prod(normalized) > MAX_SPATIAL_ELEMENTS:
        raise ValidationError(f"{field_name} exceeds the decoded element limit.")
    return normalized


def _readonly_array(
    value: np.ndarray | Sequence[Any],
    field_name: str,
    *,
    dimensions: tuple[int, ...],
) -> np.ndarray:
    array = np.asanyarray(value)
    if array.dtype == object or not (
        np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValidationError(f"{field_name} must contain numeric values without object dtype.")
    if array.ndim not in dimensions or any(item <= 0 for item in array.shape):
        expected = ", ".join(str(item) for item in dimensions)
        raise ValidationError(f"{field_name} must have {expected} dimension(s).")
    if array.size > MAX_SPATIAL_ELEMENTS or array.nbytes > MAX_DECODED_BYTES:
        raise ValidationError(f"{field_name} exceeds the decoded payload limit.")
    if np.iscomplexobj(array) or not np.all(np.isfinite(array)):
        raise ValidationError(f"{field_name} must contain finite real values.")
    copied = np.array(array, copy=True)
    copied.setflags(write=False)
    return copied


def _affine(value: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    affine = np.asarray(value, dtype=np.float64)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValidationError("affine_ras must be a finite 4x4 matrix.")
    if not np.allclose(affine[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValidationError("affine_ras must use a homogeneous [0, 0, 0, 1] final row.")
    basis = affine[:3, :3]
    if np.any(np.linalg.norm(basis, axis=0) <= 0.0) or np.isclose(np.linalg.det(basis), 0.0):
        raise ValidationError("affine_ras must describe three independent spatial axes.")
    copied = np.array(affine, copy=True)
    copied.setflags(write=False)
    return copied


def _freeze_metadata(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate provider metadata and freeze it without retaining sensitive fields."""

    def freeze_item(item: Any, depth: int) -> Any:
        if depth > 4:
            raise ValidationError("provider metadata nesting is too deep.")
        if item is None or isinstance(item, (bool, int, str)):
            if isinstance(item, str):
                if len(item) > 512 or "\0" in item or "://" in item or "\\" in item:
                    raise ValidationError("provider metadata contains an unsafe string value.")
                if "/../" in f"/{item}/" or item.startswith(("/", "~")):
                    raise ValidationError("provider metadata cannot contain filesystem paths.")
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValidationError("provider metadata cannot contain NaN or infinity.")
            return item
        if isinstance(item, Mapping):
            return freeze_mapping(item, depth + 1)
        if isinstance(item, (list, tuple)):
            if len(item) > 64:
                raise ValidationError("provider metadata sequences cannot exceed 64 items.")
            return tuple(freeze_item(child, depth + 1) for child in item)
        raise ValidationError(
            "provider metadata supports only JSON scalar, mapping, and list values."
        )

    def freeze_mapping(mapping: Mapping[str, Any], depth: int) -> Mapping[str, Any]:
        if len(mapping) > 64:
            raise ValidationError("provider metadata mappings cannot exceed 64 keys.")
        result: dict[str, Any] = {}
        for raw_key, raw_value in mapping.items():
            key = str(raw_key)
            compact = re.sub(r"[^a-z0-9]", "", key.lower())
            if not _METADATA_KEY_RE.fullmatch(key) or any(
                token in compact for token in _SENSITIVE_KEY_TOKENS
            ):
                raise ValidationError(
                    "provider metadata keys cannot identify people, credentials, paths, or URLs."
                )
            result[key] = freeze_item(raw_value, depth)
        return MappingProxyType(result)

    if not isinstance(value, Mapping):
        raise ValidationError("provider metadata must be a mapping.")
    return freeze_mapping(value, 0)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class LLMInputReference:
    """An exact, deidentified payload reference; never a filename or URL."""

    input_id: str
    kind: LLMInputKind
    payload_sha256: str
    layer_id: str | None = None
    series_id: str | None = None
    slice_index: int | None = None
    transform_sha256: str | None = None
    deidentified: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_id", _opaque_id(self.input_id, "input_id"))
        object.__setattr__(self, "kind", LLMInputKind(self.kind))
        object.__setattr__(self, "payload_sha256", _sha256(self.payload_sha256, "payload_sha256"))
        if self.layer_id is not None:
            object.__setattr__(self, "layer_id", _opaque_id(self.layer_id, "layer_id"))
        if self.series_id is not None:
            object.__setattr__(self, "series_id", _opaque_id(self.series_id, "series_id"))
        if (self.layer_id is None) != (self.series_id is None):
            raise ValidationError(
                "layer_id and series_id must either both be present or both absent."
            )
        if self.slice_index is not None:
            if (
                self.layer_id is None
                or int(self.slice_index) != self.slice_index
                or self.slice_index < 0
            ):
                raise ValidationError(
                    "slice_index requires a referenced layer and a nonnegative integer."
                )
            object.__setattr__(self, "slice_index", int(self.slice_index))
        if self.transform_sha256 is not None:
            object.__setattr__(
                self,
                "transform_sha256",
                _sha256(self.transform_sha256, "transform_sha256"),
            )
        if self.deidentified is not True:
            raise ValidationError("LLM inputs must be explicitly marked deidentified.")

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "input_id": self.input_id,
            "kind": self.kind.value,
            "payload_sha256": self.payload_sha256,
            "layer_id": self.layer_id,
            "series_id": self.series_id,
            "slice_index": self.slice_index,
            "transform_sha256": self.transform_sha256,
            "deidentified": True,
        }


@dataclass(frozen=True, slots=True)
class ReconstructionEvidence:
    """Physics evidence required before an output may be named reconstruction."""

    sampling_operator_sha256: str
    data_consistency: DataConsistencyMethod
    acquisition_parameters_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sampling_operator_sha256",
            _sha256(self.sampling_operator_sha256, "sampling_operator_sha256"),
        )
        object.__setattr__(self, "data_consistency", DataConsistencyMethod(self.data_consistency))
        object.__setattr__(
            self,
            "acquisition_parameters_sha256",
            _sha256(self.acquisition_parameters_sha256, "acquisition_parameters_sha256"),
        )

    def canonical_dict(self) -> dict[str, str]:
        return {
            "sampling_operator_sha256": self.sampling_operator_sha256,
            "data_consistency": self.data_consistency.value,
            "acquisition_parameters_sha256": self.acquisition_parameters_sha256,
        }


_TASK_OUTPUT_TYPES: Mapping[LLMTaskKind, frozenset[LLMArtifactType]] = MappingProxyType(
    {
        LLMTaskKind.RECONSTRUCT: frozenset(
            {LLMArtifactType.RECONSTRUCTED_IMAGE, LLMArtifactType.RECONSTRUCTED_VOLUME}
        ),
        LLMTaskKind.RESTORE: frozenset(
            {LLMArtifactType.RECONSTRUCTED_IMAGE, LLMArtifactType.RECONSTRUCTED_VOLUME}
        ),
        LLMTaskKind.ENHANCE: frozenset(
            {LLMArtifactType.RECONSTRUCTED_IMAGE, LLMArtifactType.RECONSTRUCTED_VOLUME}
        ),
        LLMTaskKind.GENERATE: frozenset(
            {LLMArtifactType.RECONSTRUCTED_IMAGE, LLMArtifactType.RECONSTRUCTED_VOLUME}
        ),
        LLMTaskKind.SEGMENT: frozenset({LLMArtifactType.MASK_2D, LLMArtifactType.MASK_3D}),
        LLMTaskKind.CLASSIFY: frozenset({LLMArtifactType.CLASS_SCORES}),
        LLMTaskKind.LABELS: frozenset({LLMArtifactType.LABELS}),
    }
)


@dataclass(frozen=True, slots=True)
class LLMTaskRequest:
    """A provider-neutral task bound to exact reviewed transfer bytes.

    The raw prompt is intentionally absent.  ``prompt_sha256`` and
    ``transfer_plan_sha256`` let audit records bind the task without copying
    possibly sensitive text or any local file location.
    """

    request_id: str
    task: LLMTaskKind
    inputs: tuple[LLMInputReference, ...]
    transfer_plan_sha256: str
    prompt_sha256: str
    requested_artifact_types: tuple[LLMArtifactType, ...]
    reconstruction_evidence: ReconstructionEvidence | None = None
    created_at: datetime = field(default_factory=_utc_now)
    request_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        request_id = _opaque_id(self.request_id, "request_id")
        task = LLMTaskKind(self.task)
        inputs = tuple(self.inputs)
        if not inputs or not all(isinstance(item, LLMInputReference) for item in inputs):
            raise ValidationError("inputs must contain at least one LLMInputReference.")
        if len({item.input_id for item in inputs}) != len(inputs):
            raise ValidationError("LLM input IDs must be unique within a request.")
        requested = tuple(LLMArtifactType(item) for item in self.requested_artifact_types)
        if not requested or len(set(requested)) != len(requested):
            raise ValidationError("requested_artifact_types must be non-empty and unique.")
        unsupported = set(requested) - _TASK_OUTPUT_TYPES[task]
        if unsupported:
            names = ", ".join(sorted(item.value for item in unsupported))
            raise ValidationError(f"Task {task.value!r} cannot request artifact type(s): {names}.")

        evidence = self.reconstruction_evidence
        raw_inputs = {LLMInputKind.KSPACE, LLMInputKind.SINOGRAM}
        if task is LLMTaskKind.RECONSTRUCT:
            if evidence is None or not any(item.kind in raw_inputs for item in inputs):
                raise ValidationError(
                    "A reconstruction task requires k-space or sinogram input plus sampling "
                    "operator and data-consistency evidence."
                )
        elif evidence is not None:
            raise ValidationError(
                "Reconstruction evidence is valid only for the reconstruct task; use restore, "
                "enhance, or generate for ordinary image-to-image output."
            )

        transfer_digest = _sha256(self.transfer_plan_sha256, "transfer_plan_sha256")
        prompt_digest = _sha256(self.prompt_sha256, "prompt_sha256")
        created_at = _aware_utc(self.created_at, "created_at")
        canonical = {
            "request_id": request_id,
            "task": task.value,
            "inputs": [item.canonical_dict() for item in inputs],
            "transfer_plan_sha256": transfer_digest,
            "prompt_sha256": prompt_digest,
            "requested_artifact_types": [item.value for item in requested],
            "reconstruction_evidence": None if evidence is None else evidence.canonical_dict(),
        }
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "transfer_plan_sha256", transfer_digest)
        object.__setattr__(self, "prompt_sha256", prompt_digest)
        object.__setattr__(self, "requested_artifact_types", requested)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(
            self, "request_sha256", hashlib.sha256(_canonical_json(canonical)).hexdigest()
        )

    def validate_transfer_plan(self, plan: Any) -> None:
        """Verify an existing TransferPlan without importing or copying its payload."""

        from .types import TransferPlan

        if not isinstance(plan, TransferPlan):
            raise ValidationError("plan must be a TransferPlan.")
        if plan.plan_id != self.transfer_plan_sha256:
            raise ValidationError("The task does not match the reviewed TransferPlan digest.")
        if plan.prompt_sha256 != self.prompt_sha256:
            raise ValidationError("The task prompt digest differs from the reviewed TransferPlan.")
        if plan.task != self.task.value:
            raise ValidationError("The task kind differs from the reviewed TransferPlan.")
        planned_hashes = {item.sha256 for item in plan.items}
        missing = [
            item.input_id for item in self.inputs if item.payload_sha256 not in planned_hashes
        ]
        if missing:
            raise ValidationError("One or more task inputs are absent from the TransferPlan.")


@dataclass(frozen=True, slots=True)
class ProviderResponseMetadata:
    """Authenticated response provenance with no endpoint, credential, or raw headers."""

    provider_id: str
    model_id: str
    response_id: str
    authenticated: bool
    received_at: datetime = field(default_factory=_utc_now)
    latency_ms: float | None = None
    adapter_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_id", _opaque_id(self.provider_id, "provider_id"))
        object.__setattr__(self, "model_id", _model_id(self.model_id))
        object.__setattr__(self, "response_id", _opaque_id(self.response_id, "response_id"))
        if self.authenticated is not True:
            raise ValidationError(
                "Structured artifacts require an authenticated provider response."
            )
        object.__setattr__(self, "received_at", _aware_utc(self.received_at, "received_at"))
        if self.latency_ms is not None:
            latency = float(self.latency_ms)
            if not math.isfinite(latency) or latency < 0:
                raise ValidationError("latency_ms must be finite and nonnegative.")
            object.__setattr__(self, "latency_ms", latency)
        object.__setattr__(self, "adapter_metadata", _freeze_metadata(self.adapter_metadata))


@dataclass(frozen=True, slots=True)
class TextArtifact:
    text: str
    language: str
    citation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        text = str(self.text)
        if not text.strip() or len(text.encode("utf-8")) > 1_048_576 or "\0" in text:
            raise ValidationError("Text artifacts require non-empty UTF-8 text up to 1 MiB.")
        citations = tuple(_opaque_id(item, "citation_id") for item in self.citation_ids)
        if len(set(citations)) != len(citations):
            raise ValidationError("citation_ids must be unique.")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "language", _language(self.language))
        object.__setattr__(self, "citation_ids", citations)

    def canonical_bytes(self) -> bytes:
        return self.text.encode("utf-8")


@dataclass(frozen=True, slots=True)
class ClassScore:
    class_id: str
    class_name: str
    score: float
    semantics: ScoreSemantics
    threshold: float | None = None
    calibration: CalibrationMethod = CalibrationMethod.NOT_APPLICABLE
    calibration_note_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "class_id", _opaque_id(self.class_id, "class_id"))
        object.__setattr__(
            self, "class_name", _short_text(self.class_name, "class_name", maximum=128)
        )
        score = float(self.score)
        semantics = ScoreSemantics(self.semantics)
        if not math.isfinite(score):
            raise ValidationError("Class scores must be finite.")
        if semantics is ScoreSemantics.PROBABILITY and not 0.0 <= score <= 1.0:
            raise ValidationError("Scores declared as probability must be in [0, 1].")
        threshold = None if self.threshold is None else float(self.threshold)
        if threshold is not None:
            if not math.isfinite(threshold):
                raise ValidationError("Class-score thresholds must be finite.")
            if semantics is ScoreSemantics.PROBABILITY and not 0.0 <= threshold <= 1.0:
                raise ValidationError("Probability thresholds must be in [0, 1].")
        calibration = CalibrationMethod(self.calibration)
        note = self.calibration_note_sha256
        if note is not None:
            note = _sha256(note, "calibration_note_sha256")
        if calibration is CalibrationMethod.PROVIDER_DECLARED and note is None:
            raise ValidationError("Provider-declared calibration requires a note digest.")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "semantics", semantics)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "calibration", calibration)
        object.__setattr__(self, "calibration_note_sha256", note)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "score": self.score,
            "semantics": self.semantics.value,
            "threshold": self.threshold,
            "calibration": self.calibration.value,
            "calibration_note_sha256": self.calibration_note_sha256,
        }


@dataclass(frozen=True, slots=True)
class ClassScoresArtifact:
    scores: tuple[ClassScore, ...]

    def __post_init__(self) -> None:
        scores = tuple(self.scores)
        if not scores or not all(isinstance(item, ClassScore) for item in scores):
            raise ValidationError("scores must contain at least one ClassScore.")
        if len({item.class_id for item in scores}) != len(scores):
            raise ValidationError("Class-score IDs must be unique.")
        object.__setattr__(self, "scores", scores)

    def canonical_bytes(self) -> bytes:
        return _canonical_json({"scores": [item.canonical_dict() for item in self.scores]})


@dataclass(frozen=True, slots=True)
class SemanticLabel:
    label_id: str
    name: str
    confidence: float | None = None
    region_reference_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "label_id", _opaque_id(self.label_id, "label_id"))
        object.__setattr__(self, "name", _short_text(self.name, "label name", maximum=128))
        if self.confidence is not None:
            confidence = float(self.confidence)
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValidationError("Label confidence must be in [0, 1].")
            object.__setattr__(self, "confidence", confidence)
        regions = tuple(
            _opaque_id(item, "region_reference_id") for item in self.region_reference_ids
        )
        if len(set(regions)) != len(regions):
            raise ValidationError("region_reference_ids must be unique.")
        object.__setattr__(self, "region_reference_ids", regions)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "label_id": self.label_id,
            "name": self.name,
            "confidence": self.confidence,
            "region_reference_ids": list(self.region_reference_ids),
        }


@dataclass(frozen=True, slots=True)
class LabelsArtifact:
    labels: tuple[SemanticLabel, ...]
    language: str

    def __post_init__(self) -> None:
        labels = tuple(self.labels)
        if not labels or not all(isinstance(item, SemanticLabel) for item in labels):
            raise ValidationError("labels must contain at least one SemanticLabel.")
        if len({item.label_id for item in labels}) != len(labels):
            raise ValidationError("Semantic label IDs must be unique.")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "language", _language(self.language))

    def canonical_bytes(self) -> bytes:
        return _canonical_json(
            {
                "language": self.language,
                "labels": [item.canonical_dict() for item in self.labels],
            }
        )


@dataclass(frozen=True, slots=True)
class ArtifactLabelDefinition:
    value: int
    label_id: str
    name: str
    color: str

    def __post_init__(self) -> None:
        value = int(self.value)
        if value < 0 or value != self.value:
            raise ValidationError("Mask label values must be nonnegative integers.")
        color = str(self.color).upper()
        if not _HEX_COLOR_RE.fullmatch(color):
            raise ValidationError("Mask label colors must use #RRGGBB notation.")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "label_id", _opaque_id(self.label_id, "label_id"))
        object.__setattr__(self, "name", _short_text(self.name, "label name", maximum=128))
        object.__setattr__(self, "color", color)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "label_id": self.label_id,
            "name": self.name,
            "color": self.color,
        }


def _validated_label_schema(
    value: Sequence[ArtifactLabelDefinition],
) -> tuple[ArtifactLabelDefinition, ...]:
    labels = tuple(value)
    if not labels or not all(isinstance(item, ArtifactLabelDefinition) for item in labels):
        raise ValidationError("label_schema must contain at least one ArtifactLabelDefinition.")
    if len({item.value for item in labels}) != len(labels) or len(
        {item.label_id for item in labels}
    ) != len(labels):
        raise ValidationError("Mask label values and IDs must be unique.")
    return labels


@dataclass(frozen=True, slots=True, eq=False)
class SpatialArtifactReference:
    """Geometry and source-layer identity used to validate an inbound spatial artifact."""

    series_id: str
    layer_id: str
    shape_zyx: tuple[int, int, int]
    affine_ras: np.ndarray
    frame_of_reference_uid: str | None = None
    slice_axis: int | None = None
    slice_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "series_id", _opaque_id(self.series_id, "series_id"))
        object.__setattr__(self, "layer_id", _opaque_id(self.layer_id, "layer_id"))
        object.__setattr__(self, "shape_zyx", _shape(self.shape_zyx, "shape_zyx", dimensions=(3,)))
        object.__setattr__(self, "affine_ras", _affine(self.affine_ras))
        if self.frame_of_reference_uid is not None:
            uid = str(self.frame_of_reference_uid).strip()
            if len(uid) > 64 or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", uid):
                raise ValidationError("frame_of_reference_uid must be a valid DICOM UID.")
            object.__setattr__(self, "frame_of_reference_uid", uid)
        if (self.slice_axis is None) != (self.slice_index is None):
            raise ValidationError(
                "slice_axis and slice_index must either both be present or absent."
            )
        if self.slice_axis is not None:
            axis = int(self.slice_axis)
            index = int(self.slice_index)  # type: ignore[arg-type]
            if axis not in {0, 1, 2} or axis != self.slice_axis:
                raise ValidationError("slice_axis must be 0, 1, or 2.")
            if index != self.slice_index or not 0 <= index < self.shape_zyx[axis]:
                raise ValidationError("slice_index is outside the referenced geometry.")
            object.__setattr__(self, "slice_axis", axis)
            object.__setattr__(self, "slice_index", index)

    @property
    def slice_shape(self) -> tuple[int, int] | None:
        if self.slice_axis is None:
            return None
        return tuple(size for axis, size in enumerate(self.shape_zyx) if axis != self.slice_axis)  # type: ignore[return-value]


def _validate_mask_values(
    array: np.ndarray,
    semantics: MaskValueSemantics,
    label_schema: tuple[ArtifactLabelDefinition, ...],
) -> None:
    if semantics in {MaskValueSemantics.DISCRETE, MaskValueSemantics.BINARY}:
        if not np.all(np.equal(array, np.floor(array))):
            raise ValidationError("Discrete masks must contain integer-valued samples.")
        allowed = {item.value for item in label_schema}
        observed = {int(item) for item in np.unique(array)}
        if not observed.issubset(allowed):
            raise ValidationError("Mask samples contain values absent from label_schema.")
        if semantics is MaskValueSemantics.BINARY and not observed.issubset({0, 1}):
            raise ValidationError("Binary masks may contain only 0 and 1.")
    elif np.any(array < 0.0) or np.any(array > 1.0):
        raise ValidationError("Fractional masks must be in [0, 1].")


@dataclass(frozen=True, slots=True, eq=False)
class Mask2DArtifact:
    array: np.ndarray
    reference: SpatialArtifactReference
    label_schema: tuple[ArtifactLabelDefinition, ...]
    value_semantics: MaskValueSemantics = MaskValueSemantics.DISCRETE

    def __post_init__(self) -> None:
        array = _readonly_array(self.array, "mask_2d array", dimensions=(2,))
        if (
            not isinstance(self.reference, SpatialArtifactReference)
            or self.reference.slice_shape is None
        ):
            raise ValidationError("mask_2d requires a reference with a slice axis and index.")
        if array.shape != self.reference.slice_shape:
            raise ValidationError("mask_2d shape does not match the referenced slice.")
        labels = _validated_label_schema(self.label_schema)
        semantics = MaskValueSemantics(self.value_semantics)
        _validate_mask_values(array, semantics, labels)
        object.__setattr__(self, "array", array)
        object.__setattr__(self, "label_schema", labels)
        object.__setattr__(self, "value_semantics", semantics)


@dataclass(frozen=True, slots=True, eq=False)
class Mask3DArtifact:
    array: np.ndarray
    reference: SpatialArtifactReference
    label_schema: tuple[ArtifactLabelDefinition, ...]
    value_semantics: MaskValueSemantics = MaskValueSemantics.DISCRETE

    def __post_init__(self) -> None:
        array = _readonly_array(self.array, "mask_3d array", dimensions=(3,))
        if not isinstance(self.reference, SpatialArtifactReference):
            raise ValidationError("mask_3d requires a SpatialArtifactReference.")
        if self.reference.slice_axis is not None or array.shape != self.reference.shape_zyx:
            raise ValidationError("mask_3d shape must match the complete referenced volume.")
        labels = _validated_label_schema(self.label_schema)
        semantics = MaskValueSemantics(self.value_semantics)
        _validate_mask_values(array, semantics, labels)
        object.__setattr__(self, "array", array)
        object.__setattr__(self, "label_schema", labels)
        object.__setattr__(self, "value_semantics", semantics)


def _validated_intensity_semantics(value: IntensitySemantics | str) -> IntensitySemantics:
    semantics = IntensitySemantics(value)
    if semantics in {
        IntensitySemantics.LABEL,
        IntensitySemantics.DISCRETE_LABEL,
        IntensitySemantics.PROBABILITY,
    }:
        raise ValidationError("Reconstructed image payloads require non-label intensity semantics.")
    return semantics


@dataclass(frozen=True, slots=True, eq=False)
class ReconstructedImageArtifact:
    array: np.ndarray
    reference: SpatialArtifactReference
    intensity_semantics: IntensitySemantics | str

    def __post_init__(self) -> None:
        array = _readonly_array(self.array, "reconstructed image array", dimensions=(2, 3))
        if (
            not isinstance(self.reference, SpatialArtifactReference)
            or self.reference.slice_shape is None
        ):
            raise ValidationError("reconstructed_image requires a referenced slice.")
        spatial_shape = array.shape if array.ndim == 2 else array.shape[:2]
        if spatial_shape != self.reference.slice_shape:
            raise ValidationError("Reconstructed image shape does not match the referenced slice.")
        if array.ndim == 3 and array.shape[2] not in {3, 4}:
            raise ValidationError("Color reconstructed images must have three or four channels.")
        object.__setattr__(self, "array", array)
        object.__setattr__(
            self,
            "intensity_semantics",
            _validated_intensity_semantics(self.intensity_semantics),
        )


@dataclass(frozen=True, slots=True, eq=False)
class ReconstructedVolumeArtifact:
    array: np.ndarray
    reference: SpatialArtifactReference
    intensity_semantics: IntensitySemantics | str

    def __post_init__(self) -> None:
        array = _readonly_array(self.array, "reconstructed volume array", dimensions=(3, 4))
        if not isinstance(self.reference, SpatialArtifactReference):
            raise ValidationError("reconstructed_volume requires a SpatialArtifactReference.")
        if self.reference.slice_axis is not None or array.shape[-3:] != self.reference.shape_zyx:
            raise ValidationError(
                "Reconstructed volume spatial shape must match the complete reference geometry."
            )
        object.__setattr__(self, "array", array)
        object.__setattr__(
            self,
            "intensity_semantics",
            _validated_intensity_semantics(self.intensity_semantics),
        )


ArtifactPayload: TypeAlias = (
    TextArtifact
    | ClassScoresArtifact
    | LabelsArtifact
    | Mask2DArtifact
    | Mask3DArtifact
    | ReconstructedImageArtifact
    | ReconstructedVolumeArtifact
)

_PAYLOAD_TYPES: Mapping[LLMArtifactType, type[ArtifactPayload]] = MappingProxyType(
    {
        LLMArtifactType.TEXT: TextArtifact,
        LLMArtifactType.CLASS_SCORES: ClassScoresArtifact,
        LLMArtifactType.LABELS: LabelsArtifact,
        LLMArtifactType.MASK_2D: Mask2DArtifact,
        LLMArtifactType.MASK_3D: Mask3DArtifact,
        LLMArtifactType.RECONSTRUCTED_IMAGE: ReconstructedImageArtifact,
        LLMArtifactType.RECONSTRUCTED_VOLUME: ReconstructedVolumeArtifact,
    }
)

_ALLOWED_MIME_TYPES: Mapping[LLMArtifactType, frozenset[str]] = MappingProxyType(
    {
        LLMArtifactType.TEXT: frozenset({"text/plain; charset=utf-8"}),
        LLMArtifactType.CLASS_SCORES: frozenset({"application/json"}),
        LLMArtifactType.LABELS: frozenset({"application/json"}),
        LLMArtifactType.MASK_2D: frozenset({"image/png", "application/x-npy"}),
        LLMArtifactType.MASK_3D: frozenset(
            {"application/x-npy", "application/x-nifti", "application/nifti"}
        ),
        LLMArtifactType.RECONSTRUCTED_IMAGE: frozenset({"image/png", "application/x-npy"}),
        LLMArtifactType.RECONSTRUCTED_VOLUME: frozenset(
            {"application/x-npy", "application/x-nifti", "application/nifti"}
        ),
    }
)


def encode_npy(array: np.ndarray | Sequence[Any]) -> bytes:
    """Return a pickle-free NPY representation suitable for an artifact envelope."""

    value = np.asanyarray(array)
    if value.dtype == object:
        raise ValidationError("Object arrays cannot be encoded as LLM artifacts.")
    output = BytesIO()
    np.save(output, value, allow_pickle=False)
    encoded = output.getvalue()
    if len(encoded) > MAX_ARTIFACT_BYTES:
        raise ValidationError("Encoded NPY artifact exceeds the byte limit.")
    return encoded


def _validate_png_chunks(data: bytes) -> None:
    if not data.startswith(_PNG_SIGNATURE):
        raise ValidationError("PNG artifact signature is invalid.")
    offset = len(_PNG_SIGNATURE)
    saw_header = saw_data = saw_end = False
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValidationError("PNG artifact is truncated.")
        length = int.from_bytes(data[offset : offset + 4], "big")
        chunk_type = data[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(data) or chunk_type not in _SAFE_PNG_CHUNKS:
            raise ValidationError("PNG artifact contains truncated or unsafe metadata chunks.")
        payload = data[offset + 8 : offset + 8 + length]
        expected_crc = int.from_bytes(data[offset + 8 + length : end], "big")
        if binascii.crc32(chunk_type + payload) & 0xFFFFFFFF != expected_crc:
            raise ValidationError("PNG artifact failed its integrity check.")
        if not saw_header:
            if chunk_type != b"IHDR" or length != 13:
                raise ValidationError("PNG artifact must begin with one IHDR chunk.")
            saw_header = True
        elif chunk_type == b"IHDR":
            raise ValidationError("PNG artifact contains multiple IHDR chunks.")
        elif chunk_type == b"IDAT":
            saw_data = True
        elif chunk_type == b"IEND":
            if length != 0 or end != len(data):
                raise ValidationError("PNG artifact contains trailing data.")
            saw_end = True
            break
        offset = end
    if not (saw_header and saw_data and saw_end):
        raise ValidationError("PNG artifact is incomplete.")


def _decode_png(data: bytes, *, mask: bool) -> np.ndarray:
    _validate_png_chunks(data)
    try:
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_SPATIAL_ELEMENTS:
                raise ValidationError("PNG artifact dimensions exceed the safety limit.")
            if mask and image.mode not in {"1", "L", "I", "I;16", "I;16B", "I;16L"}:
                raise ValidationError("PNG masks must use a single-channel integer mode.")
            if not mask and image.mode not in {
                "1",
                "L",
                "I",
                "I;16",
                "I;16B",
                "I;16L",
                "RGB",
                "RGBA",
            }:
                raise ValidationError("PNG image mode is not supported.")
            image.load()
            decoded = np.array(image, copy=True)
    except ValidationError:
        raise
    except (OSError, ValueError):
        raise ValidationError("PNG artifact pixel data could not be decoded safely.") from None
    return decoded


def _decode_npy(data: bytes) -> np.ndarray:
    stream = BytesIO(data)
    try:
        decoded = np.load(stream, allow_pickle=False)
    except (OSError, ValueError):
        raise ValidationError("NPY artifact could not be decoded with pickle disabled.") from None
    if not isinstance(decoded, np.ndarray) or decoded.dtype == object:
        raise ValidationError("NPY artifact must contain one non-object ndarray.")
    if stream.tell() != len(data):
        raise ValidationError("NPY artifact contains trailing data.")
    return decoded


def _bounded_gzip_decode(data: bytes) -> bytes:
    try:
        with gzip.GzipFile(fileobj=BytesIO(data)) as stream:
            decoded = stream.read(MAX_DECODED_BYTES + 1)
    except (EOFError, OSError):
        raise ValidationError("Compressed NIfTI artifact is invalid.") from None
    if len(decoded) > MAX_DECODED_BYTES:
        raise ValidationError("Compressed NIfTI exceeds the decoded byte limit.")
    return decoded


def _decode_nifti(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    try:
        import nibabel as nib
    except ImportError:
        raise MissingDependencyError(
            "NIfTI LLM artifacts require the optional 'nifti' dependency."
        ) from None
    raw = _bounded_gzip_decode(data) if data.startswith(b"\x1f\x8b") else data
    if len(raw) < 352:
        raise ValidationError("NIfTI artifact is truncated.")
    sizes = {int.from_bytes(raw[:4], "little"), int.from_bytes(raw[:4], "big")}
    image_type: type[Any]
    if 348 in sizes and raw[344:347] == b"n+1":
        image_type = nib.Nifti1Image
    elif 540 in sizes and raw[4:7] == b"n+2":
        image_type = nib.Nifti2Image
    else:
        raise ValidationError("NIfTI artifact header or magic is invalid.")
    try:
        image = nib.as_closest_canonical(image_type.from_bytes(raw))
        shape = tuple(int(item) for item in image.shape)
        if len(shape) not in {3, 4} or math.prod(shape) > MAX_SPATIAL_ELEMENTS:
            raise ValidationError("NIfTI artifact must contain a bounded 3-D or 4-D array.")
        storage_dtype = np.dtype(image.get_data_dtype())
        if storage_dtype.hasobject or math.prod(shape) * storage_dtype.itemsize > MAX_DECODED_BYTES:
            raise ValidationError("NIfTI artifact exceeds the decoded byte limit.")
        xyz = np.asanyarray(image.dataobj)
        affine = np.asarray(image.affine, dtype=np.float64)
    except ValidationError:
        raise
    except Exception:
        raise ValidationError("NIfTI artifact could not be decoded safely.") from None
    zyx = np.transpose(xyz, (2, 1, 0)) if xyz.ndim == 3 else np.transpose(xyz, (3, 2, 1, 0))
    return zyx, affine


def _spatial_payload_array(payload: ArtifactPayload) -> np.ndarray | None:
    if isinstance(
        payload,
        (Mask2DArtifact, Mask3DArtifact, ReconstructedImageArtifact, ReconstructedVolumeArtifact),
    ):
        return payload.array
    return None


def _validate_encoded_payload(
    artifact_type: LLMArtifactType,
    payload: ArtifactPayload,
    mime_type: str,
    encoded: bytes,
) -> None:
    if isinstance(payload, (TextArtifact, ClassScoresArtifact, LabelsArtifact)):
        if encoded != payload.canonical_bytes():
            raise ValidationError(
                "Structured text/JSON bytes do not match the normalized typed payload."
            )
        return
    expected = _spatial_payload_array(payload)
    if expected is None:  # pragma: no cover - protected by the type map
        raise ValidationError("Unsupported artifact payload.")
    decoded_affine: np.ndarray | None = None
    if mime_type == "image/png":
        decoded = _decode_png(encoded, mask=artifact_type is LLMArtifactType.MASK_2D)
    elif mime_type == "application/x-npy":
        decoded = _decode_npy(encoded)
    else:
        decoded, decoded_affine = _decode_nifti(encoded)
    if decoded.shape != expected.shape or not np.array_equal(decoded, expected, equal_nan=False):
        raise ValidationError("Decoded artifact pixels do not match the typed payload array.")
    if decoded_affine is not None:
        reference = payload.reference
        if not np.allclose(decoded_affine, reference.affine_ras, rtol=1e-5, atol=1e-5):
            raise ValidationError("NIfTI affine does not match the artifact reference geometry.")


@dataclass(frozen=True, slots=True)
class ArtifactReview:
    decision: ArtifactReviewDecision
    reviewer_id: str
    note_sha256: str
    reviewed_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision", ArtifactReviewDecision(self.decision))
        object.__setattr__(self, "reviewer_id", _opaque_id(self.reviewer_id, "reviewer_id"))
        object.__setattr__(self, "note_sha256", _sha256(self.note_sha256, "note_sha256"))
        object.__setattr__(self, "reviewed_at", _aware_utc(self.reviewed_at, "reviewed_at"))


@dataclass(frozen=True, slots=True, eq=False)
class DerivedLayerDescription:
    """Validated recipe for creating a new immutable LLM-derived layer."""

    layer_id: str
    series_id: str
    name: str
    layer_kind: DerivedLayerKind
    array: np.ndarray = field(repr=False)
    reference: SpatialArtifactReference
    derived_from_layer_ids: tuple[str, ...]
    artifact_sha256: str
    provider_id: str
    model_id: str
    request_sha256: str
    prompt_sha256: str
    transfer_plan_sha256: str
    task: LLMTaskKind
    mime_type: str
    label_schema: tuple[ArtifactLabelDefinition, ...] = ()
    intensity_semantics: IntensitySemantics | None = None
    validation_status: ArtifactValidationStatus = ArtifactValidationStatus.UNVERIFIED
    immutable: bool = field(default=True, init=False)
    replace_original: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        layer_id = _opaque_id(self.layer_id, "layer_id")
        series_id = _opaque_id(self.series_id, "series_id")
        kind = DerivedLayerKind(self.layer_kind)
        derived = tuple(
            _opaque_id(item, "derived_from_layer_id") for item in self.derived_from_layer_ids
        )
        if not derived or len(set(derived)) != len(derived) or layer_id in derived:
            raise ValidationError(
                "A derived layer needs unique source layer IDs and a new layer ID."
            )
        if not isinstance(self.reference, SpatialArtifactReference):
            raise ValidationError("reference must be a SpatialArtifactReference.")
        if series_id != self.reference.series_id or self.reference.layer_id not in derived:
            raise ValidationError("Derived layer geometry must reference one of its source layers.")
        dimensions = (2, 3) if kind is DerivedLayerKind.SEGMENTATION else (2, 3, 4)
        array = _readonly_array(self.array, "derived layer array", dimensions=dimensions)
        labels = tuple(self.label_schema)
        intensity = self.intensity_semantics
        if kind is DerivedLayerKind.SEGMENTATION:
            labels = _validated_label_schema(labels)
            if intensity is not None:
                raise ValidationError(
                    "Segmentation layer descriptions cannot declare intensity semantics."
                )
        else:
            if labels:
                raise ValidationError("Volume layer descriptions cannot carry a mask label schema.")
            if intensity is None:
                raise ValidationError("Volume layer descriptions require intensity semantics.")
            intensity = _validated_intensity_semantics(intensity)
        status = ArtifactValidationStatus(self.validation_status)
        if status is ArtifactValidationStatus.REJECTED:
            raise ValidationError("Rejected artifacts cannot be described as derived layers.")
        object.__setattr__(self, "layer_id", layer_id)
        object.__setattr__(self, "series_id", series_id)
        object.__setattr__(self, "name", _short_text(self.name, "layer name", maximum=160))
        object.__setattr__(self, "layer_kind", kind)
        object.__setattr__(self, "array", array)
        object.__setattr__(self, "derived_from_layer_ids", derived)
        object.__setattr__(self, "label_schema", labels)
        object.__setattr__(self, "intensity_semantics", intensity)
        object.__setattr__(
            self, "artifact_sha256", _sha256(self.artifact_sha256, "artifact_sha256")
        )
        object.__setattr__(self, "provider_id", _opaque_id(self.provider_id, "provider_id"))
        object.__setattr__(self, "model_id", _model_id(self.model_id))
        object.__setattr__(self, "request_sha256", _sha256(self.request_sha256, "request_sha256"))
        object.__setattr__(self, "prompt_sha256", _sha256(self.prompt_sha256, "prompt_sha256"))
        object.__setattr__(
            self,
            "transfer_plan_sha256",
            _sha256(self.transfer_plan_sha256, "transfer_plan_sha256"),
        )
        object.__setattr__(self, "task", LLMTaskKind(self.task))
        object.__setattr__(self, "validation_status", status)


@dataclass(frozen=True, slots=True)
class LLMArtifactResponse:
    """One authenticated, hash-bound, mutually exclusive structured response."""

    artifact_id: str
    request_id: str
    request_sha256: str
    artifact_type: LLMArtifactType
    payload: ArtifactPayload
    provider: ProviderResponseMetadata
    mime_type: str
    encoded_bytes: bytes = field(repr=False)
    artifact_sha256: str
    warnings: tuple[str, ...] = ()
    validation_status: ArtifactValidationStatus = ArtifactValidationStatus.UNVERIFIED
    reviews: tuple[ArtifactReview, ...] = ()

    def __post_init__(self) -> None:
        artifact_type = LLMArtifactType(self.artifact_type)
        expected_type = _PAYLOAD_TYPES[artifact_type]
        if not isinstance(self.payload, expected_type):
            raise ValidationError(
                f"artifact_type {artifact_type.value!r} requires {expected_type.__name__}; "
                "text or scores cannot masquerade as a spatial artifact."
            )
        if not isinstance(self.provider, ProviderResponseMetadata):
            raise ValidationError("provider must be authenticated ProviderResponseMetadata.")
        mime_type = str(self.mime_type).strip().lower()
        if mime_type not in _ALLOWED_MIME_TYPES[artifact_type]:
            raise ValidationError(
                f"MIME type {mime_type!r} is not allowed for {artifact_type.value!r}."
            )
        encoded = bytes(self.encoded_bytes)
        if not encoded or len(encoded) > MAX_ARTIFACT_BYTES:
            raise ValidationError("Artifact bytes must be non-empty and within the 32 MiB limit.")
        digest = _sha256(self.artifact_sha256, "artifact_sha256")
        if hashlib.sha256(encoded).hexdigest() != digest:
            raise ValidationError("Artifact SHA-256 does not match the exact encoded bytes.")
        _validate_encoded_payload(artifact_type, self.payload, mime_type, encoded)

        warnings = tuple(
            _short_text(item, "artifact warning", maximum=512) for item in self.warnings
        )
        reviews = tuple(self.reviews)
        if not all(isinstance(item, ArtifactReview) for item in reviews):
            raise ValidationError("reviews must contain ArtifactReview values.")
        status = ArtifactValidationStatus(self.validation_status)
        if status is ArtifactValidationStatus.USER_CONFIRMED:
            if not reviews or reviews[-1].decision is not ArtifactReviewDecision.CONFIRMED:
                raise ValidationError("USER_CONFIRMED status requires a final confirming review.")
        elif status is ArtifactValidationStatus.REJECTED:
            if not reviews or reviews[-1].decision is not ArtifactReviewDecision.REJECTED:
                raise ValidationError("REJECTED status requires a final rejecting review.")
        elif reviews:
            raise ValidationError("UNVERIFIED artifacts cannot already contain review records.")

        object.__setattr__(self, "artifact_id", _opaque_id(self.artifact_id, "artifact_id"))
        object.__setattr__(self, "request_id", _opaque_id(self.request_id, "request_id"))
        object.__setattr__(self, "request_sha256", _sha256(self.request_sha256, "request_sha256"))
        object.__setattr__(self, "artifact_type", artifact_type)
        object.__setattr__(self, "mime_type", mime_type)
        object.__setattr__(self, "encoded_bytes", encoded)
        object.__setattr__(self, "artifact_sha256", digest)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "validation_status", status)
        object.__setattr__(self, "reviews", reviews)

    @classmethod
    def from_normalized_payload(
        cls,
        *,
        artifact_id: str,
        request: LLMTaskRequest,
        artifact_type: LLMArtifactType,
        payload: ArtifactPayload,
        provider: ProviderResponseMetadata,
        mime_type: str | None = None,
        encoded_bytes: bytes | None = None,
        warnings: Sequence[str] = (),
    ) -> LLMArtifactResponse:
        """Build an unverified response while retaining exact normalized bytes.

        Text and JSON payloads have a canonical encoding.  Spatial adapters
        must pass the authenticated PNG/NIfTI bytes, or may explicitly choose
        the safe pickle-free NPY normalization by omitting ``encoded_bytes``.
        """

        normalized_type = LLMArtifactType(artifact_type)
        if isinstance(payload, (TextArtifact, ClassScoresArtifact, LabelsArtifact)):
            expected_mime = (
                "text/plain; charset=utf-8"
                if isinstance(payload, TextArtifact)
                else "application/json"
            )
            if encoded_bytes is None:
                encoded_bytes = payload.canonical_bytes()
            if mime_type is None:
                mime_type = expected_mime
        else:
            if encoded_bytes is None:
                encoded_bytes = encode_npy(payload.array)
                mime_type = "application/x-npy"
            elif mime_type is None:
                raise ValidationError("Spatial response bytes require an explicit MIME type.")
        digest = hashlib.sha256(encoded_bytes).hexdigest()
        response = cls(
            artifact_id=artifact_id,
            request_id=request.request_id,
            request_sha256=request.request_sha256,
            artifact_type=normalized_type,
            payload=payload,
            provider=provider,
            mime_type=mime_type,
            encoded_bytes=encoded_bytes,
            artifact_sha256=digest,
            warnings=tuple(warnings),
        )
        response.validate_against(request)
        return response

    @property
    def is_spatial(self) -> bool:
        return _spatial_payload_array(self.payload) is not None

    @property
    def produced_structured_image(self) -> bool:
        return self.is_spatial

    def validate_against(self, request: LLMTaskRequest) -> None:
        if not isinstance(request, LLMTaskRequest):
            raise ValidationError("request must be an LLMTaskRequest.")
        if self.request_id != request.request_id or self.request_sha256 != request.request_sha256:
            raise ValidationError("Artifact response does not match the exact task request.")
        if self.artifact_type is not LLMArtifactType.TEXT and (
            self.artifact_type not in request.requested_artifact_types
        ):
            raise ValidationError("Provider returned an artifact type not requested by this task.")
        payload = self.payload
        if isinstance(
            payload,
            (
                Mask2DArtifact,
                Mask3DArtifact,
                ReconstructedImageArtifact,
                ReconstructedVolumeArtifact,
            ),
        ):
            matching = [
                item
                for item in request.inputs
                if item.layer_id == payload.reference.layer_id
                and item.series_id == payload.reference.series_id
            ]
            if not matching:
                raise ValidationError(
                    "Spatial artifact references a layer absent from the request."
                )

    def with_review(self, review: ArtifactReview) -> LLMArtifactResponse:
        """Append an immutable review without erasing provider provenance."""

        if not isinstance(review, ArtifactReview):
            raise ValidationError("review must be an ArtifactReview.")
        if self.validation_status is not ArtifactValidationStatus.UNVERIFIED:
            raise ValidationError("A completed artifact review cannot be overwritten.")
        status = (
            ArtifactValidationStatus.USER_CONFIRMED
            if review.decision is ArtifactReviewDecision.CONFIRMED
            else ArtifactValidationStatus.REJECTED
        )
        return replace(self, reviews=(*self.reviews, review), validation_status=status)

    def describe_derived_layer(
        self,
        request: LLMTaskRequest,
        *,
        layer_id: str,
        name: str | None = None,
    ) -> DerivedLayerDescription:
        """Describe a new immutable layer; never overwrite a source layer.

        This method intentionally returns a recipe instead of mutating an
        ``ImageStudy``.  The application can commit it atomically after its own
        geometry and user-review workflow.
        """

        self.validate_against(request)
        payload = self.payload
        if not isinstance(
            payload,
            (
                Mask2DArtifact,
                Mask3DArtifact,
                ReconstructedImageArtifact,
                ReconstructedVolumeArtifact,
            ),
        ):
            raise ValidationError(
                f"{self.artifact_type.value!r} does not contain structured spatial data; "
                "text, scores, and language labels cannot create image layers."
            )
        if self.validation_status is ArtifactValidationStatus.REJECTED:
            raise ValidationError("A rejected artifact cannot create a derived layer.")
        derived_ids = tuple(
            dict.fromkeys(item.layer_id for item in request.inputs if item.layer_id is not None)
        )
        if payload.reference.layer_id not in derived_ids:
            raise ValidationError("The artifact reference is not among the derived input layers.")
        if isinstance(payload, (Mask2DArtifact, Mask3DArtifact)):
            layer_kind = DerivedLayerKind.SEGMENTATION
            default_name = f"LLM {request.task.value} segmentation"
            labels = payload.label_schema
            intensity = None
        else:
            layer_kind = DerivedLayerKind.VOLUME
            default_name = f"LLM {request.task.value} derived image"
            labels = ()
            intensity = payload.intensity_semantics
        return DerivedLayerDescription(
            layer_id=layer_id,
            series_id=payload.reference.series_id,
            name=default_name if name is None else name,
            layer_kind=layer_kind,
            array=payload.array,
            reference=payload.reference,
            derived_from_layer_ids=derived_ids,
            artifact_sha256=self.artifact_sha256,
            provider_id=self.provider.provider_id,
            model_id=self.provider.model_id,
            request_sha256=request.request_sha256,
            prompt_sha256=request.prompt_sha256,
            transfer_plan_sha256=request.transfer_plan_sha256,
            task=request.task,
            mime_type=self.mime_type,
            label_schema=labels,
            intensity_semantics=intensity,
            validation_status=self.validation_status,
        )


__all__ = [
    "ArtifactLabelDefinition",
    "ArtifactPayload",
    "ArtifactReview",
    "ArtifactReviewDecision",
    "ArtifactValidationStatus",
    "CalibrationMethod",
    "ClassScore",
    "ClassScoresArtifact",
    "DataConsistencyMethod",
    "DerivedLayerDescription",
    "DerivedLayerKind",
    "LLMArtifactResponse",
    "LLMArtifactType",
    "LLMInputKind",
    "LLMInputReference",
    "LLMTaskKind",
    "LLMTaskRequest",
    "LabelsArtifact",
    "MAX_ARTIFACT_BYTES",
    "MAX_DECODED_BYTES",
    "Mask2DArtifact",
    "Mask3DArtifact",
    "MaskValueSemantics",
    "ProviderResponseMetadata",
    "ReconstructedImageArtifact",
    "ReconstructedVolumeArtifact",
    "ReconstructionEvidence",
    "ScoreSemantics",
    "SemanticLabel",
    "SpatialArtifactReference",
    "TextArtifact",
    "encode_npy",
]
