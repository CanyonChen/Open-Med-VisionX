"""Typed, runtime-agnostic results returned by model plugins."""

# ``dataclass(slots=True)`` replaces the class object, so zero-argument super()
# is unsafe on supported Python versions (CPython gh-90562).
# ruff: noqa: UP008

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

from .enums import BoxFormat, CoordinateSystem, RuntimeKind, Task

if TYPE_CHECKING:
    from dicom_viewer.domain.transforms import TransformRecord


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _scores(values: Mapping[str, float], name: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in values.items():
        label = str(key).strip()
        if not label:
            raise ValueError(f"{name} labels must not be empty")
        result[label] = _finite(value, f"{name}[{label!r}]")
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class InferenceProvenance:
    """Non-sensitive runtime provenance attached to every result."""

    model_name: str
    model_version: str
    runtime: RuntimeKind
    request_id: str | None = None
    duration_ms: float | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        if not self.model_name.strip() or not self.model_version.strip():
            raise ValueError("model_name and model_version must not be empty")
        if not isinstance(self.runtime, RuntimeKind):
            object.__setattr__(self, "runtime", RuntimeKind.coerce(self.runtime))
        if self.duration_ms is not None:
            duration = _finite(self.duration_ms, "duration_ms")
            if duration < 0:
                raise ValueError("duration_ms must be non-negative")
            object.__setattr__(self, "duration_ms", duration)


@dataclass(frozen=True, slots=True, kw_only=True)
class UncertaintyResult:
    """Typed uncertainty payload shared by task-specific results."""

    score: float | None = None
    interval: tuple[float, float] | None = None
    map: Any | None = None
    calibrated: bool = False
    method: str | None = None

    def __post_init__(self) -> None:
        if self.score is not None:
            object.__setattr__(self, "score", _finite(self.score, "uncertainty.score"))
        if self.interval is not None:
            if len(self.interval) != 2:
                raise ValueError("uncertainty interval must contain [lower, upper]")
            lower = _finite(self.interval[0], "uncertainty.interval.lower")
            upper = _finite(self.interval[1], "uncertainty.interval.upper")
            if lower > upper:
                raise ValueError("uncertainty interval lower bound exceeds upper bound")
            object.__setattr__(self, "interval", (lower, upper))


@dataclass(frozen=True, slots=True, kw_only=True)
class BoundingBox:
    coordinates: tuple[float, ...]
    format: BoxFormat
    coordinate_system: CoordinateSystem
    label: str | None = None
    score: float | None = None
    class_id: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.format, BoxFormat):
            object.__setattr__(self, "format", BoxFormat.coerce(self.format))
        if not isinstance(self.coordinate_system, CoordinateSystem):
            object.__setattr__(
                self,
                "coordinate_system",
                CoordinateSystem.coerce(self.coordinate_system),
            )
        expected = 6 if self.format is BoxFormat.XYZXYZ else 4
        if len(self.coordinates) != expected:
            raise ValueError(f"{self.format.value} boxes require {expected} coordinates")
        coords = tuple(_finite(value, "box coordinate") for value in self.coordinates)
        object.__setattr__(self, "coordinates", coords)
        if self.format is BoxFormat.XYXY and (coords[2] < coords[0] or coords[3] < coords[1]):
            raise ValueError("xyxy maximum coordinates must not be less than minima")
        if self.format is BoxFormat.XYZXYZ and (
            coords[3] < coords[0] or coords[4] < coords[1] or coords[5] < coords[2]
        ):
            raise ValueError("xyzxyz maximum coordinates must not be less than minima")
        if self.score is not None:
            score = _finite(self.score, "box score")
            if not 0.0 <= score <= 1.0:
                raise ValueError("box score must be in [0, 1]")
            object.__setattr__(self, "score", score)


@dataclass(frozen=True, slots=True, kw_only=True)
class Keypoint:
    coordinates: tuple[float, ...]
    coordinate_system: CoordinateSystem
    label: str | None = None
    score: float | None = None

    def __post_init__(self) -> None:
        if len(self.coordinates) not in {2, 3}:
            raise ValueError("a keypoint must have 2D or 3D coordinates")
        object.__setattr__(
            self,
            "coordinates",
            tuple(_finite(value, "keypoint coordinate") for value in self.coordinates),
        )
        if not isinstance(self.coordinate_system, CoordinateSystem):
            object.__setattr__(
                self,
                "coordinate_system",
                CoordinateSystem.coerce(self.coordinate_system),
            )
        if self.score is not None:
            score = _finite(self.score, "keypoint score")
            if not 0.0 <= score <= 1.0:
                raise ValueError("keypoint score must be in [0, 1]")
            object.__setattr__(self, "score", score)


@dataclass(frozen=True, slots=True, kw_only=True)
class SamplingState:
    step: int
    sample: Any
    time: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("sampling step must be non-negative")
        if self.time is not None:
            object.__setattr__(self, "time", _finite(self.time, "sampling time"))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True, kw_only=True)
class TrackObservation:
    frame_index: int
    box: BoundingBox | None = None
    mask: Any | None = None
    keypoints: tuple[Keypoint, ...] = ()
    score: float | None = None

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if self.box is None and self.mask is None and not self.keypoints:
            raise ValueError("a track observation requires a box, mask, or keypoints")
        if self.score is not None:
            score = _finite(self.score, "track observation score")
            if not 0.0 <= score <= 1.0:
                raise ValueError("track observation score must be in [0, 1]")
            object.__setattr__(self, "score", score)


@dataclass(frozen=True, slots=True, kw_only=True)
class Track:
    track_id: str
    observations: tuple[TrackObservation, ...]
    label: str | None = None

    def __post_init__(self) -> None:
        if not str(self.track_id).strip():
            raise ValueError("track_id must not be empty")
        if not self.observations:
            raise ValueError("track must contain at least one observation")
        frame_indices = [item.frame_index for item in self.observations]
        if frame_indices != sorted(frame_indices) or len(frame_indices) != len(set(frame_indices)):
            raise ValueError("track observations must have unique ascending frame indices")


@dataclass(frozen=True, slots=True, kw_only=True)
class InferenceResult:
    """Extensible base result; task-specific subclasses add typed payloads."""

    task: Task
    provenance: InferenceProvenance
    transform_record: TransformRecord | None = None
    uncertainty: UncertaintyResult | None = None
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    allowed_tasks: ClassVar[frozenset[Task] | None] = None

    def __post_init__(self) -> None:
        if not isinstance(self.task, Task):
            object.__setattr__(self, "task", Task.coerce(self.task))
        if self.allowed_tasks is not None and self.task not in self.allowed_tasks:
            allowed = ", ".join(
                task.value for task in sorted(self.allowed_tasks, key=lambda item: item.value)
            )
            raise ValueError(
                f"{type(self).__name__} cannot represent task "
                f"{self.task.value!r}; expected {allowed}"
            )
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True, kw_only=True)
class ClassificationResult(InferenceResult):
    task: Task = field(default=Task.CLASSIFICATION, init=False)
    scores: Mapping[str, float]
    predicted_label: str | None = None
    logits: Any | None = None
    saliency_maps: Mapping[str, Any] = field(default_factory=dict)

    allowed_tasks: ClassVar[frozenset[Task]] = frozenset({Task.CLASSIFICATION})

    def __post_init__(self) -> None:
        super(ClassificationResult, self).__post_init__()
        normalized = _scores(self.scores, "scores")
        if not normalized:
            raise ValueError("classification scores must not be empty")
        if self.predicted_label is not None and self.predicted_label not in normalized:
            raise ValueError("predicted_label must be present in scores")
        object.__setattr__(self, "scores", normalized)
        object.__setattr__(self, "saliency_maps", dict(self.saliency_maps))


@dataclass(frozen=True, slots=True, kw_only=True)
class SegmentationResult(InferenceResult):
    task: Task = field(default=Task.SEGMENTATION, init=False)
    masks: Any
    labels: Mapping[int, str] = field(default_factory=dict)
    probabilities: Any | None = None
    logits: Any | None = None
    instances: tuple[Mapping[str, Any], ...] = ()

    allowed_tasks: ClassVar[frozenset[Task]] = frozenset({Task.SEGMENTATION})

    def __post_init__(self) -> None:
        super(SegmentationResult, self).__post_init__()
        if self.masks is None:
            raise ValueError("segmentation masks must not be None")
        object.__setattr__(self, "labels", {int(k): str(v) for k, v in self.labels.items()})
        object.__setattr__(self, "instances", tuple(dict(item) for item in self.instances))


@dataclass(frozen=True, slots=True, kw_only=True)
class DetectionResult(InferenceResult):
    task: Task = field(default=Task.DETECTION, init=False)
    boxes: tuple[BoundingBox, ...]
    keypoints: tuple[Keypoint, ...] = ()
    masks: Any | None = None

    allowed_tasks: ClassVar[frozenset[Task]] = frozenset({Task.DETECTION})

    def __post_init__(self) -> None:
        super(DetectionResult, self).__post_init__()
        if any(not isinstance(box, BoundingBox) for box in self.boxes):
            raise TypeError("boxes must contain BoundingBox values")
        if any(not isinstance(point, Keypoint) for point in self.keypoints):
            raise TypeError("keypoints must contain Keypoint values")


@dataclass(frozen=True, slots=True, kw_only=True)
class RepresentationResult(InferenceResult):
    task: Task = field(default=Task.REPRESENTATION, init=False)
    embeddings: Any
    feature_maps: Mapping[str, Any] = field(default_factory=dict)
    attention_maps: Mapping[str, Any] = field(default_factory=dict)

    allowed_tasks: ClassVar[frozenset[Task]] = frozenset({Task.REPRESENTATION})

    def __post_init__(self) -> None:
        super(RepresentationResult, self).__post_init__()
        if self.embeddings is None:
            raise ValueError("representation embeddings must not be None")
        object.__setattr__(self, "feature_maps", dict(self.feature_maps))
        object.__setattr__(self, "attention_maps", dict(self.attention_maps))


@dataclass(frozen=True, slots=True, kw_only=True)
class RegistrationResult(InferenceResult):
    task: Task = field(default=Task.REGISTRATION, init=False)
    warped_image: Any | None = None
    vector_field: Any | None = None
    affine_matrix: Any | None = None
    jacobian: Any | None = None
    inverse_consistency: Any | None = None

    allowed_tasks: ClassVar[frozenset[Task]] = frozenset({Task.REGISTRATION})

    def __post_init__(self) -> None:
        super(RegistrationResult, self).__post_init__()
        if self.warped_image is None and self.vector_field is None and self.affine_matrix is None:
            raise ValueError(
                "registration result requires warped_image, vector_field, or affine_matrix"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReconstructionResult(InferenceResult):
    task: Task = field(default=Task.RECONSTRUCTION, init=False)
    reconstructed_image: Any
    data_consistency_residual: Any | None = None
    spectrum: Any | None = None
    intermediate_images: tuple[Any, ...] = ()
    sampling_trajectory: tuple[SamplingState, ...] = ()

    allowed_tasks: ClassVar[frozenset[Task]] = frozenset({Task.RECONSTRUCTION})

    def __post_init__(self) -> None:
        super(ReconstructionResult, self).__post_init__()
        if self.reconstructed_image is None:
            raise ValueError("reconstructed_image must not be None")


@dataclass(frozen=True, slots=True, kw_only=True)
class RestorationResult(InferenceResult):
    task: Task = field(default=Task.RESTORATION, init=False)
    restored_image: Any
    noise_estimate: Any | None = None
    error_map: Any | None = None
    intermediate_images: tuple[Any, ...] = ()

    allowed_tasks: ClassVar[frozenset[Task]] = frozenset({Task.RESTORATION})

    def __post_init__(self) -> None:
        super(RestorationResult, self).__post_init__()
        if self.restored_image is None:
            raise ValueError("restored_image must not be None")


@dataclass(frozen=True, slots=True, kw_only=True)
class GenerationResult(InferenceResult):
    task: Task = field(default=Task.GENERATION, init=False)
    samples: tuple[Any, ...]
    latents: Any | None = None
    sampling_trajectory: tuple[SamplingState, ...] = ()
    safety_warnings: tuple[str, ...] = ()

    allowed_tasks: ClassVar[frozenset[Task]] = frozenset({Task.GENERATION})

    def __post_init__(self) -> None:
        super(GenerationResult, self).__post_init__()
        if not self.samples:
            raise ValueError("generation result must contain at least one sample")
        object.__setattr__(
            self,
            "safety_warnings",
            tuple(str(item) for item in self.safety_warnings),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class MultimodalResult(InferenceResult):
    task: Task = Task.MULTIMODAL
    text_output: str | None = None
    similarity_scores: Mapping[str, float] = field(default_factory=dict)
    embeddings: Any | None = None
    grounding_boxes: tuple[BoundingBox, ...] = ()

    allowed_tasks: ClassVar[frozenset[Task]] = frozenset(
        {Task.MULTIMODAL, Task.RETRIEVAL, Task.VQA, Task.REPORT_GENERATION}
    )

    def __post_init__(self) -> None:
        super(MultimodalResult, self).__post_init__()
        scores = _scores(self.similarity_scores, "similarity_scores")
        object.__setattr__(self, "similarity_scores", scores)
        if (
            self.text_output is None
            and not scores
            and self.embeddings is None
            and not self.grounding_boxes
        ):
            raise ValueError("multimodal result must contain at least one output payload")


@dataclass(frozen=True, slots=True, kw_only=True)
class TrackingResult(InferenceResult):
    task: Task = field(default=Task.TRACKING, init=False)
    tracks: tuple[Track, ...]
    vector_fields: Any | None = None
    memory_summary: Mapping[str, Any] = field(default_factory=dict)

    allowed_tasks: ClassVar[frozenset[Task]] = frozenset({Task.TRACKING})

    def __post_init__(self) -> None:
        super(TrackingResult, self).__post_init__()
        if any(not isinstance(track, Track) for track in self.tracks):
            raise TypeError("tracks must contain Track values")
        object.__setattr__(self, "memory_summary", dict(self.memory_summary))


@dataclass(frozen=True, slots=True, kw_only=True)
class AnomalyDetectionResult(InferenceResult):
    task: Task = field(default=Task.ANOMALY_DETECTION, init=False)
    anomaly_score: float
    anomaly_map: Any | None = None
    threshold: float | None = None
    is_anomaly: bool | None = None
    ood_score: float | None = None

    allowed_tasks: ClassVar[frozenset[Task]] = frozenset({Task.ANOMALY_DETECTION})

    def __post_init__(self) -> None:
        super(AnomalyDetectionResult, self).__post_init__()
        object.__setattr__(self, "anomaly_score", _finite(self.anomaly_score, "anomaly_score"))
        if self.threshold is not None:
            object.__setattr__(self, "threshold", _finite(self.threshold, "threshold"))
        if self.ood_score is not None:
            object.__setattr__(self, "ood_score", _finite(self.ood_score, "ood_score"))


@dataclass(frozen=True, slots=True, kw_only=True)
class WSIMILResult(InferenceResult):
    task: Task = field(default=Task.WSI_MIL, init=False)
    slide_scores: Mapping[str, float]
    attention_map: Any | None = None
    key_instances: tuple[Mapping[str, Any], ...] = ()
    embeddings: Any | None = None

    allowed_tasks: ClassVar[frozenset[Task]] = frozenset({Task.WSI_MIL})

    def __post_init__(self) -> None:
        super(WSIMILResult, self).__post_init__()
        scores = _scores(self.slide_scores, "slide_scores")
        if not scores:
            raise ValueError("WSI MIL slide_scores must not be empty")
        object.__setattr__(self, "slide_scores", scores)
        object.__setattr__(self, "key_instances", tuple(dict(item) for item in self.key_instances))


InferenceResultType: TypeAlias = (
    ClassificationResult
    | SegmentationResult
    | DetectionResult
    | RepresentationResult
    | RegistrationResult
    | ReconstructionResult
    | RestorationResult
    | GenerationResult
    | MultimodalResult
    | TrackingResult
    | AnomalyDetectionResult
    | WSIMILResult
    | InferenceResult
)
