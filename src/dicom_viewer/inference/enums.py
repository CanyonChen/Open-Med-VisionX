"""Controlled string enumerations used by inference manifests and results."""

from __future__ import annotations

from enum import Enum
from typing import TypeVar

_EnumT = TypeVar("_EnumT", bound="ControlledStringEnum")


class ControlledStringEnum(str, Enum):
    """A JSON/YAML-friendly enum with a consistent coercion helper."""

    def __str__(self) -> str:
        return self.value

    @classmethod
    def coerce(cls: type[_EnumT], value: object) -> _EnumT:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ValueError(f"expected a string, got {type(value).__name__}")
        normalized = value.strip().lower()
        aliases = {
            normalized,
            normalized.replace("_", "-"),
            normalized.replace("-", "_"),
        }
        for member in cls:
            if member.value in aliases:
                return member
        allowed = ", ".join(member.value for member in cls)
        raise ValueError(f"unsupported value {value!r}; expected one of: {allowed}")


class Task(ControlledStringEnum):
    """Tasks accepted by the stable model-plugin protocol."""

    REPRESENTATION = "representation"
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    REGISTRATION = "registration"
    RECONSTRUCTION = "reconstruction"
    RESTORATION = "restoration"
    GENERATION = "generation"
    RETRIEVAL = "retrieval"
    VQA = "vqa"
    REPORT_GENERATION = "report_generation"
    WSI_MIL = "wsi_mil"
    TRACKING = "tracking"
    ANOMALY_DETECTION = "anomaly_detection"
    MULTIMODAL = "multimodal"


class RuntimeKind(ControlledStringEnum):
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    PYTHON_ADAPTER = "python-adapter"


class DeviceKind(ControlledStringEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    DIRECTML = "directml"


class Modality(ControlledStringEnum):
    GENERIC_IMAGE = "generic-image"
    CT = "ct"
    MR = "mr"
    XRAY = "xray"
    MAMMOGRAPHY = "mammography"
    ULTRASOUND = "ultrasound"
    PET = "pet"
    SPECT = "spect"
    PATHOLOGY = "pathology"
    MICROSCOPY = "microscopy"
    ENDOSCOPY = "endoscopy"
    FUNDUS = "fundus"
    DERMOSCOPY = "dermoscopy"
    VIDEO = "video"
    TEXT = "text"
    TABULAR = "tabular"
    MULTIMODAL = "multimodal"


class Dimensionality(ControlledStringEnum):
    TWO_D = "2d"
    TWO_POINT_FIVE_D = "2.5d"
    THREE_D = "3d"
    FOUR_D = "4d"


class InputSemantic(ControlledStringEnum):
    IMAGE = "image"
    IMAGE_SEQUENCE = "image-sequence"
    VOLUME = "volume"
    MASK = "mask"
    POINT_PROMPTS = "point-prompts"
    BOX_PROMPTS = "box-prompts"
    TEXT = "text"
    FEATURES = "features"
    SINOGRAM = "sinogram"
    KSPACE = "kspace"
    TABULAR = "tabular"


class OutputSemantic(ControlledStringEnum):
    CLASS_SCORES = "class-scores"
    MASKS = "masks"
    BOXES = "boxes"
    KEYPOINTS = "keypoints"
    EMBEDDINGS = "embeddings"
    FEATURE_MAPS = "feature-maps"
    ATTENTION_MAPS = "attention-maps"
    VECTOR_FIELDS = "vector-fields"
    AFFINE_TRANSFORMS = "affine-transforms"
    RECONSTRUCTED_IMAGES = "reconstructed-images"
    RESTORED_IMAGES = "restored-images"
    GENERATED_SAMPLES = "generated-samples"
    SAMPLING_TRAJECTORIES = "sampling-trajectories"
    ANOMALY_SCORES = "anomaly-scores"
    ANOMALY_MAPS = "anomaly-maps"
    TRACKS = "tracks"
    TEXT = "text"
    SIMILARITY_SCORES = "similarity-scores"
    UNCERTAINTY = "uncertainty"


class TensorLayout(ControlledStringEnum):
    HW = "hw"
    HWC = "hwc"
    CHW = "chw"
    NHWC = "nhwc"
    NCHW = "nchw"
    DHW = "dhw"
    DHWC = "dhwc"
    CDHW = "cdhw"
    NDHWC = "ndhwc"
    NCDHW = "ncdhw"
    THWC = "thwc"
    NTHWC = "nthwc"
    NTCHW = "ntchw"


class TensorDType(ControlledStringEnum):
    UINT8 = "uint8"
    UINT16 = "uint16"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    BOOL = "bool"


class ColorSpace(ControlledStringEnum):
    GRAYSCALE = "grayscale"
    RGB = "rgb"
    BGR = "bgr"
    RGBA = "rgba"
    BGRA = "bgra"
    HSV = "hsv"
    LAB = "lab"
    YCBCR = "ycbcr"
    NATIVE = "native"


class AlphaHandling(ControlledStringEnum):
    REJECT = "reject"
    DROP = "drop"
    PRESERVE = "preserve"
    COMPOSITE_BLACK = "composite-black"
    COMPOSITE_WHITE = "composite-white"
    PREMULTIPLY = "premultiply"


class OrientationHandling(ControlledStringEnum):
    APPLY_EXIF = "apply-exif"
    REQUIRE_CANONICAL = "require-canonical"
    IGNORE = "ignore"


class SpatialOperationKind(ControlledStringEnum):
    NONE = "none"
    RESIZE = "resize"
    CENTER_CROP = "center-crop"
    CROP = "crop"
    LETTERBOX = "letterbox"
    FIT_SHORTER_SIDE = "fit-shorter-side"
    FIT_LONGER_SIDE = "fit-longer-side"


class InterpolationMode(ControlledStringEnum):
    NEAREST = "nearest"
    LINEAR = "linear"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    AREA = "area"
    LANCZOS = "lanczos"


class CropAnchor(ControlledStringEnum):
    CENTER = "center"
    TOP_LEFT = "top-left"
    TOP_RIGHT = "top-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_RIGHT = "bottom-right"


class CoordinateSystem(ControlledStringEnum):
    NOT_APPLICABLE = "not-applicable"
    SOURCE_PIXEL = "source-pixel"
    MODEL_INPUT_PIXEL = "model-input-pixel"
    NORMALIZED = "normalized"
    PATIENT_RAS = "patient-ras"
    PATIENT_LPS = "patient-lps"
    WORLD = "world"
    FEATURE_GRID = "feature-grid"
    DETECTOR = "detector"


class BoxFormat(ControlledStringEnum):
    XYXY = "xyxy"
    XYWH = "xywh"
    CXCYWH = "cxcywh"
    XYZXYZ = "xyzxyz"


class ActivationKind(ControlledStringEnum):
    NONE = "none"
    SOFTMAX = "softmax"
    SIGMOID = "sigmoid"


class UncertaintyKind(ControlledStringEnum):
    NONE = "none"
    SCORE = "score"
    PROBABILITY = "probability"
    ENTROPY = "entropy"
    VARIANCE = "variance"
    CONFIDENCE_INTERVAL = "confidence-interval"
    EVIDENTIAL = "evidential"
    ENSEMBLE = "ensemble"


class ValidationSeverity(ControlledStringEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class VisualizationKind(ControlledStringEnum):
    IMAGE = "image"
    MASK_OVERLAY = "mask-overlay"
    BOX_OVERLAY = "box-overlay"
    KEYPOINT_OVERLAY = "keypoint-overlay"
    HEATMAP = "heatmap"
    VECTOR_FIELD = "vector-field"
    TRAJECTORY = "trajectory"
    TEXT = "text"
    TABLE = "table"
    PLOT = "plot"
