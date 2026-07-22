"""Dataset evaluation and reproducible experiment contracts."""

from .advanced_metrics import (
    DetectionMetrics,
    FrocPoint,
    RegistrationMetrics,
    box_iou_matrix,
    compute_detection_metrics,
    compute_registration_metrics,
)
from .brats_records import create_brats_experiment_record
from .contracts import (
    DatasetManifest,
    DatasetSample,
    FrozenJson,
    JsonScalar,
    SplitValidationReport,
    dataset_manifest_from_mapping,
    load_dataset_manifest,
    thaw_json,
    validate_group_splits,
)
from .metrics import (
    BinaryClassificationMetrics,
    CalibrationBin,
    ConfidenceInterval,
    ConfusionMatrix,
    SegmentationMetrics,
    ThresholdPoint,
    bootstrap_classification_interval,
    compute_binary_classification_metrics,
    compute_segmentation_metrics,
    threshold_sweep,
)
from .records import (
    ArtifactReference,
    ExperimentRecord,
    TransformStep,
    export_experiment_record,
)

__all__ = [
    "ArtifactReference",
    "BinaryClassificationMetrics",
    "CalibrationBin",
    "ConfidenceInterval",
    "ConfusionMatrix",
    "DatasetManifest",
    "DatasetSample",
    "DetectionMetrics",
    "ExperimentRecord",
    "FrozenJson",
    "FrocPoint",
    "JsonScalar",
    "RegistrationMetrics",
    "SegmentationMetrics",
    "SplitValidationReport",
    "ThresholdPoint",
    "TransformStep",
    "bootstrap_classification_interval",
    "box_iou_matrix",
    "compute_binary_classification_metrics",
    "create_brats_experiment_record",
    "compute_detection_metrics",
    "compute_registration_metrics",
    "compute_segmentation_metrics",
    "dataset_manifest_from_mapping",
    "export_experiment_record",
    "load_dataset_manifest",
    "thaw_json",
    "threshold_sweep",
    "validate_group_splits",
]
