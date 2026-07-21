from __future__ import annotations

import importlib.util
import unittest
from copy import deepcopy
from pathlib import Path

from dicom_viewer.inference import (
    AnomalyDetectionResult,
    BoundingBox,
    BoxFormat,
    ClassificationResult,
    CoordinateSystem,
    DetectionResult,
    GenerationResult,
    InferenceProvenance,
    InferenceRequest,
    ManifestBackedModelPlugin,
    ManifestDependencyError,
    ManifestValidationError,
    ModelManifest,
    MultimodalResult,
    PluginLoadContext,
    ReconstructionResult,
    RegistrationResult,
    RepresentationResult,
    RestorationResult,
    RuntimeKind,
    SegmentationResult,
    Task,
    Track,
    TrackingResult,
    TrackObservation,
    UntrustedPluginError,
    WSIMILResult,
    load_manifest,
    require_python_adapter_consent,
    validate_manifest_references,
)


def valid_manifest_mapping() -> dict:
    return {
        "schema_version": "1.0",
        "name": "contract-test-model",
        "version": "1.0.0",
        "family": "test-family",
        "source": {"name": "unit-test", "model_id": "test-family-classifier"},
        "description": "A runtime-free contract fixture.",
        "tasks": ["classification"],
        "subtasks": ["binary-classification"],
        "license": {"code": "MIT", "model": "MIT", "weights": "MIT"},
        "runtime": {"kind": "onnx", "device": "cpu", "options": {}},
        "entrypoint": {"path": "models/model.onnx"},
        "weights": [
            {
                "name": "model",
                "path": "models/model.onnx",
                "format": "onnx",
                "required": True,
            }
        ],
        "inputs": [
            {
                "name": "image",
                "semantic": "image",
                "modalities": ["generic-image"],
                "dimensionality": "2d",
                "shape": [1, 3, 32, 32],
                "spacing": {"required": False, "unit": "pixel"},
                "preprocessing": {
                    "layout": "nchw",
                    "color_space": "rgb",
                    "channel_order": ["R", "G", "B"],
                    "alpha_handling": "drop",
                    "dtype": "float32",
                    "value_range": [0, 255],
                    "normalization": {
                        "scale": 0.00392156862745098,
                        "mean": [0, 0, 0],
                        "std": [1, 1, 1],
                    },
                    "spatial": [
                        {
                            "operation": "resize",
                            "size": [32, 32],
                            "interpolation": "bilinear",
                        }
                    ],
                    "orientation": "apply-exif",
                },
            }
        ],
        "outputs": [
            {
                "name": "scores",
                "semantic": "class-scores",
                "dtype": "float32",
                "shape": [1, 2],
                "coordinate_system": "not-applicable",
                "labels": {"0": "negative", "1": "positive"},
                "postprocessing": {
                    "activation": "softmax",
                    "threshold": None,
                    "nms_iou_threshold": None,
                    "interpolation": None,
                    "discrete_labels": False,
                },
                "uncertainty": {
                    "kind": "none",
                    "description": "",
                    "calibrated": False,
                },
            }
        ],
        "capabilities": {
            "prompts": False,
            "intermediate_features": False,
            "attention": False,
            "embeddings": False,
            "multiscale_outputs": False,
            "vector_fields": False,
            "sampling_trajectory": False,
            "multimodal_text": False,
            "uncertainty": False,
        },
    }


class ContractTests(unittest.TestCase):
    def test_valid_manifest_round_trip(self) -> None:
        manifest = ModelManifest.from_mapping(valid_manifest_mapping())
        self.assertEqual(manifest.tasks, (Task.CLASSIFICATION,))
        self.assertEqual(manifest.inputs[0].preprocessing.layout.value, "nchw")
        self.assertEqual(
            ModelManifest.from_mapping(manifest.to_mapping()).to_mapping(),
            manifest.to_mapping(),
        )

    def test_remote_weight_is_rejected(self) -> None:
        mapping = deepcopy(valid_manifest_mapping())
        mapping["weights"][0]["path"] = "https://example.invalid/model.onnx"
        with self.assertRaises(ManifestValidationError):
            ModelManifest.from_mapping(mapping)

    def test_two_dimensional_image_requires_preprocessing(self) -> None:
        mapping = deepcopy(valid_manifest_mapping())
        del mapping["inputs"][0]["preprocessing"]
        with self.assertRaises(ManifestValidationError):
            ModelManifest.from_mapping(mapping)

    def test_entrypoint_cannot_escape_plugin_root(self) -> None:
        mapping = deepcopy(valid_manifest_mapping())
        mapping["entrypoint"]["path"] = "../model.onnx"
        manifest = ModelManifest.from_mapping(mapping)
        with self.assertRaises(ManifestValidationError):
            validate_manifest_references(manifest, Path("plugin"))

    def test_python_adapter_requires_explicit_consent_and_subprocess(self) -> None:
        mapping = deepcopy(valid_manifest_mapping())
        mapping["runtime"] = {
            "kind": "python-adapter",
            "device": "cpu",
            "python": {"conda_environment": "user-model-env", "subprocess": True},
        }
        mapping["entrypoint"] = {"path": "adapter.py", "object": "Plugin"}
        manifest = ModelManifest.from_mapping(mapping)
        with self.assertRaises(UntrustedPluginError):
            require_python_adapter_consent(
                manifest,
                user_consented=False,
                subprocess_isolated=True,
            )
        with self.assertRaises(UntrustedPluginError):
            require_python_adapter_consent(
                manifest,
                user_consented=True,
                subprocess_isolated=False,
            )

    def test_typed_result_and_plugin_boundary(self) -> None:
        manifest = ModelManifest.from_mapping(valid_manifest_mapping())
        provenance = InferenceProvenance(
            model_name=manifest.name,
            model_version=manifest.version,
            runtime=RuntimeKind.ONNX,
        )
        result = ClassificationResult(
            provenance=provenance,
            scores={"negative": 0.2, "positive": 0.8},
            predicted_label="positive",
        )

        class MockPlugin(ManifestBackedModelPlugin):
            def load(self, context: PluginLoadContext) -> None:
                self.authorize_load(context)

            def predict(self, request: InferenceRequest) -> ClassificationResult:
                self.validate_result(result)
                return result

            def visualize(self, result, context=None):
                return ()

        plugin = MockPlugin(manifest)
        plugin.load(PluginLoadContext(plugin_root=Path(".")))
        predicted = plugin.predict(InferenceRequest(inputs={"image": object()}))
        self.assertIs(predicted, result)

    def test_all_public_result_families_are_runtime_agnostic(self) -> None:
        provenance = InferenceProvenance(
            model_name="contract-test-model",
            model_version="1.0.0",
            runtime=RuntimeKind.ONNX,
        )
        box = BoundingBox(
            coordinates=(1.0, 2.0, 8.0, 9.0),
            format=BoxFormat.XYXY,
            coordinate_system=CoordinateSystem.SOURCE_PIXEL,
            label="finding",
            score=0.9,
        )
        observation = TrackObservation(frame_index=0, box=box)
        results = (
            SegmentationResult(provenance=provenance, masks=object()),
            DetectionResult(provenance=provenance, boxes=(box,)),
            RepresentationResult(provenance=provenance, embeddings=object()),
            RegistrationResult(provenance=provenance, vector_field=object()),
            ReconstructionResult(provenance=provenance, reconstructed_image=object()),
            RestorationResult(provenance=provenance, restored_image=object()),
            GenerationResult(provenance=provenance, samples=(object(),)),
            MultimodalResult(
                task=Task.VQA,
                provenance=provenance,
                text_output="contract response",
            ),
            TrackingResult(
                provenance=provenance,
                tracks=(Track(track_id="one", observations=(observation,)),),
            ),
            AnomalyDetectionResult(provenance=provenance, anomaly_score=0.4),
            WSIMILResult(provenance=provenance, slide_scores={"negative": 0.6}),
        )
        self.assertEqual(
            {result.task for result in results},
            {
                Task.SEGMENTATION,
                Task.DETECTION,
                Task.REPRESENTATION,
                Task.REGISTRATION,
                Task.RECONSTRUCTION,
                Task.RESTORATION,
                Task.GENERATION,
                Task.VQA,
                Task.TRACKING,
                Task.ANOMALY_DETECTION,
                Task.WSI_MIL,
            },
        )

    def test_yaml_example_or_clear_optional_dependency_error(self) -> None:
        example = Path(__file__).parents[1] / "examples" / "manifest.yaml"
        if importlib.util.find_spec("yaml") is None:
            with self.assertRaises(ManifestDependencyError):
                load_manifest(example)
        else:
            self.assertEqual(load_manifest(example).name, "example-local-classifier")


if __name__ == "__main__":
    unittest.main()
