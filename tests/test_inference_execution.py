from __future__ import annotations

import builtins
import sys
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from workbench.domain.images import (
    ColorSpace as ImageColorSpace,
)
from workbench.domain.images import (
    IntensitySemantics,
    RasterImage2D,
    SourceType,
)
from workbench.domain.transforms import TransformRecord
from workbench.inference import (
    AlphaHandling,
    AnomalyDetectionResult,
    ClassificationResult,
    ColorSpace,
    CropAnchor,
    DetectionResult,
    DeviceKind,
    GenerationResult,
    InferenceCancelledError,
    InferenceRequest,
    InterpolationMode,
    ModelManifest,
    MultimodalResult,
    NormalizationSpec,
    NumericRange,
    OnnxModelPlugin,
    OrientationHandling,
    PluginContractError,
    PluginLoadContext,
    Preprocessing2DSpec,
    PythonAdapterProxy,
    ReconstructionResult,
    RegistrationResult,
    RepresentationResult,
    RestorationResult,
    SegmentationResult,
    SpatialOperation2D,
    SpatialOperationKind,
    TensorDType,
    TensorLayout,
    TorchScriptModelPlugin,
    TrackingResult,
    VisualizationKind,
    WSIMILResult,
    build_typed_result,
    prepare_input_2d,
)
from workbench.io import RasterImageLoader


def _postprocessing(
    *,
    activation: str = "none",
    threshold: float | None = None,
    interpolation: str | None = None,
    discrete: bool = False,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "activation": activation,
        "threshold": threshold,
        "nms_iou_threshold": None,
        "interpolation": interpolation,
        "discrete_labels": discrete,
        "parameters": parameters or {},
    }


def _output(
    name: str,
    semantic: str,
    shape: list[int | str | None],
    *,
    coordinate_system: str = "not-applicable",
    labels: dict[str, str] | None = None,
    postprocessing: dict[str, Any] | None = None,
    dtype: str = "float32",
) -> dict[str, Any]:
    return {
        "name": name,
        "semantic": semantic,
        "dtype": dtype,
        "shape": shape,
        "coordinate_system": coordinate_system,
        "labels": labels or {},
        "postprocessing": postprocessing or _postprocessing(),
        "uncertainty": {"kind": "none", "description": "", "calibrated": False},
    }


def _manifest_mapping(
    *,
    tasks: list[str],
    outputs: list[dict[str, Any]],
    runtime: str = "onnx",
    entrypoint: str = "model.onnx",
    input_shape: list[int | str | None] | None = None,
    input_size: tuple[int, int] = (8, 8),
    python_executable: str | None = None,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    runtime_mapping: dict[str, Any] = {
        "kind": runtime,
        "device": "cpu",
        "options": {},
    }
    entrypoint_mapping: dict[str, Any] = {"path": entrypoint}
    if runtime == "python-adapter":
        assert python_executable is not None
        runtime_mapping["python"] = {
            "python_executable": python_executable,
            "subprocess": True,
        }
        runtime_mapping["options"] = {"timeout_seconds": timeout_seconds}
        entrypoint_mapping["object"] = "Adapter"
    return {
        "schema_version": "1.0",
        "name": "inference-test-model",
        "version": "1.0.0",
        "family": "test-family",
        "source": {"name": "unit-test", "model_id": "generated-at-runtime"},
        "description": "Generated inference contract fixture.",
        "tasks": tasks,
        "subtasks": [],
        "license": {"code": "MIT", "model": "MIT", "weights": "MIT"},
        "runtime": runtime_mapping,
        "entrypoint": entrypoint_mapping,
        "weights": [
            {
                "name": "local-model",
                "path": entrypoint,
                "format": runtime,
                "required": True,
            }
        ],
        "inputs": [
            {
                "name": "image",
                "semantic": "image",
                "modalities": ["generic-image"],
                "dimensionality": "2d",
                "shape": input_shape or [1, 1, input_size[0], input_size[1]],
                "spacing": {"required": False, "unit": "pixel"},
                "preprocessing": {
                    "layout": "nchw",
                    "color_space": "grayscale",
                    "channel_order": ["Y"],
                    "alpha_handling": "drop",
                    "dtype": "float32",
                    "value_range": [0, 255],
                    "normalization": {"scale": 1.0, "mean": [0], "std": [1]},
                    "spatial": [
                        {
                            "operation": "resize",
                            "size": list(input_size),
                            "interpolation": "nearest",
                        }
                    ],
                    "orientation": "apply-exif",
                },
            }
        ],
        "outputs": outputs,
        "capabilities": {
            "prompts": False,
            "intermediate_features": True,
            "attention": True,
            "embeddings": True,
            "multiscale_outputs": False,
            "vector_fields": True,
            "sampling_trajectory": True,
            "multimodal_text": True,
            "uncertainty": False,
        },
    }


def _preprocessing_with_geometry() -> Preprocessing2DSpec:
    return Preprocessing2DSpec(
        layout=TensorLayout.NCHW,
        color_space=ColorSpace.GRAYSCALE,
        channel_order=("Y",),
        alpha_handling=AlphaHandling.DROP,
        dtype=TensorDType.FLOAT32,
        value_range=NumericRange(0, 255),
        normalization=NormalizationSpec(mean=(0,), std=(1,)),
        spatial=(
            SpatialOperation2D(
                operation=SpatialOperationKind.RESIZE,
                size=(12, 10),
                interpolation=InterpolationMode.NEAREST,
            ),
            SpatialOperation2D(
                operation=SpatialOperationKind.CROP,
                size=(10, 8),
                anchor=CropAnchor.CENTER,
            ),
            SpatialOperation2D(
                operation=SpatialOperationKind.LETTERBOX,
                size=(12, 12),
                interpolation=InterpolationMode.NEAREST,
            ),
        ),
        orientation=OrientationHandling.APPLY_EXIF,
    )


def _oriented_raster(raw: np.ndarray, orientation: int = 6) -> RasterImage2D:
    transform = TransformRecord.from_exif_orientation(orientation, raw.shape)
    canonical = np.empty(transform.output_shape, dtype=raw.dtype)
    for y in range(raw.shape[0]):
        for x in range(raw.shape[1]):
            target_x, target_y = np.rint(transform.forward((x, y))).astype(int)
            canonical[target_y, target_x] = raw[y, x]
    return RasterImage2D(
        array=canonical,
        source_type=SourceType.RASTER,
        intensity_semantics=IntensitySemantics.GRAYSCALE,
        bit_depth=8,
        color_space=ImageColorSpace.GRAYSCALE,
        channel_order=("Y",),
        transform_record=transform,
    )


def test_resize_crop_letterbox_and_exif_transform_round_trip() -> None:
    raw = np.zeros((6, 8), dtype=np.uint8)
    raw[2, 3] = 7
    prepared = prepare_input_2d(_oriented_raster(raw), _preprocessing_with_geometry())

    points = np.asarray([[3.0, 2.0], [4.0, 3.0], [2.5, 2.5]])
    np.testing.assert_allclose(
        prepared.transform_record.inverse(prepared.transform_record.forward(points)),
        points,
        atol=1e-10,
    )
    assert prepared.tensor.shape == (1, 1, 12, 12)
    assert [item.name for item in prepared.transform_record.operations] == [
        "exif_orientation",
        "resize",
        "crop",
        "resize",
        "letterbox_pad",
    ]


def test_masks_boxes_keypoints_and_heatmaps_map_to_original_pixels() -> None:
    raw = np.zeros((6, 8), dtype=np.uint8)
    raw[2, 3] = 2
    prepared = prepare_input_2d(_oriented_raster(raw), _preprocessing_with_geometry())
    transform = prepared.transform_record
    request = InferenceRequest(
        inputs={"image": raw},
        transform_records={"image": transform},
    )

    mask_output = _output(
        "masks",
        "masks",
        [1, 1, 12, 12],
        coordinate_system="model-input-pixel",
        labels={"0": "background", "2": "finding"},
        postprocessing=_postprocessing(
            interpolation="nearest",
            discrete=True,
            parameters={"layout": "nchw"},
        ),
    )
    segmentation_manifest = ModelManifest.from_mapping(
        _manifest_mapping(tasks=["segmentation"], outputs=[mask_output], input_size=(12, 12))
    )
    segmentation = build_typed_result(
        segmentation_manifest,
        {"masks": prepared.tensor.astype(np.uint8)},
        request,
        runtime=segmentation_manifest.runtime.kind,
        duration_ms=1.0,
        device="cpu",
    )
    assert isinstance(segmentation, SegmentationResult)
    assert segmentation.masks.shape == raw.shape
    assert segmentation.masks[2, 3] == 2
    assert set(np.unique(segmentation.masks)) <= {0, 2}

    source_box = np.asarray([[2.0, 1.0, 5.0, 4.0]])
    model_box = transform.forward_boxes(source_box)[0]
    source_point = np.asarray([[3.0, 2.0]])
    model_point = transform.forward(source_point)[0]
    detection_manifest = ModelManifest.from_mapping(
        _manifest_mapping(
            tasks=["detection"],
            input_size=(12, 12),
            outputs=[
                _output(
                    "boxes",
                    "boxes",
                    [1, 6],
                    coordinate_system="model-input-pixel",
                    labels={"1": "finding"},
                    postprocessing=_postprocessing(parameters={"box_format": "xyxy"}),
                ),
                _output(
                    "keypoints",
                    "keypoints",
                    [1, 4],
                    coordinate_system="model-input-pixel",
                    labels={"1": "centre"},
                ),
            ],
        )
    )
    detection = build_typed_result(
        detection_manifest,
        {
            "boxes": np.asarray([[*model_box, 0.9, 1.0]], dtype=np.float32),
            "keypoints": np.asarray([[*model_point, 0.8, 1.0]], dtype=np.float32),
        },
        request,
        runtime=detection_manifest.runtime.kind,
        duration_ms=1.0,
        device="cpu",
    )
    assert isinstance(detection, DetectionResult)
    np.testing.assert_allclose(detection.boxes[0].coordinates, source_box[0], atol=1e-5)
    np.testing.assert_allclose(detection.keypoints[0].coordinates, source_point[0], atol=1e-5)
    assert detection.boxes[0].coordinate_system.value == "source-pixel"
    assert detection.keypoints[0].coordinate_system.value == "source-pixel"
    artifacts = OnnxModelPlugin(detection_manifest, Path(".")).visualize(detection)
    assert {artifact.kind for artifact in artifacts} == {
        VisualizationKind.BOX_OVERLAY,
        VisualizationKind.KEYPOINT_OVERLAY,
    }

    representation_manifest = ModelManifest.from_mapping(
        _manifest_mapping(
            tasks=["representation"],
            input_size=(12, 12),
            outputs=[
                _output("embedding", "embeddings", [1, 2]),
                _output(
                    "attention",
                    "attention-maps",
                    [1, 1, 12, 12],
                    coordinate_system="feature-grid",
                    postprocessing=_postprocessing(
                        interpolation="bilinear",
                        parameters={"layout": "nchw"},
                    ),
                ),
            ],
        )
    )
    representation = build_typed_result(
        representation_manifest,
        {
            "embedding": np.asarray([[1.0, 2.0]], dtype=np.float32),
            "attention": prepared.tensor,
        },
        request,
        runtime=representation_manifest.runtime.kind,
        duration_ms=1.0,
        device="cpu",
    )
    assert isinstance(representation, RepresentationResult)
    heatmap = representation.attention_maps["attention"]
    assert heatmap.shape == (1, 1, *raw.shape)
    assert np.unravel_index(np.argmax(heatmap[0, 0]), raw.shape) == (2, 3)


def test_loader_exif_transform_composes_with_preprocessing_to_raw_pixels(
    tmp_path: Path,
) -> None:
    raw = np.zeros((6, 8), dtype=np.uint8)
    raw[2, 3] = 2
    path = tmp_path / "oriented.png"
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(raw).save(path, exif=exif)
    loaded = RasterImageLoader().load(path)
    assert isinstance(loaded, RasterImage2D)
    prepared = prepare_input_2d(loaded, _preprocessing_with_geometry())
    manifest = ModelManifest.from_mapping(
        _manifest_mapping(
            tasks=["segmentation"],
            input_size=(12, 12),
            outputs=[
                _output(
                    "masks",
                    "masks",
                    [1, 1, 12, 12],
                    coordinate_system="model-input-pixel",
                    postprocessing=_postprocessing(
                        interpolation="nearest",
                        discrete=True,
                        parameters={"layout": "nchw"},
                    ),
                    dtype="uint8",
                )
            ],
        )
    )
    result = build_typed_result(
        manifest,
        {"masks": prepared.tensor.astype(np.uint8)},
        InferenceRequest(
            inputs={"image": loaded},
            transform_records={"image": prepared.transform_record},
        ),
        runtime=manifest.runtime.kind,
        duration_ms=1.0,
        device="cpu",
    )

    assert result.masks.shape == raw.shape
    assert result.masks[2, 3] == 2
    assert np.count_nonzero(result.masks == 2) == 1


def _write_tiny_onnx(path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    weights = np.zeros((16, 2), dtype=np.float32)
    weights[:, 1] = 0.05
    graph = helper.make_graph(
        [
            helper.make_node("Flatten", ["image"], ["flat"], axis=1),
            helper.make_node("Gemm", ["flat", "weights", "bias"], ["scores"]),
        ],
        "tiny-runtime-test",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 1, 4, 4])],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 2])],
        [
            numpy_helper.from_array(weights, name="weights"),
            numpy_helper.from_array(np.zeros(2, dtype=np.float32), name="bias"),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="openmedvisionx-tests",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _classification_output() -> dict[str, Any]:
    return _output(
        "scores",
        "class-scores",
        [1, 2],
        labels={"0": "negative", "1": "positive"},
        postprocessing=_postprocessing(activation="softmax"),
    )


def test_tiny_onnx_runs_and_input_shape_mismatch_is_clear(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    model_path = tmp_path / "model.onnx"
    _write_tiny_onnx(model_path)
    mapping = _manifest_mapping(
        tasks=["classification"],
        outputs=[_classification_output()],
        input_size=(4, 4),
    )
    manifest = ModelManifest.from_mapping(mapping)
    plugin = OnnxModelPlugin(manifest, tmp_path)
    plugin.load(PluginLoadContext(plugin_root=tmp_path, device=DeviceKind.CPU))
    try:
        result = plugin.predict(InferenceRequest(inputs={"image": np.ones((4, 4), dtype=np.uint8)}))
        assert isinstance(result, ClassificationResult)
        assert result.predicted_label == "positive"

        bad_mapping = deepcopy(mapping)
        bad_mapping["inputs"][0]["shape"] = [1, 1, 5, 5]
        bad_plugin = OnnxModelPlugin(ModelManifest.from_mapping(bad_mapping), tmp_path)
        bad_plugin.load(PluginLoadContext(plugin_root=tmp_path, device=DeviceKind.CPU))
        try:
            with pytest.raises(PluginContractError, match="axis 2.*expected 5"):
                bad_plugin.predict(
                    InferenceRequest(inputs={"image": np.ones((4, 4), dtype=np.uint8)})
                )
        finally:
            bad_plugin.close()
    finally:
        plugin.close()


def test_onnx_missing_dependency_and_unavailable_cuda_are_clear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "model.onnx").write_bytes(b"generated test placeholder")
    manifest = ModelManifest.from_mapping(
        _manifest_mapping(
            tasks=["classification"],
            outputs=[_classification_output()],
            input_size=(4, 4),
        )
    )
    real_import = builtins.__import__

    def missing_ort(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "onnxruntime":
            raise ImportError("simulated missing optional dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_ort)
    with pytest.raises(PluginContractError, match="ONNX Runtime is not installed"):
        OnnxModelPlugin(manifest, tmp_path).load(PluginLoadContext(plugin_root=tmp_path))
    monkeypatch.setattr(builtins, "__import__", real_import)

    fake_ort = SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    with pytest.raises(PluginContractError, match="CUDA.*unavailable"):
        OnnxModelPlugin(manifest, tmp_path).load(
            PluginLoadContext(plugin_root=tmp_path, device=DeviceKind.CUDA)
        )


def test_tiny_torchscript_runtime_when_pytorch_is_available(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def forward(self, image):
            flat = image.flatten(1)
            positive = flat.mean(1, keepdim=True)
            return torch.cat((-positive, positive), dim=1)

    path = tmp_path / "model.pt"
    torch.jit.trace(TinyModel(), torch.zeros((1, 1, 4, 4))).save(str(path))
    manifest = ModelManifest.from_mapping(
        _manifest_mapping(
            tasks=["classification"],
            outputs=[_classification_output()],
            runtime="torchscript",
            entrypoint="model.pt",
            input_size=(4, 4),
        )
    )
    plugin = TorchScriptModelPlugin(manifest, tmp_path)
    plugin.load(PluginLoadContext(plugin_root=tmp_path, device=DeviceKind.CPU))
    try:
        result = plugin.predict(InferenceRequest(inputs={"image": np.ones((4, 4), dtype=np.uint8)}))
        assert isinstance(result, ClassificationResult)
        assert result.predicted_label == "positive"
    finally:
        plugin.close()


def _write_adapter(path: Path, body: str) -> None:
    path.write_text("import numpy as np\n" + body, encoding="utf-8")


def _python_manifest(
    tmp_path: Path,
    *,
    tasks: list[str],
    outputs: list[dict[str, Any]],
    timeout_seconds: float = 5.0,
) -> ModelManifest:
    return ModelManifest.from_mapping(
        _manifest_mapping(
            tasks=tasks,
            outputs=outputs,
            runtime="python-adapter",
            entrypoint="adapter.py",
            input_size=(8, 8),
            python_executable=sys.executable,
            timeout_seconds=timeout_seconds,
        )
    )


def _load_python_proxy(manifest: ModelManifest, root: Path) -> PythonAdapterProxy:
    proxy = PythonAdapterProxy(manifest, root)
    proxy.load(
        PluginLoadContext(
            plugin_root=root,
            device=DeviceKind.CPU,
            user_consented_python_code=True,
            subprocess_isolated=True,
        )
    )
    return proxy


def test_mock_python_adapter_returns_all_major_typed_results(tmp_path: Path) -> None:
    outputs = [
        _classification_output(),
        _output(
            "masks",
            "masks",
            [1, 1, 8, 8],
            coordinate_system="model-input-pixel",
            postprocessing=_postprocessing(
                interpolation="nearest",
                discrete=True,
                parameters={"layout": "nchw"},
            ),
            dtype="uint8",
        ),
        _output(
            "boxes",
            "boxes",
            [1, 6],
            coordinate_system="model-input-pixel",
            labels={"1": "finding"},
            postprocessing=_postprocessing(parameters={"box_format": "xyxy"}),
        ),
        _output(
            "keypoints",
            "keypoints",
            [1, 4],
            coordinate_system="model-input-pixel",
        ),
        _output("embedding", "embeddings", [1, 4]),
        _output(
            "attention",
            "attention-maps",
            [1, 1, 8, 8],
            coordinate_system="feature-grid",
            postprocessing=_postprocessing(interpolation="bilinear", parameters={"layout": "nchw"}),
        ),
        _output(
            "vector",
            "vector-fields",
            [1, 2, 8, 8],
            coordinate_system="model-input-pixel",
        ),
        _output(
            "reconstructed",
            "reconstructed-images",
            [1, 1, 8, 8],
            coordinate_system="model-input-pixel",
            postprocessing=_postprocessing(interpolation="bilinear", parameters={"layout": "nchw"}),
        ),
        _output(
            "restored",
            "restored-images",
            [1, 1, 8, 8],
            coordinate_system="model-input-pixel",
            postprocessing=_postprocessing(interpolation="bilinear", parameters={"layout": "nchw"}),
        ),
        _output("generated", "generated-samples", [2, 1, 8, 8]),
        _output("trajectory", "sampling-trajectories", [2, 1, 8, 8]),
        _output("text", "text", [1]),
        _output(
            "similarity",
            "similarity-scores",
            [1, 2],
            labels={"0": "left", "1": "right"},
        ),
        _output(
            "tracks",
            "tracks",
            [1],
            coordinate_system="model-input-pixel",
            postprocessing=_postprocessing(parameters={"box_format": "xyxy"}),
        ),
        _output("anomaly_score", "anomaly-scores", [1]),
        _output(
            "anomaly_map",
            "anomaly-maps",
            [1, 1, 8, 8],
            coordinate_system="model-input-pixel",
            postprocessing=_postprocessing(interpolation="bilinear", parameters={"layout": "nchw"}),
        ),
    ]
    tasks = [
        "classification",
        "segmentation",
        "detection",
        "representation",
        "registration",
        "reconstruction",
        "restoration",
        "generation",
        "vqa",
        "tracking",
        "anomaly_detection",
        "wsi_mil",
    ]
    _write_adapter(
        tmp_path / "adapter.py",
        """
class Adapter:
    def predict(self, request):
        image = np.asarray(request["inputs"]["image"], dtype=np.float32)
        field = np.zeros((1, 2, 8, 8), dtype=np.float32)
        heatmap = np.zeros((1, 1, 8, 8), dtype=np.float32)
        heatmap[0, 0, 3, 3] = 1.0
        mask = (image > 0).astype(np.uint8)
        return {
            "scores": np.asarray([[0.2, 0.8]], dtype=np.float32),
            "masks": mask,
            "boxes": np.asarray([[1, 1, 5, 5, 0.9, 1]], dtype=np.float32),
            "keypoints": np.asarray([[3, 3, 0.8, 1]], dtype=np.float32),
            "embedding": np.asarray([[1, 2, 3, 4]], dtype=np.float32),
            "attention": heatmap,
            "vector": field,
            "reconstructed": image,
            "restored": image,
            "generated": np.concatenate((image, image), axis=0),
            "trajectory": np.concatenate((image, image), axis=0),
            "text": ["teaching response"],
            "similarity": np.asarray([[0.25, 0.75]], dtype=np.float32),
            "tracks": [{
                "track_id": "lesion",
                "observations": [{
                    "frame_index": 0,
                    "box": [1, 1, 5, 5, 0.9, 1],
                    "keypoints": [[3, 3, 0.8, 1]],
                }],
            }],
            "anomaly_score": np.asarray([0.7], dtype=np.float32),
            "anomaly_map": heatmap,
        }
""",
    )
    manifest = _python_manifest(tmp_path, tasks=tasks, outputs=outputs)
    proxy = _load_python_proxy(manifest, tmp_path)
    expected_types = {
        "classification": ClassificationResult,
        "segmentation": SegmentationResult,
        "detection": DetectionResult,
        "representation": RepresentationResult,
        "registration": RegistrationResult,
        "reconstruction": ReconstructionResult,
        "restoration": RestorationResult,
        "generation": GenerationResult,
        "vqa": MultimodalResult,
        "tracking": TrackingResult,
        "anomaly_detection": AnomalyDetectionResult,
        "wsi_mil": WSIMILResult,
    }
    results: dict[str, Any] = {}
    try:
        for task, expected_type in expected_types.items():
            result = proxy.predict(
                InferenceRequest(
                    inputs={"image": np.ones((8, 8), dtype=np.uint8)},
                    parameters={"task": task},
                )
            )
            assert isinstance(result, expected_type)
            results[task] = result
        detection_artifacts = proxy.visualize(results["detection"])
        assert {artifact.kind for artifact in detection_artifacts} == {
            VisualizationKind.BOX_OVERLAY,
            VisualizationKind.KEYPOINT_OVERLAY,
            VisualizationKind.MASK_OVERLAY,
        }
        assert {artifact.coordinate_system for artifact in detection_artifacts} == {"source-pixel"}
        reconstruction = proxy.predict(
            InferenceRequest(
                inputs={"image": np.ones((8, 8), dtype=np.uint8)},
                parameters={"task": "reconstruction"},
            )
        )
        generation = proxy.predict(
            InferenceRequest(
                inputs={"image": np.ones((8, 8), dtype=np.uint8)},
                parameters={"task": "generation"},
            )
        )
        assert len(reconstruction.sampling_trajectory) == 2
        assert len(generation.sampling_trajectory) == 2
        assert VisualizationKind.TRAJECTORY in {
            artifact.kind for artifact in proxy.visualize(reconstruction)
        }
        assert VisualizationKind.TRAJECTORY in {
            artifact.kind for artifact in proxy.visualize(generation)
        }
        assert VisualizationKind.VECTOR_FIELD in {
            artifact.kind for artifact in proxy.visualize(results["registration"])
        }
        assert VisualizationKind.VECTOR_FIELD in {
            artifact.kind for artifact in proxy.visualize(results["tracking"])
        }
    finally:
        proxy.close()


def test_python_adapter_crash_and_cancellation_are_contained(tmp_path: Path) -> None:
    output = [_classification_output()]
    _write_adapter(
        tmp_path / "adapter.py",
        """
import os
class Adapter:
    def predict(self, request):
        os._exit(17)
""",
    )
    crash_proxy = _load_python_proxy(
        _python_manifest(tmp_path, tasks=["classification"], outputs=output), tmp_path
    )
    with pytest.raises(PluginContractError, match="exited|crashed"):
        crash_proxy.predict(InferenceRequest(inputs={"image": np.ones((8, 8), np.uint8)}))
    crash_proxy.close()

    _write_adapter(
        tmp_path / "adapter.py",
        """
import time
from pathlib import Path
class Adapter:
    def predict(self, request):
        Path("started.flag").write_text("started", encoding="utf-8")
        time.sleep(30)
        return {"scores": np.asarray([[0.5, 0.5]], dtype=np.float32)}
""",
    )
    cancel_proxy = _load_python_proxy(
        _python_manifest(tmp_path, tasks=["classification"], outputs=output), tmp_path
    )
    captured: list[BaseException] = []

    def predict() -> None:
        try:
            cancel_proxy.predict(
                InferenceRequest(inputs={"image": np.ones((8, 8), dtype=np.uint8)})
            )
        except BaseException as exc:  # test captures the worker-thread result
            captured.append(exc)

    thread = Thread(target=predict)
    thread.start()
    deadline = time.monotonic() + 3
    while not (tmp_path / "started.flag").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert (tmp_path / "started.flag").exists()
    assert cancel_proxy.cancel()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert len(captured) == 1
    assert isinstance(captured[0], InferenceCancelledError)
    cancel_proxy.close()


def test_pre_cancelled_request_never_runs_a_model(tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"placeholder")
    manifest = ModelManifest.from_mapping(
        _manifest_mapping(
            tasks=["classification"],
            outputs=[_classification_output()],
            input_size=(4, 4),
        )
    )
    plugin = OnnxModelPlugin(manifest, tmp_path)
    plugin._loaded = True  # exercise the public pre-run cancellation boundary without ORT
    plugin._session = SimpleNamespace()
    with pytest.raises(InferenceCancelledError):
        plugin.predict(
            InferenceRequest(
                inputs={"image": np.ones((4, 4), dtype=np.uint8)},
                cancellation_check=lambda: True,
            )
        )
