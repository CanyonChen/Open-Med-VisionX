from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from dicom_viewer.inference import (
    CoordinateSystem,
    InferenceRequest,
    ModelManifest,
    OnnxModelPlugin,
    PluginContractError,
    RuntimeKind,
    SegmentationResult,
    build_typed_result,
    prepare_input_2d,
)
from dicom_viewer.inference.execution import validate_tensor_dtype
from dicom_viewer.inference.standard_runtimes import _prepared_inputs, _visualizations

from .test_contracts import valid_manifest_mapping


def _output(
    *,
    name: str,
    semantic: str,
    dtype: str,
    shape: list[int | str | None],
    coordinate_system: str,
    discrete_labels: bool,
    interpolation: str | None,
    activation: str = "none",
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "semantic": semantic,
        "dtype": dtype,
        "shape": shape,
        "coordinate_system": coordinate_system,
        "labels": {"0": "background", "1": "finding", "2": "other"},
        "postprocessing": {
            "activation": activation,
            "threshold": None,
            "nms_iou_threshold": None,
            "interpolation": interpolation,
            "discrete_labels": discrete_labels,
            "parameters": parameters or {},
        },
        "uncertainty": {
            "kind": "none",
            "description": "",
            "calibrated": False,
        },
    }


def _spatial_manifest(
    *,
    task: str,
    source_operation: str,
    target: tuple[int, int],
    output: dict[str, Any],
) -> ModelManifest:
    mapping = deepcopy(valid_manifest_mapping())
    mapping["tasks"] = [task]
    mapping["subtasks"] = [f"test-{task}"]
    mapping["inputs"][0]["shape"] = [1, 3, target[0], target[1]]
    mapping["inputs"][0]["preprocessing"]["spatial"] = [
        {
            "operation": source_operation,
            "size": list(target),
            "interpolation": "nearest",
        }
    ]
    mapping["outputs"] = [output]
    return ModelManifest.from_mapping(mapping)


def _result(
    manifest: ModelManifest,
    source: np.ndarray,
    output: np.ndarray,
) -> Any:
    preprocessing = manifest.inputs[0].preprocessing
    assert preprocessing is not None
    prepared = prepare_input_2d(source, preprocessing)
    request = InferenceRequest(
        inputs={"image": source},
        transform_records={"image": prepared.transform_record},
    )
    return build_typed_result(
        manifest,
        {manifest.outputs[0].name: output},
        request,
        runtime=RuntimeKind.ONNX,
        duration_ms=1.0,
        device="cpu",
    )


def test_resize_mask_inverse_mapping_uses_nearest_neighbor() -> None:
    manifest = _spatial_manifest(
        task="segmentation",
        source_operation="resize",
        target=(4, 4),
        output=_output(
            name="masks",
            semantic="masks",
            dtype="int64",
            shape=[1, 1, 4, 4],
            coordinate_system="model-input-pixel",
            discrete_labels=True,
            interpolation="nearest",
            parameters={"layout": "nchw"},
        ),
    )
    source = np.zeros((2, 2, 3), dtype=np.uint8)
    expected = np.asarray([[1, 0], [0, 2]], dtype=np.int64)
    model_mask = np.repeat(np.repeat(expected, 2, axis=0), 2, axis=1)[None, None]

    result = _result(manifest, source, model_mask)

    assert isinstance(result, SegmentationResult)
    np.testing.assert_array_equal(result.masks, expected)
    assert set(np.unique(result.masks)) == {0, 1, 2}


def test_letterbox_mask_inverse_mapping_removes_padding() -> None:
    manifest = _spatial_manifest(
        task="segmentation",
        source_operation="letterbox",
        target=(4, 4),
        output=_output(
            name="masks",
            semantic="masks",
            dtype="int64",
            shape=[1, 1, 4, 4],
            coordinate_system="normalized",
            discrete_labels=True,
            interpolation="nearest",
            parameters={"layout": "nchw"},
        ),
    )
    source = np.zeros((2, 4, 3), dtype=np.uint8)
    expected = np.asarray([[1, 1, 0, 0], [2, 2, 1, 0]], dtype=np.int64)
    model_mask = np.zeros((1, 1, 4, 4), dtype=np.int64)
    model_mask[0, 0, 1:3] = expected

    result = _result(manifest, source, model_mask)

    np.testing.assert_array_equal(result.masks, expected)


def test_resize_box_inverse_mapping_returns_source_pixels() -> None:
    manifest = _spatial_manifest(
        task="detection",
        source_operation="resize",
        target=(4, 8),
        output=_output(
            name="boxes",
            semantic="boxes",
            dtype="float32",
            shape=[1, 6],
            coordinate_system="model-input-pixel",
            discrete_labels=False,
            interpolation=None,
        ),
    )
    source = np.zeros((2, 4, 3), dtype=np.uint8)
    preprocessing = manifest.inputs[0].preprocessing
    assert preprocessing is not None
    transform = prepare_input_2d(source, preprocessing).transform_record
    source_box = np.asarray([[0.25, 0.25, 3.25, 1.25]], dtype=float)
    model_box = transform.forward_boxes(source_box).astype(np.float32)
    raw = np.column_stack((model_box, [[0.9, 1.0]])).astype(np.float32)

    result = _result(manifest, source, raw)

    np.testing.assert_allclose(result.boxes[0].coordinates, source_box[0], atol=1e-6)
    assert result.boxes[0].coordinate_system is CoordinateSystem.SOURCE_PIXEL
    assert _visualizations(result)[0].coordinate_system == "source-pixel"


def test_letterbox_normalized_box_inverse_mapping_returns_source_pixels() -> None:
    manifest = _spatial_manifest(
        task="detection",
        source_operation="letterbox",
        target=(4, 4),
        output=_output(
            name="boxes",
            semantic="boxes",
            dtype="float32",
            shape=[1, 6],
            coordinate_system="normalized",
            discrete_labels=False,
            interpolation=None,
        ),
    )
    source = np.zeros((2, 4, 3), dtype=np.uint8)
    preprocessing = manifest.inputs[0].preprocessing
    assert preprocessing is not None
    transform = prepare_input_2d(source, preprocessing).transform_record
    source_box = np.asarray([[0.5, 0.25, 3.5, 1.75]], dtype=float)
    model_box = transform.forward_boxes(source_box)
    model_box /= np.asarray([4.0, 4.0, 4.0, 4.0])
    raw = np.column_stack((model_box, [[0.8, 1.0]])).astype(np.float32)

    result = _result(manifest, source, raw)

    np.testing.assert_allclose(result.boxes[0].coordinates, source_box[0], atol=1e-6)
    assert result.boxes[0].coordinate_system is CoordinateSystem.SOURCE_PIXEL


def test_nchw_softmax_and_argmax_use_the_channel_axis() -> None:
    manifest = _spatial_manifest(
        task="segmentation",
        source_operation="resize",
        target=(2, 2),
        output=_output(
            name="masks",
            semantic="masks",
            dtype="float32",
            shape=[1, 2, 2, 2],
            coordinate_system="model-input-pixel",
            discrete_labels=False,
            interpolation="bilinear",
            activation="softmax",
            parameters={"layout": "nchw"},
        ),
    )
    source = np.zeros((2, 2, 3), dtype=np.uint8)
    logits = np.asarray(
        [[[[4.0, 1.0], [1.0, 4.0]], [[1.0, 4.0], [4.0, 1.0]]]],
        dtype=np.float32,
    )

    result = _result(manifest, source, logits)

    np.testing.assert_allclose(np.sum(result.probabilities, axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(result.masks, [[0, 1], [1, 0]])


class _FakeOnnxSession:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="image")]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="scores")]

    def run(self, output_names: list[str], tensors: dict[str, Any], options: Any) -> list[Any]:
        assert output_names == ["scores"]
        assert tensors["image"].dtype == np.float32
        return [self.output]


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (np.zeros((1, 3), dtype=np.float32), "axis 1 has size 3"),
        (np.zeros((1, 2), dtype=np.float64), "dtype 'float64'"),
    ],
)
def test_onnx_runtime_rejects_declared_output_shape_and_dtype_mismatches(
    output: np.ndarray,
    message: str,
) -> None:
    manifest = ModelManifest.from_mapping(valid_manifest_mapping())
    plugin = OnnxModelPlugin(manifest, Path("."))
    plugin._session = _FakeOnnxSession(output)
    plugin._loaded = True

    with pytest.raises(PluginContractError, match=message):
        plugin.predict(InferenceRequest(inputs={"image": np.zeros((32, 32, 3), dtype=np.uint8)}))


def test_runtime_rejects_declared_input_shape_and_dtype_mismatches(monkeypatch) -> None:
    shape_mapping = deepcopy(valid_manifest_mapping())
    shape_mapping["inputs"][0]["shape"] = [1, 3, 16, 16]
    shape_manifest = ModelManifest.from_mapping(shape_mapping)
    request = InferenceRequest(inputs={"image": np.zeros((32, 32, 3), dtype=np.uint8)})
    with pytest.raises(PluginContractError, match="axis 2 has size 32"):
        _prepared_inputs(shape_manifest, request)

    dtype_manifest = ModelManifest.from_mapping(valid_manifest_mapping())
    preprocessing = dtype_manifest.inputs[0].preprocessing
    assert preprocessing is not None
    prepared = prepare_input_2d(request.inputs["image"], preprocessing)
    bad_prepared = replace(prepared, tensor=prepared.tensor.astype(np.float64))
    monkeypatch.setattr(
        "dicom_viewer.inference.standard_runtimes.prepare_input_2d",
        lambda source, spec: bad_prepared,
    )
    with pytest.raises(PluginContractError, match="dtype 'float64'"):
        _prepared_inputs(dtype_manifest, request)


def test_input_context_validates_modality_and_required_spacing() -> None:
    mapping = deepcopy(valid_manifest_mapping())
    mapping["inputs"][0]["modalities"] = ["ct"]
    mapping["inputs"][0]["spacing"] = {
        "required": True,
        "values": [0.7, 0.8],
        "unit": "mm",
        "tolerance": 0.01,
    }
    manifest = ModelManifest.from_mapping(mapping)
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    with pytest.raises(PluginContractError, match="does not accept modality"):
        _prepared_inputs(manifest, InferenceRequest(inputs={"image": image}))
    with pytest.raises(PluginContractError, match="requires spacing"):
        _prepared_inputs(
            manifest,
            InferenceRequest(
                inputs={"image": image},
                parameters={"input_context": {"image": {"modality": "ct"}}},
            ),
        )
    with pytest.raises(PluginContractError, match="does not match"):
        _prepared_inputs(
            manifest,
            InferenceRequest(
                inputs={"image": image},
                parameters={"input_context": {"image": {"modality": "ct", "spacing": [1.0, 1.0]}}},
            ),
        )

    tensors, _ = _prepared_inputs(
        manifest,
        InferenceRequest(
            inputs={"image": image},
            parameters={
                "input_context": {
                    "image": {
                        "modality": "ct",
                        "spacing": [0.705, 0.795],
                        "spacing_unit": "mm",
                    }
                }
            },
        ),
    )
    assert tensors["image"].shape == (1, 3, 32, 32)


def test_lanczos_is_rejected_instead_of_silently_using_cubic() -> None:
    mapping = deepcopy(valid_manifest_mapping())
    mapping["inputs"][0]["preprocessing"]["spatial"][0]["interpolation"] = "lanczos"
    manifest = ModelManifest.from_mapping(mapping)
    preprocessing = manifest.inputs[0].preprocessing
    assert preprocessing is not None

    with pytest.raises(PluginContractError, match="cannot implement Lanczos exactly"):
        prepare_input_2d(np.zeros((16, 16, 3), dtype=np.uint8), preprocessing)


def test_validate_tensor_dtype_requires_an_exact_manifest_match() -> None:
    with pytest.raises(PluginContractError, match="expected 'float32'"):
        validate_tensor_dtype(
            np.zeros((1,), dtype=np.float64),
            "float32",
            label="Output 'value'",
        )
