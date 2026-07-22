"""Optional ONNX and TorchScript implementations of the stable plugin contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from threading import Event, Lock
from time import perf_counter
from typing import Any

import numpy as np

from ..domain.images import RasterImage2D
from .enums import CoordinateSystem, DeviceKind, Modality, RuntimeKind, VisualizationKind
from .errors import InferenceCancelledError, PluginContractError, PluginNotLoadedError
from .execution import (
    build_typed_result,
    prepare_input_2d,
    validate_tensor_dtype,
    validate_tensor_shape,
)
from .manifest import ModelManifest
from .plugin import (
    InferenceRequest,
    ManifestBackedModelPlugin,
    PluginLoadContext,
    PluginValidationContext,
    PluginValidationReport,
    VisualizationArtifact,
    VisualizationContext,
)
from .results import (
    AnomalyDetectionResult,
    ClassificationResult,
    DetectionResult,
    GenerationResult,
    InferenceResult,
    MultimodalResult,
    ReconstructionResult,
    RegistrationResult,
    RepresentationResult,
    RestorationResult,
    SegmentationResult,
    TrackingResult,
    WSIMILResult,
)
from .security import resolve_local_reference


def _validate_input_context(
    manifest: ModelManifest,
    request: InferenceRequest,
) -> None:
    raw_context = request.parameters.get("input_context", {})
    if raw_context is None:
        raw_context = {}
    if not isinstance(raw_context, Mapping):
        raise PluginContractError("request.parameters['input_context'] must be a mapping.")
    declared_names = {item.name for item in manifest.inputs}
    extra_names = set(raw_context) - declared_names
    if extra_names:
        raise PluginContractError(
            "Input context contains undeclared inputs: "
            f"{', '.join(sorted(str(item) for item in extra_names))}."
        )

    for spec in manifest.inputs:
        context = raw_context.get(spec.name, {})
        if context is None:
            context = {}
        if not isinstance(context, Mapping):
            raise PluginContractError(f"Input context for {spec.name!r} must be a mapping.")
        try:
            modality = Modality.coerce(context.get("modality", Modality.GENERIC_IMAGE.value))
        except (TypeError, ValueError) as exc:
            raise PluginContractError(
                f"Input context for {spec.name!r} has an invalid modality."
            ) from exc
        if modality not in spec.modalities:
            supported = ", ".join(item.value for item in spec.modalities)
            raise PluginContractError(
                f"Model input {spec.name!r} does not accept modality {modality.value!r}; "
                f"expected one of: {supported}."
            )

        spacing_value = context.get("spacing")
        spacing_unit: Any = context.get("spacing_unit", context.get("unit"))
        source = request.inputs.get(spec.name)
        if (
            spacing_value is None
            and isinstance(source, RasterImage2D)
            and (spec.spacing.required or spec.spacing.values is not None)
        ):
            spacing_value = source.pixel_spacing
            if spacing_value is not None:
                spacing_unit = "mm"
        if spacing_value is None:
            if spec.spacing.required:
                raise PluginContractError(
                    f"Model input {spec.name!r} requires spacing in {spec.spacing.unit}."
                )
            continue
        if isinstance(spacing_value, (str, bytes)) or not isinstance(spacing_value, Sequence):
            raise PluginContractError(
                f"Input context spacing for {spec.name!r} must be a numeric sequence."
            )
        try:
            spacing = tuple(float(item) for item in spacing_value)
        except (TypeError, ValueError) as exc:
            raise PluginContractError(
                f"Input context spacing for {spec.name!r} must be numeric."
            ) from exc
        if len(spacing) < 2 or any(not np.isfinite(item) or item <= 0 for item in spacing):
            raise PluginContractError(
                f"Input context spacing for {spec.name!r} must contain at least two "
                "positive finite values."
            )
        actual_unit = spec.spacing.unit if spacing_unit is None else str(spacing_unit)
        if actual_unit.casefold() != spec.spacing.unit.casefold():
            raise PluginContractError(
                f"Model input {spec.name!r} spacing uses {actual_unit!r}, "
                f"expected {spec.spacing.unit!r}."
            )
        expected = spec.spacing.values
        if expected is not None:
            if len(spacing) != len(expected):
                raise PluginContractError(
                    f"Model input {spec.name!r} spacing has {len(spacing)} values, "
                    f"expected {len(expected)}."
                )
            tolerance = 0.0 if spec.spacing.tolerance is None else spec.spacing.tolerance
            if any(
                abs(observed - declared) > tolerance
                for observed, declared in zip(spacing, expected, strict=True)
            ):
                raise PluginContractError(
                    f"Model input {spec.name!r} spacing {spacing} does not match "
                    f"the declared {expected} within tolerance {tolerance}."
                )


def _prepared_inputs(
    manifest: ModelManifest,
    request: InferenceRequest,
) -> tuple[dict[str, Any], InferenceRequest]:
    request.raise_if_cancelled()
    _validate_input_context(manifest, request)
    tensors: dict[str, Any] = {}
    transforms = dict(request.transform_records)
    for spec in manifest.inputs:
        if spec.name not in request.inputs:
            raise PluginContractError(f"Missing required model input {spec.name!r}.")
        value = request.inputs[spec.name]
        if spec.preprocessing is not None:
            if not isinstance(value, (RasterImage2D, np.ndarray)):
                raise PluginContractError(
                    f"2-D input {spec.name!r} must be RasterImage2D or ndarray."
                )
            prepared = prepare_input_2d(value, spec.preprocessing)
            tensors[spec.name] = prepared.tensor
            transforms[spec.name] = prepared.transform_record
        else:
            tensors[spec.name] = np.asarray(value) if hasattr(value, "__array__") else value
        validate_tensor_shape(
            tensors[spec.name],
            spec.shape,
            label=f"Model input {spec.name!r}",
        )
        if spec.preprocessing is not None:
            validate_tensor_dtype(
                tensors[spec.name],
                spec.preprocessing.dtype,
                label=f"Model input {spec.name!r}",
            )
    extra = set(request.inputs) - {item.name for item in manifest.inputs}
    if extra:
        raise PluginContractError(f"Undeclared model inputs: {', '.join(sorted(extra))}.")
    updated = replace(request, transform_records=transforms)
    updated.raise_if_cancelled()
    return tensors, updated


def _visualizations(result: InferenceResult) -> tuple[VisualizationArtifact, ...]:
    artifacts: list[VisualizationArtifact] = []
    if isinstance(result, ClassificationResult):
        artifacts.append(
            VisualizationArtifact(
                kind=VisualizationKind.TABLE,
                title="Class scores",
                payload=dict(result.scores),
            )
        )
        for name, value in result.saliency_maps.items():
            artifacts.append(
                VisualizationArtifact(kind=VisualizationKind.HEATMAP, title=name, payload=value)
            )
    elif isinstance(result, SegmentationResult):
        artifacts.append(
            VisualizationArtifact(
                kind=VisualizationKind.MASK_OVERLAY,
                title="Segmentation masks",
                payload=result.masks,
                coordinate_system="source-pixel",
            )
        )
    elif isinstance(result, DetectionResult):
        coordinate_systems = {box.coordinate_system for box in result.boxes}
        coordinate_system = (
            next(iter(coordinate_systems)).value
            if len(coordinate_systems) == 1
            else (
                CoordinateSystem.SOURCE_PIXEL.value
                if not coordinate_systems and result.transform_record is not None
                else None
            )
        )
        if result.boxes:
            artifacts.append(
                VisualizationArtifact(
                    kind=VisualizationKind.BOX_OVERLAY,
                    title="Detection boxes",
                    payload=result.boxes,
                    coordinate_system=coordinate_system,
                )
            )
        if result.keypoints:
            point_coordinates = {point.coordinate_system for point in result.keypoints}
            point_coordinate_system = (
                next(iter(point_coordinates)).value if len(point_coordinates) == 1 else None
            )
            artifacts.append(
                VisualizationArtifact(
                    kind=VisualizationKind.KEYPOINT_OVERLAY,
                    title="Detection keypoints",
                    payload=result.keypoints,
                    coordinate_system=point_coordinate_system,
                )
            )
        if result.masks is not None:
            artifacts.append(
                VisualizationArtifact(
                    kind=VisualizationKind.MASK_OVERLAY,
                    title="Detection masks",
                    payload=result.masks,
                    coordinate_system=CoordinateSystem.SOURCE_PIXEL.value,
                )
            )
    elif isinstance(result, RepresentationResult):
        artifacts.append(
            VisualizationArtifact(
                kind=VisualizationKind.PLOT,
                title="Embeddings",
                payload=result.embeddings,
            )
        )
        for name, value in result.attention_maps.items():
            artifacts.append(
                VisualizationArtifact(kind=VisualizationKind.HEATMAP, title=name, payload=value)
            )
    elif isinstance(result, RegistrationResult):
        if result.warped_image is not None:
            artifacts.append(
                VisualizationArtifact(
                    kind=VisualizationKind.IMAGE,
                    title="Warped image",
                    payload=result.warped_image,
                )
            )
        if result.vector_field is not None:
            artifacts.append(
                VisualizationArtifact(
                    kind=VisualizationKind.VECTOR_FIELD,
                    title="Vector field",
                    payload=result.vector_field,
                )
            )
    elif isinstance(result, ReconstructionResult):
        artifacts.append(
            VisualizationArtifact(
                kind=VisualizationKind.IMAGE,
                title="Reconstructed image",
                payload=result.reconstructed_image,
            )
        )
        if result.sampling_trajectory:
            artifacts.append(
                VisualizationArtifact(
                    kind=VisualizationKind.TRAJECTORY,
                    title="Reconstruction sampling trajectory",
                    payload=result.sampling_trajectory,
                )
            )
    elif isinstance(result, RestorationResult):
        artifacts.append(
            VisualizationArtifact(
                kind=VisualizationKind.IMAGE,
                title="Restored image",
                payload=result.restored_image,
            )
        )
    elif isinstance(result, GenerationResult):
        for index, sample in enumerate(result.samples):
            artifacts.append(
                VisualizationArtifact(
                    kind=VisualizationKind.IMAGE,
                    title=f"Generated sample {index + 1}",
                    payload=sample,
                )
            )
        if result.sampling_trajectory:
            artifacts.append(
                VisualizationArtifact(
                    kind=VisualizationKind.TRAJECTORY,
                    title="Generation sampling trajectory",
                    payload=result.sampling_trajectory,
                )
            )
    elif isinstance(result, MultimodalResult) and result.text_output:
        artifacts.append(
            VisualizationArtifact(
                kind=VisualizationKind.TEXT,
                title="Model text output",
                payload=result.text_output,
            )
        )
    elif isinstance(result, AnomalyDetectionResult):
        if result.anomaly_map is not None:
            artifacts.append(
                VisualizationArtifact(
                    kind=VisualizationKind.HEATMAP,
                    title="Anomaly map",
                    payload=result.anomaly_map,
                )
            )
    elif isinstance(result, WSIMILResult):
        artifacts.append(
            VisualizationArtifact(
                kind=VisualizationKind.TABLE,
                title="Slide scores",
                payload=dict(result.slide_scores),
            )
        )
    elif isinstance(result, TrackingResult):
        artifacts.append(
            VisualizationArtifact(
                kind=VisualizationKind.TRAJECTORY,
                title="Tracks",
                payload=result.tracks,
            )
        )
        if result.vector_fields is not None:
            artifacts.append(
                VisualizationArtifact(
                    kind=VisualizationKind.VECTOR_FIELD,
                    title="Tracking vector fields",
                    payload=result.vector_fields,
                )
            )
    return tuple(artifacts)


class _StandardRuntimePlugin(ManifestBackedModelPlugin):
    expected_runtime: RuntimeKind

    def __init__(self, manifest: ModelManifest, plugin_root: str | Path) -> None:
        super().__init__(manifest)
        if manifest.runtime.kind is not self.expected_runtime:
            raise PluginContractError(
                f"{type(self).__name__} requires runtime {self.expected_runtime.value!r}."
            )
        self.plugin_root = Path(plugin_root).expanduser().resolve(strict=False)
        self._loaded = False
        self._device: str | None = None
        self._cancel_event = Event()
        self._active_lock = Lock()
        self._prediction_lock = Lock()
        self._active = False

    def validate(self, context: PluginValidationContext | None = None) -> PluginValidationReport:
        return super().validate(context or PluginValidationContext(plugin_root=self.plugin_root))

    def _model_path(self, context: PluginLoadContext) -> Path:
        root = context.plugin_root.resolve(strict=False)
        if root != self.plugin_root:
            raise PluginContractError(
                "PluginLoadContext.plugin_root does not match the plugin instance root."
            )
        return resolve_local_reference(
            self.describe().entrypoint.path,
            root,
            field="entrypoint.path",
            require_within_root=True,
        )

    def _begin_prediction(self, request: InferenceRequest) -> None:
        request.raise_if_cancelled()
        self._cancel_event.clear()
        with self._active_lock:
            self._active = True

    def _end_prediction(self) -> None:
        with self._active_lock:
            self._active = False

    def _raise_if_cancelled(self, request: InferenceRequest) -> None:
        request.raise_if_cancelled()
        if self._cancel_event.is_set():
            raise InferenceCancelledError("Model inference was cancelled.")

    def cancel(self) -> bool:
        """Request cancellation; runtimes may only stop at a safe boundary."""

        with self._active_lock:
            active = self._active
        if active:
            self._cancel_event.set()
        return active

    def visualize(
        self,
        result: InferenceResult,
        context: VisualizationContext | None = None,
    ) -> Sequence[VisualizationArtifact]:
        self.validate_result(result)
        return _visualizations(result)


class OnnxModelPlugin(_StandardRuntimePlugin):
    """Run a user-supplied local ONNX graph without downloading anything."""

    expected_runtime = RuntimeKind.ONNX

    def __init__(self, manifest: ModelManifest, plugin_root: str | Path) -> None:
        super().__init__(manifest, plugin_root)
        self._session: Any | None = None
        self._run_options: Any | None = None
        self._run_options_lock = Lock()

    def load(self, context: PluginLoadContext) -> None:
        self.authorize_load(context)
        report = self.validate(PluginValidationContext(plugin_root=context.plugin_root))
        report.raise_for_errors()
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise PluginContractError(
                "ONNX Runtime is not installed; install the 'onnx' optional dependency."
            ) from exc
        model_path = self._model_path(context)
        available = set(ort.get_available_providers())
        options = dict(self.describe().runtime.options)
        options.update(context.runtime_options)
        requested = options.get("providers")
        if requested is None and self.describe().runtime.provider is not None:
            requested = [self.describe().runtime.provider]
        if requested is None:
            if context.device is DeviceKind.CUDA:
                if "CUDAExecutionProvider" not in available:
                    raise PluginContractError("CUDA was requested but ONNX CUDA is unavailable.")
                requested = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif context.device is DeviceKind.DIRECTML:
                if "DmlExecutionProvider" not in available:
                    raise PluginContractError(
                        "DirectML was requested but its ONNX provider is unavailable."
                    )
                requested = ["DmlExecutionProvider", "CPUExecutionProvider"]
            elif context.device is DeviceKind.MPS:
                raise PluginContractError("MPS is not an ONNX Runtime execution provider.")
            else:
                requested = ["CPUExecutionProvider"]
        if isinstance(requested, str):
            requested = [requested]
        elif not isinstance(requested, Sequence):
            raise PluginContractError("ONNX providers must be a sequence of provider names.")
        requested_names = [str(item) for item in requested]
        missing = [item for item in requested_names if item not in available]
        if missing:
            raise PluginContractError(
                f"Requested ONNX providers are unavailable: {missing}; "
                f"available: {sorted(available)}."
            )
        providers = requested_names
        if not providers:
            raise PluginContractError(
                f"Requested ONNX providers are unavailable; available: {sorted(available)}."
            )
        try:
            self._session = ort.InferenceSession(str(model_path), providers=providers)
        except Exception as exc:
            raise PluginContractError(
                f"ONNX model could not be loaded ({type(exc).__name__})."
            ) from None
        self._device = ",".join(self._session.get_providers())
        self._loaded = True

    def predict(self, request: InferenceRequest) -> InferenceResult:
        if not self._loaded or self._session is None:
            raise PluginNotLoadedError("Load the ONNX plugin before prediction.")
        with self._prediction_lock:
            self._begin_prediction(request)
            try:
                return self._predict_loaded(request)
            finally:
                with self._run_options_lock:
                    self._run_options = None
                self._end_prediction()

    def _predict_loaded(self, request: InferenceRequest) -> InferenceResult:
        assert self._session is not None
        tensors, updated_request = _prepared_inputs(self.describe(), request)
        expected_names = {item.name for item in self._session.get_inputs()}
        if set(tensors) != expected_names:
            raise PluginContractError(
                f"ONNX input names {sorted(expected_names)} do not match "
                f"manifest {sorted(tensors)}."
            )
        output_names = [item.name for item in self._session.get_outputs()]
        try:
            import onnxruntime as ort

            run_options = ort.RunOptions()
        except (ImportError, AttributeError):  # pragma: no cover - load proved dependency
            run_options = None
        with self._run_options_lock:
            self._run_options = run_options
        self._raise_if_cancelled(updated_request)
        started = perf_counter()
        try:
            values = self._session.run(output_names, tensors, run_options)
        except Exception as exc:
            if self._cancel_event.is_set():
                raise InferenceCancelledError("ONNX inference was cancelled.") from None
            raise PluginContractError(
                f"ONNX prediction failed ({type(exc).__name__}); verify input shapes and dtypes."
            ) from None
        duration = (perf_counter() - started) * 1000.0
        self._raise_if_cancelled(updated_request)
        outputs = dict(zip(output_names, values, strict=True))
        declared = {item.name for item in self.describe().outputs}
        if not declared.issubset(outputs):
            raise PluginContractError(
                f"ONNX outputs {sorted(outputs)} do not contain "
                f"manifest outputs {sorted(declared)}."
            )
        for spec in self.describe().outputs:
            validate_tensor_shape(
                outputs[spec.name],
                spec.shape,
                label=f"ONNX output {spec.name!r}",
            )
            validate_tensor_dtype(
                outputs[spec.name],
                spec.dtype,
                label=f"ONNX output {spec.name!r}",
            )
        result = build_typed_result(
            self.describe(),
            outputs,
            updated_request,
            runtime=RuntimeKind.ONNX,
            duration_ms=duration,
            device=self._device,
        )
        self.validate_result(result)
        return result

    def cancel(self) -> bool:
        active = super().cancel()
        with self._run_options_lock:
            run_options = self._run_options
            if run_options is not None:
                with suppress(AttributeError, RuntimeError):
                    run_options.terminate = True
        return active

    def close(self) -> None:
        self.cancel()
        self._session = None
        self._loaded = False


class TorchScriptModelPlugin(_StandardRuntimePlugin):
    """Run an exported TorchScript graph; plain .pth/.ckpt files are rejected."""

    expected_runtime = RuntimeKind.TORCHSCRIPT

    def __init__(self, manifest: ModelManifest, plugin_root: str | Path) -> None:
        super().__init__(manifest, plugin_root)
        self._model: Any | None = None
        self._torch: Any | None = None

    def load(self, context: PluginLoadContext) -> None:
        self.authorize_load(context)
        report = self.validate(PluginValidationContext(plugin_root=context.plugin_root))
        report.raise_for_errors()
        try:
            import torch
        except ImportError as exc:
            raise PluginContractError(
                "TorchScript requires PyTorch; install the 'pytorch-plugin' optional dependency."
            ) from exc
        path = self._model_path(context)
        if path.suffix.lower() in {".pth", ".ckpt"}:
            raise PluginContractError(
                "Plain .pth/.ckpt files do not define architecture/preprocessing; "
                "use a reviewed Python adapter."
            )
        requested = context.device
        if requested is DeviceKind.CUDA:
            if not torch.cuda.is_available():
                raise PluginContractError("CUDA was requested but is unavailable.")
            device = "cuda"
        elif requested is DeviceKind.MPS:
            if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
                raise PluginContractError("MPS was requested but is unavailable.")
            device = "mps"
        elif requested is DeviceKind.DIRECTML:
            raise PluginContractError(
                "DirectML is not supported by the TorchScript runtime adapter."
            )
        else:
            device = "cpu"
        try:
            self._model = torch.jit.load(str(path), map_location=device)
            self._model.eval()
        except Exception as exc:
            raise PluginContractError(
                f"TorchScript model could not be loaded ({type(exc).__name__})."
            ) from None
        self._torch = torch
        self._device = device
        self._loaded = True

    def predict(self, request: InferenceRequest) -> InferenceResult:
        if not self._loaded or self._model is None or self._torch is None:
            raise PluginNotLoadedError("Load the TorchScript plugin before prediction.")
        with self._prediction_lock:
            self._begin_prediction(request)
            try:
                return self._predict_loaded(request)
            finally:
                self._end_prediction()

    def _predict_loaded(self, request: InferenceRequest) -> InferenceResult:
        assert self._model is not None and self._torch is not None
        tensors, updated_request = _prepared_inputs(self.describe(), request)
        torch = self._torch
        ordered = [
            torch.from_numpy(np.asarray(tensors[item.name])).to(self._device)
            for item in self.describe().inputs
        ]
        self._raise_if_cancelled(updated_request)
        started = perf_counter()
        try:
            with torch.no_grad():
                raw = self._model(*ordered)
        except Exception as exc:
            raise PluginContractError(
                f"TorchScript prediction failed ({type(exc).__name__}); "
                "verify input shapes and dtypes."
            ) from None
        duration = (perf_counter() - started) * 1000.0
        self._raise_if_cancelled(updated_request)
        if isinstance(raw, Mapping):
            outputs = {str(key): self._to_numpy(value) for key, value in raw.items()}
        elif isinstance(raw, (tuple, list)):
            if len(raw) != len(self.describe().outputs):
                raise PluginContractError("TorchScript output count does not match the manifest.")
            outputs = {
                spec.name: self._to_numpy(value)
                for spec, value in zip(self.describe().outputs, raw, strict=True)
            }
        else:
            if len(self.describe().outputs) != 1:
                raise PluginContractError(
                    "A single TorchScript tensor cannot satisfy multiple outputs."
                )
            outputs = {self.describe().outputs[0].name: self._to_numpy(raw)}
        declared = {item.name for item in self.describe().outputs}
        if not declared.issubset(outputs):
            raise PluginContractError(
                f"TorchScript outputs {sorted(outputs)} do not contain "
                f"manifest outputs {sorted(declared)}."
            )
        for spec in self.describe().outputs:
            validate_tensor_shape(
                outputs[spec.name],
                spec.shape,
                label=f"TorchScript output {spec.name!r}",
            )
            validate_tensor_dtype(
                outputs[spec.name],
                spec.dtype,
                label=f"TorchScript output {spec.name!r}",
            )
        result = build_typed_result(
            self.describe(),
            outputs,
            updated_request,
            runtime=RuntimeKind.TORCHSCRIPT,
            duration_ms=duration,
            device=self._device,
        )
        self.validate_result(result)
        return result

    @staticmethod
    def _to_numpy(value: Any) -> Any:
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return value

    def close(self) -> None:
        self.cancel()
        self._model = None
        self._torch = None
        self._loaded = False
