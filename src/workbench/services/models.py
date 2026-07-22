"""Application facade for user-supplied model manifests and runtimes.

The UI deliberately depends on this service instead of selecting a concrete
ONNX, TorchScript, or Python adapter.  Runtime implementations remain an
inference-layer detail and can be replaced in tests through the factory map.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

from ..inference import (
    PYTHON_ADAPTER_WARNING,
    InferenceCancelledError,
    InferenceRequest,
    InferenceResult,
    ModelManifest,
    ModelPlugin,
    OnnxModelPlugin,
    PluginContractError,
    PluginLoadContext,
    PluginNotLoadedError,
    PythonAdapterProxy,
    RuntimeKind,
    TorchScriptModelPlugin,
    VisualizationArtifact,
    VisualizationContext,
    load_manifest,
)

ModelPluginFactory = Callable[[ModelManifest, Path], ModelPlugin]
CancellationCheck = Callable[[], bool]


class ManifestLoader(Protocol):
    def __call__(
        self,
        path: str | Path,
        *,
        validate_files: bool = False,
    ) -> ModelManifest: ...


def _default_plugin_factories() -> dict[RuntimeKind, ModelPluginFactory]:
    return {
        RuntimeKind.ONNX: OnnxModelPlugin,
        RuntimeKind.TORCHSCRIPT: TorchScriptModelPlugin,
        RuntimeKind.PYTHON_ADAPTER: PythonAdapterProxy,
    }


class ModelInferenceService:
    """Own one validated manifest and its replaceable runtime session.

    ``load`` and ``predict`` are synchronous so callers can run them through
    the shared background-task service.  ``cancel`` is thread-safe and may be
    called by the GUI while either operation is executing.
    """

    python_adapter_warning = PYTHON_ADAPTER_WARNING

    def __init__(
        self,
        *,
        manifest_loader: ManifestLoader = load_manifest,
        plugin_factories: Mapping[RuntimeKind, ModelPluginFactory] | None = None,
    ) -> None:
        self._manifest_loader = manifest_loader
        self._plugin_factories = dict(plugin_factories or _default_plugin_factories())
        missing = set(RuntimeKind) - set(self._plugin_factories)
        if missing:
            names = ", ".join(sorted(item.value for item in missing))
            raise ValueError(f"Model runtime factories are missing: {names}.")
        self._lock = RLock()
        self._manifest: ModelManifest | None = None
        self._manifest_path: Path | None = None
        self._plugin: ModelPlugin | None = None
        self._loaded = False
        self._reload_required = False
        self._generation = 0

    @property
    def manifest(self) -> ModelManifest | None:
        with self._lock:
            return self._manifest

    @property
    def manifest_path(self) -> Path | None:
        with self._lock:
            return self._manifest_path

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._loaded and not self._reload_required and self._plugin is not None

    @property
    def reload_required(self) -> bool:
        with self._lock:
            return self._reload_required

    @property
    def requires_python_consent(self) -> bool:
        with self._lock:
            return bool(
                self._manifest is not None
                and self._manifest.runtime.kind is RuntimeKind.PYTHON_ADAPTER
            )

    def inspect_manifest(self, path: str | Path) -> ModelManifest:
        """Strictly parse a manifest without loading code, weights, or a runtime."""

        manifest_path = Path(path).expanduser()
        manifest = self._manifest_loader(manifest_path, validate_files=False)
        self.unload()
        with self._lock:
            self._manifest = manifest
            self._manifest_path = manifest_path
            self._reload_required = False
            self._generation += 1
        return manifest

    def load(
        self,
        *,
        user_consented_python_code: bool = False,
        cancellation_check: CancellationCheck | None = None,
    ) -> ModelManifest:
        """Select, validate, and load the runtime declared by the manifest."""

        self._raise_if_cancelled(cancellation_check)
        with self._lock:
            manifest = self._manifest
            manifest_path = self._manifest_path
            if manifest is None or manifest_path is None:
                raise PluginContractError("Inspect a model manifest before loading its runtime.")
            if (
                manifest.runtime.kind is RuntimeKind.PYTHON_ADAPTER
                and not user_consented_python_code
            ):
                raise PluginContractError(
                    "Python adapter execution requires explicit user consent."
                )
            factory = self._plugin_factories[manifest.runtime.kind]
            generation = self._generation + 1
            self._generation = generation
            previous = self._plugin
            plugin_root = manifest_path.parent
            plugin = factory(manifest, plugin_root)
            self._plugin = plugin
            self._loaded = False
            self._reload_required = False
        if previous is not None:
            with suppress(BaseException):
                previous.close()
        context = PluginLoadContext(
            plugin_root=plugin_root,
            device=manifest.runtime.device,
            user_consented_python_code=user_consented_python_code,
            subprocess_isolated=manifest.runtime.kind is RuntimeKind.PYTHON_ADAPTER,
        )
        try:
            plugin.load(context)
            self._raise_if_cancelled(cancellation_check)
        except BaseException:
            with self._lock:
                current = self._plugin is plugin and self._generation == generation
                if current:
                    self._loaded = False
                    if manifest.runtime.kind is RuntimeKind.PYTHON_ADAPTER:
                        self._reload_required = True
            if current:
                with suppress(BaseException):
                    plugin.close()
            raise
        with self._lock:
            if (
                self._plugin is not plugin
                or self._generation != generation
                or self._reload_required
            ):
                stale = True
            else:
                stale = False
                self._loaded = True
        if stale:
            with suppress(BaseException):
                plugin.close()
            raise InferenceCancelledError("Model loading was superseded or cancelled.")
        return manifest

    def predict(
        self,
        inputs: Mapping[str, Any],
        *,
        request_id: str | None = None,
        prompts: Mapping[str, Any] | None = None,
        parameters: Mapping[str, Any] | None = None,
        cancellation_check: CancellationCheck | None = None,
    ) -> InferenceResult:
        """Build the stable request type and execute the active plugin."""

        self._raise_if_cancelled(cancellation_check)
        with self._lock:
            plugin = self._plugin
            manifest = self._manifest
            generation = self._generation
            if self._reload_required:
                raise PluginNotLoadedError(
                    "The model runtime was cancelled and must be loaded again."
                )
            if plugin is None or manifest is None or not self._loaded:
                raise PluginNotLoadedError("Load the selected model before prediction.")
        request = InferenceRequest(
            inputs=inputs,
            request_id=request_id,
            prompts=prompts or {},
            parameters=parameters or {},
            cancellation_check=cancellation_check,
        )
        try:
            result = plugin.predict(request)
            self._raise_if_cancelled(cancellation_check)
        except BaseException:
            if manifest.runtime.kind is RuntimeKind.PYTHON_ADAPTER:
                self._mark_reload_required(plugin, generation)
            raise
        with self._lock:
            if self._plugin is not plugin or self._generation != generation:
                raise InferenceCancelledError("Model prediction was superseded.")
        return result

    def visualize(
        self,
        result: InferenceResult,
        context: VisualizationContext | None = None,
    ) -> Sequence[VisualizationArtifact]:
        with self._lock:
            plugin = self._plugin
            if plugin is None:
                raise PluginNotLoadedError("No model runtime is available for visualization.")
        return plugin.visualize(result, context)

    def cancel(self) -> bool:
        """Request runtime cancellation and report whether a hook was invoked.

        Cancelling the subprocess-backed Python adapter terminates its
        interpreter, so that session is never reused without another explicit
        load (and another consent decision in the GUI).
        """

        with self._lock:
            plugin = self._plugin
            manifest = self._manifest
            if plugin is None:
                return False
            if manifest is not None and manifest.runtime.kind is RuntimeKind.PYTHON_ADAPTER:
                self._loaded = False
                self._reload_required = True
            cancel = getattr(plugin, "cancel", None)
        if not callable(cancel):
            return False
        with suppress(BaseException):
            return bool(cancel())
        return False

    def unload(self) -> None:
        """Release the runtime while retaining the inspected manifest."""

        with self._lock:
            plugin = self._plugin
            self._plugin = None
            self._loaded = False
            self._reload_required = False
            self._generation += 1
        if plugin is not None:
            with suppress(BaseException):
                plugin.close()

    def close(self) -> None:
        self.cancel()
        self.unload()

    def _mark_reload_required(self, plugin: ModelPlugin, generation: int) -> None:
        with self._lock:
            if self._plugin is not plugin or self._generation != generation:
                return
            self._loaded = False
            self._reload_required = True
        with suppress(BaseException):
            plugin.close()

    @staticmethod
    def _raise_if_cancelled(cancellation_check: CancellationCheck | None) -> None:
        if cancellation_check is not None and cancellation_check():
            raise InferenceCancelledError("Model operation was cancelled.")


__all__ = [
    "CancellationCheck",
    "ManifestLoader",
    "ModelInferenceService",
    "ModelPluginFactory",
]
