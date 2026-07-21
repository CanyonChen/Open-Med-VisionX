"""Subprocess proxy for user-reviewed Python research adapters."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from contextlib import suppress
from io import BytesIO
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Any

import numpy as np

from .enums import OutputSemantic, RuntimeKind
from .errors import InferenceCancelledError, PluginContractError, PluginNotLoadedError
from .execution import build_typed_result, validate_tensor_dtype, validate_tensor_shape
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
from .results import InferenceResult
from .security import resolve_local_reference
from .standard_runtimes import _prepared_inputs, _visualizations

_MAX_MESSAGE_BYTES = 256 * 1024 * 1024


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        stream = BytesIO()
        np.save(stream, value, allow_pickle=False)
        return {"__ndarray_npy__": base64.b64encode(stream.getvalue()).decode("ascii")}
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    raise PluginContractError(
        f"Python adapter request contains unsupported type {type(value).__name__}."
    )


def _decode(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"__ndarray_npy__"}:
        raw = base64.b64decode(value["__ndarray_npy__"], validate=True)
        return np.load(BytesIO(raw), allow_pickle=False)
    if isinstance(value, dict):
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


class PythonAdapterProxy(ManifestBackedModelPlugin):
    """ModelPlugin proxy whose adapter code always runs in another process.

    Process isolation contains dependency conflicts and crashes but is not a
    security sandbox. The caller must show the arbitrary-code warning and pass
    explicit consent in :class:`PluginLoadContext`.
    """

    def __init__(self, manifest: ModelManifest, plugin_root: str | Path) -> None:
        super().__init__(manifest)
        if manifest.runtime.kind is not RuntimeKind.PYTHON_ADAPTER:
            raise PluginContractError("PythonAdapterProxy requires a python-adapter manifest.")
        self.plugin_root = Path(plugin_root).expanduser().resolve(strict=False)
        self._process: subprocess.Popen[bytes] | None = None
        self._responses: Queue[bytes | None] = Queue()
        self._reader: Thread | None = None
        self._lock = Lock()
        self._cancel_event = Event()
        self._next_id = 1
        self._timeout = float(manifest.runtime.options.get("timeout_seconds", 120.0))
        self._device: str | None = None
        if self._timeout <= 0:
            raise PluginContractError("Python adapter timeout_seconds must be positive.")

    def validate(self, context: PluginValidationContext | None = None) -> PluginValidationReport:
        return super().validate(context or PluginValidationContext(plugin_root=self.plugin_root))

    def load(self, context: PluginLoadContext) -> None:
        self.authorize_load(context)
        if not context.subprocess_isolated:
            raise PluginContractError("Python adapter load requires subprocess isolation.")
        report = self.validate(PluginValidationContext(plugin_root=context.plugin_root))
        report.raise_for_errors()
        if self._process is not None:
            raise PluginContractError("Python adapter is already loaded.")
        context_root = context.plugin_root.expanduser().resolve(strict=False)
        if context_root != self.plugin_root:
            raise PluginContractError(
                "PluginLoadContext.plugin_root does not match the plugin instance root."
            )
        self._cancel_event.clear()
        self._responses = Queue()
        entrypoint = resolve_local_reference(
            self.describe().entrypoint.path,
            context.plugin_root,
            field="entrypoint.path",
            require_within_root=True,
        )
        object_name = self.describe().entrypoint.object_name
        assert object_name is not None
        command = self._worker_command(entrypoint, object_name)
        environment = self._filtered_environment()
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._process = subprocess.Popen(
                command,
                cwd=str(context_root),
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
        except OSError as exc:
            self._process = None
            raise PluginContractError(
                f"Python adapter process could not start ({type(exc).__name__})."
            ) from None
        process = self._process
        responses = self._responses
        self._reader = Thread(
            target=self._read_responses,
            args=(process, responses),
            daemon=True,
        )
        self._reader.start()
        self._device = context.device.value
        runtime_options = dict(self.describe().runtime.options)
        runtime_options.update(context.runtime_options)
        try:
            self._request(
                "load",
                {
                    "manifest": self.describe().to_mapping(),
                    "plugin_root": ".",
                    "device": context.device.value,
                    "runtime_options": runtime_options,
                },
            )
        except BaseException:
            self.cancel()
            raise

    def predict(self, request: InferenceRequest) -> InferenceResult:
        if self._process is None:
            raise PluginNotLoadedError("Load the Python adapter before prediction.")
        request.raise_if_cancelled()
        tensors, updated_request = _prepared_inputs(self.describe(), request)
        started = perf_counter()
        outputs = self._request(
            "predict",
            {
                "inputs": tensors,
                "prompts": dict(request.prompts),
                "parameters": dict(request.parameters),
                "request_id": request.request_id,
            },
        )
        if not isinstance(outputs, Mapping):
            raise PluginContractError("Python adapter returned an invalid output mapping.")
        updated_request.raise_if_cancelled()
        declared = {item.name for item in self.describe().outputs}
        undeclared = set(outputs) - declared
        if undeclared:
            raise PluginContractError(
                f"Python adapter returned undeclared outputs {sorted(undeclared)}."
            )
        for spec in self.describe().outputs:
            if spec.name not in outputs:
                continue
            validate_tensor_shape(
                outputs[spec.name],
                spec.shape,
                label=f"Python adapter output {spec.name!r}",
            )
            if spec.semantic not in {
                OutputSemantic.TEXT,
                OutputSemantic.TRACKS,
                OutputSemantic.SAMPLING_TRAJECTORIES,
            }:
                validate_tensor_dtype(
                    outputs[spec.name],
                    spec.dtype,
                    label=f"Python adapter output {spec.name!r}",
                )
        result = build_typed_result(
            self.describe(),
            outputs,
            updated_request,
            runtime=RuntimeKind.PYTHON_ADAPTER,
            duration_ms=(perf_counter() - started) * 1000.0,
            device=self._device,
        )
        self.validate_result(result)
        return result

    def visualize(
        self,
        result: InferenceResult,
        context: VisualizationContext | None = None,
    ) -> Sequence[VisualizationArtifact]:
        self.validate_result(result)
        return _visualizations(result)

    def _worker_command(self, entrypoint: Path, object_name: str) -> list[str]:
        environment = self.describe().runtime.python
        assert environment is not None
        worker = Path(__file__).with_name("_adapter_worker.py").resolve()
        arguments = [
            str(worker),
            "--plugin-root",
            str(self.plugin_root),
            "--entrypoint",
            str(entrypoint),
            "--object",
            object_name,
        ]
        if environment.python_executable:
            executable = resolve_local_reference(
                environment.python_executable,
                self.plugin_root,
                field="runtime.python.python_executable",
                require_within_root=False,
            )
            return [str(executable), *arguments]
        assert environment.conda_environment
        conda = shutil.which("conda")
        if conda is None:
            raise PluginContractError("Conda was requested for the adapter but is not available.")
        return [
            conda,
            "run",
            "--no-capture-output",
            "-n",
            environment.conda_environment,
            "python",
            *arguments,
        ]

    @staticmethod
    def _filtered_environment() -> dict[str, str]:
        allowed_exact = {
            "SYSTEMROOT",
            "WINDIR",
            "PATH",
            "PATHEXT",
            "TEMP",
            "TMP",
            "TMPDIR",
            "HOME",
            "USERPROFILE",
            "LD_LIBRARY_PATH",
            "DYLD_LIBRARY_PATH",
        }
        return {
            key: value
            for key, value in os.environ.items()
            if key.upper() in allowed_exact or key.upper().startswith("CONDA")
        }

    @staticmethod
    def _read_responses(
        process: subprocess.Popen[bytes] | None,
        responses: Queue[bytes | None],
    ) -> None:
        if process is None or process.stdout is None:
            responses.put(None)
            return
        while True:
            line = process.stdout.readline(_MAX_MESSAGE_BYTES + 1)
            if not line:
                responses.put(None)
                return
            if len(line) > _MAX_MESSAGE_BYTES or not line.endswith(b"\n"):
                responses.put(None)
                return
            responses.put(line)

    def _request(self, command: str, payload: Mapping[str, Any]) -> Any:
        with self._lock:
            process = self._process
            if process is None or process.stdin is None or process.poll() is not None:
                raise PluginContractError("Python adapter process is not running.")
            request_id = self._next_id
            self._next_id += 1
            encoded = (
                json.dumps(
                    {"id": request_id, "command": command, "payload": _encode(payload)},
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
            if len(encoded) > _MAX_MESSAGE_BYTES:
                raise PluginContractError("Python adapter request exceeds the IPC size limit.")
            try:
                process.stdin.write(encoded)
                process.stdin.flush()
                line = self._responses.get(timeout=self._timeout)
            except (BrokenPipeError, OSError, Empty):
                cancelled = self._cancel_event.is_set()
                self._terminate(mark_cancelled=False)
                if cancelled:
                    raise InferenceCancelledError(
                        "Python adapter inference was cancelled."
                    ) from None
                raise PluginContractError("Python adapter crashed or timed out.") from None
            if line is None:
                cancelled = self._cancel_event.is_set()
                self._terminate(mark_cancelled=False)
                if cancelled:
                    raise InferenceCancelledError("Python adapter inference was cancelled.")
                raise PluginContractError("Python adapter exited or violated the IPC protocol.")
            try:
                response = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._terminate(mark_cancelled=False)
                raise PluginContractError("Python adapter returned malformed IPC data.") from None
            if response.get("id") != request_id:
                self._terminate(mark_cancelled=False)
                raise PluginContractError("Python adapter returned an out-of-order response.")
            if not response.get("ok"):
                error_type = str(response.get("error_type", "AdapterError"))
                raise PluginContractError(f"Python adapter failed safely ({error_type}).")
            return _decode(response.get("payload"))

    def cancel(self) -> bool:
        return self._terminate(mark_cancelled=True)

    def _terminate(self, *, mark_cancelled: bool) -> bool:
        if mark_cancelled:
            self._cancel_event.set()
        process = self._process
        if process is None:
            return False
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        self._process = None
        self._device = None
        return True

    def close(self) -> None:
        if self._process is not None and self._process.poll() is None:
            with suppress(PluginContractError):
                self._request("close", {})
        self._terminate(mark_cancelled=False)
