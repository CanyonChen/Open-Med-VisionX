"""Private JSON-lines worker for explicitly trusted Python model adapters.

This process boundary contains crashes and dependency conflicts; it is not an
OS security sandbox.  The host shows that distinction before launching it.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import inspect
import json
import sys
from contextlib import redirect_stdout
from io import BytesIO
from pathlib import Path
from typing import Any


def _encode(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]
    if np is not None and isinstance(value, np.ndarray):
        stream = BytesIO()
        np.save(stream, value, allow_pickle=False)
        return {"__ndarray_npy__": base64.b64encode(stream.getvalue()).decode("ascii")}
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    raise TypeError(f"Adapter returned unsupported value type {type(value).__name__}.")


def _decode(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"__ndarray_npy__"}:
        import numpy as np

        raw = base64.b64decode(value["__ndarray_npy__"], validate=True)
        return np.load(BytesIO(raw), allow_pickle=False)
    if isinstance(value, dict):
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _load_object(plugin_root: Path, entrypoint: Path, object_name: str) -> Any:
    module_name = "_openmedvisionx_external_adapter"
    spec = importlib.util.spec_from_file_location(module_name, entrypoint)
    if spec is None or spec.loader is None:
        raise ImportError("Adapter module cannot be loaded.")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(plugin_root))
    try:
        with redirect_stdout(sys.stderr):
            spec.loader.exec_module(module)
    finally:
        if sys.path and sys.path[0] == str(plugin_root):
            sys.path.pop(0)
    target: Any = module
    for component in object_name.split("."):
        target = getattr(target, component)
    if inspect.isclass(target):
        with redirect_stdout(sys.stderr):
            target = target()
    if not callable(getattr(target, "predict", None)):
        raise TypeError("Adapter object must provide predict(request_mapping).")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--plugin-root", required=True)
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--object", required=True)
    args = parser.parse_args()
    plugin_root = Path(args.plugin_root).resolve()
    entrypoint = Path(args.entrypoint).resolve()
    try:
        entrypoint.relative_to(plugin_root)
    except ValueError:
        return 2
    try:
        adapter = _load_object(plugin_root, entrypoint, args.object)
    except BaseException as exc:
        response = {"id": None, "ok": False, "error_type": type(exc).__name__}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
        return 3

    for line in sys.stdin:
        request_id: Any = None
        try:
            envelope = json.loads(line)
            request_id = envelope.get("id")
            command = envelope.get("command")
            if command == "load":
                load_method = getattr(adapter, "load", None)
                if callable(load_method):
                    with redirect_stdout(sys.stderr):
                        load_method(_decode(envelope.get("payload", {})))
                payload: Any = {"loaded": True}
            elif command == "predict":
                with redirect_stdout(sys.stderr):
                    payload = adapter.predict(_decode(envelope.get("payload", {})))
                if not isinstance(payload, dict):
                    raise TypeError("Adapter predict() must return a named-output mapping.")
            elif command == "close":
                close_method = getattr(adapter, "close", None)
                if callable(close_method):
                    with redirect_stdout(sys.stderr):
                        close_method()
                payload = {"closed": True}
                sys.stdout.write(
                    json.dumps({"id": request_id, "ok": True, "payload": payload}) + "\n"
                )
                sys.stdout.flush()
                return 0
            else:
                raise ValueError("Unsupported worker command.")
            response = {"id": request_id, "ok": True, "payload": _encode(payload)}
        except BaseException as exc:
            # Do not echo adapter exception text: it may contain paths, PHI, or secrets.
            response = {"id": request_id, "ok": False, "error_type": type(exc).__name__}
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
