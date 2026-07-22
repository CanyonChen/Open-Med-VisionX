"""Strict, pickle-free NumPy archives for reviewed model tensors.

The application only opens hash-verified ``.npz`` files and always passes
``allow_pickle=False``.  This module also provides the deterministic writer
used by the maintainer conversion scripts so the reviewed payload hashes are
reproducible across runs with the same NumPy/zlib implementation.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class SafeNpzError(ValueError):
    """A reviewed NPZ payload does not match its fixed tensor contract."""


@dataclass(frozen=True, slots=True)
class ArraySpec:
    """Exact shape and dtype required for one tensor."""

    shape: tuple[int, ...]
    dtype: str = "float32"
    finite: bool = True


def write_deterministic_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    """Write a compressed NPZ with stable member order and timestamps."""

    if not arrays:
        raise SafeNpzError("an NPZ payload must contain at least one array")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name in sorted(arrays):
            if not name or name.startswith(("/", "\\")) or ".." in Path(name).parts:
                raise SafeNpzError(f"unsafe NPZ tensor name: {name!r}")
            array = np.asarray(arrays[name])
            if array.dtype.hasobject:
                raise SafeNpzError(f"object arrays are forbidden: {name}")
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)


def load_safe_npz(
    path: Path,
    expected: Mapping[str, ArraySpec],
) -> dict[str, np.ndarray]:
    """Load only the exact declared numeric arrays, without pickle support."""

    if not expected:
        raise SafeNpzError("an NPZ contract must contain at least one array")
    try:
        with np.load(path, allow_pickle=False) as payload:
            actual_names = set(payload.files)
            expected_names = set(expected)
            if actual_names != expected_names:
                missing = sorted(expected_names - actual_names)
                extra = sorted(actual_names - expected_names)
                raise SafeNpzError(
                    f"NPZ keys do not match the reviewed contract; missing={missing}, extra={extra}"
                )
            result: dict[str, np.ndarray] = {}
            for name, spec in expected.items():
                array = np.asarray(payload[name])
                if array.dtype.hasobject:
                    raise SafeNpzError(f"object arrays are forbidden: {name}")
                if array.dtype != np.dtype(spec.dtype):
                    raise SafeNpzError(
                        f"{name} dtype mismatch: expected {spec.dtype}, got {array.dtype}"
                    )
                if tuple(array.shape) != spec.shape:
                    raise SafeNpzError(
                        f"{name} shape mismatch: expected {spec.shape}, got {tuple(array.shape)}"
                    )
                if spec.finite and not np.isfinite(array).all():
                    raise SafeNpzError(f"{name} contains a non-finite value")
                result[name] = np.array(array, copy=True, order="C")
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, SafeNpzError):
            raise
        raise SafeNpzError(f"cannot read reviewed NPZ {path}: {exc}") from exc
    return result
