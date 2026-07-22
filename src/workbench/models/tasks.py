"""Task-aware adapters for the reviewed offline teaching models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np

from .bundled import load_bundled_model

DeviceRequest = Literal["auto", "cpu", "cuda"]
MRISimulationSource = Literal["image-derived-simulation", "synthetic-phantom"]


@dataclass(frozen=True, slots=True)
class ModelRunResult:
    """Immutable arrays plus visible execution and scientific provenance."""

    bundle_id: str
    device: str
    elapsed_seconds: float
    outputs: Mapping[str, np.ndarray]
    output_channels: tuple[str, ...]
    fallback_reason: str | None
    warnings: tuple[str, ...]


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.array(array, dtype=np.float32, copy=True)
    result.setflags(write=False)
    return result


def _numeric_finite(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(array, dtype=np.float32)


def run_dival_fbp_unet(
    fbp_image: np.ndarray,
    *,
    device: DeviceRequest = "auto",
) -> ModelRunResult:
    """Post-process one fixed-geometry DIVal LoDoPaB FBP reconstruction."""

    array = _numeric_finite(fbp_image, name="fbp_image")
    if array.shape == (362, 362):
        array = array[None, None]
    if array.shape != (1, 1, 362, 362):
        raise ValueError("DIVal FBP-U-Net requires shape 362x362 or 1x1x362x362")
    runtime = load_bundled_model("dival-lodopab-fbpunet", device=device)
    (raw,), elapsed = runtime.run(np.ascontiguousarray(array))
    reconstruction = _readonly(raw.numpy()[0, 0])
    return ModelRunResult(
        bundle_id=runtime.record.bundle_id,
        device=runtime.device,
        elapsed_seconds=elapsed,
        outputs=MappingProxyType({"reconstruction": reconstruction}),
        output_channels=("FBP post-processing reconstruction",),
        fallback_reason=runtime.fallback_reason,
        warnings=(
            "Input must come from the documented LoDoPaB FBP operator; it is not HU.",
            "Education and research only; not for diagnosis.",
        ),
    )


def _complex_channels(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        if array.ndim != 2:
            raise ValueError(f"complex {name} must have shape HxW")
        array = np.stack((array.real, array.imag), axis=0)[None]
    elif array.ndim == 3 and array.shape[0] == 2:
        array = array[None]
    if array.ndim != 4 or array.shape[0:2] != (1, 2):
        raise ValueError(f"{name} must have shape HxW complex, 2xHxW, or 1x2xHxW")
    if min(array.shape[-2:]) < 16:
        raise ValueError(f"{name} height and width must both be at least 16")
    return _numeric_finite(array, name=name)


def run_deepinv_mri_modl(
    kspace: np.ndarray,
    mask: np.ndarray,
    *,
    source_kind: MRISimulationSource,
    device: DeviceRequest = "auto",
) -> ModelRunResult:
    """Reconstruct reviewed single-coil, image-derived or synthetic k-space."""

    if source_kind not in {"image-derived-simulation", "synthetic-phantom"}:
        raise ValueError("source_kind must be 'image-derived-simulation' or 'synthetic-phantom'")
    kspace_array = _complex_channels(kspace, name="kspace")
    mask_array = np.asarray(mask)
    if mask_array.ndim == 2:
        mask_array = np.broadcast_to(mask_array, (1, 2, *mask_array.shape))
    elif mask_array.ndim == 3 and mask_array.shape[0] in {1, 2}:
        mask_array = np.broadcast_to(mask_array[None], kspace_array.shape)
    if mask_array.shape != kspace_array.shape:
        raise ValueError("mask must match k-space spatial dimensions and channel layout")
    if not np.logical_or(mask_array == 0, mask_array == 1).all():
        raise ValueError("mask must be binary")
    if not np.array_equal(mask_array[:, 0], mask_array[:, 1]):
        raise ValueError("MRI sampling mask must be identical for real and imaginary channels")
    mask_array = np.asarray(mask_array, dtype=np.float32)

    runtime = load_bundled_model("deepinv-mri-modl", device=device)
    (raw,), elapsed = runtime.run(
        np.ascontiguousarray(kspace_array),
        np.ascontiguousarray(mask_array),
    )
    complex_channels = _readonly(raw.numpy()[0])
    magnitude = _readonly(np.sqrt(complex_channels[0] ** 2 + complex_channels[1] ** 2))
    source_warning = (
        "Image-derived simulation - not scanner raw data."
        if source_kind == "image-derived-simulation"
        else "Synthetic phantom simulation - not scanner raw data."
    )
    return ModelRunResult(
        bundle_id=runtime.record.bundle_id,
        device=runtime.device,
        elapsed_seconds=elapsed,
        outputs=MappingProxyType({"complex_channels": complex_channels, "magnitude": magnitude}),
        output_channels=("real", "imaginary"),
        fallback_reason=runtime.fallback_reason,
        warnings=(source_warning, "Education and research only; not for diagnosis."),
    )


def _normalize_brats(volume: np.ndarray) -> np.ndarray:
    array = _numeric_finite(volume, name="volume")
    if array.ndim == 4 and array.shape[0] == 4:
        array = array[None]
    if array.ndim != 5 or array.shape[0:2] != (1, 4):
        raise ValueError("BraTS patch must have shape 4xDxHxW or 1x4xDxHxW")
    if any(size < 16 or size % 8 for size in array.shape[-3:]):
        raise ValueError("BraTS patch dimensions must be at least 16 and divisible by 8")
    normalized = np.array(array, dtype=np.float32, copy=True)
    for channel_index, channel_name in enumerate(("T1ce", "T1", "T2", "FLAIR")):
        channel = normalized[0, channel_index]
        foreground = channel != 0
        if not foreground.any():
            raise ValueError(f"BraTS {channel_name} channel has no nonzero voxels")
        values = channel[foreground]
        deviation = float(values.std(dtype=np.float64))
        if deviation <= np.finfo(np.float32).eps:
            raise ValueError(f"BraTS {channel_name} nonzero voxels have zero variance")
        channel[foreground] = (values - float(values.mean(dtype=np.float64))) / deviation
    return np.ascontiguousarray(normalized)


def run_monai_brats_patch(
    volume: np.ndarray,
    *,
    device: DeviceRequest = "auto",
    threshold: float = 0.5,
) -> ModelRunResult:
    """Run one aligned MRI patch and expose WT/TC/ET rather than native order."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    normalized = _normalize_brats(volume)
    runtime = load_bundled_model("monai-brats-segmentation", device=device)
    (raw,), elapsed = runtime.run(normalized)
    logits_native = raw.numpy()[0]
    probabilities_native = np.empty_like(logits_native, dtype=np.float32)
    positive = logits_native >= 0
    probabilities_native[positive] = 1.0 / (1.0 + np.exp(-logits_native[positive]))
    negative_exponential = np.exp(logits_native[~positive])
    probabilities_native[~positive] = negative_exponential / (1.0 + negative_exponential)
    # Upstream order is TC, WT, ET. Product order is visibly WT, TC, ET.
    probabilities = _readonly(probabilities_native[[1, 0, 2]])
    masks = np.array(probabilities >= threshold, dtype=bool, copy=True)
    masks.setflags(write=False)
    return ModelRunResult(
        bundle_id=runtime.record.bundle_id,
        device=runtime.device,
        elapsed_seconds=elapsed,
        outputs=MappingProxyType({"probabilities": probabilities, "masks": masks}),
        output_channels=("WT", "TC", "ET"),
        fallback_reason=runtime.fallback_reason,
        warnings=(
            "Patch-level inference; preserve the verified source affine when mapping outputs.",
            "Trained on BraTS 2018 glioma MRI; not a general brain-lesion model.",
            "Education and research only; not for diagnosis.",
        ),
    )
