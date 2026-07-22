"""Background-safe synthetic demonstrations for the reviewed model bundles.

The demonstrations deliberately generate their inputs in memory.  They do not
scan user files, fetch network resources, or imply that simulated acquisition
data is equivalent to scanner-native data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..algorithms import (
    FilteredBackProjection,
    ReconstructionRequest,
    ReconstructionSourceKind,
    generate_ct_phantom,
    generate_sinogram,
)
from ..models import run_deepinv_mri_modl, run_dival_fbp_unet
from ..runtime import TaskContext

DeviceRequest = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True, slots=True)
class TorchDeviceStatus:
    """Runtime device facts discovered without changing global Torch settings."""

    torch_version: str
    cuda_available: bool
    device_name: str | None
    total_vram_gib: float | None


@dataclass(frozen=True, slots=True)
class BundledDemoResult:
    """Three comparable views plus visible runtime and scientific provenance."""

    bundle_id: str
    source: np.ndarray
    prepared_input: np.ndarray
    output: np.ndarray
    device: str
    elapsed_seconds: float
    fallback_reason: str | None
    warnings: tuple[str, ...]


def inspect_torch_device(context: TaskContext) -> TorchDeviceStatus:
    """Probe the optional PyTorch/CUDA runtime on a worker thread."""

    context.report_progress(0.1, message="Loading the local PyTorch runtime")
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "PyTorch is unavailable. Install the documented GPU or CPU dependency first."
        ) from exc
    context.raise_if_cancelled()
    cuda_available = bool(torch.cuda.is_available())
    device_name: str | None = None
    total_vram_gib: float | None = None
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        device_name = str(properties.name)
        total_vram_gib = float(properties.total_memory) / (1024.0**3)
    context.report_progress(1.0, message="Local device check complete")
    return TorchDeviceStatus(
        torch_version=str(torch.__version__),
        cuda_available=cuda_available,
        device_name=device_name,
        total_vram_gib=total_vram_gib,
    )


def run_dival_synthetic_demo(
    context: TaskContext,
    *,
    device: DeviceRequest = "auto",
) -> BundledDemoResult:
    """Run an explicitly out-of-domain CT stress test through the DIVal adapter."""

    context.report_progress(0.02, message="Generating a synthetic attenuation phantom")
    phantom = generate_ct_phantom(362)
    context.raise_if_cancelled()
    sinogram = generate_sinogram(
        phantom.array,
        projection_count=1000,
        angle_range=180,
        circle=True,
        source_kind=ReconstructionSourceKind.IMAGE_DERIVED_SIMULATION,
        cancel=lambda: context.cancelled,
        progress=lambda fraction, message: context.report_progress(
            0.03 + 0.22 * fraction,
            message=message,
        ),
    )
    request = ReconstructionRequest.from_sinogram_result(sinogram, output_size=362)
    reconstruction = FilteredBackProjection("hann").reconstruct(
        request,
        cancel=lambda: context.cancelled,
        progress=lambda fraction, message: context.report_progress(
            0.25 + 0.20 * fraction,
            message=message,
        ),
    )
    context.report_progress(0.48, message="Running the reviewed DIVal task adapter")
    model_result = run_dival_fbp_unet(
        np.asarray(reconstruction.image, dtype=np.float32),
        device=device,
    )
    context.raise_if_cancelled()
    context.report_progress(1.0, message="Synthetic DIVal stress test complete")
    return BundledDemoResult(
        bundle_id=model_result.bundle_id,
        source=_readonly(phantom.array),
        prepared_input=_readonly(reconstruction.image),
        output=_readonly(model_result.outputs["reconstruction"]),
        device=model_result.device,
        elapsed_seconds=model_result.elapsed_seconds,
        fallback_reason=model_result.fallback_reason,
        warnings=(
            "Out-of-domain stress test: this self-generated image-domain phantom is not an "
            "official LoDoPaB observation.",
            "Pipeline: synthetic attenuation map -> Radon transform -> Hann FBP -> reviewed "
            "DIVal task adapter.",
            *model_result.warnings,
        ),
    )


def run_deepinv_synthetic_demo(
    context: TaskContext,
    *,
    device: DeviceRequest = "auto",
) -> BundledDemoResult:
    """Run a deterministic single-coil MRI simulation through the DeepInverse adapter."""

    context.report_progress(0.05, message="Generating a synthetic image phantom")
    phantom = np.asarray(generate_ct_phantom(128).array, dtype=np.float32)
    maximum = float(np.max(phantom))
    if maximum > 0.0:
        phantom = phantom / maximum
    context.raise_if_cancelled()

    context.report_progress(0.18, message="Simulating single-coil k-space and sampling mask")
    full_kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(phantom), norm="ortho")).astype(
        np.complex64
    )
    mask = np.zeros(phantom.shape, dtype=np.float32)
    mask[::4, :] = 1.0
    centre = phantom.shape[0] // 2
    mask[centre - 4 : centre + 4, :] = 1.0
    sampled_kspace = full_kspace * mask
    zero_filled = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(sampled_kspace), norm="ortho"))
    kspace_channels = np.stack(
        (sampled_kspace.real, sampled_kspace.imag),
        axis=0,
    ).astype(np.float32)
    context.raise_if_cancelled()

    context.report_progress(0.32, message="Running the reviewed DeepInverse task adapter")
    model_result = run_deepinv_mri_modl(
        kspace_channels,
        mask,
        source_kind="synthetic-phantom",
        device=device,
    )
    context.raise_if_cancelled()
    context.report_progress(1.0, message="Synthetic MRI demonstration complete")
    return BundledDemoResult(
        bundle_id=model_result.bundle_id,
        source=_readonly(phantom),
        prepared_input=_readonly(np.abs(zero_filled)),
        output=_readonly(model_result.outputs["magnitude"]),
        device=model_result.device,
        elapsed_seconds=model_result.elapsed_seconds,
        fallback_reason=model_result.fallback_reason,
        warnings=(
            "Synthetic phantom -> single-coil FFT -> deterministic undersampling mask -> "
            "reviewed DeepInverse task adapter.",
            "The k-space is simulated from an image phantom; it is not scanner raw data.",
            *model_result.warnings,
        ),
    )


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.float32, copy=True)
    result.setflags(write=False)
    return result


__all__ = [
    "BundledDemoResult",
    "TorchDeviceStatus",
    "inspect_torch_device",
    "run_deepinv_synthetic_demo",
    "run_dival_synthetic_demo",
]
