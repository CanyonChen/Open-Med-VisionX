"""Trusted PyTorch graphs for the two pickle-free NPZ model bundles.

The class definitions are versioned source code, while the reviewed NPZ files
contain numeric arrays only.  Runtime loading never calls ``torch.load``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ._safe_npz import ArraySpec, load_safe_npz


class _InBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.conv(value)


class _DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.conv(value)


class _UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int) -> None:
        super().__init__()
        self.skip_conv = nn.Sequential(
            nn.Conv2d(out_channels, skip_channels, 1),
            nn.BatchNorm2d(skip_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels + skip_channels),
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, lower: Tensor, skip: Tensor) -> Tensor:
        lower = self.up(lower)
        skip = self.skip_conv(skip)
        target_height = min(lower.shape[-2], skip.shape[-2])
        target_width = min(lower.shape[-1], skip.shape[-1])
        lower_height = (lower.shape[-2] - target_height) // 2
        lower_width = (lower.shape[-1] - target_width) // 2
        skip_height = (skip.shape[-2] - target_height) // 2
        skip_width = (skip.shape[-1] - target_width) // 2
        lower = lower[
            ...,
            lower_height : lower_height + target_height,
            lower_width : lower_width + target_width,
        ]
        skip = skip[
            ...,
            skip_height : skip_height + target_height,
            skip_width : skip_width + target_width,
        ]
        return self.conv(torch.cat((lower, skip), dim=1))


class _OutBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, image: Tensor) -> Tensor:
        return self.conv(image)


class DivalFBPUNet(nn.Module):
    """DIVal LoDoPaB FBP-U-Net at the pinned reviewed source revision."""

    def __init__(self) -> None:
        super().__init__()
        channels = (32, 32, 64, 64, 128)
        self.inc = _InBlock(1, channels[0])
        self.down = nn.ModuleList(
            [_DownBlock(channels[index - 1], channels[index]) for index in range(1, 5)]
        )
        self.up = nn.ModuleList(
            [
                _UpBlock(128, 64, 4),
                _UpBlock(64, 64, 4),
                _UpBlock(64, 32, 4),
                _UpBlock(32, 32, 4),
            ]
        )
        self.outc = _OutBlock(32, 1)

    def forward(self, image: Tensor) -> Tensor:
        x0 = self.inc(image)
        x1 = self.down[0](x0)
        x2 = self.down[1](x1)
        x3 = self.down[2](x2)
        x4 = self.down[3](x3)
        value = self.up[0](x4, x3)
        value = self.up[1](value, x2)
        value = self.up[2](value, x1)
        value = self.up[3](value, x0)
        return self.outc(value)


class _TwoLayerDnCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_conv = nn.Conv2d(2, 64, 3, padding=1)
        self.out_conv = nn.Conv2d(64, 2, 3, padding=1)

    def forward(self, image: Tensor) -> Tensor:
        return self.out_conv(torch.relu(self.in_conv(image))) + image


def _to_complex(value: Tensor) -> Tensor:
    return torch.view_as_complex(value.movedim(1, -1).contiguous())


def _from_complex(value: Tensor) -> Tensor:
    return torch.view_as_real(value).movedim(-1, 1)


def _centered_fft(value: Tensor) -> Tensor:
    complex_value = _to_complex(value)
    complex_value = torch.fft.ifftshift(complex_value, dim=(-2, -1))
    complex_value = torch.fft.fftn(complex_value, dim=(-2, -1), norm="ortho")
    complex_value = torch.fft.fftshift(complex_value, dim=(-2, -1))
    return _from_complex(complex_value)


def _centered_ifft(value: Tensor) -> Tensor:
    complex_value = _to_complex(value)
    complex_value = torch.fft.ifftshift(complex_value, dim=(-2, -1))
    complex_value = torch.fft.ifftn(complex_value, dim=(-2, -1), norm="ortho")
    complex_value = torch.fft.fftshift(complex_value, dim=(-2, -1))
    return _from_complex(complex_value)


class DeepInvMRIMoDL(nn.Module):
    """Exact two-iteration DeepInverse 0.4.1 MRI-tour MoDL inference graph."""

    stepsizes: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.denoiser = _TwoLayerDnCNN()
        self.register_buffer("stepsizes", torch.ones(2, dtype=torch.float32))

    def forward(self, kspace: Tensor, mask: Tensor) -> Tensor:
        adjoint = _centered_ifft(mask * kspace)
        image = adjoint
        for index in range(2):
            inverse_step = torch.reciprocal(self.stepsizes[index])
            right_hand_side = adjoint + inverse_step * image
            scaling = torch.conj(mask) * mask + inverse_step
            data_consistent = _centered_ifft(_centered_fft(right_hand_side) / scaling)
            image = self.denoiser(data_consistent)
        return image


def _state_specs(model: nn.Module) -> dict[str, ArraySpec]:
    specs: dict[str, ArraySpec] = {}
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().numpy()
        specs[name] = ArraySpec(shape=tuple(array.shape), dtype=str(array.dtype))
    return specs


def load_npz_model(bundle_id: str, artifact_path: Path, device: str) -> nn.Module:
    """Build one trusted graph and populate it from a strict numeric NPZ."""

    if bundle_id == "dival-lodopab-fbpunet":
        model: nn.Module = DivalFBPUNet()
    elif bundle_id == "deepinv-mri-modl":
        model = DeepInvMRIMoDL()
    else:  # pragma: no cover - caller is allow-listed
        raise ValueError(f"no NPZ adapter for {bundle_id!r}")

    arrays = load_safe_npz(artifact_path, _state_specs(model))
    state: dict[str, Any] = {name: torch.from_numpy(array) for name, array in arrays.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


__all__ = ["DeepInvMRIMoDL", "DivalFBPUNet", "load_npz_model"]
