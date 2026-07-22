"""Convert the pinned DeepInverse MRI-tour checkpoint to safe numeric NPZ.

This maintainer-only command is the sole place that deserializes the reviewed
upstream pickle.  It uses ``weights_only=True``, rebuilds the official
DeepInverse 0.4.1 model, and proves the project adapter is numerically
equivalent on the same deterministic input before writing release metadata.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from workbench.models._adapters import DeepInvMRIMoDL, load_npz_model
from workbench.models._safe_npz import write_deterministic_npz

SOURCE_SIZE = 38_274
SOURCE_SHA256 = "e756b303917babeca8c32896597c9e4989fdd6a9792de3a79e9965f1f5cdbaff"
_STATE_KEYS = {
    "init_params_algo.g_param.0",
    "init_params_algo.g_param.1",
    "init_params_algo.lambda.0",
    "init_params_algo.lambda.1",
    "init_params_algo.stepsize.0",
    "init_params_algo.stepsize.1",
    "params_algo.g_param.0",
    "params_algo.g_param.1",
    "params_algo.lambda.0",
    "params_algo.lambda.1",
    "params_algo.stepsize.0",
    "params_algo.stepsize.1",
    "prior.0.denoiser.in_conv.weight",
    "prior.0.denoiser.in_conv.bias",
    "prior.0.denoiser.out_conv.weight",
    "prior.0.denoiser.out_conv.bias",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_source(path: Path) -> None:
    if path.stat().st_size != SOURCE_SIZE or _sha256(path) != SOURCE_SHA256:
        raise ValueError("DeepInverse source checkpoint does not match the reviewed artifact")


def _official_model(state: dict[str, torch.Tensor]) -> torch.nn.Module:
    try:
        import deepinv
        from deepinv.models import DnCNN, MoDL
    except ImportError as exc:
        raise RuntimeError("DeepInverse 0.4.1 is required for equivalence conversion") from exc
    if deepinv.__version__ != "0.4.1":
        raise RuntimeError(f"DeepInverse 0.4.1 is required, but {deepinv.__version__} is installed")
    model = MoDL(
        DnCNN(
            in_channels=2,
            out_channels=2,
            pretrained=None,
            depth=2,
            nf=64,
        ),
        num_iter=2,
    ).eval()
    model.load_state_dict(state, strict=True)
    return model


def _review_state(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"state_dict", "optimizer"}:
        raise TypeError("DeepInverse checkpoint must contain only state_dict and optimizer")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or set(state) != _STATE_KEYS:
        raise TypeError("DeepInverse state_dict does not match the reviewed key allow-list")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError("DeepInverse state_dict values must all be tensors")
    for prefix in ("g_param", "lambda", "stepsize"):
        for index in range(2):
            initial = state[f"init_params_algo.{prefix}.{index}"]
            current = state[f"params_algo.{prefix}.{index}"]
            if initial.shape or initial.dtype != torch.float32 or not torch.equal(initial, current):
                raise ValueError(f"DeepInverse {prefix}.{index} is not the reviewed scalar")
    for index in range(2):
        if float(state[f"init_params_algo.g_param.{index}"]) != _float32_point_zero_one():
            raise ValueError("DeepInverse denoiser parameter changed from the reviewed value")
        if float(state[f"init_params_algo.lambda.{index}"]) != 1.0:
            raise ValueError("DeepInverse lambda changed from the reviewed value")
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("DeepInverse state_dict contains non-finite values")
    return state


def _float32_point_zero_one() -> float:
    """Return the exact float32 representation stored by the official checkpoint."""

    return float(np.float32(0.01))


def _safe_state(state: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    stepsizes = torch.stack(
        [state["init_params_algo.stepsize.0"], state["init_params_algo.stepsize.1"]]
    )
    return {
        "stepsizes": stepsizes.detach().cpu().numpy(),
        "denoiser.in_conv.weight": state["prior.0.denoiser.in_conv.weight"].detach().cpu().numpy(),
        "denoiser.in_conv.bias": state["prior.0.denoiser.in_conv.bias"].detach().cpu().numpy(),
        "denoiser.out_conv.weight": state["prior.0.denoiser.out_conv.weight"]
        .detach()
        .cpu()
        .numpy(),
        "denoiser.out_conv.bias": state["prior.0.denoiser.out_conv.bias"].detach().cpu().numpy(),
    }


def _golden_input() -> tuple[torch.Tensor, torch.Tensor]:
    coordinate = torch.linspace(-1.0, 1.0, 32, dtype=torch.float32)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    real = torch.exp(-4.0 * (xx.square() + yy.square())) + 0.1 * torch.cos(7.0 * xx)
    imaginary = 0.08 * torch.sin(5.0 * yy) * torch.exp(-2.0 * xx.square())
    image = torch.stack((real, imaginary), dim=0).unsqueeze(0)
    mask = torch.zeros_like(image)
    mask[..., ::4, :] = 1.0
    mask[..., 14:18, :] = 1.0
    return image, mask


def convert(
    source: Path,
    destination: Path,
    golden_destination: Path | None = None,
) -> dict[str, tuple[int, str]]:
    _check_source(source)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    state = _review_state(checkpoint)
    official = _official_model(state)

    try:
        from deepinv.physics import MRI
    except ImportError as exc:
        raise RuntimeError("DeepInverse 0.4.1 is required for equivalence conversion") from exc
    image, mask = _golden_input()
    physics = MRI(mask=mask, device="cpu")
    with torch.inference_mode():
        kspace = physics(image)
        official_output = official(kspace, physics)

    safe_state = _safe_state(state)
    direct_adapter = DeepInvMRIMoDL().eval()
    direct_adapter.load_state_dict(
        {name: torch.from_numpy(array) for name, array in safe_state.items()},
        strict=True,
    )
    with torch.inference_mode():
        direct_output = direct_adapter(kspace, mask)
    torch.testing.assert_close(direct_output, official_output, rtol=1e-6, atol=1e-7)

    write_deterministic_npz(destination, safe_state)
    reloaded = load_npz_model("deepinv-mri-modl", destination, "cpu")
    with torch.inference_mode():
        reloaded_output = reloaded(kspace, mask)
    torch.testing.assert_close(reloaded_output, official_output, rtol=1e-6, atol=1e-7)

    golden_destination = golden_destination or destination.with_name("golden.npz")
    write_deterministic_npz(
        golden_destination,
        {
            "input.kspace": kspace.detach().cpu().numpy(),
            "input.mask": mask.detach().cpu().numpy(),
            "output.reconstruction": official_output.detach().cpu().numpy(),
        },
    )
    return {
        "artifact": (destination.stat().st_size, _sha256(destination)),
        "golden": (golden_destination.stat().st_size, _sha256(golden_destination)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--golden", type=Path)
    args = parser.parse_args()
    results = convert(args.source, args.destination, args.golden)
    for label, (size, digest) in results.items():
        print(f"{label}.size_bytes={size}")
        print(f"{label}.sha256={digest}")


if __name__ == "__main__":
    main()
