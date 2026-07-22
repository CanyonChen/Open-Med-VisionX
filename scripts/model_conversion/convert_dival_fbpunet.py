"""Convert the pinned DIVal LoDoPaB FBP-U-Net checkpoint to safe NPZ."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from workbench.models._adapters import DivalFBPUNet, load_npz_model
from workbench.models._safe_npz import write_deterministic_npz

SOURCE_SIZE = 2_485_382
SOURCE_SHA256 = "990b366a5053485093e2b9876f1503d92fcd786b44f8970ca6f5d9e4c1dfdf23"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_source(path: Path) -> None:
    if path.stat().st_size != SOURCE_SIZE or _sha256(path) != SOURCE_SHA256:
        raise ValueError("DIVal source checkpoint does not match the reviewed artifact")


def _review_state(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("DIVal checkpoint must contain a tensor state dictionary")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in checkpoint.items()
    ):
        raise TypeError("DIVal state_dict must contain only string-to-tensor entries")
    clean_state = {
        key.removeprefix("module."): value.detach().cpu() for key, value in checkpoint.items()
    }
    reviewed_model = DivalFBPUNet().eval()
    expected = reviewed_model.state_dict()
    if set(clean_state) != set(expected):
        raise ValueError("DIVal state_dict keys do not match the pinned source architecture")
    for name, expected_tensor in expected.items():
        tensor = clean_state[name]
        if tensor.shape != expected_tensor.shape or tensor.dtype != expected_tensor.dtype:
            raise ValueError(f"DIVal tensor contract mismatch for {name}")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"DIVal tensor contains non-finite values: {name}")
    reviewed_model.load_state_dict(clean_state, strict=True)
    return clean_state


def _golden_input() -> torch.Tensor:
    coordinate = torch.linspace(-1.0, 1.0, 362, dtype=torch.float32)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    image = (
        0.7 * torch.exp(-5.0 * (xx.square() + yy.square()))
        + 0.2 * torch.exp(-30.0 * ((xx - 0.35).square() + (yy + 0.2).square()))
        + 0.03 * torch.cos(9.0 * xx) * torch.sin(7.0 * yy)
    )
    return image.unsqueeze(0).unsqueeze(0)


def convert(
    source: Path,
    destination: Path,
    golden_destination: Path | None = None,
) -> dict[str, tuple[int, str]]:
    _check_source(source)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    state = _review_state(checkpoint)
    source_model = DivalFBPUNet().eval()
    source_model.load_state_dict(state, strict=True)
    golden_input = _golden_input()
    with torch.inference_mode():
        source_output = source_model(golden_input)

    arrays = {name: tensor.numpy() for name, tensor in state.items()}
    write_deterministic_npz(destination, arrays)
    reloaded = load_npz_model("dival-lodopab-fbpunet", destination, "cpu")
    with torch.inference_mode():
        reloaded_output = reloaded(golden_input)
    torch.testing.assert_close(reloaded_output, source_output, rtol=0.0, atol=0.0)

    golden_destination = golden_destination or destination.with_name("golden.npz")
    write_deterministic_npz(
        golden_destination,
        {
            "input.fbp": golden_input.numpy(),
            "output.reconstruction": source_output.numpy(),
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
