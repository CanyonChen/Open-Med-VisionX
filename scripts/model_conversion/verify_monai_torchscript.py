"""Verify and stage the pinned upstream MONAI BraTS TorchScript artifact."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import torch

from workbench.models._safe_npz import write_deterministic_npz

SOURCE_SIZE = 18_911_784
SOURCE_SHA256 = "729980a0bd9347bf2397701eb329e12517918dc282a2d09c40458e95b24ceed9"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_and_stage(
    source: Path,
    destination: Path,
    golden_destination: Path | None = None,
) -> dict[str, tuple[int, str]]:
    if source.stat().st_size != SOURCE_SIZE or _sha256(source) != SOURCE_SHA256:
        raise ValueError("MONAI source TorchScript does not match the reviewed artifact")
    model = torch.jit.load(str(source), map_location="cpu").eval()
    with torch.inference_mode():
        example = torch.zeros((1, 4, 32, 32, 32), dtype=torch.float32)
        output = model(example)
    if tuple(output.shape) != (1, 3, 32, 32, 32) or not torch.isfinite(output).all():
        raise ValueError("MONAI TorchScript failed the reviewed input/output contract")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    golden_destination = golden_destination or destination.with_name("golden.npz")
    write_deterministic_npz(
        golden_destination,
        {
            "input.modalities": example.numpy(),
            "output.logits": output.detach().cpu().numpy(),
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
    results = verify_and_stage(args.source, args.destination, args.golden)
    for label, (size, digest) in results.items():
        print(f"{label}.size_bytes={size}")
        print(f"{label}.sha256={digest}")


if __name__ == "__main__":
    main()
