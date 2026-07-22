"""Command-line integrity and smoke verification for installed model bundles."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from .bundled import (
    BUNDLED_MODEL_IDS,
    load_bundled_model,
    validate_bundle_golden,
    verify_all_bundles,
)


def _summary(value: Any) -> dict[str, Any]:
    array = value.detach().cpu().numpy()
    return {
        "shape": list(array.shape),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean(dtype=np.float64)),
        "finite": bool(np.isfinite(array).all()),
    }


def _inputs(bundle_id: str) -> tuple[np.ndarray, ...]:
    if bundle_id == "deepinv-mri-modl":
        kspace = np.zeros((1, 2, 32, 32), dtype=np.float32)
        mask = np.zeros_like(kspace)
        mask[..., ::4, :] = 1.0
        return kspace, mask
    if bundle_id == "dival-lodopab-fbpunet":
        return (np.zeros((1, 1, 362, 362), dtype=np.float32),)
    if bundle_id == "monai-brats-segmentation":
        return (np.zeros((1, 4, 32, 32, 32), dtype=np.float32),)
    raise AssertionError(f"unhandled reviewed bundle: {bundle_id}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="also execute fixed synthetic inputs")
    parser.add_argument(
        "--golden",
        action="store_true",
        help="execute and compare every packaged deterministic golden reference",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)

    records = verify_all_bundles()
    report: dict[str, Any] = {
        "integrity": "passed",
        "payload_bytes": sum(
            record.artifact_size_bytes + record.golden_size_bytes for record in records
        ),
        "formats": {record.bundle_id: record.artifact_format for record in records},
        "bundles": {},
    }
    if args.golden:
        report["golden"] = {}
        for bundle_id in BUNDLED_MODEL_IDS:
            result = validate_bundle_golden(bundle_id, device=args.device)
            report["golden"][bundle_id] = {
                "device": result.device,
                "fallback_reason": result.fallback_reason,
                "elapsed_seconds": result.elapsed_seconds,
                "maximum_absolute_error": result.maximum_absolute_error,
                "maximum_relative_error": result.maximum_relative_error,
            }
    if args.smoke:
        for bundle_id in BUNDLED_MODEL_IDS:
            loaded = load_bundled_model(bundle_id, device=args.device)
            outputs, elapsed = loaded.run(*_inputs(bundle_id))
            report["bundles"][bundle_id] = {
                "device": loaded.device,
                "fallback_reason": loaded.fallback_reason,
                "elapsed_seconds": elapsed,
                "outputs": [_summary(output) for output in outputs],
            }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
