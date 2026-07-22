from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import numpy as np

import workbench.services.bundled_models as demos
from workbench.models import ModelRunResult
from workbench.runtime import TaskRunner


def _model_result(bundle_id: str, key: str, value: np.ndarray) -> ModelRunResult:
    return ModelRunResult(
        bundle_id=bundle_id,
        device="cpu",
        elapsed_seconds=0.01,
        outputs=MappingProxyType({key: value}),
        output_channels=(key,),
        fallback_reason=None,
        warnings=("Education and research only; not for diagnosis.",),
    )


def test_dival_demo_builds_attenuation_radon_hann_fbp_before_task_adapter(
    monkeypatch,
) -> None:
    captured = SimpleNamespace()

    def fake_adapter(value: np.ndarray, *, device: str) -> ModelRunResult:
        captured.input = np.array(value, copy=True)
        captured.device = device
        return _model_result("dival-lodopab-fbpunet", "reconstruction", value)

    monkeypatch.setattr(demos, "run_dival_fbp_unet", fake_adapter)
    with TaskRunner(max_workers=1) as runner:
        result = runner.submit(demos.run_dival_synthetic_demo, device="auto").result(timeout=30)

    assert captured.input.shape == (362, 362)
    assert np.isfinite(captured.input).all()
    assert captured.device == "auto"
    assert result.source.shape == (362, 362)
    assert result.prepared_input.shape == (362, 362)
    assert any("out-of-domain stress test" in warning.lower() for warning in result.warnings)
    assert any("Radon transform -> Hann FBP" in warning for warning in result.warnings)
    assert any("not an official LoDoPaB observation" in warning for warning in result.warnings)


def test_deepinv_demo_builds_single_coil_masked_kspace_before_task_adapter(
    monkeypatch,
) -> None:
    captured = SimpleNamespace()

    def fake_adapter(
        kspace: np.ndarray,
        mask: np.ndarray,
        *,
        source_kind: str,
        device: str,
    ) -> ModelRunResult:
        captured.kspace = np.array(kspace, copy=True)
        captured.mask = np.array(mask, copy=True)
        captured.source_kind = source_kind
        captured.device = device
        magnitude = np.hypot(kspace[0], kspace[1]).astype(np.float32)
        return _model_result("deepinv-mri-modl", "magnitude", magnitude)

    monkeypatch.setattr(demos, "run_deepinv_mri_modl", fake_adapter)
    with TaskRunner(max_workers=1) as runner:
        result = runner.submit(demos.run_deepinv_synthetic_demo, device="cpu").result(timeout=10)

    assert captured.kspace.shape == (2, 128, 128)
    assert captured.mask.shape == (128, 128)
    assert set(np.unique(captured.mask)) == {0.0, 1.0}
    assert np.all(captured.kspace[:, captured.mask == 0] == 0)
    assert captured.source_kind == "synthetic-phantom"
    assert captured.device == "cpu"
    assert result.source.shape == (128, 128)
    assert result.prepared_input.shape == (128, 128)
    assert any("not scanner raw data" in warning for warning in result.warnings)
