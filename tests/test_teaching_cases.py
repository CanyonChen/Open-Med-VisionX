from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pytest

from workbench.cases import (
    BUNDLED_TEACHING_CASE_IDS,
    load_lodopab_case,
    verify_bundled_teaching_cases,
)

_DIVAL_REVISION = "6117cabc55d223f5b62ea43d4a40225270fb6756"


def test_lodopab_release_record_and_budget() -> None:
    records = verify_bundled_teaching_cases()

    assert tuple(record.case_id for record in records) == BUNDLED_TEACHING_CASE_IDS
    assert sum(record.artifact_size_bytes for record in records) <= 2 * 1024 * 1024
    assert records[0].license["identifier"] == "CC-BY-4.0"
    assert records[0].privacy["contains_patient_identifiers"] is False
    assert records[0].sample["index"] == 3456
    assert records[0].sample["source_member_row"] == 0
    assert records[0].source_members["observation"]["downloaded_size_bytes"] == 8 * 1024**2
    assert records[0].source_members["ground_truth"]["downloaded_size_bytes"] == 4 * 1024**2
    assert (
        records[0].source_members["observation"]["downloaded_sha256"]
        == "6379e14b597d244cdf10e7def44e7f14e9f0edb3c63ed325d96a819295e43c4e"
    )


def test_lodopab_fbp_parameter_source_is_immutable_https_and_consistent() -> None:
    record = verify_bundled_teaching_cases()[0]
    parameter_source = record.arrays["fbp"]["derivation"]["parameter_source"]
    parsed = urlsplit(parameter_source)

    assert parsed.scheme == "https"
    assert parsed.netloc == "github.com"
    assert re.fullmatch(r"[0-9a-f]{40}", _DIVAL_REVISION)
    assert f"/blob/{_DIVAL_REVISION}/" in parsed.path
    assert parsed.path.endswith("/reference_params/lodopab/lodopab_fbpunet_hyper_params.json")
    assert not parsed.query
    assert not parsed.fragment

    repository_root = Path(__file__).parents[1]
    evidence = (
        repository_root
        / "src"
        / "workbench"
        / "resources"
        / "teaching_cases"
        / "lodopab-ct"
        / "SOURCE_EVIDENCE.md"
    ).read_text(encoding="utf-8")
    converter = (
        repository_root / "scripts" / "data_conversion" / "extract_lodopab_sample.py"
    ).read_text(encoding="utf-8")
    assert _DIVAL_REVISION in evidence
    assert _DIVAL_REVISION in converter
    assert f"{_DIVAL_REVISION}d" not in evidence
    assert f"{_DIVAL_REVISION}d" not in converter


def test_lodopab_sample_is_safe_finite_and_immutable() -> None:
    case = load_lodopab_case()

    assert case.observation.shape == (1000, 513)
    assert case.fbp.shape == (362, 362)
    assert case.ground_truth.shape == (362, 362)
    assert case.observation.dtype == np.float32
    assert case.fbp.dtype == np.float32
    assert case.ground_truth.dtype == np.float32
    assert np.isfinite(case.observation).all()
    assert np.isfinite(case.fbp).all()
    assert np.isfinite(case.ground_truth).all()
    assert case.record.case_id == "lodopab-ct-test-03456"
    assert case.metadata["sample_index"] == 3456
    assert case.metadata["source_member_row"] == 0
    assert case.metadata["contains_dicom_metadata"] is False
    assert case.metadata["ground_truth_semantics"].endswith("not clinical HU")
    assert (
        case.metadata["array_sha256"]["observation"]
        == "b4f20f395d68e9755e0a250a78206351fee1a95b8c4f4ed4e236f40eb12b0be7"
    )
    assert (
        case.metadata["array_sha256"]["fbp"]
        == "2fbfeb49fc11dc239c9c44226ddc2c611bf96c93b4b89f85d6d0fc61105f1b75"
    )
    assert float(case.observation.mean()) == pytest.approx(0.0289679002, abs=1e-9)
    assert float(case.fbp.mean()) == pytest.approx(0.1626800150, abs=1e-9)
    assert float(case.ground_truth.mean()) == pytest.approx(0.1573498249, abs=1e-9)

    with pytest.raises(ValueError):
        case.ground_truth[0, 0] = 0.0
    with pytest.raises(ValueError):
        case.fbp[0, 0] = 0.0
