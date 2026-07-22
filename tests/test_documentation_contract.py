from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]

USER_DOCUMENT_PAIRS = (
    (ROOT / "README.md", ROOT / "docs" / "README.zh-CN.md"),
    (ROOT / "docs" / "QUICKSTART.md", ROOT / "docs" / "QUICKSTART.zh-CN.md"),
    (ROOT / "docs" / "USER_GUIDE.md", ROOT / "docs" / "USER_GUIDE.zh-CN.md"),
    (
        ROOT / "docs" / "TROUBLESHOOTING.md",
        ROOT / "docs" / "TROUBLESHOOTING.zh-CN.md",
    ),
    (
        ROOT / "docs" / "TEACHING_CURRICULUM.md",
        ROOT / "docs" / "TEACHING_CURRICULUM.zh-CN.md",
    ),
    (ROOT / "docs" / "LLM_SECURITY.md", ROOT / "docs" / "LLM_SECURITY.zh-CN.md"),
    (ROOT / "docs" / "MODEL_BUNDLES.md", ROOT / "docs" / "MODEL_BUNDLES.zh-CN.md"),
)

USER_GUIDE_NAMES = (
    "QUICKSTART.md",
    "USER_GUIDE.md",
    "TROUBLESHOOTING.md",
    "TEACHING_CURRICULUM.md",
    "LLM_SECURITY.md",
    "MODEL_BUNDLES.md",
)

BUNDLED_MODEL_CARDS = (
    ROOT
    / "src"
    / "workbench"
    / "resources"
    / "model_bundles"
    / "monai-brats-segmentation"
    / "MODEL_CARD.md",
    ROOT
    / "src"
    / "workbench"
    / "resources"
    / "model_bundles"
    / "dival-lodopab-fbpunet"
    / "MODEL_CARD.md",
    ROOT
    / "src"
    / "workbench"
    / "resources"
    / "model_bundles"
    / "deepinv-mri-modl"
    / "MODEL_CARD.md",
)

FORBIDDEN_USER_TERM = "\u5b66\u751f"
GPU_REQUIREMENT_COMMAND = "python -m pip install -r requirements/torch-cu130.txt"
MODEL_PAYLOAD_DIGESTS = (
    "702658708828d13135228e32fef980ba1048e200f5c2fa4ebf54fa12d653f8ab",
    "2b709a7c16aedd65110aaf929bb2c6cc35db1c94d9fe01b751a29b06634d29af",
    "8b18fd2a88355ddec043ae7c737ddf3321424e2ba52102869d3dbaf6bf68504c",
    "6789c93592ab6cfd3d4924e6d077ce8966d0e7fd7bb0b7f1a7305d66f742a3df",
    "729980a0bd9347bf2397701eb329e12517918dc282a2d09c40458e95b24ceed9",
    "d7982ada82f56b28615ed6ad170641ee1f3f0cb6a819285598c0380efa957e45",
)
TEACHING_CASE_DIGEST = "b323cdef2529927336069b3385605d1049117fe69e59583072861fa573493846"

MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)", re.MULTILINE)
HTML_LINK = re.compile(r"(?:href|src)=[\"']([^\"']+)[\"']", re.IGNORECASE)
DOCUMENTATION_EXCLUDED_PARTS = {
    ".agents",
    ".codex",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".tmp_ui_qa",
    "__pycache__",
    "build",
}


def repository_documentation() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.md")
        if not DOCUMENTATION_EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)
    )


def terminology_contract_files() -> list[Path]:
    ui_sources = (ROOT / "src" / "workbench" / "ui").rglob("*.py")
    return sorted({*repository_documentation(), *ui_sources})


def relative_link(source: Path, target: Path) -> str:
    return os.path.relpath(target, source.parent).replace(os.sep, "/")


@pytest.mark.parametrize(("english", "chinese"), USER_DOCUMENT_PAIRS)
def test_user_document_pair_is_reciprocal_and_structurally_aligned(
    english: Path,
    chinese: Path,
) -> None:
    assert english.is_file(), f"missing English user document: {english.relative_to(ROOT)}"
    assert chinese.is_file(), f"missing Chinese user document: {chinese.relative_to(ROOT)}"

    english_text = english.read_text(encoding="utf-8")
    chinese_text = chinese.read_text(encoding="utf-8")
    assert relative_link(english, chinese) in english_text
    assert relative_link(chinese, english) in chinese_text

    english_sections = re.findall(r"^##\s+", english_text, flags=re.MULTILINE)
    chinese_sections = re.findall(r"^##\s+", chinese_text, flags=re.MULTILINE)
    assert len(english_sections) == len(chinese_sections)


def test_public_quickstarts_use_the_named_conda_environment() -> None:
    required = (
        ROOT / "README.md",
        ROOT / "docs" / "README.zh-CN.md",
        ROOT / "docs" / "QUICKSTART.md",
        ROOT / "docs" / "QUICKSTART.zh-CN.md",
    )
    for path in required:
        assert "conda activate openmedvisionx" in path.read_text(encoding="utf-8")

    all_user_docs = {path for pair in USER_DOCUMENT_PAIRS for path in pair}
    for path in all_user_docs:
        text = path.read_text(encoding="utf-8")
        environment_names = re.findall(r"\bconda\s+activate\s+([A-Za-z0-9_.-]+)", text)
        assert set(environment_names) <= {"openmedvisionx"}, (
            f"unexpected Conda environment in {path.relative_to(ROOT)}: {environment_names}"
        )


def test_public_setup_recommends_the_supported_python_version() -> None:
    required = (
        ROOT / "README.md",
        ROOT / "docs" / "README.zh-CN.md",
        ROOT / "docs" / "QUICKSTART.md",
        ROOT / "docs" / "QUICKSTART.zh-CN.md",
        ROOT / "docs" / "TROUBLESHOOTING.md",
        ROOT / "docs" / "TROUBLESHOOTING.zh-CN.md",
    )
    for path in required:
        assert "Python 3.11" in path.read_text(encoding="utf-8")


def test_public_window_size_is_expressed_in_logical_and_physical_pixels() -> None:
    required = (
        ROOT / "README.md",
        ROOT / "docs" / "README.zh-CN.md",
        ROOT / "docs" / "QUICKSTART.md",
        ROOT / "docs" / "QUICKSTART.zh-CN.md",
    )
    for path in required:
        text = path.read_text(encoding="utf-8")
        for value in ("900", "620", "150%", "1350", "930", "1024", "680"):
            assert value in text

    english = (ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (ROOT / "docs" / "README.zh-CN.md").read_text(encoding="utf-8")
    assert "logical pixels" in english and "physical pixels" in english
    assert "逻辑像素" in chinese and "物理像素" in chinese


def test_removed_maintenance_documents_are_absent_and_unlinked() -> None:
    removed = ("CODE_OF_CONDUCT.md", "CHANGELOG.md", "CONTRIBUTING.md")
    assert all(not (ROOT / name).exists() for name in removed)
    overviews = (
        (ROOT / "README.md").read_text(encoding="utf-8"),
        (ROOT / "docs" / "README.zh-CN.md").read_text(encoding="utf-8"),
        (ROOT / "MANIFEST.in").read_text(encoding="utf-8"),
    )
    for name in removed:
        assert all(name not in text for text in overviews)


def test_project_overviews_link_the_classified_bilingual_guides() -> None:
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (ROOT / "docs" / "README.zh-CN.md").read_text(encoding="utf-8")
    for name in USER_GUIDE_NAMES:
        assert f"docs/{name}" in english
        chinese_name = name.replace(".md", ".zh-CN.md")
        assert chinese_name in chinese


def test_gpu_installation_is_explicitly_machine_scoped() -> None:
    detailed_guides = (
        ROOT / "docs" / "QUICKSTART.md",
        ROOT / "docs" / "QUICKSTART.zh-CN.md",
        ROOT / "docs" / "MODEL_BUNDLES.md",
        ROOT / "docs" / "MODEL_BUNDLES.zh-CN.md",
    )
    for path in detailed_guides:
        text = path.read_text(encoding="utf-8")
        assert GPU_REQUIREMENT_COMMAND in text
        assert re.search(r"RTX\s+4060 Laptop", text)

    # The project overviews stay focused on a portable five-minute start.
    for path in (ROOT / "README.md", ROOT / "docs" / "README.zh-CN.md"):
        text = path.read_text(encoding="utf-8")
        assert GPU_REQUIREMENT_COMMAND not in text
        assert not re.search(r"RTX\s+4060 Laptop", text)

    requirement = (ROOT / "requirements" / "torch-cu130.txt").read_text(encoding="utf-8")
    assert "https://download.pytorch.org/whl/cu130" in requirement
    assert "torch==2.13.0+cu130" in requirement


def test_three_offline_model_cards_are_bilingual_and_linked() -> None:
    assert len(BUNDLED_MODEL_CARDS) == 3
    linked_guides = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "docs" / "QUICKSTART.md",
            ROOT / "docs" / "QUICKSTART.zh-CN.md",
            ROOT / "docs" / "USER_GUIDE.md",
            ROOT / "docs" / "USER_GUIDE.zh-CN.md",
        )
    )
    for card in BUNDLED_MODEL_CARDS:
        assert card.is_file()
        text = card.read_text(encoding="utf-8")
        assert "## English" in text
        assert "## 中文" in text
        relative_from_docs = card.relative_to(ROOT).as_posix()
        assert f"../{relative_from_docs}" in linked_guides


def test_model_and_teaching_case_guides_publish_frozen_integrity_facts() -> None:
    guides = (
        ROOT / "docs" / "MODEL_BUNDLES.md",
        ROOT / "docs" / "MODEL_BUNDLES.zh-CN.md",
    )
    for path in guides:
        text = path.read_text(encoding="utf-8")
        assert "lodopab-ct-test-03456" in text
        assert TEACHING_CASE_DIGEST in text
        assert "1,912,858" in text
        assert "CC BY 4.0" in text
        assert "allow_pickle=False" in text
        assert "22,502,245" in text
        for digest in MODEL_PAYLOAD_DIGESTS:
            assert digest in text


def test_github_introduction_is_bilingual() -> None:
    text = (ROOT / "docs" / "INTRODUCTION.md").read_text(encoding="utf-8")
    assert "## English" in text
    assert "## 简体中文" in text
    assert "OpenMedVisionX" in text
    assert "LoDoPaB-CT" in text
    assert "DeepInverse" in text
    assert "MONAI" in text
    assert "four co-registered" in text
    assert "四个完成配准" in text


def test_structured_artifact_guides_match_the_desktop_review_boundary() -> None:
    english_paths = (
        ROOT / "README.md",
        ROOT / "docs" / "USER_GUIDE.md",
        ROOT / "docs" / "LLM_SECURITY.md",
        ROOT / "docs" / "INTRODUCTION.md",
    )
    chinese_paths = (
        ROOT / "docs" / "README.zh-CN.md",
        ROOT / "docs" / "USER_GUIDE.zh-CN.md",
        ROOT / "docs" / "LLM_SECURITY.zh-CN.md",
        ROOT / "docs" / "INTRODUCTION.md",
    )
    for path in english_paths:
        assert "Structured artifacts" in path.read_text(encoding="utf-8")
    for path in chinese_paths:
        assert "结构化产物" in path.read_text(encoding="utf-8")

    english_security = (ROOT / "docs" / "LLM_SECURITY.md").read_text(encoding="utf-8")
    chinese_security = (ROOT / "docs" / "LLM_SECURITY.zh-CN.md").read_text(encoding="utf-8")
    for artifact_type in (
        "text",
        "class_scores",
        "labels",
        "mask_2d",
        "mask_3d",
        "reconstructed_image",
        "reconstructed_volume",
    ):
        assert artifact_type in english_security
        assert artifact_type in chinese_security
    assert re.search(r"does not\s+create a layer", english_security)
    assert re.search(r"not automatically\s+adapted into typed artifacts", english_security)
    assert "no desktop artifact importer" in english_security
    assert "不会创建图层" in chinese_security
    assert "不会自动转换为类型化产物" in chinese_security
    assert "没有通往该预览的桌面产物导入器" in chinese_security

    english_guide = (ROOT / "docs" / "USER_GUIDE.md").read_text(encoding="utf-8")
    chinese_guide = (ROOT / "docs" / "USER_GUIDE.zh-CN.md").read_text(encoding="utf-8")
    assert "Teaching chat" in english_guide
    assert "Confirm artifact" in english_guide and "Reject artifact" in english_guide
    assert "教学对话" in chinese_guide
    assert "确认产物" in chinese_guide and "拒绝产物" in chinese_guide


def test_bundled_model_guides_match_desktop_capability_boundaries() -> None:
    detailed_pairs = (
        (
            ROOT / "docs" / "QUICKSTART.md",
            ROOT / "docs" / "QUICKSTART.zh-CN.md",
        ),
        (
            ROOT / "docs" / "USER_GUIDE.md",
            ROOT / "docs" / "USER_GUIDE.zh-CN.md",
        ),
        (
            ROOT / "docs" / "MODEL_BUNDLES.md",
            ROOT / "docs" / "MODEL_BUNDLES.zh-CN.md",
        ),
    )
    for english_path, chinese_path in detailed_pairs:
        english = english_path.read_text(encoding="utf-8")
        chinese = chinese_path.read_text(encoding="utf-8")

        assert "DIVal" in english and "LoDoPaB" in english
        assert "DeepInverse" in english and "synthetic MRI" in english
        assert "MONAI" in english and "disabled" in english
        assert re.search(r"external(?:-|\s+)manifest", english, flags=re.IGNORECASE)

        assert "DIVal" in chinese and "LoDoPaB" in chinese
        assert "DeepInverse" in chinese and "合成 MRI" in chinese
        assert "MONAI" in chinese and "禁用" in chinese
        assert "外部清单" in chinese


def test_documentation_uses_user_facing_terminology() -> None:
    violations: list[str] = []
    for path in terminology_contract_files():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if FORBIDDEN_USER_TERM in line:
                violations.append(f"{path.relative_to(ROOT).as_posix()}:{line_number}")
    assert not violations, "forbidden user-facing term found at " + ", ".join(violations)


def test_repository_has_no_ci_workflow_files() -> None:
    workflow_root = ROOT / ".github" / "workflows"
    assert not workflow_root.exists() or not any(
        path.is_file() for path in workflow_root.rglob("*")
    )


def local_link_targets(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    candidates = [*MARKDOWN_LINK.findall(text), *HTML_LINK.findall(text)]
    local_targets: list[str] = []
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate.startswith("<") and candidate.endswith(">"):
            candidate = candidate[1:-1]
        # Markdown permits an optional quoted title after the destination.
        candidate = re.split(r"\s+[\"']", candidate, maxsplit=1)[0]
        parsed = urlsplit(candidate)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        local_targets.append(unquote(parsed.path))
    return local_targets


def test_local_documentation_links_exist() -> None:
    missing: list[str] = []
    for path in repository_documentation():
        for target in local_link_targets(path):
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                source = path.relative_to(ROOT).as_posix()
                missing.append(f"{source} -> {target}")
    assert not missing, "broken local documentation links:\n" + "\n".join(missing)
