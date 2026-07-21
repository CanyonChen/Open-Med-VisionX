#!/usr/bin/env python3
"""Fail when a source checkout contains unsafe or non-source artifacts."""

from __future__ import annotations

import argparse
import gzip
import os
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_FILE_MB = 5.0
MAX_SECRET_SCAN_BYTES = 2 * 1024 * 1024

SKIPPED_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    ".tox",
    ".nox",
}

FORBIDDEN_DIRECTORIES = {
    "__pycache__": "Python bytecode cache",
    ".mypy_cache": "type-checker cache",
    ".pytest_cache": "test cache",
    ".ruff_cache": "linter cache",
    ".test-deps": "local dependency installation",
    "build": "build output",
    "dist": "distribution output",
}

FORBIDDEN_EXACT_NAMES = {
    ".env": "environment file that may contain credentials",
    "credentials.json": "credential file",
    "secrets.json": "secret file",
    "id_dsa": "private key",
    "id_ecdsa": "private key",
    "id_ed25519": "private key",
    "id_rsa": "private key",
}

FORBIDDEN_SUFFIXES = {
    # Medical formats
    ".dcm": "DICOM file",
    ".dicom": "DICOM file",
    ".nii": "NIfTI file",
    ".nii.gz": "compressed NIfTI file",
    ".mha": "medical image file",
    ".mhd": "medical image file",
    ".nrrd": "medical image file",
    ".nhdr": "medical image file",
    # Archives
    ".zip": "archive",
    ".rar": "archive",
    ".7z": "archive",
    ".tar": "archive",
    ".tgz": "archive",
    # Models and weights
    ".pth": "model weights",
    ".pt": "model weights or TorchScript model",
    ".ckpt": "model checkpoint",
    ".safetensors": "model weights",
    ".onnx": "serialized ONNX model",
    ".torchscript": "serialized TorchScript model",
    ".weights": "model weights",
    ".h5": "serialized model or data artifact",
    ".hdf5": "serialized model or data artifact",
    ".pb": "serialized model",
    ".tflite": "serialized model",
    ".engine": "accelerator engine",
    ".plan": "accelerator engine",
    ".trt": "accelerator engine",
    ".joblib": "serialized Python artifact",
    ".pkl": "serialized Python artifact",
    # Executables and compiled output
    ".exe": "executable",
    ".msi": "installer",
    ".dll": "compiled library",
    ".pyd": "compiled Python extension",
    ".so": "compiled library",
    ".dylib": "compiled library",
    ".pyc": "Python bytecode",
    ".class": "compiled bytecode",
    # Credential containers
    ".pem": "private key or certificate bundle",
    ".key": "private key",
    ".p12": "credential container",
    ".pfx": "credential container",
}

TEXT_SUFFIXES = {
    "",
    ".bat",
    ".cfg",
    ".cmd",
    ".css",
    ".gitignore",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".ts",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    "OpenAI-style API key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "Anthropic API key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    "GitHub token": re.compile(r"\bgh[oprsu]_[A-Za-z0-9]{30,}\b"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "Google API key": re.compile(r"\bAIza[A-Za-z0-9_-]{30,}\b"),
}

GENERIC_SECRET_ASSIGNMENT = re.compile(
    r"""(?im)\b(api[_-]?key|secret[_-]?key|access[_-]?token|password)\b"""
    r"""\s*[:=]\s*(?:(["'])([^"'\n]{12,})\2|([A-Za-z0-9_+/=-]{20,}))"""
)

PLACEHOLDER_MARKERS = {
    "$" + "{",
    "<",
    "changeme",
    "dummy",
    "example",
    "keyring",
    "os.environ",
    "placeholder",
    "redacted",
    "secret_ref",
    "test-only",
    "your-",
}


@dataclass(frozen=True, order=True)
class Violation:
    path: str
    category: str
    detail: str


def relative_display(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def iter_repository_files(root: Path, violations: list[Violation]) -> Iterable[Path]:
    """Walk without following links and report forbidden source directories."""
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        retained_directories: list[str] = []

        for name in directory_names:
            if name in SKIPPED_DIRECTORIES:
                continue
            directory_path = current_path / name
            if name in FORBIDDEN_DIRECTORIES:
                violations.append(
                    Violation(
                        relative_display(directory_path, root),
                        "forbidden directory",
                        FORBIDDEN_DIRECTORIES[name],
                    )
                )
                continue
            retained_directories.append(name)

        directory_names[:] = retained_directories

        for name in file_names:
            yield current_path / name


def matching_forbidden_suffix(name: str) -> tuple[str, str] | None:
    lowered = name.lower()
    for suffix in sorted(FORBIDDEN_SUFFIXES, key=len, reverse=True):
        if lowered.endswith(suffix):
            return suffix, FORBIDDEN_SUFFIXES[suffix]
    return None


def inspect_symlink(path: Path, root: Path, violations: list[Violation]) -> bool:
    """Return True when the caller should skip reading this path."""
    if not path.is_symlink():
        return False
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        violations.append(
            Violation(
                relative_display(path, root),
                "unsafe link",
                "broken symbolic link",
            )
        )
        return True

    try:
        resolved.relative_to(root)
    except ValueError:
        violations.append(
            Violation(
                relative_display(path, root),
                "unsafe link",
                "symbolic link resolves outside the repository",
            )
        )
    return True


def read_prefix(path: Path, length: int = 512) -> bytes:
    try:
        with path.open("rb") as stream:
            return stream.read(length)
    except OSError:
        return b""


def detect_binary_signature(path: Path, prefix: bytes) -> str | None:
    if len(prefix) >= 132 and prefix[128:132] == b"DICM":
        return "DICOM signature"
    if len(prefix) >= 348 and prefix[344:348] in {b"n+1\x00", b"ni1\x00"}:
        return "NIfTI signature"
    if prefix.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        return "ZIP archive signature"
    if prefix.startswith(b"Rar!\x1a\x07"):
        return "RAR archive signature"
    if prefix.startswith(b"7z\xbc\xaf\x27\x1c"):
        return "7-Zip archive signature"
    if prefix.startswith(b"MZ"):
        return "Windows executable signature"
    if prefix.startswith(b"\x7fELF"):
        return "ELF executable or shared-library signature"

    # Detect a compressed NIfTI even when its extension is misleading. Reading
    # only the header bounds decompression and avoids materializing the payload.
    if prefix.startswith(b"\x1f\x8b"):
        try:
            with gzip.open(path, "rb") as stream:
                header = stream.read(352)
        except (OSError, EOFError):
            return "gzip archive signature"
        if len(header) >= 348 and header[344:348] in {b"n+1\x00", b"ni1\x00"}:
            return "gzip-compressed NIfTI signature"
        return "gzip archive signature"
    return None


def line_number(text: str, position: int) -> int:
    return text.count("\n", 0, position) + 1


def is_placeholder(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def scan_text_for_secrets(path: Path, root: Path, violations: list[Violation]) -> None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return
    try:
        if path.stat().st_size > MAX_SECRET_SCAN_BYTES:
            return
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return

    display_path = relative_display(path, root)
    for label, pattern in SECRET_PATTERNS.items():
        for match in pattern.finditer(text):
            matched_value = match.group(0)
            if is_placeholder(matched_value):
                continue
            violations.append(
                Violation(
                    display_path,
                    "possible secret",
                    f"{label} at line {line_number(text, match.start())}",
                )
            )

    for match in GENERIC_SECRET_ASSIGNMENT.finditer(text):
        value = match.group(3) or match.group(4)
        if is_placeholder(value):
            continue
        violations.append(
            Violation(
                display_path,
                "possible secret",
                f"credential assignment at line {line_number(text, match.start())}",
            )
        )


def inspect_file(
    path: Path,
    root: Path,
    max_file_bytes: int,
    violations: list[Violation],
) -> None:
    if inspect_symlink(path, root, violations):
        return

    display_path = relative_display(path, root)
    lowered_name = path.name.lower()

    if lowered_name in FORBIDDEN_EXACT_NAMES:
        violations.append(
            Violation(
                display_path,
                "forbidden file",
                FORBIDDEN_EXACT_NAMES[lowered_name],
            )
        )
    elif lowered_name.startswith(".env.") and lowered_name != ".env.example":
        violations.append(
            Violation(
                display_path,
                "forbidden file",
                "environment file that may contain credentials",
            )
        )

    suffix_match = matching_forbidden_suffix(lowered_name)
    if suffix_match is not None:
        _, reason = suffix_match
        violations.append(Violation(display_path, "forbidden file", reason))

    try:
        size = path.stat().st_size
    except OSError as error:
        violations.append(Violation(display_path, "unreadable file", error.__class__.__name__))
        return

    if size > max_file_bytes:
        size_mb = size / (1024 * 1024)
        limit_mb = max_file_bytes / (1024 * 1024)
        violations.append(
            Violation(
                display_path,
                "oversized file",
                f"{size_mb:.2f} MiB exceeds {limit_mb:.2f} MiB",
            )
        )

    signature = detect_binary_signature(path, read_prefix(path))
    if signature is not None:
        violations.append(Violation(display_path, "forbidden signature", signature))

    scan_text_for_secrets(path, root, violations)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reject medical data, model artifacts, secrets, build output, "
            "executables, archives, and oversized files from the source tree."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (defaults to the parent of scripts/)",
    )
    parser.add_argument(
        "--max-file-mb",
        type=float,
        default=float(os.environ.get("OPENMEDVISIONX_REPOSITORY_MAX_FILE_MB", DEFAULT_MAX_FILE_MB)),
        help=(
            "maximum permitted file size in MiB "
            f"(default: {DEFAULT_MAX_FILE_MB:g}, environment override: "
            "OPENMEDVISIONX_REPOSITORY_MAX_FILE_MB)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()

    if not root.is_dir():
        print(f"error: repository root is not a directory: {root}", file=sys.stderr)
        return 2
    if args.max_file_mb <= 0:
        print("error: --max-file-mb must be positive", file=sys.stderr)
        return 2

    max_file_bytes = int(args.max_file_mb * 1024 * 1024)
    violations: list[Violation] = []
    files = list(iter_repository_files(root, violations))
    for path in files:
        inspect_file(path, root, max_file_bytes, violations)

    unique_violations = sorted(set(violations))
    if unique_violations:
        print(
            f"Repository policy failed with {len(unique_violations)} violation(s):",
            file=sys.stderr,
        )
        for violation in unique_violations:
            print(
                f"  - {violation.path}: {violation.category} ({violation.detail})",
                file=sys.stderr,
            )
        print(
            "\nRemove these files from the working tree and, if previously "
            "committed, from Git history before publishing.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Repository policy passed: scanned {len(files)} file(s), "
        f"maximum size {args.max_file_mb:g} MiB."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
