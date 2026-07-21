# Contributing to OpenMedVisionX

Thank you for helping improve OpenMedVisionX. Contributions should preserve its
education-first, local-first, and source-only boundaries.

## Before opening a change

1. Search existing issues and pull requests.
2. Keep the change focused on one problem.
3. For a public-interface or architecture change, describe the proposed
   contract before coupling UI code to an implementation.
4. Never attach or commit patient data, medical images, model weights, API keys,
   executable files, or build output.

Do not use a real patient file as a bug reproducer, even if obvious DICOM fields
have been removed. Pixel data can contain burned-in identifiers and private
metadata can be easy to miss. Build a minimal synthetic fixture at test runtime
instead.

## Development setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,nifti,onnx]"
```

Alternatively:

```bash
conda env create -f environment.yml
conda activate openmedvisionx
python -m pip install -e ".[dev,nifti,onnx]"
```

Use `OpenMedVisionX` for the project display name and `openmedvisionx` for the
Conda environment, distribution, and command.

PyTorch and cloud-provider SDKs are intentionally optional. Install
`pytorch-plugin` or `llm` only when working on those adapters.

## Architecture rules

- Domain objects own data meaning and capabilities.
- IO code probes and decodes formats but does not drive Qt widgets.
- Algorithms accept domain data or arrays through documented contracts and do
  not read files or manipulate the GUI.
- Services orchestrate use cases for the UI.
- UI code presents state and visualization; it does not implement algorithms.
- Slow work uses `BackgroundTask` and must support cancellation.
- Concrete model runtimes and LLM vendors stay behind plugin/provider
  interfaces.

The intended public extension points are documented in
[README.md](README.md#public-extension-points). A change to one of them needs:

- contract tests;
- an entry in [CHANGELOG.md](CHANGELOG.md);
- updated English and Chinese documentation;
- a migration note when compatibility cannot be preserved.

## Data and coordinate correctness

Tests must exercise semantics rather than merely confirming that an array was
returned:

- preserve source dtype and dynamic range at the IO boundary;
- distinguish pixel coordinates from physical coordinates;
- keep ordinary 2D images free of invented HU, spacing, affine, and 3D tools;
- verify DICOM LPS to internal RAS+ conversion and NIfTI affine handling;
- apply DICOM rescale consistently to display, measurement, and reconstruction;
- verify reversible EXIF, resize, crop, and letterbox transforms;
- use nearest-neighbour interpolation for discrete label maps.

Generate small DICOM, NIfTI, PNG, JPEG, TIFF, model, and mock HTTP fixtures
during the test. Temporary files must be removed by the test framework.

## Model and provider contributions

A model contribution implements the protocol; it must not add a checkpoint or
download one automatically. Keep weight paths external. Document the licenses
for adapter code, model definition, and weights separately.

A Python adapter executes in a separate process but remains arbitrary code.
Preserve the first-load warning, timeout, cancellation, and crash isolation.

Provider tests use a local mock HTTP server. Automated tests must never contact
a real vendor, spend account credit, or require an API key. Redact credentials
from errors, logs, test snapshots, and recorded requests.

## Required checks

Run from the project root:

```bash
python scripts/check_repository.py
ruff check src tests scripts
python -m pytest
python -m build
```

The repository check rejects medical formats, archives, model weights,
executables, build output, likely secrets, and oversized files. Do not weaken
the check to make an unsafe fixture pass.

## Documentation

`README.md` is the English primary page and links to
`docs/README.zh-CN.md`; the Chinese page links back. User-facing behavior,
installation commands, safety controls, and public interfaces should remain
consistent in both files.

Examples must use generated arrays, phantoms, mock tensors, placeholder model
paths, and placeholder credential references.

## Pull requests

A pull request should:

- explain the problem and the observable outcome;
- list the affected public interfaces;
- include focused tests;
- note optional-dependency or license changes;
- update documentation and the changelog when user-visible behavior changes;
- pass repository policy and test checks.

By contributing, you agree that your contribution is licensed under the
project's [MIT License](LICENSE).
