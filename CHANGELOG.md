# OpenMedVisionX Changelog

All notable changes to this project are documented here.

The format follows Keep a Changelog, and releases use Semantic Versioning once
the first source release is tagged.

## [Unreleased]

### Added

- Source-package metadata with a lightweight base install and optional NIfTI,
  ONNX, PyTorch-plugin, LLM, and development dependency groups.
- English and Simplified Chinese project documentation with architecture,
  format semantics, plugin/provider boundaries, and a staged roadmap.
- MIT License, contribution guide, code of conduct, and security policy.
- Repository policy checks and CI protection against medical files, archives,
  model weights, credentials, build products, executables, and oversized files.
- Conda environment and command-line entry point both named `openmedvisionx`.
- Unified raster/sequence/RAS+ volume domains; safe PNG/JPEG/TIFF, DICOM ZIP,
  DICOM series, and optional NIfTI loading; reversible EXIF/model transforms.
- Cancellable background services, capability-driven 2-D/tri-planar viewing,
  measurements, overlays, local pairing, safe exports, and structured lessons.
- Corrected Radon, BP, FBP, direct-Fourier, and SART teaching workflows with
  intermediate views, shared-range metrics, ROI analysis, and error maps.
- Runtime-neutral model manifests, typed inference results, ONNX/TorchScript
  runtimes, and explicitly consented subprocess Python adapters.
- Offline-by-default OpenAI Responses, Anthropic, Kimi/Moonshot, GLM,
  DeepSeek, and generic OpenAI-compatible teaching-assistant providers.

### Changed

- Replaced the legacy monolithic GUI implementation with IO, domain,
  algorithm, inference, LLM, runtime, service, and UI layers.

### Fixed

- Applied DICOM rescale slope/intercept consistently, sorted oblique slices by
  their direction normal, converted LPS geometry to RAS+, and cleared all
  dataset-derived state on a new import.
- Made DFR interpolation selection effective and aligned reconstruction angle,
  circle, output-size, and metric semantics across algorithms.

### Security

- Documented source-only distribution, cloud-image consent, credential storage,
  arbitrary Python plugin risk, and non-diagnostic medical-use boundaries.
- Required runtime-generated synthetic fixtures instead of committed medical
  images, model files, or weights.
