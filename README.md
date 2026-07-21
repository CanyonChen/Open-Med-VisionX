<p align="center">
  <img src="figs/logo_full.png" alt="OpenMedVisionX" width="900">
</p>

<p align="center">
  <a href="docs/README.zh-CN.md">简体中文</a>
</p>

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyQt5](https://img.shields.io/badge/PyQt5-5.15+-41CD52.svg)](https://pypi.org/project/PyQt5/)
[![pydicom](https://img.shields.io/badge/pydicom-2.4+-f37626.svg)](https://pydicom.github.io/)
[![SciPy](https://img.shields.io/badge/SciPy-1.11+-8CAAE6.svg)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# OpenMedVisionX

An Open Interactive Platform for Medical Computer Vision Learning and Exploration

---

OpenMedVisionX is a local, GUI-driven teaching platform for medical imaging,
computed-tomography reconstruction, and external computer-vision inference. It
is designed to make image semantics, spatial geometry, preprocessing, algorithm
intermediates, and model outputs visible to students instead of hiding them
behind a single prediction button.

> [!WARNING]
> This software is for education and research only. It is not a medical device,
> has not been validated for clinical use, and must not be used to diagnose,
> treat, triage, or make decisions about a patient.

The repository contains source code only. It intentionally does **not** include
medical images, datasets, model weights, API keys, executables, or packaged
build artifacts.

## Project status

The project is an alpha-stage refactor of an earlier CT teaching viewer. The
source release focuses first on a correct image-domain model, safe local
loading, traditional reconstruction, a capability-driven PyQt interface, and
testable extension contracts. Model runtimes, research adapters, and cloud LLM
providers remain optional and never become requirements of the base viewer.

Pre-1.0 public interfaces are documented below so that examples and plugins can
converge on a stable boundary. Compatibility changes are recorded in
[CHANGELOG.md](CHANGELOG.md).

## Design commitments

- Local-first operation with explicit user-selected inputs and export paths.
- Separate semantics for a 2D raster, a 2D page/frame sequence, and a physical
  medical volume.
- RAS+ internal coordinates for spatial volumes; no fabricated spacing, affine,
  Z axis, millimetres, or HU semantics for ordinary 2D images.
- Correct DICOM rescale, orientation, spacing, and multiplanar geometry.
- Original dtype and dynamic range preserved by IO; display mapping and model
  preprocessing are separate transformations.
- Cancellable background work for loading, reconstruction, inference, and
  network requests.
- External models and LLM services are optional, user-configured extensions.
- No automatic dataset scanning, model downloading, cloud upload, or overwrite
  of a source image.

## Format policy

| Format | Domain object | Installation and semantics |
| --- | --- | --- |
| DICOM folder or validated DICOM ZIP | `ImageVolume` | Base install. ZIP entries and decoded data are checked against safety limits. CT intensities use rescale slope/intercept. |
| NIfTI `.nii` / `.nii.gz` | `ImageVolume` | Install the `nifti` extra. Affines are normalized to internal RAS+. |
| PNG | `RasterImage2D` | Base install. Supports common grayscale, RGB, RGBA, and palette images while retaining source dtype. |
| JPEG `.jpg` / `.jpeg` | `RasterImage2D` | Base install. EXIF orientation is normalized and lossy-compression status is exposed. |
| TIFF `.tif` / `.tiff` | `RasterImage2D` or `ImageSequence2D` | The base backend preserves high-bit-depth pages and offers cancellable, on-demand page, thumbnail, and flat tile access. Decoder and WSI backends remain replaceable. |
| MHA/MHD, NRRD, BMP, WebP, WSI | Plugin-defined | Not mandatory base dependencies. A loader plugin must declare and validate its capabilities. |

A multipage TIFF is a page sequence unless validated spatial metadata proves
otherwise. It does not acquire coronal, sagittal, volume-rendering, or 3D
measurement tools merely because it contains multiple pages.

For a high-resolution flat image, `RasterTileSource` reads only the requested
page result, thumbnail, or canonical-coordinate region and retains it in a
thread-safe LRU bounded by both entry count and decoded bytes. Source opening
checks the extension and signature, every page is checked against dimension,
pixel, frame, and decoded-byte limits, and calls accept the same cooperative
cancel check used by background tasks. EXIF orientation, palette/color
conversion, and alpha semantics remain visible in the returned transform and
runtime trace; paths and pixel payloads are never copied into runtime metadata.
The default `ImageService` routes a flat raster whose estimated decoded payload
exceeds 64 MiB through this backend and presents a bounded 2048-pixel thumbnail
with a reversible source-coordinate transform instead of retaining the full page.

This base API models flat pages, not pathology pyramids. Pillow codecs may need
to decode backing strips while producing a requested region, so this is not a
memory-mapped or WSI-streaming guarantee. True multiresolution WSI readers must
implement the separate `WsiPyramidTileSource` plugin boundary and declare their
own level geometry, scheduler, and limits. Ordinary TIFF files are never
auto-promoted to WSI.

## Architecture

```text
src/dicom_viewer/
├── domain/       ImageData, geometry, display mapping, transforms, results
├── io/           format probes and safe DICOM, NIfTI, and raster loaders
├── algorithms/   Radon, DFR, BP, FBP, SART, and image-quality metrics
├── inference/    manifests, preprocessing, runtimes, adapters, postprocessing
├── llm/          provider contracts and teaching-assistant adapters
├── runtime/      tasks, cancellation, configuration, credentials, and errors
├── services/     application use cases consumed by the GUI
└── ui/           PyQt pages, state presentation, and visualizations
```

The UI depends on services and public contracts. It must not import a concrete
file decoder, model runtime, or vendor API adapter. Tool availability comes
from the loaded object's capabilities rather than from its filename extension.
`ModelInferenceService` owns manifest inspection, runtime factory selection,
load/predict/cancel state, and Python-adapter reload requirements.
`TeachingAssistantService` and `LLMProviderRegistry` own provider defaults,
transport selection, and concrete provider construction; the UI sees only the
service facade and provider-neutral value types.

### Public extension points

- `ImageLoader.can_load/probe/load`
- `ImageData`, `RasterImage2D`, `ImageSequence2D`, and `ImageVolume`
- `TransformRecord.forward/inverse`
- `ReconstructionAlgorithm.reconstruct`
- `ModelPlugin.describe/validate/capabilities/load/predict/visualize`
- typed `InferenceResult` variants
- `LLMProvider.chat/stream/capabilities`
- `BackgroundTask.cancel/progress/result`

Plugins should use these contracts instead of reaching into a concrete loader,
Qt widget, model implementation, or provider adapter.

## Installation

Python 3.10 or newer is required; supported versions are 3.10–3.12.

### Conda

```bash
conda env create -f environment.yml
conda activate openmedvisionx
openmedvisionx
```

The project display name is `OpenMedVisionX`; the Conda environment,
distribution, and command use `openmedvisionx`.

### pip

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
openmedvisionx
```

The default installation is the `base` viewer. Install only the capabilities
you need:

```bash
python -m pip install -e ".[nifti]"
python -m pip install -e ".[onnx]"
python -m pip install -e ".[pytorch-plugin]"
python -m pip install -e ".[llm]"
python -m pip install -e ".[dev,nifti,onnx]"
```

Installing the base viewer does not install PyTorch, ONNX Runtime, a cloud SDK,
CUDA, model code, or model weights.

## Image semantics and measurement

`RasterImage2D` accepts only `H x W` grayscale or `H x W x C` color
arrays. It records bit depth, color space, channel order, and alpha handling.
It uses a top-left pixel coordinate system and defaults to pixel distance and
pixel area. Physical measurement is enabled only after the user supplies
trusted pixel spacing; the UI labels that source as user-provided. DPI is never
treated as medical spacing.

`ImageSequence2D` represents pages or frames without inventing physical
volume geometry. `ImageVolume` represents data with validated spatial
semantics and carries affine, spacing, origin, direction, modality, and
intensity meaning.

Display mapping never mutates the source array. EXIF correction, resize, crop,
and letterbox preprocessing produce a reversible `TransformRecord` so masks,
boxes, keypoints, heatmaps, and probability maps can be mapped back to source
pixels. Discrete label maps use nearest-neighbour interpolation when mapped.

## Local mask, annotation, and experiment files

Pairing is always explicit and applies only to the currently visible plane.
A mask must be a lossless, single-page grayscale PNG/TIFF with exactly the
same `H x W`; non-zero values form the overlay and are never resized. Annotation
JSON is limited to 4 MiB, rejects unknown fields and external paths, and uses
top-left pixel coordinates:

```json
{
  "schema_version": 1,
  "coordinate_system": "pixel_xy_top_left",
  "image_size": [512, 512],
  "boxes": [{"x1": 20, "y1": 30, "x2": 120, "y2": 160, "label": "example"}],
  "points": [{"x": 64, "y": 80, "label": "landmark"}]
}
```

The size must match the current plane; boxes satisfy `x1 < x2` and `y1 < y2`
inside the image, and points are in bounds. Rendered PNG and experiment JSON
exports use exclusive file creation and never overwrite an existing file or the
source image. Experiment records contain only a non-pixel image summary,
parameters, and numeric metrics; raw pixels and source metadata are rejected.

## CT reconstruction teaching

The traditional algorithm area covers:

- Radon transform and sinogram generation;
- direct Fourier reconstruction and its interpolation choices;
- unfiltered backprojection and filtered backprojection;
- SART iteration and convergence;
- the redundancy of 180-degree and 360-degree parallel-beam acquisition;
- consistent MSE, PSNR, SSIM, difference images, error heatmaps, and ROI
  metrics;
- intermediate frequency filtering, backprojection, and iteration states.

Tests and lessons generate phantoms at runtime. No DICOM or other medical
example image is committed for a tutorial.

## External model plugins

The platform does not ship a model implementation or weights. A user can point
to a local ONNX/TorchScript model or select a research plugin containing a
`manifest.yaml` and Python adapter.

Every manifest must describe:

- model identity, version, family, task, subtask, and licenses;
- runtime, Python/Conda environment, adapter entry point, and external weight
  reference;
- modality and 2D/2.5D/3D/4D dimensionality;
- input size, layout, color space, channel order, alpha policy, dtype, numeric
  range, spacing, normalization, and resize/crop/letterbox behavior;
- output tensor meaning, coordinates, labels, postprocessing, and uncertainty;
- optional prompts, intermediate features, attention, embeddings, multiscale
  outputs, vector fields, trajectories, and text outputs.

Schema version `1.0` requires the top-level fields `schema_version`,
`name`, `version`, `family`, `source`, `description`, `tasks`,
`subtasks`, `license`, `runtime`, `entrypoint`, `weights`, `inputs`,
`outputs`, and `capabilities`. See the source-only
[example manifest](src/dicom_viewer/inference/examples/manifest.yaml); its model
path is deliberately only a user-local reference.

Task names use a controlled vocabulary such as `classification`,
`segmentation`, `detection`, `registration`, `reconstruction`,
`restoration`, `generation`, `retrieval`, `vqa`,
`report_generation`, `wsi_mil`, `tracking`, and
`anomaly_detection`. Ambiguous family names must include their source and
task; for example, a self-supervised DINO encoder is not a DINO detector or
Grounding DINO.

> [!CAUTION]
> A Python plugin can execute arbitrary code. It runs in a separate process and
> a user-selected environment, but process separation is not a security
> sandbox. Inspect and trust the code before loading it.

The application never downloads a model automatically or copies weights into
the repository. Each plugin must display the licenses for its code, model, and
weights. This project's MIT license does not cover third-party plugins or
weights.

Whole-slide pathology requires a future streaming pyramid and tile scheduler.
Until that loader exists, WSI plugins may consume only user-provided pre-tiles
or extracted features. The GUI performs import, inference, and visualization;
it is not a training, validation, or checkpoint-resume system.

## LLM teaching assistant

The provider contract is designed for OpenAI, Anthropic Claude, Moonshot/Kimi,
Zhipu GLM, DeepSeek, and user-configured OpenAI-compatible endpoints. Other
vendors can be added through provider plugins.

Provider settings contain the endpoint, a user-entered model ID, capability
flags, timeout, and a credential reference. The code does not hard-code a
"latest" model. The OpenAI adapter uses the Responses API.

A saved provider profile contains a reference, never the key itself:

```toml
provider_id = "openai"
endpoint = "https://provider.example.invalid/v1"
model_id = "user-selected-model-id"
credential_ref = "env:OPENAI_API_KEY"
supports_vision = false
timeout = 30
```

Credential references use `env:VARIABLE_NAME` or
`keyring:service/account-name`. Enter the real endpoint and model ID shown by
your provider. Declare vision support only after confirming it for that model.

API keys belong in the operating-system credential store or an environment
variable. Configuration files store only credential references. Keys must
never appear in source, logs, tracebacks, exported conversations, screenshots,
or bug reports.

Cloud image transfer is off by default and requires per-provider consent that
can be revoked. When enabled, the GUI keeps a visible warning on screen. Only a
rendered slice explicitly selected and previewed by the user may be sent.
Original DICOM/NIfTI files, full series, and DICOM metadata are never sent.
Metadata is stripped before transmission, but burned-in pixel text may still
contain identifying information.

Assistant responses show the provider, model ID, time, and an education-only
disclaimer. They must not be presented as medical advice or a diagnosis.

## Repository and data safety

Before contributing, run:

```bash
python scripts/check_repository.py
```

The repository policy check rejects:

- medical volumes and DICOM files, including files hidden behind a wrong
  extension when a known signature is detectable;
- archives, executables, compiled output, and build directories;
- model weights and serialized model binaries;
- likely API keys, access tokens, private keys, and passwords;
- files larger than the repository policy limit.

If sensitive data or a secret was ever committed, deleting the working-tree
file is insufficient. Revoke exposed credentials, remove the object from Git
history before publishing, and follow [SECURITY.md](SECURITY.md).

The application reads only files selected by the user. It does not recursively
scan a device for medical data. Derived images are written only to a
user-selected local directory and do not overwrite source images by default.

## Further documentation

- [Architecture and dependency rules](docs/ARCHITECTURE.md)
- [Model plugin development](docs/PLUGIN_DEVELOPMENT.md)
- [LLM providers and cloud-image security](docs/LLM_SECURITY.md)
- [Teaching curriculum: from pixels to multimodal AI](docs/TEACHING_CURRICULUM.md)

## Development and testing

```bash
python -m pip install -e ".[dev,nifti,onnx]"
python scripts/check_repository.py
python -m pytest
ruff check src tests scripts
```

PyTorch is intentionally optional in local development. The tiny TorchScript
integration test is reported as skipped when PyTorch is unavailable; install
the plugin extra to run it explicitly:

```bash
python -m pip install -e ".[pytorch-plugin]"
python -m pytest -q \
  tests/test_inference_execution.py::test_tiny_torchscript_runtime_when_pytorch_is_available
```

Tests create small DICOM, NIfTI, raster, ONNX/TorchScript, and mock-provider
fixtures at runtime in temporary directories. Do not add fixture images or
weights to the repository.

See [CONTRIBUTING.md](CONTRIBUTING.md) for architecture and pull-request rules,
[SECURITY.md](SECURITY.md) for private reporting, and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations.

## Roadmap

1. Repository hygiene, licensing, bilingual documentation, and commit guards.
2. Unified image domains and safe DICOM, NIfTI, PNG, JPEG, and TIFF loaders.
3. Capability-driven 2D/3D PyQt UI, background tasks, and traditional CT
   teaching visualizations.
4. ONNX, TorchScript, and isolated Python model plugins.
5. Typed adapters and visualizations across classification, segmentation,
   detection, registration, reconstruction, restoration, generation,
   multimodal, WSI, temporal/4D, and anomaly tasks.
6. Multi-provider LLM assistant, consent controls, and secure credentials.
7. Automated testing, plugin authoring guidance, and the first source-only
   release.

## License

The platform source is available under the [MIT License](LICENSE). Imported
data, plugins, models, and weights retain their own terms and are the user's
responsibility.
