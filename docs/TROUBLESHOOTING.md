# OpenMedVisionX Troubleshooting

[简体中文](TROUBLESHOOTING.zh-CN.md) · [Documentation home](INDEX.md) · [Quickstart](QUICKSTART.md)

Find the symptom below, try the checks in order, and stop when the expected
state is restored. An error normally leaves the current source data unchanged.

> [!IMPORTANT]
> Never attach patient data, credentials, private paths, or restricted model
> files to a public issue. Use the [security policy](../SECURITY.md) for a
> suspected vulnerability or disclosure.

## Find your symptom

| Symptom | Go to |
| --- | --- |
| `conda` or `openmedvisionx` is not found; no window appears | [Installation and launch](#installation-and-launch) |
| An image, volume, folder, or ZIP is rejected | [Opening images and volumes](#opening-images-and-volumes) |
| A mask, SEG, RTSTRUCT, or label map is rejected/hidden | [Annotations and layers](#annotations-and-layers) |
| CT Lab cannot use the active image or metrics look wrong | [CT Lab](#ct-lab) |
| PyTorch, a bundled model, external manifest, or BraTS fails | [Models](#models) |
| Evaluation input or export fails | [Evaluation and exports](#evaluation-and-exports) |
| AI send/attachment is disabled or credentials fail | [AI Assistant](#ai-assistant) |

First read the complete status line and dialog. It usually names the invalid
field, missing dependency, or unsupported capability. Do not repeatedly change
unrelated settings; change one condition at a time.

## Installation and launch

### `conda` is not recognized

Open **Anaconda Prompt**, **Miniforge Prompt**, or another terminal initialized
for Conda. If Conda is installed but a normal shell cannot see it, run the
platform-appropriate `conda init`, close the terminal completely, and open a
new one.

### The environment does not exist

From the project directory:

```bash
conda env create -f environment.yml
conda activate openmedvisionx
```

The supported interpreter is **Python 3.11 only**. Check the active environment:

```bash
python --version
```

If it is not Python 3.11, deactivate the current environment and activate
`openmedvisionx` instead of installing the application into an unrelated one.

### `openmedvisionx` is not found

Confirm and repair the editable installation:

```bash
conda activate openmedvisionx
python -m pip show openmedvisionx
python -m pip install -e .
openmedvisionx
```

As a temporary fallback from the project root:

```bash
python main.py
```

### The environment was created before the project changed

Update the editable installation and any optional features you actually use:

```bash
conda activate openmedvisionx
python -m pip install -e .
```

Recreate the environment only if dependency repair fails; do not delete an
environment that contains unrelated work without first checking its contents.

### No window appears on Linux or remote SSH

OpenMedVisionX needs a graphical desktop, a working display server, and the Qt
system libraries required by PyQt5. A headless SSH session without display
forwarding cannot show the application. Start it in a local graphical session
or configure a supported remote desktop/display environment.

### Text or controls are clipped

The minimum is 900 × 620 logical pixels. Maximize the window. At high operating
system scaling, use a larger display or a lower standard scaling level, then
restart so Qt can recalculate the layout. Avoid custom fractional scaling while
diagnosing.

## Opening images and volumes

### NIfTI reports a missing dependency

Install the optional loader in the active environment and restart:

```bash
conda activate openmedvisionx
python -m pip install -e ".[nifti]"
```

### A 4-D NIfTI asks for a volume selection

This is expected. Select one time point/channel explicitly. The dialog uses
human-friendly numbering starting at 1; OpenMedVisionX does not guess which
3-D volume is intended and does not provide full 4-D playback in this release.

### A DICOM folder contains several series

Select one row in **Choose a DICOM series** and review modality, dimensions,
frame count, geometry state, and warnings. OpenMedVisionX does not silently
choose a series. Unsupported rows remain unavailable.

### A DICOM folder is rejected

Check that the selected folder represents one consistent image series:

- monochrome images with consistent rows/columns and pixel spacing;
- compatible orientation and unique, regularly ordered slice positions;
- no mixture of an Enhanced multi-frame object and independent single-frame
  instances;
- an installed decoder for the file's compressed transfer syntax, when needed.

Use the narrowest folder that contains the intended series. OpenMedVisionX
does not recursively treat every neighboring DICOM file as one study.

### An Enhanced multi-frame object is rejected

Only supported, geometrically consistent monochrome Enhanced CT/MR objects are
accepted. Color, unsupported functional groups, inconsistent frame geometry,
or a mixed multi-frame/single-frame selection is rejected. “Enhanced” does not
mean every multi-frame object is supported.

### A DICOM ZIP is rejected

The archive must be unencrypted and contain safe relative member paths. Very
large member counts, expanded sizes, or compression ratios are blocked to
prevent archive abuse. Extract it yourself to a controlled local directory,
inspect the contents, then use **Open DICOM folder**. Do not bypass the check
with an untrusted archive.

### CT does not show HU, or PET does not show SUV

OpenMedVisionX labels HU only when CT modality and valid rescale evidence support
that meaning. PET SUV likewise needs the required units and correction evidence.
Without it, values remain arbitrary/unknown. Do not relabel them manually just
because the image looks like CT or PET.

### A TIFF behaves like pages rather than a volume

This is intentional. Page order alone does not provide spacing, orientation,
or a patient-space affine. Use the page control; do not interpret orthogonal
anatomy or millimetres unless a trusted spatial format supplies the geometry.

### A very large image looks lower resolution

The viewer can use a bounded preview for very large flat images. Check the
image details before measuring or using the active image as model input. A
model run requires an additional explicit preview override when applicable;
cancel if preview pixels are not valid for the task.

For exact support boundaries, see [Data and formats](DATA_FORMATS.md).

## Annotations and layers

### A current-plane mask is rejected

Use a single-page, lossless grayscale PNG or TIFF whose height and width match
the current active plane exactly. JPEG, color data, multipage files, and
shape-mismatched masks are rejected. Non-zero values form the overlay; the
application does not resize it.

### Annotation JSON is rejected

Confirm schema version 1, `pixel_xy_top_left`, exact `image_size`, and only the
supported `boxes` and `points` fields. Remove external paths and unknown keys.
See the example in [Data and formats](DATA_FORMATS.md#annotations-masks-and-label-layers).

### DICOM SEG or RTSTRUCT cannot be imported

Open the referenced DICOM image series first. Then confirm that the annotation
references the same Series, Frame of Reference, SOP instances, and compatible
geometry. RTSTRUCT support is limited to supported closed planar contours and
remains a vector contour layer.

### A PNG, TIFF, or NIfTI label map stays hidden

The layer is awaiting a geometry decision. Choose one of the offered actions:

- keep the original unmatched layer hidden;
- cancel the import; or
- explicitly create a derived display-resampled layer when that option is
  scientifically appropriate.

OpenMedVisionX never silently resamples. Discrete labels use nearest-neighbour
interpolation; the original source layer remains unchanged. JPEG label maps are
always rejected.

### Changing opacity or label visibility seems not to change the source

That is expected. Visibility, opacity, locking, and per-label visibility are
presentation state. They do not edit the imported source array. The Layers
panel separates Study → Series → Layer provenance from display choices.

## CT Lab

### The first experiment does not need an image

Keep **Synthetic phantom (recommended)** selected and choose **Run first
phantom experiment**. This local deterministic path is the recovery route when
an active image is absent or incompatible.

### **Active image (advanced)** is unavailable

Open an image in **Images**, click the intended plane so the active-view line
identifies it, then return to CT Lab. The input must be a finite, non-negative,
non-constant, square 2-D grayscale domain compatible with the projection
simulation. Color, negative HU, unsupported support, or a large bounded preview
is not silently converted, cropped, or relabelled.

### The advanced workflow shows a sinogram but no reconstruction

The active-image path is deliberately two-stage: generate/inspect the
projection data first, then choose **2. Reconstruct**. The first-phantom button
runs both stages automatically.

### A circular crop or support warning appears

The selected reconstruction assumes a defined circular support. Return to the
synthetic phantom or use a valid square non-negative attenuation map with the
documented support. Do not use an ordinary clinical HU slice as if it were a
scanner attenuation object.

### Metrics disagree with visual appearance

Check whether you are comparing one shared reference display range and whether
the metric uses the declared original data range. Independently auto-stretched
images can look similar despite different errors. Compare MAE/RMSE/bias and
error views as well as PSNR/SSIM; confirm the same geometry and ROI are used.

## Models

### PyTorch is unavailable or CUDA is not selected

Install the runtime for your computer by following [Install
PyTorch](MODEL_BUNDLES.md#install-pytorch), then inspect the environment:

```bash
conda activate openmedvisionx
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -m workbench.models --smoke --device auto
```

`auto` prefers CUDA when PyTorch reports it available and otherwise records CPU
fallback. A generic ONNX Runtime installation likewise does not imply GPU
support.

### A bundled model file is missing or fails its hash

Run the non-executing integrity check:

```bash
python -m workbench.models
```

Do not edit a bundled weight or its registered hash. Restore the exact file
from a trusted copy of the same release. A smoke pass proves runtime/artifact
compatibility, not medical accuracy.

### An external manifest validates but **Run inference** is disabled

The generic desktop path currently accepts one supported active 2-D `image`
input. It cannot supply multiple modalities, a complete 3-D/4-D volume, prompt,
k-space/mask, sinogram, temporal, or WSI input. Use a compatible model or its
own task-specific workflow; do not falsify the manifest. See [Custom local
models](CUSTOM_MODELS.md).

### ONNX or TorchScript runtime is unavailable

For ONNX:

```bash
python -m pip install -e ".[onnx]"
```

For TorchScript, install a compatible PyTorch runtime. Restart the application
after changing dependencies.

### A Python adapter asks for trust confirmation

This is expected because it executes third-party code. Review source, origin,
hashes, environment, dependencies, and licenses. A subprocess is not a security
sandbox. Decline if any item is unclear.

### MONAI BraTS remains disabled

Open **Set up local BraTS 2021…** and select one authorized directory containing
uniquely identifiable T1, T1ce, T2, FLAIR, and SEG NIfTI files. All must be
finite 3-D volumes with matching shape, spacing, affine/orientation, world
coverage, and usable qform/sform; SEG labels must be integers in `{0,1,2,4}`.
The desktop requires all five files. Complete in-session validation before
running. The application does not download, copy, rename, or upload the case.

### BraTS is slow or runs out of GPU memory

Full-volume sliding-window inference is substantially heavier than the two
synthetic demonstrations. Close other GPU work, confirm the selected device,
or use CPU and expect a much longer run. Do not change patch geometry or resize
the case merely to force it through; that changes the reviewed contract.

### DeepInverse has no experiment JSON button

This is a current limitation, not a failed run. The deterministic synthetic
MRI demonstration shows phantom, zero-filled IFFT, model magnitude, error, and
metrics but does not export JSON in this release. DIVal LoDoPaB and validated
BraTS workflows support pixel-free experiment records.

## Evaluation and exports

### A dataset manifest is rejected

Use JSON or YAML with bounded pseudonymous sample IDs/hashes and no image paths.
Every sample belongs to one group, and a group must not cross train,
validation, and test splits. Fix duplicate IDs, unknown splits, invalid hashes,
or group leakage before continuing.

### Binary evaluation is rejected

Reference labels and predicted probabilities must contain the same number of
finite values. Labels are only `0` or `1`; probabilities lie in `[0,1]`; both
classes must be present. Commas or new lines are accepted separators. Select a
decision threshold explicitly.

### A metric is blank or unexpected

Some intervals or metrics are undefined for small/degenerate samples. Confirm
both classes are present and read the warning. AUROC does not use the selected
operating threshold; sensitivity, specificity, predictive values, accuracy,
and F1 do. Calibration depends on probability values, not hard predictions.

### An export fails or asks about an existing file

Choose a writable local directory and a new filename. Follow the native
overwrite prompt for the specific export. Do not overwrite source data. Image
exports are rendered planes rather than viewport screenshots, and experiment
records omit input pixels/vectors but can still contain model/task metadata;
review them before sharing.

## AI Assistant

### Credentials cannot be resolved

The field accepts `env:NAME`, `keyring:service/username`, or `none` for exact
loopback services only. It does not accept a raw key. Set the environment value
outside the repository or install `.[llm]` and create the named keyring entry.
Restart the application if the environment changed.

### The endpoint is rejected

Use HTTPS with a hostname for remote services. Plain HTTP is accepted only for
exact `localhost`, `127.0.0.1`, or `::1` loopback destinations. Remove embedded
credentials, query strings, and fragments. The OpenAI-compatible
`example.invalid` value is a placeholder and must be replaced.

### **Send** is disabled

Confirm all of these: provider selected, exact Model ID entered, valid endpoint,
credential reference present, non-empty prompt, **Enable network** on, and no
request already active. OpenMedVisionX does not supply a default model ID.

### Image attachment is disabled

Load an image, click the intended active plane, verify the provider model really
supports vision, enable **Vision input**, then choose **Attach active rendered
plane**. DeepSeek is treated as text-only. A final exact-PNG review and one-shot
authorization are still required before dispatch.

### The preview does not include overlays or zoom

This is intentional. The outgoing PNG is the complete active 2-D plane after
display mapping; viewport zoom/pan, measurements, annotations, and overlays are
excluded. Inspect the final confirmation preview—not the viewport—for burned-in
private text.

### **Structured artifacts · API preview** stays empty

This is expected in an ordinary desktop session. Teaching chat and the file
picker do not populate it, and this release has no desktop artifact importer.
Only a trusted host integration can supply a typed request/response. Local
Confirm/Reject does not send data or create a layer.

### Cancellation or revocation did not recall a request

Cancellation can stop pending work but cannot recall data already transmitted.
Provider deletion/incident procedures are required after a completed transfer.
An image retry needs a new exact plan and authorization. See [AI Assistant,
privacy, and cloud images](LLM_SECURITY.md#cancel-revoke-or-change-your-mind).

## Still blocked?

Before asking for help, record only non-sensitive diagnostic facts:

- operating system and display scaling;
- `python --version` and installation method;
- the exact workspace and action;
- the complete redacted error text;
- input format and non-identifying dimensions (not the file itself);
- optional runtime versions and requested/actual device;
- the checks already attempted.

Search or open a public report at the project's [GitHub issue
page](https://github.com/CanyonChen/Open-Med-VisionX/issues/new/choose) only when
the report contains no secrets or restricted data. For security or disclosure
concerns, follow [SECURITY.md](../SECURITY.md) privately.

Back to the [documentation home](INDEX.md).
