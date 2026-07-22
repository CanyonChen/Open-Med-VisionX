# OpenMedVisionX Quickstart

[简体中文](QUICKSTART.zh-CN.md) · [Documentation home](INDEX.md) · [Project overview](../README.md)

This guide takes you from a fresh checkout to a successful first experiment.
The first workflow needs no image, model runtime, network connection, or patient
data.

> [!WARNING]
> Use only data you are permitted to handle. OpenMedVisionX is for learning and
> research, not clinical use.

## 1. Before you begin

You need:

- Git;
- Conda through Miniconda, Anaconda, or Miniforge;
- a graphical desktop session;
- enough display space for the application window.

The project supports **Python 3.11 only** (`>=3.11,<3.12`). You do not need to
install that Python separately when you create the provided Conda environment.

The minimum window is **900 × 620 logical pixels**. At 150% operating-system
scaling this is approximately **1350 × 930 physical pixels**. A complete fit is
not promised on a 1024 × 680 physical-pixel display at 150%. If controls are
clipped, maximize the window, use a larger display, or lower the scaling level
and restart the application.

## 2. Install and launch

Run these commands once:

```bash
git clone https://github.com/CanyonChen/Open-Med-VisionX.git
cd Open-Med-VisionX
conda env create -f environment.yml
```

Activate the environment whenever you open a new terminal, then launch:

```bash
conda activate openmedvisionx
openmedvisionx
```

The `openmedvisionx` command can be run from any directory after installation.
If it is not found, return to the project directory and repair the editable
installation:

```bash
conda activate openmedvisionx
python -m pip install -e .
openmedvisionx
```

As a temporary compatibility launcher from the project directory, use:

```bash
python main.py
```

When the window opens, you should see six workspaces: **Images**, **CT Lab**,
**Models**, **Learn**, **Evaluate**, and **AI Assistant**. Use the header button
to switch between English and Simplified Chinese. The switch preserves current
state. `Alt+1` through `Alt+6` opens the corresponding workspace directly.

If the window does not appear, follow [Installation and launch
troubleshooting](TROUBLESHOOTING.md#installation-and-launch).

## 3. Complete your first session

### A. Run a data-free CT experiment

1. Open **Learn**.
2. Choose **Start first experiment**.
3. OpenMedVisionX switches to **CT Lab**, keeps **Synthetic phantom
   (recommended)** selected, and runs the deterministic default experiment.
4. Wait until the status reports completion. The interface stays responsive
   while the calculation runs.
5. Inspect the four primary views: input attenuation map, sinogram,
   reconstruction, and absolute error.
6. Read the reported metrics. The image comparison uses one shared display
   range, so one result is not made to look better by stretching it separately.
7. Change exactly one parameter—projection count, reconstruction method,
   filter, or SART iterations—and run again.

**Success check:** you can explain what changed, see a reconstruction and error
view, and identify the parameter responsible. No local or network data has
been used.

### B. Inspect an optional local image

If you have a non-sensitive PNG, JPEG, TIFF, or DICOM file you may use:

1. Open **Images** and choose **Open image / DICOM ZIP** (`Ctrl+O`). For a
   DICOM directory, use **Open DICOM folder** (`Ctrl+Shift+O`).
2. Review source type, dtype, dimensions, numeric range, and warnings.
3. Click a visible image plane to make it the **active view**. CT Lab, external
   models, and the AI image-preview workflow use this exact active 2-D plane.
4. Open **Histogram**, adjust the display range, and confirm that the recorded
   source range does not change.
5. Try **Distance** or **Area / ROI**, then choose **Clear marks**.
6. If you export the rendered plane, choose a new filename and read any
   overwrite confirmation carefully. The exported PNG is not a screenshot: it
   excludes viewport zoom/pan, measurements, annotations, and overlays.

For DICOM/NIfTI volumes, 4-D selection, layers, or segmentation imports, follow
[Data and formats](DATA_FORMATS.md) before proceeding.

### C. Run the built-in evaluation example

1. Open **Evaluate**.
2. Keep the example reference labels and predicted probabilities.
3. Keep the explicit decision threshold at `0.50` and choose **Evaluate**.
4. Compare threshold-dependent sensitivity/specificity/F1 with AUROC, AUPRC,
   Brier score, and calibration bins.
5. Optionally export the experiment record to JSON or YAML. It contains hashes,
   parameters, and metrics, not the entered vectors or image pixels.

**Success check:** the status reports a completed evaluation and both the
**Metrics** and **Calibration** tabs contain results.

## 4. Add optional capabilities

Activate the same environment before installing only what you need:

```bash
conda activate openmedvisionx
python -m pip install -e ".[nifti]"          # NIfTI loading with nibabel
python -m pip install -e ".[onnx]"           # ONNX and ONNX Runtime
python -m pip install -e ".[llm]"            # OS keyring credential references
```

You may combine them, for example:

```bash
python -m pip install -e ".[nifti,onnx,llm]"
```

Restart OpenMedVisionX after adding an optional dependency.

### NIfTI

After installing `nifti`, use **Open image / DICOM ZIP** for `.nii` or
`.nii.gz`. OpenMedVisionX normalizes spatial volumes to RAS+. A 4-D file always
asks you to select one time point/channel; the application does not guess.
NIfTI alone does not prove modality, HU, or another intensity meaning. See
[Data and formats](DATA_FORMATS.md#nifti-volumes).

### Reviewed offline models

PyTorch is separate because the correct build depends on your computer. The
portable CPU path for this release is:

```bash
python -m pip install -r requirements/torch-cpu.txt
python -m workbench.models
```

The following GPU command is deliberately machine-scoped: it was tested on a
Windows development machine with an **NVIDIA RTX 4060 Laptop GPU**, 8 GiB VRAM,
and a CUDA 13.0-compatible driver. Do not assume it matches another computer.

```bash
python -m pip install -r requirements/torch-cu130.txt
python -m workbench.models --smoke --device auto
```

For another GPU/driver/operating-system combination, choose the matching build
from PyTorch's official [Start Locally](https://pytorch.org/get-started/locally/)
page, then run the verification command above.

The three reviewed workflows have different, non-interchangeable inputs:

- [DIVal LoDoPaB FBP-U-Net](../src/workbench/resources/model_bundles/dival-lodopab-fbpunet/MODEL_CARD.md) runs the included public LoDoPaB case in its fixed Hann-FBP benchmark domain—not an arbitrary CT or HU image.
- [DeepInverse MRI MoDL](../src/workbench/resources/model_bundles/deepinv-mri-modl/MODEL_CARD.md) runs a deterministic **synthetic MRI** demonstration using image-derived single-coil k-space—not scanner raw data.
- [MONAI BraTS 3D Segmentation](../src/workbench/resources/model_bundles/monai-brats-segmentation/MODEL_CARD.md) remains **disabled** until the guided desktop setup validates a user-managed local T1ce/T1/T2/FLAIR/SEG case. No BraTS case is included or downloaded.

Follow the [bundled model guide](MODEL_BUNDLES.md) before running them.

### Another local model

The **External manifests** path does not download a model or install its
dependencies. Its current desktop capability accepts one supported active 2-D
image input. Compound k-space/mask inputs, prompts, multiple modalities,
complete 3-D/4-D volumes, temporal input, sinograms, and whole-slide pipelines
are rejected even when an external manifest is valid. The maintained [example
manifest](../src/workbench/inference/examples/manifest.yaml) is not runnable
because its example model file is intentionally absent. Follow the [custom
model guide](CUSTOM_MODELS.md).

### AI Assistant

No provider is needed for the first session. Before enabling a provider, read
[AI Assistant, privacy, and cloud images](LLM_SECURITY.md). Network use and
image attachment are separate opt-in decisions; an image additionally requires
review and one-request authorization of the exact outgoing PNG.

## 5. Choose the next guide

- To learn all six workspaces, continue to the [User guide](USER_GUIDE.md).
- To prepare DICOM, NIfTI, masks, or label layers, use [Data and
  formats](DATA_FORMATS.md).
- To build knowledge in a deliberate order, follow the [Learning
  curriculum](TEACHING_CURRICULUM.md).
- To look up a term, use the [Glossary](GLOSSARY.md).
- If anything failed, open [Troubleshooting](TROUBLESHOOTING.md).

Back to the [documentation home](INDEX.md).
