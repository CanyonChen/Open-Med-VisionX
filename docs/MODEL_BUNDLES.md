# OpenMedVisionX Bundled Models

[简体中文](MODEL_BUNDLES.zh-CN.md) · [Documentation index](INDEX.md) · [Data formats](DATA_FORMATS.md) · [Custom models](CUSTOM_MODELS.md)

This guide takes you from installing the model runtime to completing all three
reviewed desktop workflows and understanding their results. No programming is
required. Once PyTorch is installed, the bundled models and their prepared
teaching inputs run locally without downloading a model.

> [!WARNING]
> The bundled models are fixed learning and research references. They are not
> medical devices, have not been clinically validated in OpenMedVisionX, and
> must not be used for diagnosis, treatment, triage, or any patient decision.

## Install PyTorch

First complete the [Quickstart](QUICKSTART.md) and activate the project
environment:

```bash
conda activate openmedvisionx
```

All three bundled models require PyTorch. For a portable CPU-only installation
that matches this release, run:

```bash
python -m pip install -r requirements/torch-cpu.txt
```

CPU mode is the simplest choice and works without an NVIDIA GPU. For another
operating system or a GPU installation, choose a command that matches the exact
OS, GPU, and driver on PyTorch's official
[Start Locally](https://pytorch.org/get-started/locally/) page. Do not guess a
CUDA wheel. The command for the repository's one tested GPU configuration is
recorded, with its narrow scope, in the [release integrity appendix](#tested-gpu-reference).

Start the application after installation:

```bash
openmedvisionx
```

OpenMedVisionX does not download a bundled model, a BraTS case, or an additional
runtime when a workflow starts. If a required component is absent, the workflow
stops and explains what is missing.

## Verify the installation

In the desktop application, open **Models → Bundled models** and select
**Verify catalog & device**. This verifies the three reviewed bundles and shows
the local PyTorch device that can be used.

You can perform the same checks from a terminal. Begin with an integrity-only
check, which does not execute a model:

```bash
python -m workbench.models
```

After PyTorch is installed, load every model and run a small deterministic
input on the preferred device:

```bash
python -m workbench.models --smoke --device auto
```

For the strongest packaged numerical check, compare every result with its
reviewed expected output:

```bash
python -m workbench.models --golden --device auto
```

`auto` uses CUDA when PyTorch reports it available and otherwise uses CPU. A
smoke or golden check can take noticeably longer on CPU, especially for the 3-D
MONAI model. A successful check proves that the packaged files and local
runtime work together; it does not prove medical accuracy, clinical validity,
or generalization.

If verification reports a missing file, size mismatch, hash mismatch, or
unexpected bundle ID, do not bypass the check or replace one file by hand. Use
a fresh copy from a trusted project release and verify again.

## Understand each model's input

Each model has a dedicated workflow because its input has a different physical
meaning. An arbitrary open image is never silently converted into one of these
task-specific inputs.

| Model | Where its input comes from | What you provide | Scientific boundary |
| --- | --- | --- | --- |
| [DIVal LoDoPaB FBP-U-Net](../src/workbench/resources/model_bundles/dival-lodopab-fbpunet/MODEL_CARD.md) | The packaged public LoDoPaB-CT teaching case | Nothing | Accepts only the fixed `1×1×362×362` DIVal Hann-filtered FBP domain; it is normalized attenuation, not clinical HU |
| [DeepInverse MRI MoDL teaching model](../src/workbench/resources/model_bundles/deepinv-mri-modl/MODEL_CARD.md) | A deterministic synthetic MRI demonstration generated in memory | Nothing | Uses simulated single-coil complex k-space and a binary mask; it is not scanner raw data |
| [MONAI BraTS 3D Segmentation](../src/workbench/resources/model_bundles/monai-brats-segmentation/MODEL_CARD.md) | One local BraTS-style case selected and validated by you | T1, T1ce, T2, FLAIR, and Task 1 SEG NIfTI files | A BraTS 2018 glioma model evaluated on a BraTS 2021 Task 1 case; it is not a general brain-lesion model |

The model packages do not contain a training dataset, patient image, BraTS
case, or scanner acquisition. DIVal uses one separately reviewed public
benchmark case, DeepInverse creates a simulation, and MONAI uses only the local
case that you select.

## Workflow 1 — DIVal LoDoPaB CT reconstruction

### Before you start

This workflow needs PyTorch but needs no user data. It uses the packaged
`lodopab-ct-test-03456` benchmark example so that the observation, analytic
baseline, model result, and reference all belong to one reviewed domain.

### Run the workflow

1. Open **Models → Bundled models**.
2. Select **DIVal LoDoPaB FBP-U-Net** and review the input contract and model
   card shown in the panel.
3. Select **Run LoDoPaB case**.
4. Wait for case verification, model inference, and reference metrics to
   finish. The status area records the device actually used and any fallback.
5. To keep a reproducible, pixel-free summary, select **Export experiment
   JSON…** and choose an appropriate destination.

### Read this workflow's views

| View | Meaning |
| --- | --- |
| Observation | The `1000×513` simulated low-dose parallel-beam line-integral data |
| Hann FBP | The fixed `362×362` analytic reconstruction supplied as the model input |
| Model output | The FBP-U-Net post-processed reconstruction in the same normalized benchmark domain |
| Ground truth | The reviewed `362×362` benchmark reference, not a clinical HU image |
| Absolute error | The per-pixel absolute difference between model output and ground truth, shown with a fixed reference range |

The page also reports MAE, RMSE, and PSNR against this single reference. Lower
MAE and RMSE and higher PSNR indicate a closer numerical match for this case
and this fixed data range. They do not show performance on another scanner,
anatomy, dose level, or patient population.

Do not supply an arbitrary CT image, an HU slice, or a scanner sinogram to this
model. The learned component is tied to the documented DIVal LoDoPaB FBP
operator and benchmark geometry.

## Workflow 2 — DeepInverse synthetic MRI reconstruction

### Before you start

This workflow also needs PyTorch but no file from you. It generates a
deterministic `128×128` digital phantom in memory, transforms it to simulated
single-coil k-space, and applies a fixed undersampling mask. The demonstration
is designed to make undersampling and learned data consistency visible.

### Run the workflow

1. Open **Models → Bundled models**.
2. Select **DeepInverse MRI MoDL Teaching Model** and read its k-space/mask
   contract.
3. Select **Run synthetic MRI demo**.
4. Compare the source phantom, zero-filled baseline, MoDL result, absolute
   error, and metrics.

This demonstration currently has no JSON export. Its compound k-space and mask
input is created only by the dedicated bundled workflow; the generic external
manifest screen cannot borrow it.

### Read this workflow's views

| View | Meaning |
| --- | --- |
| Digital phantom | The deterministic source image used to create the simulation |
| Zero-filled IFFT | The direct inverse transform after missing k-space samples are filled with zero; this is the baseline |
| MoDL magnitude | The displayed magnitude derived from the model's two-channel complex reconstruction |
| Absolute error | The per-pixel absolute difference between the MoDL magnitude and source phantom |

MAE, RMSE, and PSNR describe agreement with the generated phantom. They are
useful for this controlled mechanism demonstration only. The simulated k-space
must remain labelled as image-derived simulation, not as a clinical acquisition
or scanner raw data.

## Workflow 3 — MONAI BraTS 3-D segmentation

### Before you start

The desktop workflow requires one user-managed directory in which the following
five 3-D NIfTI files can each be identified uniquely:

- T1;
- T1ce;
- T2;
- FLAIR;
- Task 1 SEG with integer labels limited to `{0,1,2,4}`.

Names such as `case_t1.nii.gz`, `case_t1ce.nii.gz`, `case_t2.nii.gz`,
`case_flair.nii.gz`, and `case_seg.nii.gz` make the modalities unambiguous. The
four MRI volumes and SEG must have matching shape, spacing, affine, orientation,
world coverage, qform, and sform. The model path requires co-registered,
aligned 1 mm data. It does not silently register or resample a case.

No BraTS case is bundled. Obtain and use data only when you are authorized to
do so. Setup validates the selected files in place; it does not download, copy,
rename, or upload them.

### Set up and run the workflow

1. Open **Models → Bundled models** and select **MONAI BraTS 3D Segmentation**.
2. Select **Set up local BraTS 2021…**, then choose the directory containing
   the case.
3. Review the validation report. Fix any missing or duplicate modality,
   unreadable data, non-finite value, geometry mismatch, qform/sform mismatch,
   or invalid SEG label before continuing.
4. Saving the optional anonymous local manifest requires accepting the terms
   shown in the dialog. The manifest contains hashes and geometry, not file
   paths, names, or person identifiers. A valid in-session report is enough to
   run; saving this manifest is optional.
5. After validation, MONAI is no longer disabled. Select **Run BraTS
   segmentation**.
6. Review the actual device and parameters. The reviewed desktop defaults are
   a `96×96×96` sliding-window patch, `0.5` overlap, and `0.5` probability
   threshold.
7. To keep a pixel-free record of parameters and metrics, select **Export
   experiment JSON…** after the run.

### Read this workflow's views

The workflow canonicalizes the aligned inputs to RAS+, applies a nonzero-voxel
z-score separately to each modality, performs full-volume sliding-window
inference, and maps results back to source geometry. It shows:

- **WT probability**, **TC probability**, and **ET probability**. The adapter
  explicitly reorders the model's native TC/WT/ET output to the displayed
  WT/TC/ET order.
- **WT / TC / ET masks**, created with the visible threshold. WT uses reference
  labels `{1,2,4}`, TC uses `{1,4}`, and ET uses `{4}`.
- Dice, HD95 in millimetres, and absolute volume error in millilitres for each
  region against the supplied SEG.

This model was trained for the documented BraTS 2018 glioma domain. Running it
on a BraTS 2021 Task 1 case is an explicit domain shift, not external clinical
validation. It is not a general model for other lesions, diseases, scanners,
or acquisition protocols.

## Read the results

Use the displayed images and metrics together. No single number is sufficient
to establish that a result is trustworthy.

| Term | How to read it |
| --- | --- |
| Baseline | FBP for DIVal or zero-filled IFFT for DeepInverse; it shows what the learned result is being compared with |
| Model output | A task-specific estimate, not a diagnosis and not automatically a physical measurement |
| Ground truth / reference | The prepared benchmark or supplied SEG used for comparison; it is not proof that the model generalizes |
| Absolute error | Where and by how much the estimate differs from the reference; darker or brighter meaning depends on the displayed scale |
| MAE / RMSE | Average numerical error; lower is closer for the same case and scale |
| PSNR | Signal-to-error ratio in decibels; higher is closer only when the reference range and task are the same |
| Probability | A model output after sigmoid, not calibrated clinical confidence; the BraTS mask threshold defaults to `0.5` |
| Dice | Region overlap from `0` to `1`; higher is more overlap, but small structures can be unstable |
| HD95 | A robust boundary-distance measure in millimetres; lower is closer, and `n/a` can occur for empty-region cases |
| Absolute volume error | Difference in segmented volume in millilitres; closer to zero is better for that reference |

Always check the input contract, preprocessing, actual device, warnings,
display scale, and reference before interpreting a metric. A visually
convincing reconstruction or mask can still suppress, shift, or invent
structure. A single case, smoke test, or golden check cannot establish clinical
performance.

## Choose a device

The desktop workflow records the device it actually used. The command-line
checks accept these choices:

| Choice | Behaviour | When to use it |
| --- | --- | --- |
| `auto` | Prefers CUDA when PyTorch reports it available; otherwise uses CPU. A CUDA load or inference error is reported before a CPU retry. | Recommended default |
| `cpu` | Always uses CPU | Portable checks, systems without a compatible GPU, or recovery from GPU memory pressure |
| `cuda` | Requires CUDA and fails clearly when it is unavailable; it does not silently pass on CPU | A deliberate CUDA-only check |

DIVal and DeepInverse are small teaching workflows. Full-volume MONAI
segmentation can be much slower on CPU, so a compatible GPU is recommended for
that workflow. If a GPU run exhausts memory, close other GPU workloads and use
CPU if necessary. Do not change modality order, spacing, normalization, patch
contract, or other scientific inputs merely to make a run finish.

## Solve common problems

| Problem | What to do |
| --- | --- |
| `torch` is missing or the model buttons cannot run | Activate `openmedvisionx`, install the CPU runtime or the correct platform-specific PyTorch build, restart the application, and verify again |
| Integrity verification fails | Restore the complete project from a trusted release; do not edit, rename, or mix bundle files from different versions |
| CUDA is unavailable | Use `auto` or `cpu`, or reinstall a PyTorch build that matches the exact OS, GPU, and driver |
| MONAI remains disabled | Complete **Set up local BraTS 2021…** successfully in the current session; a saved manifest alone does not replace revalidation |
| BraTS validation fails | Make each of T1, T1ce, T2, FLAIR, and SEG uniquely identifiable, then fix shape, spacing, affine, orientation, qform/sform, finite-data, or label errors reported by the dialog |
| BraTS is very slow or runs out of memory | Confirm the actual device, close other GPU work, and retry on CPU if needed; CPU full-volume inference is expected to be slower |
| Export is unavailable | Finish a supported LoDoPaB or BraTS run first. DeepInverse currently has no JSON export |
| A valid external manifest cannot use these inputs | This is expected: the current generic path accepts one active 2-D image, not compound k-space/mask input, four MRI volumes, prompts, or a complete 3-D/4-D volume |
| A result looks plausible but inconsistent | Recheck the model card, input source, modality/channel order, units, preprocessing, warnings, display range, and reference before repeating the run |

For additional startup, dependency, data, and GPU symptoms, use the
[Troubleshooting guide](TROUBLESHOOTING.md).

## Related guides

- [Documentation index](INDEX.md) — choose the shortest path for the task you
  want to complete.
- [Quickstart](QUICKSTART.md) — install and launch the base application.
- [User guide](USER_GUIDE.md) — learn the complete desktop workspace.
- [Data formats](DATA_FORMATS.md) — prepare DICOM, NIfTI, raster images, masks,
  and annotations safely.
- [Custom local models](CUSTOM_MODELS.md) — create and run a local model
  manifest within the current desktop capability boundary.
- [Troubleshooting](TROUBLESHOOTING.md) — diagnose installation, runtime, and
  input-validation failures.

**Models → External manifests** is separate from the three reviewed bundled
workflows. It currently accepts one supported active 2-D image or plane. It
cannot reuse DeepInverse's synthetic MRI k-space/mask generator or collect the
four modalities used by MONAI. A visible capability rejection is intentional:
it prevents a scientifically invalid input from being fabricated.

## Release integrity appendix

The details in this appendix are frozen release facts used to identify the
exact reviewed assets. Most users only need the verification commands above.

### Frozen model payloads

The allow-list contains exactly these three IDs:

- `dival-lodopab-fbpunet`;
- `deepinv-mri-modl`;
- `monai-brats-segmentation`.

| Bundle | File | Exact size | SHA-256 |
| --- | --- | ---: | --- |
| DeepInverse | `weights.npz` | 9,929 bytes | `702658708828d13135228e32fef980ba1048e200f5c2fa4ebf54fa12d653f8ab` |
| DeepInverse | `golden.npz` | 11,251 bytes | `2b709a7c16aedd65110aaf929bb2c6cc35db1c94d9fe01b751a29b06634d29af` |
| DIVal | `weights.npz` | 2,318,110 bytes | `8b18fd2a88355ddec043ae7c737ddf3321424e2ba52102869d3dbaf6bf68504c` |
| DIVal | `golden.npz` | 920,732 bytes | `6789c93592ab6cfd3d4924e6d077ce8966d0e7fd7bb0b7f1a7305d66f742a3df` |
| MONAI | `model.ts` | 18,911,784 bytes | `729980a0bd9347bf2397701eb329e12517918dc282a2d09c40458e95b24ceed9` |
| MONAI | `golden.npz` | 330,439 bytes | `d7982ada82f56b28615ed6ad170641ee1f3f0cb6a819285598c0380efa957e45` |

The six model and golden payloads total exactly **22,502,245 bytes**, below the
25 MiB release budget. DIVal and DeepInverse use reviewed numeric NPZ payloads
loaded with `allow_pickle=False`; MONAI uses the pinned TorchScript artifact.
The verifier also checks each machine-readable record, license evidence, file
size, digest, allow-list membership, and total budget.

### Frozen LoDoPaB teaching case

The separately packaged case is `lodopab-ct-test-03456` (case version 2), test
index 3456 from [LoDoPaB-CT](https://doi.org/10.5281/zenodo.3384092). Its
`sample.npz` is exactly **1,912,858 bytes** with SHA-256
`b323cdef2529927336069b3385605d1049117fe69e59583072861fa573493846`.
It contains finite `float32` arrays for the `1000×513` observation, `362×362`
fixed Hann FBP, and `362×362` ground truth, plus canonical metadata. It is
redistributed under **CC BY 4.0** with the bundled
[NOTICE and attribution](../src/workbench/resources/teaching_cases/lodopab-ct/NOTICE.md).

The safe loader checks the case record, exact size and hash, NPZ entry
allow-list, and every array's hash, shape, and dtype before opening it with
`allow_pickle=False`. The case contains no DICOM metadata, patient identifier,
or source path. One public benchmark example is not evidence of
generalization or clinical validity.

### Tested GPU reference

The repository's recorded Windows GPU test used an **NVIDIA RTX 4060 Laptop
GPU with 8 GiB VRAM** and PyTorch `2.13.0+cu130`:

```bash
python -m pip install -r requirements/torch-cu130.txt
```

This command is only for that tested driver/GPU configuration. It is not a
universal CUDA recommendation. For another OS, NVIDIA GPU, or driver, use the
official PyTorch selector; for a CPU-only installation, use the CPU command in
the first section of this guide.
