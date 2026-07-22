# OpenMedVisionX User Guide

[简体中文](USER_GUIDE.zh-CN.md) · [Documentation home](INDEX.md) · [Quickstart](QUICKSTART.md) · [Data and formats](DATA_FORMATS.md) · [Troubleshooting](TROUBLESHOOTING.md)

This guide follows the desktop application from left to right. It explains what
ordinary users can do, what each action needs, what success looks like, and
where the scientific and privacy boundaries are.

> [!WARNING]
> OpenMedVisionX is for learning and research. Its outputs are not clinical
> findings and must not be used for diagnosis or patient decisions.

## Before you begin

Complete the [Quickstart](QUICKSTART.md), launch the desktop application, and
keep the terminal open so you can read any dependency or device error. The
application opens in English. Select **中文** in the header to switch to
Simplified Chinese; select **English** to switch back. Changing language keeps
the current image, experiment, form values, and results in place.

The header says **Local · Research**. This means:

- local images and model files are not uploaded automatically;
- the CT, image, learning, evaluation, and bundled-model workflows can run
  locally;
- the AI Assistant sends a request only after **Enable network** is selected;
- sending an image requires a second opt-in and a one-time review of the exact
  outbound PNG; and
- opening an official website or a link in an assistant response is a separate
  user action.

There is no traditional menu bar. The six main workspaces are tabs:

| Shortcut | Workspace | Main use |
| --- | --- | --- |
| `Alt+1` | **Images** | Open, inspect, measure, annotate, and export images. |
| `Alt+2` | **CT Lab** | Learn projection and reconstruction with a phantom or active plane. |
| `Alt+3` | **Models** | Run reviewed bundled demonstrations or a compatible local manifest. |
| `Alt+4` | **Learn** | Follow six structured lessons. |
| `Alt+5` | **Evaluate** | Validate a dataset manifest and evaluate binary probabilities. |
| `Alt+6` | **AI Assistant** | Ask a teaching provider or inspect structured artifact contracts. |

## Navigation and the active view

The **Images** workspace can show one raster image, a page sequence, or several
views of a 3-D volume. One visible view is always the **active view**. Click or
focus a view to activate it. The context line above the images names the active
view and its current page or slice.

The active view is important because it is the shared input for later actions:

- **CT Lab → Active image (advanced)** uses the active 2-D plane;
- **Models → External manifests** uses the active 2-D plane;
- an area drawn with **Area / ROI** can become the current CT comparison ROI;
  and
- **AI Assistant → Attach active rendered plane** sends the full active plane
  after display mapping.

For a validated 3-D volume, Axial, Coronal, and Sagittal views use RAS+
coordinates and show patient-direction markers. Double-click a location in
**Pan / zoom** mode to move all three planes to the same point. Hovering shows
pixel values and, when valid geometry exists, RAS+ coordinates in millimetres.

The active rendered plane is not a screenshot of the viewport. Zoom, pan,
crosshairs, measurements, imported overlays, and annotations are not part of
the PNG used by export or the assistant.

## Images

### Purpose

Use **Images** to open a local raster, DICOM source, or NIfTI volume; inspect
decoded values and geometry; navigate pages or orthogonal planes; make teaching
measurements; pair a plane-local overlay; import a referenced clinical layer;
and save a rendered plane or pixel-free record.

### Prerequisites

No image is required when the application starts. For NIfTI input, install the
optional NIfTI dependencies described in the [Quickstart](QUICKSTART.md). Use
only data you are authorized to access. Keep a backup of source data; the
viewer does not edit it.

### Steps

1. **Open the source.**

   - Select **Open image / DICOM ZIP** (`Ctrl+O`) for `.dcm`, `.dicom`, `.zip`,
     `.nii`, `.nii.gz`, `.png`, `.jpg`, `.jpeg`, `.tif`, or `.tiff`.
   - Select **Open DICOM folder** (`Ctrl+Shift+O`) for a directory containing a
     DICOM study. Discovery is limited to the selected file, folder, or ZIP.
   - While decoding is active, select **Cancel loading** to request a safe
     cancellation. The previously loaded image remains usable until a
     replacement finishes successfully.

2. **Resolve an ambiguous medical source.**

   - If a DICOM source contains several series, the **Choose a DICOM series**
     dialog appears. Compare Status, Modality, Description, Series no.,
     Instances / slices / frames, Dimensions, Pixel spacing, Slice thickness,
     Geometry, and Warnings. Unsupported rows are disabled. Select one
     available row and choose **Open selected series**.
   - The dialog does not display patient fields, paths, raw UIDs, or hashes.
   - If a NIfTI contains several 3-D volumes, choose the required one in
     **Time point / channel (1 to N)**. The number shown to users starts at 1.

3. **Navigate and inspect.**

   - A single 2-D raster uses the main view only.
   - A page sequence shows **Slice / page** and, when supported, **Play pages**.
   - A 3-D volume shows Axial Z, Coronal Y, and Sagittal X sliders. An axial
     maximum-intensity projection appears when one is available.
   - Use the mouse wheel to zoom and drag in **Pan / zoom** mode. Select
     **Fit views** (`Ctrl+0`) to fit every visible view.
   - Select **Histogram** to inspect decoded values in the active plane.
   - Read **Image details** for type, source, shape, data type, intensity
     semantics, capabilities, color/bit depth, spacing, modality, origin,
     direction, and privacy-filtered runtime metadata.

4. **Adjust display only.**

   - Grayscale data exposes **Lower** and **Upper** bounds. CT data with valid
     HU semantics labels the control **HU window**.
   - Color data exposes **RGB brightness**, **RGB contrast**, and **RGB gamma**.
   - Select **Auto range** to return to a useful automatic display range.

   These controls change the view, not the decoded array or its quantitative
   meaning.

5. **Measure or mark the active plane.**

   - Choose **Distance**, drag between two points, and read the distance in
     `px` or `mm` in the status strip.
   - Choose **Area / ROI**, drag a rectangle, and read area in `px²` or `mm²`.
     Numeric images also report mean and standard deviation. A rectangle at
     least 3 × 3 pixels on the current active plane can be used by CT Lab for
     ROI metrics.
   - Choose **Annotation** and click to place point marks.
   - Press `Esc` to cancel an unfinished drag. Select **Clear marks** to remove
     interactive marks and the explicitly paired plane-local overlay.
   - Ordinary rasters have pixel units unless spacing is known. Select
     **Set raster spacing…**, enter positive **X spacing (mm)** and
     **Y spacing (mm)**, and choose **Apply** only when the values come from a
     trusted source. OpenMedVisionX never infers medical spacing from DPI.

6. **Choose the correct annotation workflow.**

   **Plane-local teaching overlays** are quick comparisons tied to exactly one
   visible plane:

   - **Mask…** accepts a lossless, single-page, grayscale PNG or TIFF whose
     height and width exactly match the active plane. Every non-zero value is
     displayed as one binary mask. It is never resized.
   - **Annotations…** accepts a UTF-8 JSON file of at most 4 MiB. It must use
     top-left pixel coordinates, match the active plane, contain at least one
     box or point, and contain no unknown top-level fields or external
     references.

   A minimal annotation file is:

   ```json
   {
     "schema_version": 1,
     "coordinate_system": "pixel_xy_top_left",
     "image_size": [512, 512],
     "boxes": [
       {"x1": 20, "y1": 30, "x2": 120, "y2": 160, "label": "example"}
     ],
     "points": [
       {"x": 64, "y": 80, "label": "landmark"}
     ]
   }
   ```

   `image_size` is `[width, height]`. Box coordinates satisfy
   `0 ≤ x1 < x2 ≤ width` and `0 ≤ y1 < y2 ≤ height`; point coordinates must be
   inside the image. A plane-local overlay disappears when another page or
   slice becomes active and reappears when you return to its plane.

   **Referenced study layers** are validated against a loaded DICOM or NIfTI
   series and are managed in the **Layers** sidebar:

   - **DICOM SEG / RTSTRUCT…** accepts one `.dcm` or `.dicom` annotation after
     the referenced DICOM image series is open.
   - **Label map…** accepts a lossless PNG, lossless TIFF, `.nii`, or `.nii.gz`
     integer label map after its intended DICOM or NIfTI reference is open.
     JPEG, color label maps, floating-point label storage, and a silent reference
     guess are rejected. A 4-D label-map NIfTI requires an explicit volume.
   - If the imported geometry differs from the display grid, review the source
     and display shapes, differing components, interpolation, and outside
     value. Choose **Create resampled preview** to retain the original hidden
     layer and add a visible derived layer; choose **Keep original hidden** to
     retain only the unresampled hidden layer; or choose **Cancel** to change
     nothing. Discrete labels use nearest-neighbour interpolation; continuous
     fractional data uses the proposed continuous interpolation.
   - In **Layers**, select a layer to review Type, Created by, Validation,
     Source format, Geometry, and Revision. Use **Visible**, **Locked**, and
     **Opacity** (0–100%). Segmentation labels have individual visibility
     checkboxes and written color values. The base image itself cannot be
     edited with these presentation controls.

7. **Export a derived result.**

   - **Export PNG…** writes the full active plane with the current grayscale or
     color display mapping to a new PNG. It does not include marks, crosshairs,
     masks, contours, or layer overlays.
   - **Save record…** writes a JSON record containing plane selection, display
     parameters, pairing flags, and the latest numeric measurements. It does
     not contain image pixels or a source path.

### How to tell it worked

- The status strip reports `Loaded <kind>: <shape>` and **Image details** is
  populated.
- The context line names the expected **Active view** and page or slice.
- A plane-local import reports an explicit pairing on the current plane.
- A referenced import appears under **Layers** with a validation and geometry
  state.
- A completed export reports the new filename. Open it separately if you need
  to verify the exact saved pixels.

### Boundaries and cautions

- A page sequence is not promoted to a 3-D volume without validated geometry.
- A very large flat raster may be represented by a bounded preview. The warning
  identifies its original and displayed size; full-resolution inference is not
  available in that session.
- JPEG can contain lossy artifacts. Display adjustment cannot restore source
  information or create HU/SUV semantics.
- Changing page or slice clears interactive measurements because their
  coordinate frame changed.
- Plane-local overlays and referenced study layers are different workflows.
  Pairing a JSON box does not create a study layer; importing a clinical layer
  is not the same as pairing the current plane.
- Exports are derived files and should use a new destination. They never modify
  the opened source.

## CT Lab

### Purpose

Use **CT Lab** to learn the Radon transform, backprojection, filtering, direct
Fourier reconstruction, and SART. The recommended route uses a deterministic,
non-negative synthetic attenuation phantom and requires no medical file.

### Prerequisites

Nothing is required for **Synthetic phantom (recommended)**. For
**Active image (advanced)**, first open an image in **Images** and activate the
intended 2-D plane. The active plane must be finite, non-negative, non-constant,
square, and grayscale. If **Circular support** is selected, values outside the
reconstruction circle must be zero.

### Steps

1. Choose **Input**:

   - **Synthetic phantom (recommended)** enables **Phantom size**, from 64 to
     384 px in 32 px steps; the default is 192 px.
   - **Active image (advanced)** uses the current active plane exactly as a
     mathematical image-domain input.

2. Set the simulated acquisition:

   - **Range °** is `180` or `360`; the default is `180`.
   - **Angles** is the number of projections, from 8 to 1440; the default is
     180.
   - **Circular support** is selected by default.

3. Choose **Algorithm** and its visible options:

   | Algorithm | Options |
   | --- | --- |
   | `FBP` | **FBP filter**: `ramp`, `shepp-logan`, `cosine`, `hamming`, or `hann`. |
   | `BP` | No additional parameters. |
   | `DFR` | **DFR interpolation**: `nearest`, `linear` (default), or `cubic`. |
   | `SART` | **SART iterations**: 1–100 (default 5); **Relaxation**: 0.01–1.00 (default 0.15). |

4. Run the workflow:

   - In phantom mode, select **Run first phantom experiment**. It generates the
     phantom, simulates the sinogram, and automatically reconstructs with the
     selected algorithm.
   - In active-image mode, select **Run image-domain simulation**. When the
     sinogram is ready, select **2. Reconstruct**.
   - Select **Cancel** during a background operation to request safe
     cancellation.

5. Review the four main views: **Input**, **Sinogram**, **Reconstruction**, and
   **Absolute error heatmap**. Use **Intermediate process** to inspect available
   Radon projection snapshots, detector spectra or filtered projections, BP/FBP
   progress, DFR spectra, SART iterations, signed normalized difference, and
   absolute error.

6. Read the metrics. Input and reconstruction use one shared evaluation range.
   The summary includes MSE, PSNR, and SSIM; when a meaningful raw unit exists,
   it also includes MAE, RMSE, and bias. A valid current **Area / ROI** from
   **Images** adds ROI-specific MSE, PSNR, and SSIM.

7. Select **Export PNG…** to save the reconstruction image, or
   **Save record…** to save source mode, geometry settings, algorithm options,
   evaluation range, and numeric metrics as a pixel-free JSON record.

### How to tell it worked

- Phantom mode progresses from **Running in the background** to
  **Experiment complete** without asking for an image.
- The sinogram view fills before reconstruction; all four views fill after the
  full experiment.
- The metric line starts with **Evaluation range** and shows MSE, PSNR, and
  SSIM.
- **Export PNG…** and **Save record…** become enabled only after a
  reconstruction exists.

### Boundaries and cautions

- The phantom uses illustrative `mm⁻¹` values and a fixed 0–0.03 `mm⁻¹`
  evaluation range. It is not scanner raw data.
- Active-image mode is an image-domain teaching simulation. It does not convert
  HU to linear attenuation, recover scanner projections, silently convert RGB,
  crop a non-square image, or discard negative values.
- A 360° parallel-beam scan contains redundant information; the application
  folds and averages the redundant half.
- Changing phantom size, angular range, angle count, or circular support
  invalidates the sinogram and reconstruction. Changing only algorithm options
  keeps the sinogram but invalidates the reconstruction.
- A pleasant display is not proof of a better reconstruction. Compare results
  only on the shared range and with the reported metrics.

## Models

### Purpose

Use **Models** for two separate paths:

- **Bundled models** presents three reviewed, fixed teaching references with
  visible contracts, model cards, integrity facts, device choice, and dedicated
  demonstrations.
- **External manifests** inspects a user-selected YAML manifest, loads its
  existing local model, and runs one compatible active 2-D image.

### Prerequisites

Install the model runtime required by the workflow; see the
[Bundled model guide](MODEL_BUNDLES.md). Keep all user-managed weights and
datasets outside the repository unless their own terms say otherwise.

For an external model, you need a reviewed `.yaml` or `.yml` manifest and every
required local weight file it references. For the MONAI workflow, obtain an
authorized local BraTS 2021 case through the official process; OpenMedVisionX
does not download it.

### Steps

1. **Choose a bundled execution device.** In **Bundled models**, set
   **Execution device** to **Auto · prefer GPU**, **CPU**, or **CUDA GPU**.
   Select **Verify catalog & device** to verify all three registered SHA-256
   values and inspect the local Torch device. Every later model load remains
   integrity-gated even if this optional session-wide check has not been run.

2. **Inspect the three reviewed cards and run the appropriate dedicated path.**

   | Bundled reference | Run action and result |
   | --- | --- |
   | [DIVal LoDoPaB FBP-U-Net](../src/workbench/resources/model_bundles/dival-lodopab-fbpunet/MODEL_CARD.md) | Select **Run LoDoPaB case**. Compare the public LoDoPaB observation, reviewed Hann FBP, FBP-U-Net output, ground truth, and fixed-reference absolute error. Read MAE, RMSE, PSNR, requested/actual device, timing, fallback, and warnings. **Export experiment JSON…** writes a pixel-free reproducibility record. |
   | [DeepInverse MRI MoDL](../src/workbench/resources/model_bundles/deepinv-mri-modl/MODEL_CARD.md) | Select **Run synthetic MRI demo**. The dedicated path creates a deterministic digital phantom, simulates single-coil k-space and a sampling mask, and compares the phantom, zero-filled IFFT, MoDL magnitude, and absolute error. Read MAE, RMSE, PSNR, timing, fallback, and warnings. |
   | [MONAI BraTS 3D Segmentation](../src/workbench/resources/model_bundles/monai-brats-segmentation/MODEL_CARD.md) | MONAI starts **disabled** for inference until one local case is validated. Select **Set up local BraTS 2021…**, complete the guided validation, then select **Run BraTS segmentation**. |

3. **Set up the local MONAI BraTS case when needed.**

   - In **1 · Official access**, optionally open the official page after the
     external-link confirmation. Opening it does not download data or accept
     terms.
   - In **2 · Select and validate**, choose one local case folder containing
     T1, T1ce, T2, FLAIR, and SEG NIfTI volumes. Validation begins in the
     background. Use **Validate again** (`Ctrl+R`) after correcting a problem,
     or **Cancel validation** to stop safely.
   - In **3 · Review validation**, check each input's status, shape, spacing,
     orientation, SEG label counts, and every warning or error. The five inputs
     must share compatible geometry; SEG labels must be valid for the task.
   - A valid report enables the in-session MONAI run. Saving a manifest is
     optional. To save one, review the official terms yourself, select
     **Official terms reviewed**, and choose **Save manifest…** (`Ctrl+S`). The
     JSON contains checksums and geometry summaries, not pixels, filenames,
     absolute paths, or person identifiers.
   - Full-volume inference uses T1ce → T1 → T2 → FLAIR, non-zero-voxel z-score
     normalization, sliding-window patches, and an explicit threshold. The UI
     displays WT, TC, and ET probability maps plus a composite mask on an axial
     result slice. The details report Dice, HD95, and absolute volume error
     against SEG, device, patch settings, timing, hashes, and domain warnings.

4. **Run an external manifest.** Open **External manifests** and follow the
   numbered controls:

   1. Select **1. Choose manifest…** and open a local YAML manifest. Inspection
      validates the declared identity, task, runtime, license, inputs, outputs,
      preprocessing, capabilities, and local weight paths before model code
      runs.
   2. Read **Model resource status** and the compatibility message. Required
      weights must exist at the manifest's declared local paths. If a reviewed
      GitHub source is declared, **Open official guide** opens it in the browser;
      the application does not download the file.
   3. Select **2. Load model**. If the manifest uses a Python adapter, read the
      warning and approve only after reviewing the adapter. Its separate process
      contains crashes and dependencies but is not a security sandbox.
   4. Open and activate the intended image in **Images**. This desktop surface
      accepts exactly one declared 2-D `image` input. It checks input count,
      semantic, dimensionality, modality, and required spacing.
   5. If only a bounded raster preview is in memory, explicitly select
      **Allow preview-only teaching inference**, then review the second,
      model-specific confirmation. The result is recorded as preview-only.
   6. When the status begins **Ready: compatible**, select
      **3. Run inference**. Use **Stop** to request cancellation.
   7. Compare **Current model input** with every output tab. Typed mask, box, and
      keypoint outputs are drawn in source coordinates; image and heatmap outputs
      use declared or stable numeric ranges; text and table-like results appear
      in the summary.

### How to tell it worked

- **Verify catalog & device** reports `3/3 models verified` and shows either a
  CUDA device or an explicit CPU fallback.
- A bundled run changes its operation status to **Completed locally** and fills
  the **Experiment** views and result details.
- A valid BraTS case reports that all local inputs passed geometry and label
  validation; the run action changes from setup to segmentation.
- An external manifest progresses through **Model manifest ready**,
  **Model ready**, and then a compatible **Ready** message. After inference,
  the summary reports task, runtime, duration, input resolution, and
  visualization kinds.

### Boundaries and cautions

- Bundled models are fixed educational references, not general clinical
  models. DIVal accepts the reviewed LoDoPaB Hann-FBP domain, not arbitrary CT
  HU or scanner sinograms. DeepInverse uses image-derived simulation, not raw
  scanner k-space. MONAI was trained in a BraTS 2018 domain and is not validated
  as a general brain-lesion model or clinical system.
- The DeepInverse synthetic MRI demonstration currently has no JSON export.
  DIVal and a completed BraTS run can export pixel-free JSON records.
- The generic external-manifest GUI cannot collect multiple images, a whole
  3-D/4-D volume, prompts, k-space plus mask, sinograms, or other compound
  inputs. A valid but incompatible manifest remains blocked rather than being
  silently downgraded.
- External files, weights, dependencies, and URLs are never fetched
  automatically. Missing local resources keep **2. Load model** disabled.
- External output visualizations do not modify the source image, create a Study
  layer, or provide a desktop output-file export.

## Learn

### Purpose

Use **Learn** as the guided reading path that connects image semantics, physical
geometry, reconstruction, models, and responsible assistant use.

### Prerequisites

No file, model, credential, or network connection is required to read a lesson.
The recommended first experiment also needs none of them.

### Steps

1. Choose one item in **Experiment**:

   - **2-D pixels, bit depth, and interpolation**;
   - **DICOM and NIfTI physical geometry**;
   - **Radon transform and filtered backprojection**;
   - **SART iterative reconstruction**;
   - **External model inference**; or
   - **Multimodal AI teaching assistant**.

2. Read the seven sections in order: **Principle**, **Formula**,
   **Parameter explanation**, **Steps**, **Expected observation**,
   **Common mistakes**, and **Reflection question**.

3. For a hands-on start, select **Start first experiment**. The application
   switches to **CT Lab**, restores the recommended 180° phantom/FBP path, and
   starts the deterministic experiment.

4. Continue with the [Teaching curriculum](TEACHING_CURRICULUM.md) when you want
   a longer sequence of exercises and reflection prompts.

### How to tell it worked

- Selecting a lesson updates all seven sections and the status bar reports
  `Learning path: <title>`.
- **Start first experiment** opens CT Lab; progress and then four result views
  appear without requesting a medical file.

### Boundaries and cautions

- Lessons explain concepts and suggest experiments; a described resource is
  not necessarily generated by the desktop UI.
- Formulae and expected observations are teaching aids, not validation of a
  specific acquisition or model.
- The public LoDoPaB case contains no DICOM metadata, patient identifiers, or
  clinical narrative, but it still represents one benchmark domain only.

## Evaluate

### Purpose

Use **Evaluate** to check that pseudonymous dataset groups do not cross train,
validation, and test splits; evaluate one set of binary labels and predicted
probabilities at an explicit threshold; inspect calibration; and export a
pixel-free evidence record.

### Prerequisites

The built-in example probabilities are ready to evaluate immediately. A
dataset manifest is optional. If used, it must be UTF-8 JSON, YAML, or YML,
1 byte to 4 MiB, use schema version 1, be explicitly deidentified, contain no
filesystem paths or identity fields, and reference artifacts by lowercase
SHA-256 rather than by filename.

A minimal JSON manifest looks like this:

```json
{
  "schema": "openmedvisionx-dataset-manifest/v1",
  "schema_version": 1,
  "dataset_id": "demo-dataset",
  "dataset_version": "v1",
  "task": "classification",
  "license_id": "CC-BY-4.0",
  "deidentified": true,
  "label_schema": {},
  "provenance": {},
  "samples": [
    {
      "sample_id": "sample-001",
      "group_id": "group-a",
      "split": "train",
      "artifact_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "modality": "MR",
      "labels": {"target": 0},
      "site": null,
      "scanner": null,
      "deidentified": true
    }
  ]
}
```

Every `sample_id` must be unique. Every `group_id` must occur in only one of
`train`, `validation`, or `test`.

### Steps

1. Optionally select **Open manifest…** and choose a `.json`, `.yaml`, or `.yml`
   file. Review the sample and group counts for train, validation, and test.
2. In **Reference labels**, enter comma-separated or line-separated `0` and `1`
   values. Both classes must appear.
3. In **Predicted probabilities**, enter the same number of finite values in
   `[0, 1]`.
4. Set **Decision threshold** from 0.00 to 1.00. The default is 0.50 and the
   step is 0.05. No threshold is selected automatically from the data.
5. Select **Evaluate**.
6. Review **Metrics**:

   - Accuracy, Sensitivity, Specificity, PPV, NPV, and F1 depend on the selected
     threshold.
   - AUROC and AUPRC summarize ranking; AUROC does not choose a deployable
     operating point.
   - Brier score and Expected calibration error are lower when probability
     quality is better.
   - The **95% interval** column is shown where the workspace has a supported
     interval.

7. Review **Calibration**. Ten probability bins show count, mean score, and
   observed positive rate. Empty bins show `—`.
8. Select **Export experiment record…** and save JSON or YAML. The record stores
   a digest and shape for the entered vectors, the threshold, metrics, dataset
   manifest identity when present, and a manual-input warning. It does not
   store the raw labels, raw probabilities, image pixels, or a source path.

### How to tell it worked

- A valid manifest reports **Manifest validated. Group leakage check passed.**
  and displays dataset ID, version, and task.
- A valid evaluation reports **Evaluation complete** and fills ten metric rows
  plus ten calibration rows.
- **Export experiment record…** becomes enabled only after a valid evaluation;
  the status then reports **Experiment record exported.**

### Boundaries and cautions

- The desktop evaluation form is for binary classification only. Segmentation,
  detection, and registration metrics may appear in dedicated workflows, but
  there is no generic desktop form for them here.
- Manually entered values have no trustworthy provenance unless you record it
  elsewhere. The exported digest proves which vectors were used; it does not
  establish that the labels are correct.
- Group-safe splitting prevents one common leakage route, not every source of
  bias or dataset shift.
- A metric or confidence interval is evidence about the supplied sample, not a
  clinical claim.

## AI Assistant

### Purpose

Use **AI Assistant → Teaching chat** to ask a configured provider about concepts,
parameters, and learning results. Use **Structured artifacts · API preview** to
inspect the contracts and local review state of an already typed artifact
provided by a trusted host integration.

### Prerequisites

Teaching chat requires a provider account or an intentionally unauthenticated
loopback service. Prepare the exact model ID and a credential **reference**, not
the secret itself.

| Provider | Default endpoint | Default credential reference |
| --- | --- | --- |
| OpenAI | `https://api.openai.com/v1/responses` | `env:OPENAI_API_KEY` |
| Anthropic | `https://api.anthropic.com/v1/messages` | `env:ANTHROPIC_API_KEY` |
| Moonshot/Kimi | `https://api.moonshot.cn/v1/chat/completions` | `env:MOONSHOT_API_KEY` |
| Zhipu GLM | `https://open.bigmodel.cn/api/paas/v4/chat/completions` | `env:ZHIPU_API_KEY` |
| DeepSeek | `https://api.deepseek.com/chat/completions` | `env:DEEPSEEK_API_KEY` |
| OpenAI-compatible | Replace the placeholder with the service's exact endpoint | `env:OPENMEDVISIONX_API_KEY` |

Valid credential references are `env:NAME`,
`keyring:<service>/<username>`, and `none`. `none` is accepted only for
`localhost`, a `.localhost` name, or a loopback IP. Never paste an API key into
the **Credential reference** field.

To attach an image, first open it in **Images**, activate the intended plane,
and inspect it for burned-in private text. DeepSeek is configured as text-only
in this release, so its **Vision input** and image attachment controls are
disabled.

### Steps

1. Open **Teaching chat**. In **Provider configuration**:

   - choose **Provider**;
   - enter the exact **Model ID** supplied by that provider;
   - review or edit **Endpoint**;
   - enter a non-secret **Credential reference**; and
   - select **Enable network** only when ready to make the request.

2. For a text-only request, leave **Attach active rendered plane** clear. Type a
   learning question and select **Send**, or press `Ctrl+Enter` / `Cmd+Enter`.
   Plain `Enter` adds a line break. You can choose **Hide setup** after the
   provider is configured.

3. For an image-assisted request:

   - select **Vision input** and then **Attach active rendered plane**;
   - confirm that **Cloud image transfer** changes from **OFF** to
     **ON — one-time review required for this exact request**;
   - select **Send** and inspect **Review the exact outbound request**;
   - compare the visible PNG with the intended plane and review provider,
     destination host, endpoint, model ID, task, total bytes, prompt
     fingerprint, dimensions, MIME type, item SHA-256, transform,
     de-identification actions, burned-in-text review, and residual risks; and
   - select **Authorize once and send** only if every field and pixel is
     acceptable. **Cancel** is the default.

4. While a request is running, use **Cancel request** or `Esc` to request safe
   cancellation. You can continue drafting text, but provider settings are
   frozen until the request finishes.

5. Read the response. Provider text is rendered as safe Markdown; provider,
   model, timestamp, and disclaimer remain separate. Selecting an HTTP or HTTPS
   link asks for confirmation before opening the browser. Other link schemes
   are blocked.

6. Open **Structured artifacts · API preview** when you need to understand a
   typed-output contract. The selector covers text explanations, class scores,
   semantic labels, 2-D masks, 3-D masks, reconstructed images, and
   reconstructed volumes. The panel remains empty unless a trusted host
   integration supplies both a matching typed request and typed response.
   When an unverified response matches its request, **Confirm artifact** and
   **Reject artifact** become available. Either action records a local review;
   it does not edit source data or create a layer.

### How to tell it worked

- Before sending, the availability message changes to **Ready: review the
  provider details, then send this learning request.**
- During a request, the header reports **Generating a new response…** and the
  cancel control is enabled.
- A completed response reports **Response ready · Markdown rendered safely.**
  and shows provider/model/time metadata plus the provider disclaimer.
- For image transfer, the status reports a reviewed one-time request in
  progress; changing any bound field requires a new confirmation.
- In Structured artifacts, a matching unverified response reports that explicit
  review is required; a completed review reports **User confirmed** or
  **Rejected**.

### Boundaries and cautions

- **Enable network** and **Attach active rendered plane** are separate opt-ins.
  Text prompts are sent even when no image is attached, so remove names,
  identifiers, and other private information from the prompt.
- The attached PNG is a new encoding of the complete active 2-D plane after
  display mapping. It excludes viewport zoom/pan, measurements, crosshairs,
  masks, contours, and annotations. It is not the original DICOM/NIfTI file and
  carries no original metadata, but burned-in identifiers remain pixels.
- Authorization is valid once for the exact provider, endpoint, model, prompt,
  task, and bytes. It cannot recall a request that has already completed.
- Markdown is display content, not trusted advice. External links require a
  separate decision.
- Ordinary Teaching chat responses are not converted into Structured artifacts.
  There is no desktop artifact importer in this release. The structured review
  panel sends nothing, does not create a layer, and does not establish clinical
  validity.
- Read [AI provider and cloud-image safety](LLM_SECURITY.md) before enabling
  image transfer.

## Shortcuts and output reference

### Keyboard and mouse shortcuts

| Action | Shortcut |
| --- | --- |
| Switch workspaces | `Alt+1` through `Alt+6` |
| Open image, DICOM file/ZIP, or NIfTI | `Ctrl+O` |
| Open DICOM folder | `Ctrl+Shift+O` |
| Fit all visible image views | `Ctrl+0` |
| Zoom an image view | Mouse wheel |
| Pan an image view | Drag in **Pan / zoom** mode |
| Link 3-D orthogonal planes | Double-click a location in **Pan / zoom** mode |
| Cancel an unfinished measurement | `Esc` while the image view has focus |
| Move a focused slice/page slider | Arrow keys; `Page Up` / `Page Down`; `Home` / `End` |
| Send an assistant question | `Ctrl+Enter` or `Cmd+Enter` |
| Add a line break to the assistant prompt | `Enter` |
| Cancel an assistant request | `Esc` or **Cancel request** |
| Revalidate a BraTS folder | `Ctrl+R` in the setup dialog |
| Save an anonymous BraTS manifest | `Ctrl+S` after validation and terms confirmation |

### What each export contains

| Workspace | Output | Contains | Deliberately excludes |
| --- | --- | --- | --- |
| Images | Rendered PNG | Full active plane after display mapping | Source metadata, viewport transform, marks, and overlays |
| Images | Experiment JSON | Plane/display settings, pairing flags, numeric measurements | Pixels and source path |
| CT Lab | Reconstruction PNG | Reconstructed image | Source file and intermediate views |
| CT Lab | Experiment JSON | Acquisition/reconstruction parameters and metrics | Pixels |
| Models · DIVal | Experiment JSON | Model/case hashes, device, timing, metrics, privacy flags | Image arrays and private paths |
| Models · DeepInverse | None in this release | On-screen synthetic MRI comparison only | No JSON export |
| Models · MONAI | Experiment JSON | Anonymous case/model provenance, configuration, and region metrics | Input volumes, masks, and local paths |
| Models · External manifests | None in this release | On-screen typed visualizations and summary | No desktop result-file export |
| Learn | None | On-screen lesson text | No experiment file |
| Evaluate | JSON or YAML record | Input digest/shape, threshold, metrics, optional manifest ID | Raw vectors and pixels |
| AI Assistant | On-screen response | Provider text plus separate provenance/disclaimer | No chat or artifact export; no automatic layer creation |

When a save dialog points to an existing file, keep the existing file unless
the application and operating system both present an explicit replacement
decision. Prefer a new descriptive filename for every derived result.

## Safe working checklist

- Prefer synthetic, public, or institutionally approved data.
- Confirm the active view before CT, external-model, ROI, or assistant actions.
- Distinguish display mapping from decoded values and pixel units from trusted
  physical spacing.
- Keep plane-local comparisons separate from referenced clinical layers.
- Read every model's input contract, model card, domain warning, and device
  fallback before interpreting output.
- Split evaluation data by pseudonymous person/study group, never by slice.
- Treat metrics, overlays, and assistant text as evidence to review, not proof.
- Keep source data unchanged and save derived work under a new name.
- For failures, use [Troubleshooting](TROUBLESHOOTING.md). For a longer learning
  path, use the [Teaching curriculum](TEACHING_CURRICULUM.md).
