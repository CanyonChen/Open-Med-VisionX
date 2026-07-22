# OpenMedVisionX Learning Curriculum

[简体中文](TEACHING_CURRICULUM.zh-CN.md) · [Documentation index](INDEX.md) · [User guide](USER_GUIDE.md) · [Data formats](DATA_FORMATS.md) · [Bundled models](MODEL_BUNDLES.md) · [Custom models](CUSTOM_MODELS.md) · [AI safety](LLM_SECURITY.md) · [Glossary](GLOSSARY.md)

This curriculum turns the desktop application into a guided learning path. It
starts with pixels and physical geometry, builds CT reconstruction and
evaluation skills, and then applies the same disciplined reasoning to local
models and an optional AI assistant. No programming is required.

> [!WARNING]
> Every activity is for learning or research. Do not interpret an output as a
> diagnosis or use it for treatment, triage, or any patient decision. Use only
> data you are authorized to handle, and prefer non-sensitive examples.

## Route and estimated time

Complete the [base application setup](QUICKSTART.md) before beginning.
The full route takes about **7–12 hours**, including the capstone. It works well
as three sessions: foundations (modules 1–3), scientific reasoning (modules
4–5), and responsible model use (modules 6–7 plus the capstone).

| Stage | Central question | Main workspace | Estimated time | Learning output |
| --- | --- | --- | ---: | --- |
| 1. Pixels and display | Is the screen showing source values or a display mapping? | Images, Learn | 30–45 min | Display-versus-source note |
| 2. Medical geometry | When do slices form a physical volume? | Images, Learn | 45–60 min | Geometry worksheet |
| 3. CT projections | How do angles, filters, and iterations change reconstruction? | CT Lab | 60–90 min | Controlled reconstruction comparison |
| 4. Tasks and outputs | What does each medical imaging task consume and return? | Learn, Models | 45–60 min | Task-contract map |
| 5. Evaluation | Does the split, metric, threshold, and unit match the question? | Evaluate, CT Lab | 45–60 min | Evaluation note and pixel-free record |
| 6. Local model inference | Can another user reproduce the input and output mapping? | Models | 75–120 min | Bundled-model experiment note |
| 7. AI-assisted explanation | What leaves the computer, and what can the response support? | AI Assistant | 30–45 min | Network-boundary review |
| Capstone | Can the entire claim be reproduced and bounded? | One or more workspaces | 2–4 h | Compact experiment report |

Modules 1, 3, 4, and 5 can use built-in or ordinary non-sensitive examples.
The DIVal and DeepInverse activities in module 6 also need no user dataset.
DICOM, NIfTI, BraTS, custom-model, and image-sharing branches are optional and
must be skipped when suitable authorized material is not available.

Use the guides as references instead of repeating setup here:

- [User guide](USER_GUIDE.md) for workspace controls and end-to-end operation;
- [Data formats](DATA_FORMATS.md) for DICOM, NIfTI, raster, mask, and annotation
  contracts;
- [Bundled model guide](MODEL_BUNDLES.md) for PyTorch setup and the three
  reviewed workflows;
- [Custom model guide](CUSTOM_MODELS.md) for local manifests and capability
  limits;
- [AI safety guide](LLM_SECURITY.md) before enabling any network request;
- [Glossary](GLOSSARY.md) whenever a term or unit is unfamiliar.

## Keep one learning record

Use the same short record for every module. This makes parameter changes,
assumptions, and limits comparable instead of leaving only screenshots.

| Field | What to write |
| --- | --- |
| Question and prediction | One answerable question and what you expect before running anything |
| Source and permission | Where the input came from, its license or permitted use, and whether it is synthetic, benchmark, or local |
| Unit of analysis | Pixel, image, series, volume, lesion, slide, study, or person, as appropriate |
| Input contract | Modality, shape, dtype, range/units, geometry, channel order, and preprocessing |
| Controlled change | The one parameter, threshold, algorithm, or model being compared |
| Output and metric | What each output means, the reference used, and the metric with its data range or units |
| Failure or alternative explanation | One place the method fails, or another reason the observation could occur |
| Evidence boundary | What this activity does **not** establish |

Do not copy credentials, identifiers, source paths, or sensitive pixels into a
learning record. If a screenshot is useful, inspect it as carefully as the
original data before saving or sharing it.

## 1. Pixels, dtype, and display

### Learning goals

- Distinguish a stored array from its rendered appearance.
- Explain shape, dtype, bit depth, numeric range, color/channel order, alpha,
  and compression as separate properties.
- Recognize when a display operation changes visibility without changing source
  values.

### Preparation

- Use a local PNG, JPEG, or TIFF that you are allowed to inspect.
- If possible, prepare a lossless PNG or TIFF and a JPEG made from the same
  non-sensitive source.
- Review the raster-image section of [Data formats](DATA_FORMATS.md).

### Steps

1. Open the raster in **Images** and record its shape, dtype, source range, and
   color interpretation.
2. Open the histogram and locate the minimum, maximum, background, and any
   clipped-looking regions.
3. Change brightness, contrast, gamma, window, or display range one control at
   a time.
4. Return to the original display and confirm that the recorded source dtype
   and range did not change.
5. Compare the lossless and JPEG versions at normal scale and around sharp
   edges or low-contrast regions.
6. If the image has color or alpha, identify channel order and explain whether
   alpha is measurement data or presentation state.

### What you should observe

- Several display mappings can make the same array look very different.
- Two arrays can look similar on an 8-bit screen while preserving different
  numeric precision.
- JPEG can introduce irreversible changes even when artifacts are hard to see.
- A histogram describes stored or selected values; it does not by itself prove
  modality, units, or clinical meaning.

### Self-check

1. Can you state the source dtype and range without reading them from the screen
   colors?
2. Which controls changed only presentation, and which operation would create a
   derived pixel array?
3. Why can neither a grayscale appearance nor a filename establish that values
   are Hounsfield units?
4. Which interpolation would you use for a categorical mask, and why?

### Common misconceptions

- Calling generic grayscale values HU.
- Treating the displayed 8-bit rendering as the source array.
- Treating DPI as medical pixel spacing.
- Using bilinear interpolation for categorical labels.
- Assuming a visually clean JPEG is lossless.

## 2. Pages, volumes, and coordinates

### Learning goals

- Separate a 2-D raster, a page sequence, and a physical volume.
- Explain why spacing, origin, orientation/direction, and an affine or
  equivalent geometry are needed before millimetres and orthogonal views have
  meaning.
- Distinguish source geometry from display state and explicitly derived
  resampling.

### Preparation

- Use a multipage TIFF for the page-sequence exercise.
- DICOM or NIfTI is optional and must come from an authorized local source.
  NIfTI support requires the optional package described in
  [Data formats](DATA_FORMATS.md).
- If trying labels, use a matching lossless label map or a DICOM SEG/RTSTRUCT
  that references the open image series.

### Steps

1. Open a multipage TIFF. Record how pages are navigated and whether trusted
   spacing, origin, or orientation exists.
2. If authorized data are available, open a validated DICOM or NIfTI volume and
   record dimensions, spacing, orientation, modality, and intensity meaning.
3. Visit axial, coronal, and sagittal views. Record one voxel index, its
   physical coordinate, and the transform that connects the views.
4. Compare page navigation with orthogonal volume navigation.
5. If a NIfTI has a fourth axis, explicitly select one volume and record the
   chosen index; do not describe this as complete 4-D playback or analysis.
6. In **Layers**, identify the base image/volume, any segmentation layer, and
   any RTSTRUCT contour layer. Separate source geometry from opacity, color,
   visibility, and active-layer state.
7. Import a matching label map or referenced DICOM annotation if available.
   Confirm that matching geometry is visible and inspectable.
8. Inspect a deliberately mismatched grid only with non-sensitive test data.
   Confirm that the source remains unchanged and the overlay stays hidden or
   pending until you explicitly approve a derived display resampling.

### What you should observe

- TIFF page order alone does not establish patient-space geometry.
- A physical point can map to different voxel indices in different orthogonal
  planes while referring to the same location.
- DICOM series selection and slice order depend on identifiers and geometry,
  not filenames alone.
- NIfTI is canonicalized for viewing, but its container does not prove modality
  or intensity units.
- A discrete label resample uses nearest-neighbour interpolation and remains a
  derived display; the original label data are not silently rewritten.

### Self-check

1. What evidence makes your object a physical volume rather than a page stack?
2. Where did spacing and orientation come from, and are they source-provided or
   user-provided?
3. Why is an array axis not automatically axial, coronal, or sagittal?
4. Which identifiers and geometry must agree before a DICOM SEG or RTSTRUCT can
   be overlaid responsibly?
5. What changed after explicit resampling, and what remained immutable?

### Common misconceptions

- Sorting DICOM only by filename.
- Inferring physical orientation from array order without an affine.
- Treating a multipage TIFF as a medical volume.
- Silently resizing a mask to fit a reference.
- Assuming every NIfTI is CT, MR, or measured in HU.

## 3. From object to sinogram and back

### Learning goals

- Explain the Radon transform, angular sampling, backprojection, filtering, and
  iterative correction.
- Compare FBP and SART while controlling reference and display range.
- Distinguish an educational image-domain simulation from scanner raw data.

### Preparation

- No input file is required. Open **CT Lab** and keep **Synthetic phantom
  (recommended)** selected.
- Create a small comparison table for angle range, projection count, filter,
  iterations, elapsed time, and metrics.
- Keep the same reference and display range when comparing reconstructions.

### Steps

1. Record why the synthetic phantom's non-negative values, illustrative
   `mm⁻¹` unit, and zero background form a clearer teaching contract than an
   arbitrary open image.
2. Choose **Run first phantom experiment** with a fixed angular range and
   projection count.
3. Compare unfiltered backprojection with filtered backprojection.
4. Compare at least two FBP filters without changing the angles or reference.
5. Run SART at several iteration counts. Change only iteration count and record
   quality, noise, elapsed time, and cancellation behaviour.
6. Inspect a Radon intermediate view and the difference/error view.
7. Compare MSE, PSNR, SSIM, or other reported metrics only after recording each
   metric's data range and units.
8. Optionally inspect the advanced active-image simulation. Record any domain
   check that blocks an incompatible image and explain why accepted pixels are
   still image-derived simulation, not scanner raw acquisition.
9. Optionally run the packaged LoDoPaB example from the
   [bundled model guide](MODEL_BUNDLES.md).
   Compare its observation, fixed Hann FBP, and reference without calling its
   normalized attenuation values clinical HU.

### What you should observe

- Sparse or limited angular sampling produces structured artifacts rather than
  ordinary random noise.
- Unfiltered backprojection blurs contributions; filtering improves resolution
  while changing noise and ringing.
- More SART iterations can improve agreement at first, then amplify noise or
  cost time without a useful gain.
- Independent auto-stretching can hide differences, which is why a common
  reference range matters.
- A visually pleasing reconstruction and the lowest numerical error need not
  be the same result.

### Self-check

1. Which single parameter did you change in each comparison?
2. Why does 360° parallel-beam sampling repeat information available over
   180°?
3. What are the domain and data range of each reported metric?
4. Can you trace one displayed reconstruction back through filter, projection
   geometry, and source object?
5. Why are the synthetic and LoDoPaB examples not scanner raw data or evidence
   of clinical performance?

### Common misconceptions

- Treating a reconstructed HU slice as raw attenuation measurements.
- Comparing outputs after independently auto-stretching them.
- Changing angle count, filter, algorithm, and display range at once.
- Assuming more iterations always improve a reconstruction.
- Treating one public benchmark case as evidence of generalization.

## 4. Medical computer-vision task map

### Learning goals

- Identify the input, output, baseline, and evaluation unit of common medical
  imaging tasks.
- Separate a model-family name from an executable input contract.
- Ask which acquisition, population, labels, and validation support a proposed
  use.

### Preparation

Use the [Glossary](GLOSSARY.md) for unfamiliar terms. The model families below
are study examples, not promises that the current desktop can run them.

| Task | Input → output | Simple baseline | Representative families | Evaluation focus |
| --- | --- | --- | --- | --- |
| Classification | Image/volume → label scores | Logistic regression or small CNN | ResNet, DenseNet, EfficientNet, ViT | Patient-level split, calibration, threshold choice |
| Segmentation | Pixels/voxels → label map | Thresholding or region growing | U-Net, nnU-Net, UNETR, promptable SAM variants | Dice plus boundary and small-region error |
| Detection | Image → boxes/points/classes | Connected components | Faster R-CNN, RetinaNet, YOLO, DETR | IoU, matching rule, lesion- versus image-level unit |
| Registration | Moving + fixed image → transform/field | Rigid or affine optimization | VoxelMorph, TransMorph | Landmarks, folding, inverse consistency, interpolation |
| Reconstruction | Measurements → image | FBP, SART, zero-filled IFFT | FBPConvNet, learned primal-dual, MoDL, VarNet | Measurement consistency and acquisition assumptions |
| Restoration | Degraded image → corrected image | Classical filtering | U-Net, Restormer, diffusion approaches | Reference agreement and fabricated texture |
| WSI analysis | Tiles/features → slide result | Feature statistics | Attention MIL, CLAM, TransMIL | Slide/person-level split and tile sampling |
| Multimodal/VQA/reporting | Image + text → text/structured output | Retrieval or templates | CLIP-style encoders, VQA and report models | Grounding, unsupported claims, privacy, human review |

### Steps

1. Choose three tasks from the table, including at least one measurement-to-image
   or geometry-changing task.
2. For each task, write the exact modality, dimensionality, channel order,
   spacing/orientation, dtype, range, and normalization expected by a plausible
   model.
3. Define every output tensor or object and how it maps to source geometry.
4. Name a simple baseline before naming a learned model.
5. Choose an evaluation unit and at least two complementary metrics.
6. State which population and labels would create the training target and which
   external validation would be needed for the proposed use.
7. Record code, weight, and data licenses separately.

### What you should observe

- Similar architecture names can represent different tasks and incompatible
  preprocessing.
- Output geometry is part of the task contract, not a display afterthought.
- A high-level protocol may describe more inputs than the current desktop can
  collect safely.
- Baselines reveal whether a learned method adds value beyond a simpler method.
- AUROC, Dice, and visual quality answer different questions and cannot be
  substituted for one another.

### Self-check

For one chosen task, answer all of the following without using “whatever the
model accepts”:

1. What is the unit of input and the unit of output?
2. Which geometry and intensity assumptions must hold?
3. What does each output value mean?
4. What baseline and metric match the scientific question?
5. Which population, labels, and split define the evidence?
6. What intended use remains unsupported?

### Common misconceptions

- Treating an architecture name as a complete task definition.
- Assuming a 2-D image can stand in for a 3-D volume, k-space, or sinogram.
- Using Dice alone for boundaries or small lesions.
- Calling plausible restoration texture recovered anatomy without a reference.
- Assuming protocol-level support means the current GUI can collect every
  required input.

## 5. Evaluation without leakage

### Learning goals

- Choose splits, thresholds, metrics, and uncertainty summaries that match the
  scientific question.
- Distinguish discrimination, calibration, overlap, boundary error, and image
  fidelity.
- Separate a single-case demonstration from dataset-level generalization.

### Preparation

- Use the built-in example on **Evaluate** and one saved CT Lab result.
- A dataset manifest is optional. If used, it must be pseudonymous and must not
  expose identifiers or sensitive paths in the learning record.
- Decide the evaluation unit before looking at scores: pixel, lesion, image,
  series, volume, slide, study, or person.

### Steps

1. Write the prediction target, evaluation unit, and intended split before
   loading probabilities.
2. If using a manifest, validate that one group never crosses
   train/validation/test and that preprocessing is fitted only on training data.
3. Enter binary labels and probabilities in **Evaluate**. Run at two explicit
   thresholds and record which confusion-matrix counts and derived metrics
   change.
4. Compare threshold-dependent metrics with AUROC, calibration, and available
   confidence intervals. Explain which quantities do not depend on the selected
   operating threshold.
5. For a CT Lab result, verify shared geometry and the declared data range
   before interpreting MSE, PSNR, or SSIM.
6. For segmentation, pair region overlap such as Dice with boundary distance
   such as HD95 and a volume or lesion-level measure when appropriate.
7. Inspect at least one failure case or subgroup. Record site, scanner,
   protocol, demographic, and image-quality factors only when the data and
   permission support that analysis.
8. Export the available pixel-free `ExperimentRecord` and verify that it records
   settings and metrics rather than source pixels.

### What you should observe

- Changing a threshold changes predicted classes, sensitivity/specificity, and
  related counts, but not the ordering of the original probabilities.
- AUROC does not establish calibration or a clinically acceptable operating
  point.
- Slice-level random splitting can leak information from the same person or
  study into multiple partitions.
- A strong average can hide poor boundary accuracy, small-region failure, or a
  harmful subgroup.
- Confidence intervals describe sampling uncertainty, not every source of bias
  or domain shift.

### Self-check

1. Was the split performed before slicing, patching, augmentation, or feature
   fitting?
2. Is the threshold chosen without using the held-out test set?
3. Does every metric have a declared unit, range, reference, and evaluation
   unit?
4. Which complementary metric exposes a failure hidden by the headline metric?
5. What population or acquisition shift is not represented by this evaluation?

### Common misconceptions

- Selecting a threshold on the test set and reporting the same result as
  unbiased.
- Equating AUROC with calibration or clinical utility.
- Reporting a slice-level split as person-level generalization.
- Comparing PSNR values computed with different data ranges.
- Treating one benchmark case or one exported record as external validation.

## 6. Reproducible local model inference

### Learning goals

- Run a reviewed local model without hiding input meaning, preprocessing,
  device fallback, or output mapping.
- Explain why each bundled model needs a dedicated input path.
- Recognize when the custom-model capability gate correctly refuses a contract.

### Preparation

- Follow [Bundled models](MODEL_BUNDLES.md) for PyTorch installation and
  verification; do not repeat or guess platform-specific commands here.
- The DIVal and DeepInverse activities require no user dataset. MONAI requires
  an authorized local BraTS-style case and may be studied without running when
  one is unavailable.
- A user-supplied model is optional. If used, prepare it according to
  [Custom models](CUSTOM_MODELS.md) and review its license and source.

| Reviewed reference | Input boundary | Output boundary |
| --- | --- | --- |
| [DIVal LoDoPaB FBP-U-Net](../src/workbench/resources/model_bundles/dival-lodopab-fbpunet/MODEL_CARD.md) | Packaged fixed LoDoPaB Hann-filtered FBP, `1×1×362×362`, normalized attenuation—not arbitrary CT, clinical HU, or scanner sinogram | One benchmark-domain post-processed reconstruction; single-case metrics are not clinical performance |
| [DeepInverse MRI MoDL teaching model](../src/workbench/resources/model_bundles/deepinv-mri-modl/MODEL_CARD.md) | Locally generated deterministic synthetic MRI: simulated single-coil complex k-space plus binary mask—not scanner raw data | Two-channel complex reconstruction with magnitude derived for display; the demo currently has no JSON export |
| [MONAI BraTS 3D Segmentation](../src/workbench/resources/model_bundles/monai-brats-segmentation/MODEL_CARD.md) | User-managed, co-registered T1/T1ce/T2/FLAIR/SEG NIfTI case with the documented geometry and labels; no BraTS case or automatic registration is provided | WT/TC/ET results and reference metrics; BraTS 2018 → 2021 is an explicit domain shift, not general lesion or clinical validation |

### Steps

1. Open **Models → Bundled models**, choose **Verify catalog & device**, and
   record the usable device without interpreting verification as accuracy.
2. Run **DIVal LoDoPaB FBP-U-Net**. Compare observation, reviewed Hann FBP,
   model output, ground truth, fixed-scale error, and MAE/RMSE/PSNR. Export the
   pixel-free experiment JSON if needed.
3. Run **DeepInverse MRI MoDL Teaching Model**. Trace the digital phantom through
   single-coil FFT, fixed undersampling, zero-filled IFFT, MoDL magnitude, and
   absolute error. Keep the entire path labelled synthetic MRI.
4. Read the MONAI model card. If authorized data are available, use **Set up
   local BraTS 2021…** to validate one T1/T1ce/T2/FLAIR/SEG case, then run the
   enabled full-volume workflow and inspect WT/TC/ET plus Dice, HD95, and volume
   error. If no suitable case is available, document the contract without
   selecting substitute data.
5. Optionally open **Models → External manifests**, start from the documented
   example, and replace every placeholder with the facts of your actual local
   model rather than changing only its name.
6. Before loading a custom model, review task, modality, input count,
   dimensionality, layout, channel order, range, normalization, spatial
   transforms, outputs, labels, runtime, source, and licenses.
7. Read the capability decision. Run only if the current desktop can supply
   every required input; record a rejection as a useful result when the
   contract requires prompts, compound inputs, or whole 3-D/4-D data.
8. Record requested and actual device, fallback, model version, preprocessing,
   elapsed time, warnings, output meaning, and reference metrics for every run.

### What you should observe

- The three bundled workflows are task-aware: they do not turn an arbitrary
  active image into a plausible but invalid input.
- DIVal uses a reviewed public benchmark case; DeepInverse creates its
  deterministic input locally; MONAI remains disabled until a valid local case
  passes in-session validation.
- The displayed device may be CPU after a visible fallback; reproducibility
  requires recording the actual device, not only the requested one.
- The external-manifest path currently accepts one supported active 2-D image
  or plane and cannot borrow DeepInverse's compound input or MONAI's four-volume
  workflow.
- A valid-looking output can still suppress, shift, or fabricate structure.

### Self-check

1. For DIVal, why are normalized attenuation and fixed FBP not clinical HU or
   arbitrary scanner projections?
2. For DeepInverse, where did k-space come from, and why must it remain labelled
   simulated?
3. For MONAI, which modalities, geometry, labels, and domain shift must be
   recorded?
4. For a custom model, can the current desktop collect every declared input
   without inventing or silently transforming one?
5. Can another user reproduce preprocessing, actual device, output mapping,
   threshold, and reference from your record?

### Common misconceptions

- Assuming a `.pt`, `.pth`, or `.ckpt` extension proves a safe, complete model
  contract.
- Feeding a current 2-D plane to a model trained on a complete 3-D volume.
- Changing resize, channel order, modality order, or normalization merely to
  make a tensor fit.
- Treating CPU/GPU compatibility checks or a golden result as medical accuracy.
- Treating a capability rejection as a model failure instead of an input-safety
  boundary.

## 7. Responsible AI-assisted explanation

### Learning goals

- Separate a helpful explanation from scientific or clinical evidence.
- Make the network boundary, exact payload, destination, and consent state
  visible before any request.
- Recognize residual privacy risks such as burned-in text and an already sent
  request.

### Preparation

- Read [AI Assistant, privacy, and cloud images](LLM_SECURITY.md) in full.
- Prepare a concept-only prompt containing no report text, identifier,
  credential, or sensitive context.
- Image transfer is optional. Use only a clearly non-sensitive image that you
  are authorized to send to the selected provider.
- Keep network access disabled until the endpoint, model ID, credential
  reference, and intended task have been reviewed.

### Steps

1. Configure the provider with the exact endpoint, exact model ID, and an
   environment-variable or keyring credential reference. Never paste a raw key
   into the reference field.
2. Enable network access only for the planned request. Send the concept-only
   prompt and record provider, endpoint host, model ID, time, and task.
3. Compare the response with the visible experiment and a trusted source.
   Mark unsupported, ambiguous, or unverifiable claims.
4. If considering an image request, enable vision and attach the active
   rendered plane. Inspect the canonical PNG preview before proceeding.
5. In the final review, verify provider, exact destination, model, task, prompt
   hash, image hash, dimensions, bytes, display transform, minimization steps,
   burned-in-text review, and residual risks. Keep the default answer **No**
   unless every field is correct and sending is appropriate.
6. Change one plan field and confirm that a new review is required. Remember
   that consent is one-shot and consumed for one exact dispatch.
7. Inspect **Structured artifacts** as an integration preview. Confirming or
   rejecting an artifact records a local review only; it does not send data,
   create an image layer, or establish validity.

### What you should observe

- A text request sends the full prompt, not only selected keywords.
- An image request sends a newly encoded rendering of the complete active 2-D
  plane, not the original DICOM/NIfTI file or full series; this minimization
  still does not remove burned-in pixels.
- Provider, endpoint, model, prompt, image, or task changes invalidate the prior
  review.
- Cancelling a request cannot recall bytes that have already reached the
  provider.
- Fluent prose can be wrong, ungrounded, or outside the model's competence.

### Self-check

1. What exact text and image bytes would leave the computer?
2. Which host receives them, under which provider policy and model ID?
3. Which identifiers, metadata, overlays, and burned-in pixels remain or are
   removed?
4. Which statement in the response is supported by the experiment, and which
   requires another source?
5. What would you do if sensitive data or a credential were sent accidentally?

### Common misconceptions

- Pasting reports, identifiers, or raw API keys into a prompt or settings field.
- Treating fluent text as a validated interpretation.
- Assuming consent applies to another provider, model, prompt, task, or image.
- Believing metadata removal also removes burned-in text.
- Believing cancellation or revocation recalls an already transmitted request.
- Treating structured-artifact review as automatic import or clinical approval.

## Capstone project

### Learning goals

- Combine one clear question, a valid input contract, a controlled comparison,
  an appropriate metric, and a bounded conclusion.
- Produce a record another user can reproduce and critique.
- State privacy, bias, domain-shift, and clinical limitations without hiding a
  failed or ambiguous result.

### Preparation

Choose one non-sensitive route:

- compare two CT Lab reconstruction settings;
- analyze a DIVal or DeepInverse bundled demonstration;
- evaluate labels and probabilities with two thresholds;
- run MONAI or a custom local model only with authorized, contract-compatible
  data;
- audit a concept-only AI explanation and its network plan.

Write the question and predicted result before running the workflow. Gather the
relevant user guide, data/model card, source/license record, and a blank copy of
the learning-record fields.

### Steps

1. Define the question, intended learning use, and one claim that the activity
   is capable of testing.
2. Record source, permission, exclusion criteria, evaluation unit, input/output
   meaning, geometry, and preprocessing.
3. Select a simple baseline and one controlled algorithm, model, parameter, or
   threshold comparison.
4. If a dataset is used, define a group-safe split before slicing, patching,
   augmentation, or preprocessing fit.
5. Run the workflow and record parameters, actual device, timing, warnings,
   display mapping, reference, and metrics with units/ranges.
6. Inspect one failure case, surprising region, or alternative explanation.
7. Explain privacy, bias, acquisition, population, domain-shift, and clinical
   limitations that remain.
8. Write a conclusion limited to the evidence. Include an unsuccessful or
   inconclusive result when that is what occurred.
9. Assemble a compact report with a reproducible step list and only
   non-sensitive, necessary figures or pixel-free exports.

### What you should observe

- A narrow, controlled question produces a clearer conclusion than a broad
  “does the model work?” question.
- Input contract and display choices can change interpretation as much as the
  algorithm choice.
- A failure case often reveals more about the valid domain than the best-looking
  output.
- Reproducibility requires parameters, units, geometry, versions, and warnings,
  not only a screenshot.

### Self-check

1. Can another user repeat the workflow without guessing a hidden setting?
2. Is the baseline fair, and did only the intended factor change?
3. Does every figure and metric have a reference, unit/range, and output
   meaning?
4. Are split, threshold, and preprocessing choices independent of the held-out
   result?
5. Does the conclusion say what the evidence does not establish?
6. Does the report exclude sensitive pixels, identifiers, paths, and
   credentials?

### Common misconceptions

- Treating a visually impressive output as sufficient evidence.
- Hiding failed runs or changing several settings until one result looks good.
- Calling one case a dataset-level or clinical validation.
- Reporting a metric without its reference, unit, range, or evaluation unit.
- Writing a broader conclusion than the experiment was designed to support.

## Course outputs

At the end of the route, keep the following non-sensitive outputs:

1. seven completed learning records, including self-check answers and one
   evidence boundary for each module;
2. one pixel/display comparison and one page-versus-volume geometry worksheet;
3. one controlled CT reconstruction comparison;
4. one medical imaging task-contract map;
5. one threshold/evaluation analysis with a pixel-free record when available;
6. one bundled-model experiment note, with optional MONAI or custom-model work
   only when suitable authorized inputs exist;
7. one AI network-boundary review, which may remain a no-send plan; and
8. one capstone report containing the question, contract, controlled result,
   failure case, reproducibility information, and evidence-limited conclusion.

The route is complete when another user can reproduce the work, explain why the
inputs are scientifically valid, identify what changed, and state the limits
without relying on a visually convincing output. Return to the
[documentation index](INDEX.md) for the next user task.
