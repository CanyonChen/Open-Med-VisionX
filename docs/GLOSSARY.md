# OpenMedVisionX Glossary

[简体中文](GLOSSARY.zh-CN.md) · [Documentation home](INDEX.md) · [Learning curriculum](TEACHING_CURRICULUM.md)

This glossary explains the terms that appear in the OpenMedVisionX interface
and user guides. Definitions are practical rather than exhaustive: each entry
focuses on what a new user needs to check before trusting an image, comparison,
model output, metric, or AI-assisted explanation.

> [!NOTE]
> A familiar name does not guarantee familiar meaning. Always interpret a term
> together with the current data source, geometry, task, parameters, and
> provenance.

## How to use this glossary

Use the themed tables to look up an unfamiliar term, then follow the linked
guide for the complete workflow. UI labels are shown in **bold** where useful.
Abbreviations are expanded at first use.

| If you are asking… | Start with… |
| --- | --- |
| “Which image or slice will the next action use?” | **Active view**, **plane**, and **provenance** |
| “Are these pages a physical volume?” | **Voxel**, **spacing**, **orientation**, and **affine** |
| “Why did the picture change although the data did not?” | **Display mapping** and **windowing** |
| “What is CT Lab showing?” | **Sinogram**, **FBP**, and **SART** |
| “Can this model accept my file?” | **Input contract**, **manifest**, and **capability gate** |
| “Does this score prove the model is good?” | **Evaluation unit**, **data leakage**, **calibration**, and the named metric |
| “What exactly could leave my computer?” | **Trust boundary**, **rendered preview**, **transfer plan**, and **burned-in text** |

## Interface and workflow

| Term | Plain-language meaning | Why it matters |
| --- | --- | --- |
| **Workspace** | One of the six top-level areas: Images, CT Lab, Models, Learn, Evaluate, or AI Assistant. | Each workspace has a different input and evidence boundary; switching pages does not turn one type of data into another. |
| **Active view** | The image view most recently selected for downstream use. It includes a named plane and current slice/page. | CT Lab's advanced mode, compatible models, and the Assistant use this view; click the intended view and read the context line first. |
| **Plane** | One visible 2-D section or image, such as axial, coronal, sagittal, a TIFF page, or a flat raster. | A plane is not the same as a complete volume. Current-plane masks and generic external models operate on one plane. |
| **Slice** | One 2-D cross-section from a spatial volume. | Its physical location is meaningful only when the volume geometry is valid. |
| **Page / frame** | One item in a 2-D sequence, such as a TIFF page. | Sequence order alone does not prove 3-D spacing or patient coordinates. |
| **Study → Series → Layer** | The viewer hierarchy: a study contains image series, and a series contains its source and derived layers. | It keeps images, segmentations, contours, presentation state, and provenance associated without rewriting the source. |
| **Layer** | One source or derived object shown with a series, such as a volume, segmentation, or contour set. | Visibility and opacity can change independently; a layer is not automatically fused into the source image. |
| **Source data** | The data decoded from the file or series selected by the user. | Display adjustments, overlays, and derived resampling do not replace it. Keep the original and its license. |
| **Derived result / derived layer** | A new result calculated from source data, such as a reconstruction or explicitly resampled display layer. | It must retain its transform, parameters, and provenance so it is not mistaken for the original. |
| **Presentation state** | Visibility, opacity, lock state, label visibility, window, or other choices that affect what is shown. | It changes the view, not the underlying source values. |
| **Provenance** | A record of where data came from and what choices, versions, transforms, or devices produced a result. | Without provenance, another user cannot reproduce the result or identify a mismatch. |
| **Capability gate** | A visible compatibility check performed before an action is enabled. | It prevents an unsupported image, dimension, modality, spacing, prompt, or compound input from being made to look valid. |
| **Experiment record** | A JSON/YAML record of task context, parameters, hashes, metrics, warnings, and runtime facts. | OpenMedVisionX records are deliberately pixel-free and path-free; they support reproducibility but are not the source data. |
| **Local-first** | Viewing and core learning workflows operate on the selected local data without automatic upload or download. | Network use is a separate, explicit action; “local-first” does not mean a cloud request is impossible. |
| **Immutable** | Not modified after creation. A changed interpretation or presentation creates a new record/state instead. | This makes source and derived evidence traceable and prevents silent overwrite. |

## Image data and geometry

| Term | Plain-language meaning | Why it matters |
| --- | --- | --- |
| **Pixel** | One sample in a 2-D raster at an `(x, y)` location. | A pixel has no physical size unless trustworthy spacing is known. |
| **Voxel** | One sample in a 3-D volume at an `(x, y, z)` index. | Millimetre measurements require the voxel-to-world geometry, not just the array index. |
| **Raster image** | A rectangular grid of pixels, such as PNG, JPEG, or one TIFF page. | It is normally interpreted in pixel coordinates, not patient space. |
| **Page sequence** | Several independent 2-D pages or frames. | Equal dimensions and ordering do not establish a physical volume. |
| **Volume** | A 3-D array with validated spatial geometry. | It can support physical measurements, world coordinates, and orthogonal views. |
| **Shape** | The number of samples along each array dimension. | Equal shapes do not prove that two images cover the same physical space. |
| **dtype** | The numeric storage type, such as `uint8`, `int16`, or `float32`. | It affects range and precision; the 8-bit display is not necessarily the source dtype. |
| **Bit depth** | The number of bits used to represent a sample or channel. | A higher bit depth can preserve distinctions that disappear in an 8-bit rendering. |
| **Channel** | One component of a sample, such as red/green/blue or one MRI modality/time point. | Channel order and meaning must match the task; repeated or guessed channels are not valid preprocessing. |
| **Modality** | The acquisition type, such as CT, MR, or PET. | It constrains intensity meaning, geometry, preprocessing, and model compatibility. A filename cannot establish it safely. |
| **Spacing / pixel spacing / voxel spacing** | Physical distance between neighbouring samples, usually in millimetres. | It is required for physical length, area, volume, and surface-distance metrics. DPI is not medical spacing. |
| **Origin** | The world-coordinate location assigned to a reference voxel. | A matching spacing and shape can still describe a different location when origins differ. |
| **Orientation / direction** | How array axes point through physical space. | Array axis 0 is not automatically axial; orientation is needed to label planes correctly. |
| **Affine** | Usually a `4×4` matrix that maps voxel indices to world coordinates, including spacing, orientation, and position. | Two NIfTI arrays are physically aligned only when their affines and grids agree, not merely their shapes. |
| **World coordinate** | A physical location expressed in a coordinate convention, normally millimetres rather than array indices. | It allows the same anatomical location to be related across views and compatible layers. |
| **RAS+** | A coordinate convention whose positive axes point Right, Anterior, and Superior. OpenMedVisionX uses it internally for volumes. | It provides one consistent internal frame; conversion to RAS+ does not change the source file or prove two volumes are aligned. |
| **LPS** | A coordinate convention whose positive axes point Left, Posterior, and Superior; commonly used by DICOM patient coordinates. | DICOM contours and geometry must be converted consistently between LPS and internal RAS+. |
| **Canonicalization** | Re-expressing an image in a chosen coordinate convention, such as RAS+, while recording the mapping. | It standardizes navigation but does not invent missing geometry or permit arbitrary overlay. |
| **Geometry match** | Agreement of shape/grid, spacing, orientation, origin, affine, and relevant referenced identities. | A segmentation should be overlaid only when this relationship is validated. |
| **Resampling** | Calculating values on a different grid. | It creates derived data and can move boundaries or alter values; it must be explicit rather than silent. |
| **Interpolation** | The rule used to estimate values during resampling. | Nearest-neighbour is appropriate for discrete labels; continuous interpolation is used for fractional values, not class IDs. |
| **ROI (region of interest)** | A user-selected image region for inspection or measurement. | An ROI limits an analysis but is not automatically an anatomical label or ground truth. |
| **Mask** | Usually a binary array that marks included versus excluded pixels/voxels. | A current-plane mask is a simple overlay; it is not automatically a reusable clinical segmentation. |
| **Label map** | An integer array in which values identify background and one or more classes. | Labels need a schema, matching geometry, and nearest-neighbour handling when explicitly resampled. |
| **Contour** | A vector path describing a boundary, rather than a filled raster grid. | RTSTRUCT contours remain vector data and should not be silently rasterized. |
| **DICOM** | A family of standards for medical images and related objects, including metadata and patient-space references. | Files from the same folder are not necessarily one series; identity, geometry, and pixel data require validation. |
| **Enhanced multi-frame DICOM** | A DICOM object containing many frames with shared/per-frame geometry. | OpenMedVisionX supports bounded monochrome CT/MR cases with consistent geometry, not every Enhanced object. |
| **NIfTI** | A file format commonly used for neuroimaging arrays plus an affine. | A 4-D NIfTI needs an explicit volume choice; intensity semantics are not guaranteed by the format. |
| **DICOM SEG** | A DICOM segmentation object that references source DICOM images. | Binary/fractional values and referenced identities must be preserved and validated. |
| **RTSTRUCT** | A DICOM radiotherapy structure set containing named patient-space contours. | It must reference the intended series and remains vector geometry in OpenMedVisionX. |

For format-specific preparation and import choices, see [Data and
formats](DATA_FORMATS.md).

## Intensity and display

| Term | Plain-language meaning | Why it matters |
| --- | --- | --- |
| **Intensity** | The numeric value stored at a pixel or voxel. | Its meaning depends on modality and preprocessing; “brighter” is not a unit. |
| **Intensity semantics** | The declared interpretation of numeric values, such as HU, normalized attenuation, or unknown. | OpenMedVisionX keeps semantics unknown when evidence is insufficient rather than guessing. |
| **Hounsfield unit (HU)** | A calibrated CT scale relative to water and air. | A generic grayscale image or normalized benchmark array is not HU; valid DICOM rescale evidence is required. |
| **SUV (standardized uptake value)** | A PET uptake measure that depends on units and correction metadata. | PET pixels are not labelled SUV unless the required evidence is present. |
| **Display mapping** | The transformation from source numeric values to visible brightness or color. | It changes what you see, not the source array. Exports and AI previews use the mapped plane, so inspect it explicitly. |
| **Windowing / window level and width** | Choosing a centre and numeric interval to map into the visible grayscale range. | Structures outside the window can appear uniformly dark or bright without being absent from the source. |
| **Display range** | The lower and upper values mapped to the visible output range. | Use a shared range when visually comparing reference and result; separate auto-ranges can mislead. |
| **Brightness / contrast / gamma** | Display controls that shift, stretch, or nonlinearly remap visible values. | They affect appearance and should not be mistaken for preprocessing of the source. |
| **Histogram** | A count of how frequently values occur. | It helps reveal clipping, narrow ranges, and outliers but does not provide spatial location. |
| **Normalization** | A documented numeric transformation, such as scaling into a range or standardizing a channel. | A model expects its exact training-time convention; a convenient visual normalization may be scientifically wrong. |
| **z-score normalization** | Subtracting a mean and dividing by a standard deviation, often within a declared mask. | The population/region used to compute mean and standard deviation is part of the model contract. |
| **Lossless / lossy** | Lossless encoding preserves decoded sample values; lossy encoding, such as JPEG, may change them. | Lossy files are unsuitable for discrete labels and can alter small structures or measurements. |
| **Metadata** | Non-pixel information such as dimensions, geometry, acquisition fields, or identifiers. | Metadata can be essential for interpretation and can also contain sensitive information. |
| **Burned-in text** | Text rendered directly into image pixels, such as a name, date, accession number, or facility label. | Removing metadata does not remove it; inspect the actual rendered image before export or cloud transfer. |

## CT projection and reconstruction

| Term | Plain-language meaning | Why it matters |
| --- | --- | --- |
| **Attenuation map** | A spatial map describing how strongly material attenuates X-rays. | CT Lab's synthetic phantom uses a clear non-negative teaching domain; an arbitrary image is not automatically an attenuation map. |
| **Phantom** | A synthetic or physical test object with known structure. | The built-in phantom provides a privacy-safe, reproducible reference for comparing reconstruction settings. |
| **Projection** | Measurements of line integrals through an object at one angle in the teaching model. | A reconstructed image projected mathematically is not the original scanner acquisition. |
| **Radon transform** | The mathematical operation that forms idealized line-integral projections over angles. | It is the forward model behind the CT Lab sinogram, with assumptions that differ from a real scanner. |
| **Sinogram** | Projection values arranged by detector position and projection angle. | It is measurement-domain data, not a CT image; a clinical slice cannot be relabelled as a sinogram. |
| **Angular sampling** | The number and range of projection angles. | Too few or poorly distributed angles can produce streaks and missing information. |
| **Detector bin** | One sampled detector position in a projection. | Detector sampling and angular sampling jointly affect recoverable detail. |
| **Backprojection** | Smearing each projection back across image space along its acquisition direction. | Unfiltered backprojection is blurred; it demonstrates why filtering or iterative correction is needed. |
| **FBP (filtered backprojection)** | Filtering each projection and then backprojecting it to form an analytic reconstruction. | Filter and geometry choices strongly affect sharpness, noise, and artifacts. |
| **Direct Fourier reconstruction** | Reconstructing through a Fourier-domain relationship between projections and the image. | It offers a different analytic route and has its own interpolation/sampling assumptions. |
| **SART (simultaneous algebraic reconstruction technique)** | An iterative method that repeatedly updates an image using disagreement between measured and predicted projections. | Iteration count, geometry, noise, and stopping behavior affect both quality and runtime. |
| **Iteration** | One repeated update of an iterative algorithm. | More iterations do not guarantee a better or more clinically meaningful result. Change one setting at a time. |
| **Reconstruction filter** | A frequency weighting used by FBP, such as a ramp with optional smoothing. | Smoother filters can reduce noise while blurring detail; compare under one display range. |
| **Support** | The region in which the reconstructed object is assumed to exist, often a circle in the teaching geometry. | Violating the support assumption can create cropping or warnings. |
| **Reference / ground truth** | The target used for a controlled comparison. | A synthetic or benchmark reference supports only the stated experiment; it is not proof of clinical truth or generalization. |
| **Error map** | A spatial view of the difference between a result and reference. | It shows where errors occur, while a single summary metric can hide their location. |

## Models and inference

| Term | Plain-language meaning | Why it matters |
| --- | --- | --- |
| **Model** | A learned or fixed computation that maps declared inputs to declared outputs. | A model name alone does not define modality, preprocessing, labels, or valid use. |
| **Weights / checkpoint** | Stored learned parameters, sometimes packaged with additional runtime state. | Origin, hash, serialization safety, and license matter. A `.pt` suffix does not prove safe TorchScript. |
| **Inference** | Running a trained model to produce an output; it is not training. | A successful run proves runtime compatibility, not medical accuracy. |
| **Preprocessing** | Ordered transformations applied before inference, such as orientation, resize, channel ordering, and normalization. | It must match the model's contract and be recorded; changing it to make a shape fit can invalidate the result. |
| **Postprocessing** | Declared transformations from raw model tensors to user-facing scores, labels, masks, boxes, or images. | Thresholds, activation, interpolation, and coordinate restoration change output meaning. |
| **Input/output contract** | The complete specification of required input and returned output semantics. | Check modality, dimensions, channels, spacing, range, labels, coordinates, and limitations before running. |
| **Manifest** | A local machine-readable description of a user model, its runtime, files, licenses, input/output contract, and capabilities. | A valid manifest describes a model; it does not guarantee that the current desktop can collect every required input. |
| **Runtime** | The software used to execute a model, such as ONNX Runtime or PyTorch. | It must be installed locally and compatible with the model and device. OpenMedVisionX does not install it automatically. |
| **ONNX** | A portable model graph format commonly executed with ONNX Runtime. | It still needs a correct contract and compatible operators; the file alone does not explain preprocessing. |
| **TorchScript** | A serialized PyTorch program intended to run without the original Python model class. | It differs from arbitrary pickle checkpoints but still needs trusted origin, verification, and a compatible PyTorch runtime. |
| **Python adapter** | User-supplied Python code that prepares and runs a model. | It can execute arbitrary code. A separate process is dependency/fault isolation, not a security sandbox. |
| **Bundled offline reference** | One of the fixed, reviewed model artifacts distributed with OpenMedVisionX. | It has a narrow task-specific workflow, integrity record, model card, and input boundary; it is not a general predictor. |
| **External model** | A local model and manifest supplied by the user. | The generic desktop currently accepts one supported active 2-D input, not every task a manifest can describe. |
| **Task** | The kind of problem, such as classification, segmentation, detection, registration, or reconstruction. | Metrics and outputs are task-specific; a classification score is not a segmentation mask. |
| **Classification** | Producing class scores or probabilities for an image, volume, or other declared unit. | The unit, threshold, calibration, and class definitions must be explicit. |
| **Segmentation** | Assigning a class or probability to pixels/voxels. | Geometry, label schema, boundary errors, and small structures matter in addition to Dice. |
| **Detection** | Finding objects as boxes, points, or related locations with classes/scores. | Matching rules, IoU threshold, and evaluation unit determine the reported performance. |
| **Registration** | Estimating a transform that aligns moving and fixed data. | Check landmarks, overlap, folding, inverse consistency, and interpolation; a visually aligned view is not enough. |
| **Reconstruction model** | Mapping acquisition-domain measurements or a defined intermediate representation to an image. | It must preserve the acquisition assumptions; an ordinary image cannot replace k-space or a sinogram. |
| **CUDA / CPU** | GPU and processor execution paths used by a runtime. | `auto` can prefer CUDA and record a CPU fallback. Device choice affects speed and sometimes numerical tolerance, not task validity. |
| **Fallback** | A recorded switch from the requested path to an available alternative, such as CUDA to CPU. | Keep the actual device and reason with the experiment record; do not assume every external model supports the same policy. |
| **Smoke check** | A small deterministic run that verifies artifact/runtime compatibility. | Passing it does not establish medical accuracy or generalization. |
| **Golden check** | Comparing a deterministic output with a packaged expected output. | It detects release/runtime drift within a tolerance; it is not clinical validation. |
| **Domain** | The population, modality, protocol, preprocessing, labels, and conditions represented by the data used for a model or benchmark. | Results outside that domain may not be reliable even when input shape matches. |
| **Domain shift** | A meaningful difference between training data and current data, such as scanner, protocol, population, or BraTS release. | Performance can change; the shift must remain visible in interpretation. |
| **Generalization** | Performance on appropriate data beyond the examples used to build or tune a method. | One case, a bundled demonstration, or a smoke test cannot prove it. |

## Evaluation and metrics

| Term | Plain-language meaning | Why it matters |
| --- | --- | --- |
| **Evaluation unit** | The entity counted as one observation: pixel, lesion, image, series, slide, study, or patient. | A metric changes meaning when the unit changes; state it explicitly. |
| **Train / validation / test split** | Separate partitions for fitting, choosing settings, and final evaluation. | The held-out test set should not be used to choose preprocessing or a threshold. |
| **Group leakage / data leakage** | Information from the same patient, study, or related group appears across partitions or influences evaluation improperly. | Slice-level splitting can make performance look much better than it generalizes. |
| **Threshold** | The cutoff that turns a probability or continuous score into a decision. | Sensitivity, specificity, PPV, NPV, accuracy, and F1 depend on it; AUROC does not select it for you. |
| **Confusion matrix** | Counts of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN). | Many classification metrics are different summaries of these counts at one threshold. |
| **Sensitivity / recall / true-positive rate** | Fraction of actual positives identified as positive: `TP / (TP + FN)`. | High sensitivity can come with more false positives; report the threshold and unit. |
| **Specificity / true-negative rate** | Fraction of actual negatives identified as negative: `TN / (TN + FP)`. | It complements sensitivity and also depends on the operating threshold. |
| **Precision / PPV** | Fraction of positive predictions that are correct: `TP / (TP + FP)`. | It depends on prevalence in the evaluated data and may change across populations. |
| **NPV** | Fraction of negative predictions that are correct: `TN / (TN + FN)`. | Like PPV, it depends on prevalence and the selected threshold. |
| **Accuracy** | Fraction of all decisions that are correct. | It can look high on an imbalanced dataset while the minority class performs poorly. |
| **F1 score** | Harmonic mean of precision and recall. | It ignores true negatives and depends on the threshold; it is not a complete performance summary. |
| **AUROC** | Area under the receiver operating characteristic curve; summarizes positive-versus-negative ranking across thresholds. | It does not measure calibration or choose a clinical operating point, and both classes must be present. |
| **AUPRC / average precision** | A summary of the precision–recall relationship across thresholds. OpenMedVisionX reports average precision as its PR summary. | It emphasizes positive-class retrieval and is sensitive to prevalence; preserve the exact metric name and definition. |
| **Calibration** | Agreement between predicted probability and observed frequency. | A model can rank cases well (high AUROC) while giving overconfident or underconfident probabilities. |
| **Brier score** | Mean squared difference between predicted probabilities and binary outcomes. Lower is better. | It reflects both probability calibration and discrimination; interpret it relative to the dataset and prevalence. |
| **ECE (expected calibration error)** | A binned summary of the gap between average confidence and observed frequency. Lower is better. | It depends on binning and sample size, so always keep the bin definition and calibration plot. |
| **Confidence interval** | A range produced by a stated statistical method to express sampling uncertainty. | It is not a guarantee that the true value lies inside, and it does not account for every source of bias. |
| **Dice coefficient** | Segmentation overlap: `2 × intersection / (predicted size + reference size)`. Higher is better. | It can hide boundary displacement and small-lesion failure; empty-mask conventions must be declared. |
| **IoU (intersection over union)** | Overlap divided by the union of prediction and reference. Higher is better. | It is related to Dice but numerically different; for detection it also defines matching between boxes/objects. |
| **HD95** | The 95th percentile of bidirectional surface distances, normally reported in millimetres. Lower is better. | It describes near-worst boundary separation and requires valid spacing/geometry; it complements overlap. |
| **ASSD** | Average symmetric surface distance between prediction and reference. Lower is better. | It summarizes typical boundary error but can smooth over local large errors. |
| **Surface Dice** | Fraction of surfaces that lie within a declared distance tolerance. Higher is better. | The tolerance and physical units are part of the metric and must be reported. |
| **Volume error** | Difference between predicted and reference physical volume, absolute or relative as declared. | Similar total volumes can still be spatially wrong, so pair it with overlap and boundary measures. |
| **MAE / MSE / RMSE** | Mean absolute, mean squared, and root mean squared numeric error. Lower is better. | They require a shared numeric domain and geometry; MSE penalizes large errors more strongly. |
| **PSNR** | Peak signal-to-noise ratio derived from MSE and a declared data range. Higher is better. | It is meaningless when the peak/data range is inconsistent or hidden. |
| **SSIM** | Structural similarity index comparing local luminance, contrast, and structure. Higher is generally better. | Its window and data range matter, and a good value does not prove clinical fidelity. |
| **AP / mAP** | Average precision for one class and its mean across classes/thresholds as declared. | Detection results depend on matching, IoU thresholds, confidence filtering, and evaluation unit. |
| **FROC** | Free-response receiver operating characteristic, relating lesion sensitivity to false positives per image/scan. | It is useful for detection but requires a precise lesion-matching rule. |

## AI, privacy, and trust

| Term | Plain-language meaning | Why it matters |
| --- | --- | --- |
| **AI provider** | The external or local service that receives a request and returns a response. | The provider's policy, retention, capabilities, and destination are outside the local application boundary. |
| **Endpoint** | The exact network address used for a provider request. | Review the scheme and host; changing it changes the destination and invalidates prior image consent. |
| **Model ID** | The provider-specific identifier of the selected AI model. | Capabilities vary by model; a provider name alone does not prove vision support. |
| **Credential reference** | A pointer such as `env:OPENAI_API_KEY` or a keyring entry, not the secret value itself. | It lets the real key stay outside project files, screenshots, logs, and issue reports. |
| **Environment variable / keyring** | Two ways to make a credential available at request time; a keyring uses the operating-system credential store. | Shared machines generally benefit from keyring storage. Never print or commit the resolved value. |
| **Prompt** | Text sent to an AI model as instructions or context. | Text can contain patient or institutional information even when no image is attached. |
| **Trust boundary** | The point at which data leave the local application and enter another system or organization. | Local viewing stays inside; a provider request crosses it and must be deliberate. |
| **Network opt-in** | The user's explicit decision to enable and send a provider request. | OpenMedVisionX does not contact a provider merely because the Assistant page is open. |
| **Rendered preview** | A newly encoded PNG of the full current 2-D plane after display mapping, shown before an image request. | It is the actual image payload to inspect. It excludes source DICOM/NIfTI bytes, metadata, viewport zoom/pan, overlays, and annotations. |
| **Metadata-free PNG** | A re-encoded PNG restricted to pixel and essential image chunks, without EXIF/text/XMP/ICC/time metadata. | It reduces metadata disclosure but does not remove identifiers visible in the pixels. |
| **Transfer plan** | The review record for one exact provider, endpoint, model, task, prompt digest, and PNG payload. | Any changed field needs a new plan; approval is not a durable “always allow images” switch. |
| **One-shot authorization** | Consent that is sealed and consumed immediately before one image dispatch attempt. | It cannot be reused, and revocation cannot recall a request that has already been sent. |
| **Hash / SHA-256 / digest** | A fixed-length fingerprint calculated from bytes or canonical details. | Matching hashes help bind consent or verify an artifact; a hash is not encryption and does not prove scientific validity. |
| **De-identification** | A documented process intended to remove or reduce identifying information. | “Metadata removed” is not the same as de-identified; visible text and context can remain. |
| **Burned-in text** | Identifying text stored as image pixels rather than metadata. | It will be included in a rendered preview unless you exclude the data entirely; automatic OCR cannot prove safety. |
| **Sensitive information / PHI** | Information that can identify a person or is protected by applicable policy/law. | Do not paste it into prompts or send it in pixels without appropriate authority and safeguards. |
| **Residual risk** | Risk that remains after safeguards, such as provider retention, burned-in identifiers, or contextual re-identification. | Review it before every image transfer; no technical control makes risk zero. |
| **Structured artifact** | A typed result such as class scores, labels, a mask, or a reconstruction, bound to a declared request and geometry. | It is different from fluent chat text. Current desktop chat does not automatically convert responses into these artifacts. |
| **Confirm / Reject artifact** | A local review decision on an already supplied typed artifact. | It records a review only: it sends nothing and does not create or overwrite a Study layer. |
| **Grounding** | Connecting an AI statement to visible evidence or a trusted source. | A fluent response without grounding is an explanation to verify, not evidence or diagnosis. |
| **Hallucination** | Confident output that is unsupported, fabricated, or inconsistent with the input. | Treat provider text and plausible images as hypotheses; verify them independently. |

For the complete user workflow, read [AI Assistant, privacy, and cloud
images](LLM_SECURITY.md).

## Common distinctions

| Do not confuse… | With… | Practical test |
| --- | --- | --- |
| Source array | Display mapping | Change window/gamma: if the recorded dtype/range stays the same, only the view changed. |
| Page sequence | Physical volume | Ask whether spacing, origin, orientation, and an affine/equivalent geometry are validated. |
| Pixel index | World coordinate | Ask whether the value is an array location or a millimetre position in RAS+/LPS. |
| Binary mask | Multi-class label map | A mask marks included/not included; a label map assigns declared integer classes. |
| Label map | Vector contour | One occupies a raster grid; the other stores paths in a coordinate system. |
| Probability | Binary decision | A threshold converts one into the other; changing the threshold changes the confusion matrix. |
| AUROC | Calibration | AUROC measures ranking across thresholds; calibration measures whether probabilities match frequencies. |
| Visual quality | Numeric/scientific validity | Use a shared display range, geometry check, reference, metric, and failure inspection. |
| Runtime verification | Model accuracy | Smoke/golden checks show that packaged computation behaves as expected, not that it generalizes. |
| Local image opening | Cloud image transfer | Opening stays local; a reviewed provider request crosses the trust boundary. |
| Metadata removal | De-identification | Inspect pixels, filenames, context, and residual risk—not only metadata fields. |
| Model output | Clinical finding | OpenMedVisionX outputs are for learning/research and require independent validation; they are never a diagnosis. |

## Next steps

- Prepare and open data: [Data and formats](DATA_FORMATS.md)
- Complete a first session: [Quickstart](QUICKSTART.md)
- Learn each workspace: [User guide](USER_GUIDE.md)
- Follow a structured course: [Learning curriculum](TEACHING_CURRICULUM.md)
- Review AI privacy: [AI Assistant, privacy, and cloud images](LLM_SECURITY.md)
- Resolve a problem: [Troubleshooting](TROUBLESHOOTING.md)

Next: [follow the learning curriculum](TEACHING_CURRICULUM.md).
