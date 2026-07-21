# Teaching Curriculum

This curriculum takes a learner from pixels and bit depth through medical
spatial geometry, analytical and iterative CT reconstruction, external model
inference, and multimodal AI. Every exercise uses arrays, phantoms, mock
tensors, temporary files, or local mock HTTP services generated at runtime.
No medical image, checkpoint, weight, or real API credential is required.

> [!WARNING]
> All activities are for education and research. Outputs are not clinical
> findings and must not be used for diagnosis or patient decisions.

## Learning path

| Chapter | Main question | OpenMedVisionX contracts |
| --- | --- | --- |
| 1. Pixels and digital images | How do sampling, dtype, bit depth, and color change what is measured? | `RasterImage2D`, display mappings |
| 2. Raster files and coordinates | How do file encoding and orientation differ from canonical pixels? | `ImageLoader`, `TransformRecord` |
| 3. DICOM and NIfTI geometry | How are stored voxels related to HU and patient/world space? | `ImageVolume`, RAS+ geometry |
| 4. Radon transform | How does an object become a sinogram? | `generate_angles`, `generate_sinogram` |
| 5. BP and FBP | Why does filtering change a backprojection? | reconstruction algorithms |
| 6. SART | How does iterative correction converge, fail, and cancel? | `SARTReconstruction`, `BackgroundTask` |
| 7. Metrics and ROI | When is a numerical comparison meaningful? | `compute_metrics`, `MetricReport` |
| 8. Model inference | What must be declared to reproduce preprocessing and output mapping? | manifest, `ModelPlugin`, typed results |
| 9. Multimodal AI | What leaves the machine, and what can a cloud model legitimately explain? | `LLMProvider`, `RenderedPreview` |

## Standard experiment-page template

Every GUI lesson or written lab derived from this curriculum must include:

1. **Principles** — the concept and its limits.
2. **Formulae** — notation, units, and assumptions.
3. **Parameters** — controls, valid ranges, units, and expected tradeoffs.
4. **Procedure** — reproducible steps using generated or mock data.
5. **Expected observations** — what should change and what should remain
   invariant.
6. **Common mistakes** — likely semantic, coordinate, numerical, or privacy
   errors.
7. **Thinking questions** — questions that require explanation, prediction, or
   experimental evidence.

An experiment may save parameters, configuration, and numerical metrics.
Source pixels, derived medical images, prompts containing sensitive text,
provider responses, and model artifacts are saved only when the user explicitly
selects an appropriate local destination. Source images are not overwritten.

---

## Chapter 1 — Pixels, bit depth, color, and compression

### Principles

A digital image is a sampled and quantized signal. Array shape, dtype, numeric
range, color space, channel order, and alpha semantics are independent facts.
A display is a mapping of source values; it is not permission to replace the
source array with 8-bit pixels.

Grayscale windowing and RGB brightness/contrast/gamma are display concepts.
HU is a modality-specific intensity meaning and does not apply to a generic
grayscale or RGB image.

### Formulae

For unsigned bit depth (b), the number of representable code values is:

$$L = 2^b.$$

A histogram count for bin (k) is:

$$h(k) = \sum_{x,y} \mathbf{1}[I(x,y) \in B_k].$$

A simple normalized display mapping is:

$$D(x,y) = \operatorname{clip}\left(
\frac{I(x,y)-a}{b-a}, 0, 1\right).$$

A gamma display adjustment can be written:

$$D_\gamma = D^{\gamma}.$$

The source `I` remains unchanged.

### Parameters

- dtype: `uint8`, `uint16`, signed integer, or floating point;
- bit depth and source numeric range;
- grayscale display minimum/maximum;
- RGB brightness, contrast, and gamma;
- interpolation: nearest, bilinear, bicubic, area, or Lanczos;
- JPEG quality and repeated encode/decode count;
- alpha policy: reject, preserve, drop explicitly, or composite explicitly.

### Procedure

1. Generate a 16-bit horizontal ramp, a checkerboard, an RGB color chart, and
   an RGBA shape in memory.
2. Wrap each in the appropriate `RasterImage2D` and inspect shape, dtype,
   bit depth, channel order, and capabilities.
3. Display the 16-bit ramp with several display ranges without modifying the
   array.
4. Compare histograms before and after an intentional 8-bit conversion.
5. Enlarge the checkerboard with nearest, bilinear, and bicubic interpolation.
6. Encode the color chart to JPEG at several quality levels and compute a
   pixel difference after decoding.
7. Composite the RGBA image on black and white backgrounds and record the
   declared conversion.

### Expected observations

- A 16-bit source retains more distinct code values even if the monitor shows
  an 8-bit-looking rendering.
- Display range changes visibility but not source dtype or histogram.
- Nearest interpolation preserves hard labels but creates block edges;
  continuous interpolation smooths intensities.
- JPEG creates irreversible block/ringing or color artifacts, especially after
  repeated encoding.
- Alpha compositing depends on the chosen background and is not equivalent to
  silently dropping alpha.

### Common mistakes

- Treating an 8-bit display buffer as the loaded source.
- Calling generic grayscale values HU.
- Swapping RGB and BGR without recording it.
- Applying bilinear interpolation to a discrete class mask.
- Using DPI as medical pixel spacing.
- Interpreting JPEG artifacts as anatomy or model evidence.

### Thinking questions

1. Can two arrays with different dtypes look identical? What measurement
   reveals the difference?
2. Why should a model manifest specify both dtype and numeric range?
3. Which interpolation is appropriate for intensity, probability, and label
   images, and why?
4. How would premultiplied alpha change compositing?

---

## Chapter 2 — PNG, JPEG, TIFF, EXIF, and reversible coordinates

### Principles

A filename extension is a hint, while a file signature and successful decoder
validation provide stronger evidence. A loaded image has raw file coordinates
and canonical display coordinates. EXIF orientation, resize, crop, and
letterbox operations must be recorded so annotations and model results can
return to original pixels.

A multipage TIFF is an `ImageSequence2D` unless validated spatial metadata
establishes medical volume geometry.

### Formulae

Represent a pixel as homogeneous coordinates:

$$\tilde{p} = [x, y, 1]^T.$$

A recorded operation maps it forward:

$$\tilde{p}' = H\tilde{p},$$

and returns it to source coordinates with:

$$\tilde{p} = H^{-1}\tilde{p}'.$$

Ordered operations compose as:

$$H_{total} = H_n H_{n-1}\cdots H_1.$$

### Parameters

- extension and detected signature;
- EXIF Orientation values 1–8;
- original and canonical `(height, width)`;
- crop rectangle `left, top, width, height`;
- resize or letterbox output shape;
- color/palette/alpha conversion;
- maximum pixels, frames, and decoded bytes;
- TIFF page index and playback speed.

### Procedure

1. At runtime, create grayscale/RGB/RGBA/palette PNG, an EXIF-oriented JPEG,
   and single/multipage TIFF files.
2. Probe each with `ImageLoaderRegistry.probe`; record chosen loader,
   confidence, and format details.
3. Rename one file to a wrong extension and compare extension and signature
   evidence.
4. Load the EXIF JPEG and inspect its `TransformRecord.operations`.
5. Compose EXIF, resize, crop, and letterbox transforms.
6. Map generated points and boxes forward, then use `inverse` on their corner
   points.
7. Navigate a multipage TIFF and compare its capabilities with
   `ImageVolume.capabilities`.
8. Trigger a pixel/frame/decoded-byte limit with a generated header or mock
   decoder and cancel a background load.

### Expected observations

- The registry chooses by accepted probe confidence rather than UI extension
  logic.
- EXIF orientations 5–8 swap output height and width.
- A point round trip is numerically close to its original location.
- Letterboxing adds an offset in addition to scale.
- Multipage TIFF enables frame navigation/playback but not orthogonal views or
  volume rendering.
- Malformed, oversized, or cancelled input produces an understandable error
  without freezing the UI.

### Common mistakes

- Trusting only `.jpg` or `.tif`.
- Applying EXIF visually but not during model preprocessing.
- Forgetting operation order when composing transforms.
- Mapping only two corners of a rotated/reflected box without rebounding all
  four corners.
- Treating TIFF page spacing or DPI as patient geometry.
- Decoding the entire high-resolution image on the GUI thread.

### Thinking questions

1. Why is an inverse coordinate record needed even if the displayed image is
   already canonical?
2. Which transformations preserve axis-aligned boxes?
3. Why should a page sequence and a physical volume have distinct Python
   types?
4. What information can a probe safely read without decoding all pixels?

---

## Chapter 3 — DICOM, NIfTI, HU, and RAS+ geometry

### Principles

DICOM commonly expresses patient geometry in LPS coordinates; NIfTI uses an
affine commonly interpreted in RAS+. OpenMedVisionX normalizes validated
volumes to internal RAS+. Array indices, voxel coordinates, and world
coordinates are different coordinate systems.

DICOM stored pixel values are not automatically HU. CT rescale slope and
intercept must be applied consistently to display, point/ROI measurement, and
reconstruction input.

### Formulae

For a CT pixel with stored value (S):

$$HU = mS + b,$$

where (m) is `RescaleSlope` and (b) is `RescaleIntercept`.

LPS and RAS point coordinates are related by:

$$
\begin{bmatrix}R\\A\\S\\1\end{bmatrix}
=
\begin{bmatrix}
-1&0&0&0\\
0&-1&0&0\\
0&0&1&0\\
0&0&0&1
\end{bmatrix}
\begin{bmatrix}L\\P\\S\\1\end{bmatrix}.
$$

For voxel coordinate ([x,y,z,1]^T):

$$p_{RAS} = A[x,y,z,1]^T,$$

with:

$$A_{0:3,0:3} = D\operatorname{diag}(s_x,s_y,s_z).$$

### Parameters

- DICOM series identity and ordering;
- `ImageOrientationPatient`, `ImagePositionPatient`;
- `PixelSpacing`, inter-slice distance, and slice thickness;
- rescale slope/intercept and modality;
- NIfTI affine and orientation;
- output RAS+ grid and interpolation;
- window width/level for display only.

### Procedure

1. Generate a small synthetic CT DICOM series with known orientation,
   positions, spacing, slope, and intercept.
2. Generate a NIfTI file representing the same numeric volume and world
   geometry.
3. Load both through `ImageLoaderRegistry` and inspect array shape, affine,
   spacing, origin, direction, modality, and intensity semantics.
4. Verify several stored pixels against the HU equation.
5. Convert chosen voxel coordinates to RAS world coordinates with
   `voxel_xyz_to_world_ras`, then invert them.
6. Resample to a common RAS grid and compare axial/coronal/sagittal views.
7. Compare pixel and physical measurements.
8. Attempt the same physical/HU tools on a generated PNG and explain why its
   capabilities differ.

### Expected observations

- DICOM and NIfTI may have different stored axis order while representing the
  same RAS+ geometry.
- HU changes when slope/intercept is nontrivial; windowing changes only display.
- Oblique or anisotropic data requires geometric resampling for a physically
  correct orthogonal view.
- Volume spacing is file-derived, while generic raster spacing is absent unless
  explicitly supplied by the user.

### Common mistakes

- Sorting slices by only the third component of patient position.
- Using `SliceThickness` as the only source of inter-slice distance.
- Applying slope/intercept for reconstruction but not measurement, or vice
  versa.
- Treating array order `(z,y,x)` as world coordinate order.
- Replacing a missing affine with identity and claiming physical validity.
- Labeling generic grayscale intensities HU.

### Thinking questions

1. How can two differently oriented arrays represent the same physical object?
2. Why can slice thickness and center-to-center spacing differ?
3. Which interpolation should be used when resampling a CT intensity volume
   versus a segmentation label volume?
4. What evidence is required before enabling millimetre measurement?

---

## Chapter 4 — Radon transform and sinogram formation

### Principles

Parallel-beam CT measures line integrals through an object at multiple angles.
The sinogram organizes detector position against projection angle. Angular
sampling, detector sampling, support assumptions, and the circular mask all
affect reconstruction.

### Formulae

The 2D Radon transform is:

$$
R f(\theta,s) =
\int\!\!\int f(x,y)
\delta(s-x\cos\theta-y\sin\theta)\,dx\,dy.
$$

Parallel-beam projections obey:

$$Rf(\theta+\pi,s)=Rf(\theta,-s),$$

which explains the redundancy of 180–360 degree acquisition for a static ideal
parallel-beam object.

### Parameters

- generated phantom size and dynamic range;
- angular range: 180 or 360 degrees;
- number and spacing of projection angles;
- detector count;
- circular support mask;
- progress/cancellation interval.

### Procedure

1. Generate a Shepp–Logan or simple geometric phantom.
2. Use `generate_angles` for several angular counts across 180 degrees.
3. Call `generate_sinogram` with a fixed circle setting and inspect
   `SinogramResult`.
4. Repeat for 360 degrees with comparable angular density.
5. Compare a projection at (	heta+180^\circ) with the reversed detector
   samples at (	heta).
6. Reduce the number of views and observe the sinogram and later
   reconstruction.
7. Cancel a high-resolution generation task and inspect task state.

### Expected observations

- Smooth phantom structures trace sinusoidal paths.
- A 360-degree sinogram contains redundant ideal parallel-beam information.
- Sparse angular sampling creates structured streak artifacts after
  reconstruction.
- Changing circular support changes both valid input assumptions and output;
  it must remain consistent in all later algorithms.

### Common mistakes

- Comparing 180 and 360 degrees with different angular density.
- Passing angles whose count does not match the sinogram projection dimension.
- Generating with `circle=False` and reconstructing with `circle=True`.
- Interpreting a display color map as attenuation units.
- Reusing a sinogram after loading a new source image.

### Thinking questions

1. Why is 360-degree data redundant for ideal parallel beams but still used in
   some practical scanners?
2. How do detector sampling and angular sampling create different artifacts?
3. What changes if the object extends outside the assumed circle?
4. Which sinogram features correspond to points versus extended structures?

---

## Chapter 5 — Backprojection and filtered backprojection

### Principles

Unfiltered backprojection smears every measured projection back along its
acquisition lines. It localizes structure but introduces low-frequency blur.
Filtered backprojection first weights each projection in frequency to
compensate for that blur.

### Formulae

Ideal backprojection is:

$$
B(s)(x,y)=\int_0^\pi Rf(\theta,
x\cos\theta+y\sin\theta)\,d\theta.
$$

Filtered backprojection is:

$$
\hat f(x,y)=
\int_0^\pi
\left(h * Rf(\theta,\cdot)\right)
(x\cos\theta+y\sin\theta)\,d\theta,
$$

where the ideal ramp filter satisfies:

$$H(\omega)=|\omega|.$$

Windowed filters trade resolution for noise suppression.

### Parameters

- shared `ReconstructionRequest.theta_degrees`;
- `output_size` and `circle`;
- no filter for BP;
- ramp/Ram-Lak, Shepp–Logan, cosine, Hamming, or Hann filter for FBP;
- output display mapping independent from reconstruction values.

### Procedure

1. Reuse one validated `ReconstructionRequest` for every algorithm.
2. Run `BackProjection.reconstruct` and inspect intermediate values.
3. Run `FilteredBackProjection.reconstruct` with each filter.
4. Show original, sinogram, filtered projection/frequency response,
   backprojection, and final output in synchronized panels.
5. Keep reconstruction arrays in their algorithmic scale; use a separate
   display mapping.
6. Repeat with fewer views and with synthetic noise.
7. Compare execution time and progress traces.

### Expected observations

- BP is characteristically blurred.
- The ramp filter sharpens edges but amplifies high-frequency noise.
- Windowed filters reduce noise at the cost of spatial resolution.
- Sparse views create streaks that a smoother filter cannot fully remove.
- Consistent theta/circle/output geometry allows fair algorithm comparisons.

### Common mistakes

- Independently normalizing each result and then computing MSE.
- Using a different theta array during reconstruction.
- Cropping or resizing one output before comparison without recording it.
- Calling BP “FBP with a weak filter.”
- Inferring clinical image quality from one global metric.

### Thinking questions

1. Why does unfiltered backprojection emphasize low frequencies?
2. Which filter would you choose as noise increases, and what detail would be
   lost?
3. Can a display window make two numerically different reconstructions appear
   similar?
4. How does output size affect apparent sharpness and metric comparability?

---

## Chapter 6 — SART and iterative reconstruction

### Principles

SART alternates between forward projection, comparison with measured
projections, and a weighted image update. Intermediate states expose
convergence, instability, and the influence of initialization and relaxation.

### Formulae

For system (Ax=b), a simplified row-action correction is:

$$
x^{(k+1)} =
x^{(k)} +
\lambda A^T W
\left(b-Ax^{(k)}\right),
$$

where normalization weights are represented by (W) and
(lambda) is the relaxation factor. An implementation applies projection
groups rather than this unnormalized expression directly.

A data-consistency residual is:

$$r^{(k)} = \lVert b-Ax^{(k)}\rVert_2.$$

### Parameters

- iteration count;
- relaxation factor;
- initialization;
- projection order;
- non-negativity or other constraints when declared;
- snapshot interval;
- cancellation check interval.

### Procedure

1. Generate one phantom and sinogram with known geometry.
2. Start SART from zeros and run one iteration.
3. Record the image and residual after selected iterations.
4. Repeat for several relaxation values while holding all other parameters
   fixed.
5. Compare sparse-view and noisy cases.
6. Submit a longer run through `TaskRunner`; display monotonic progress.
7. Call `BackgroundTask.cancel()` and verify that the worker observes
   `TaskContext.raise_if_cancelled()`.

### Expected observations

- Early iterations recover coarse structure; later iterations refine or amplify
  noise.
- A moderate relaxation factor converges more smoothly than an excessive one.
- Sparse data can fit measured projections while leaving ambiguity/artifacts.
- Cancellation moves a running task through `CANCELLING` to `CANCELLED`
  only when the worker cooperates.

### Common mistakes

- Assuming more iterations always improve the true image.
- Changing relaxation and view count simultaneously.
- Reporting only the final image and hiding divergence.
- Updating Qt widgets directly from the worker callback.
- Treating task cancellation as forced thread termination.

### Thinking questions

1. Can the projection residual decrease while perceptual quality becomes worse?
2. How would a prior or regularizer change the update and interpretation?
3. What snapshot spacing best reveals early versus late convergence?
4. Where should cancellation checks be placed to balance responsiveness and
   overhead?

---

## Chapter 7 — Metrics, difference maps, and ROI analysis

### Principles

Metrics are meaningful only when reference and reconstruction share shape,
coordinates, support, and intensity scale. A global average can hide a local
failure; difference maps, heatmaps, and ROIs provide spatial context.

### Formulae

For (N) pixels:

$$MSE = \frac{1}{N}\sum_i (x_i-y_i)^2.$$

With data range (R):

$$PSNR = 10\log_{10}\left(\frac{R^2}{MSE}\right).$$

For local means (mu), variances (sigma^2), and covariance
(sigma_{xy}), SSIM is:

$$
SSIM(x,y)=
\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}
{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}.
$$

A signed difference is (d=y-x); an absolute error heatmap uses
(|d|).

### Parameters

- shared data range and normalization policy;
- support/circular mask;
- SSIM window and constants;
- signed versus absolute difference;
- ROI coordinates and physical/pixel units;
- aggregation: mean, standard deviation, maximum, or percentile.

### Procedure

1. Create a reference and controlled noisy/blurred/shifted variants.
2. Confirm identical shape and coordinates before calling `compute_metrics`.
3. Compute MSE, PSNR, and SSIM using one declared range.
4. Display signed difference and absolute error heatmap with a labeled scale.
5. Draw several ROIs over uniform, edge, and artifact regions.
6. Repeat after a deliberate one-pixel shift.
7. Compare metrics for BP, filtered FBP, and selected SART iterations without
   independent normalization.

### Expected observations

- A small shift can strongly reduce pixel metrics despite visual similarity.
- Blur and noise can trade PSNR against edge preservation.
- A global score can mask a severe local artifact.
- ROI placement changes the scientific question being answered.
- Independently normalized inputs can produce deceptively favorable scores.

### Common mistakes

- Using different intensity ranges or display buffers.
- Computing SSIM with a zero or incorrect data range.
- Comparing arrays that are spatially misregistered.
- Selecting an ROI after seeing the result without documenting that choice.
- Reporting six decimal places without uncertainty or context.

### Thinking questions

1. Why can PSNR and SSIM rank two reconstructions differently?
2. Which metric is most sensitive to a constant intensity bias?
3. How should ROI selection be made reproducible?
4. What additional metric would help for a task-specific evaluation?

---

## Chapter 8 — Reproducible model preprocessing and inference

### Principles

A checkpoint alone is not a reproducible model. Runtime, layout, color, dtype,
numeric range, normalization, geometry, output meaning, labels, and
postprocessing must all be declared. OpenMedVisionX therefore treats the
manifest and adapter contract as the compatibility boundary.

Typed results and reversible transforms make model evidence inspectable in
source-image coordinates.

### Formulae

A declared scale/offset and channel normalization can be represented as:

$$z_c = \frac{s x_c + o - \mu_c}{\sigma_c}.$$

For logits (a_k), softmax probability is:

$$p_k = \frac{e^{a_k}}{\sum_j e^{a_j}}.$$

For two boxes or masks (A) and (B):

$$IoU = \frac{|A\cap B|}{|A\cup B|}.$$

Coordinates return to the source with:

$$p_{source}=H^{-1}p_{model}.$$

### Parameters

- runtime kind and device;
- model ID, version, family/source, task, and licenses;
- modality and 2D/2.5D/3D/4D dimensionality;
- input shape, layout, color/channel/alpha, dtype, range, mean/std;
- resize/crop/letterbox and interpolation;
- output semantic, coordinates, activation, threshold, NMS, labels;
- uncertainty and optional prompts/features/attention/embeddings;
- Python consent and subprocess isolation.

### Procedure

1. Inspect the maintained
   [example manifest](../src/dicom_viewer/inference/examples/manifest.yaml).
2. Load it with `load_plugin_manifest` without importing model code.
3. Display source, tasks, runtime, and code/model/weight licenses.
4. Prepare a generated image and retain the named `TransformRecord`.
5. Use a mock `ModelPlugin` to return classification, segmentation, or
   detection results with valid `InferenceProvenance`.
6. Call `validate_prediction_result`.
7. Map masks/boxes/keypoints/heatmaps back to source pixels and overlay them.
8. Trigger a wrong layout, color, shape, spacing, task, or provenance error.
9. For a mock Python adapter, demonstrate that consent and
   `subprocess_isolated=True` are both required.

### Expected observations

- Manifest validation happens before adapter import or model load.
- Changing RGB/BGR, range, normalization, or resize policy changes predictions.
- Letterbox padding must be removed through inverse mapping before overlay.
- A result whose task or provenance disagrees with the manifest is rejected.
- Python process separation prevents a crash from directly becoming a Qt
  widget exception, but it does not make untrusted code safe.

### Common mistakes

- Claiming that any `.pth` or `.ckpt` can run directly.
- Downloading missing weights automatically.
- Copying user weights into the project.
- Trusting a manifest as proof that code is benign.
- Dropping alpha or changing channel order silently.
- Resizing a mask back with bilinear interpolation.
- Letting a plugin draw directly on the GUI.

### Thinking questions

1. Which manifest fields are necessary to reproduce a published result?
2. Why is a capability protocol more future-proof than a model-name whitelist?
3. What does subprocess isolation protect against, and what does it not?
4. How should uncertainty be shown when a model is not calibrated?

---

## Chapter 9 — Multimodal AI and responsible cloud interaction

### Principles

An LLM/VLM can support concept explanation, parameter discussion, retrieval,
and reflection. It does not establish clinical truth. The provider, model,
prompt, transferred pixels, time, and disclaimer are part of the result's
provenance.

Cloud transfer changes the trust boundary. Text-only chat and image-enabled
chat are separate capabilities; image transfer is disabled until the user
authorizes one provider.

### Formulae

For normalized image/text embeddings (v) and (t), cosine similarity is:

$$
\operatorname{sim}(v,t)=
\frac{v\cdot t}{\lVert v\rVert_2\lVert t\rVert_2}.
$$

Similarity is not a probability or diagnosis. A temperature-scaled retrieval
distribution may be:

$$p_i = \frac{\exp(s_i/\tau)}{\sum_j\exp(s_j/\tau)},$$

but its interpretation depends on model training and calibration.

### Parameters

- Provider and HTTPS endpoint;
- user-selected model ID;
- text, streaming, and vision capability flags;
- credential reference, never the raw key;
- timeout and output-token limit;
- per-provider image authorization;
- rendered preview size and selected slice;
- prompt wording and explicit educational goal.

### Procedure

1. Inject a local mock `Transport`; do not configure a real API.
2. Create a provider with a placeholder model ID and environment credential
   reference supplied by the mock resolver.
3. Call text-only `chat` and `stream`; inspect the provenance footer.
4. Try to send an image with vision disabled and with consent absent.
5. Generate a small metadata-free PNG from a displayed synthetic slice and
   construct `RenderedPreview`.
6. Enable vision in the mock configuration, authorize that provider, preview
   the exact pixels, and send.
7. Verify another provider remains unauthorized.
8. Revoke authorization and confirm the next image request is rejected.
9. Add a burned-in synthetic identifier and discuss why metadata removal cannot
   remove it.
10. Compare a grounded explanation with an intentionally misleading prompt and
    identify unsupported claims.

### Expected observations

- No transport runs until explicitly injected.
- Text works without granting image transfer.
- Raw bytes or metadata-bearing PNG chunks are rejected.
- Consent applies only to one provider instance and can be revoked.
- Responses display provider, model, UTC time, and disclaimer.
- Fluent answers can still be incorrect, overconfident, or sensitive to prompt
  wording.

### Common mistakes

- Putting a key in configuration, source, logs, or screenshots.
- Hard-coding a “latest” model ID.
- Authorizing all providers when the user approved one.
- Sending original DICOM/NIfTI, a whole series, or DICOM metadata.
- Assuming a metadata-free PNG cannot contain burned-in identity text.
- Treating cosine similarity or generated prose as clinical probability.
- Running a real paid API in CI.

### Thinking questions

1. Which information must a learner see to reproduce or critique an assistant
   answer?
2. How does revocable per-provider consent differ from a single global toggle?
3. What evidence would distinguish a grounded explanation from a hallucination?
4. Which educational tasks should remain text-only even when vision is
   available?

---

## Optional advanced modules

After the core sequence, plugins can support:

- representation learning with embeddings, feature maps, attention, Grad-CAM,
  similarity, and retrieval;
- calibrated classification and uncertainty;
- semantic/instance/prompted segmentation;
- 2D/3D detection and grounding;
- registration with warped images, displacement/Jacobian fields, and inverse
  consistency;
- restoration and generation with trajectories and failure modes;
- WSI MIL using user-provided pre-tiles/features until a streaming loader
  exists;
- video/4D tracking and propagation;
- anomaly/OOD maps and reliability analysis.

Each advanced module uses the same seven-section experiment template and must
state what the current loader/runtime genuinely supports.

## Assessment guidance

Assess explanation and experimental evidence, not only a final screenshot.
A strong submission:

- predicts an outcome before changing a parameter;
- records input semantics, geometry, transforms, and units;
- compares one variable at a time;
- distinguishes source data from display mapping;
- shows intermediate states and failure cases;
- reports metrics with range, support, and ROI;
- identifies model/plugin/provider trust boundaries;
- includes no patient data, real key, or bundled weight;
- explains why results are educational rather than diagnostic.
