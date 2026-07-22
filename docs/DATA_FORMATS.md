# Data and Formats

[简体中文](DATA_FORMATS.zh-CN.md) · [Documentation home](INDEX.md) · [Images workspace](USER_GUIDE.md#images)

This guide helps you choose, prepare, open, and interpret data in
OpenMedVisionX. Start here when you are unsure which button to use, whether a
file will be treated as a page or a volume, or how an annotation relates to the
image already on screen.

> [!WARNING]
> Use only data that you are permitted to handle. OpenMedVisionX is a learning
> and research tool, not a medical device or a DICOM anonymizer. Keep an
> unchanged backup of every source and never use an output for a patient
> decision.

## Before you choose data

For a first session, prefer the built-in synthetic CT phantom or a public,
non-sensitive raster image. When you bring your own data:

1. Confirm that its license, consent, and institutional rules permit local use.
2. Keep patient data, private model files, credentials, and exports outside the
   source repository.
3. Select the narrowest useful input: one file, one intended series directory,
   or one minimal ZIP rather than a broad archive.
4. Keep the original unchanged. OpenMedVisionX reads the path you select and
   creates display state or derived results separately.
5. Know what the numbers mean. Record modality, units, spacing, orientation,
   and any preprocessing supplied with the data.
6. Inspect both metadata and visible pixels for identifiers. Removing a
   filename or metadata field does not remove text burned into an image.

OpenMedVisionX validates content rather than trusting a filename extension. A
renamed, malformed, ambiguous, oversized, or unsupported file can therefore be
rejected even when its suffix looks correct.

## Support matrix

| Input | How to open it | Installation | How it is interpreted | Important boundary |
| --- | --- | --- | --- | --- |
| PNG or JPEG image | **Open image / DICOM ZIP** | Base install | One 2-D raster image | No medical spacing, modality, or HU is inferred. JPEG is lossy and is not accepted as a label map. |
| TIFF image | **Open image / DICOM ZIP** | Base install | One 2-D image or an independent page sequence | Page order and DPI do not establish a physical volume. Very large flat images may use a bounded preview. |
| One DICOM image file | **Open image / DICOM ZIP** | Base install | A validated image or supported multi-frame object | Content, pixel data, modality, and geometry must be supported; a `.dcm` suffix alone proves nothing. |
| DICOM directory | **Open DICOM folder** | Base install | One explicitly selected image series | If several series are present, choose the intended series after checking its safe summary. Do not select a broad patient archive. |
| DICOM ZIP | **Open image / DICOM ZIP** | Base install | One selected series from a bounded archive | Member paths, counts, sizes, compression ratio, and decoded size must pass safety limits. |
| NIfTI `.nii` or `.nii.gz` | **Open image / DICOM ZIP** | Optional `nifti` extra | A spatial volume represented internally in RAS+ | A 4-D file requires an explicit time-point/channel choice. Intensity meaning is unknown unless a trusted workflow declared it. |
| Current-plane mask, PNG or TIFF | **Mask…** | Base install | A quick binary overlay on exactly one visible plane | Must be lossless, grayscale, single-page, and exactly the same height and width. Non-zero values become the overlay. |
| Current-plane annotation JSON | **Annotations…** | Base install | Boxes and points in top-left pixel coordinates on one plane | Must use the OpenMedVisionX schema, match the current image size, and contain no external references. |
| DICOM SEG or RTSTRUCT | **DICOM SEG / RTSTRUCT…** after opening its DICOM reference | Base install | An immutable segmentation or contour layer in the active study | Referenced series, frame of reference, SOP instances, and geometry must match. RTSTRUCT remains vector contours. |
| PNG or lossless TIFF label map | **Label map…** after choosing a reference series | Base install | An immutable discrete segmentation layer | Values must be finite non-negative integers. JPEG, color masks, and continuous-valued maps are rejected. |
| NIfTI label map | **Label map…** after choosing a reference series | Optional `nifti` extra | A spatial discrete segmentation layer with its own affine | A 4-D label map requires an explicit volume choice; geometry is never silently forced to match. |

The support matrix describes the desktop workflow, not every format that a
library might be able to decode. Unsupported acquisition objects, compressed
archives, temporal workflows, or annotation variants are rejected rather than
being interpreted by guesswork.

## Open DICOM data

### Choose a file, folder, or ZIP

Use **Open image / DICOM ZIP** for one DICOM file or a deliberately prepared
DICOM ZIP. Use **Open DICOM folder** for a local directory containing the
intended series.

- A single file can represent one classic image or one supported Enhanced
  multi-frame object.
- A folder can contain one or more series. OpenMedVisionX inspects the selected
  location so that you can choose a series; it does not recursively scan
  unrelated neighbouring directories.
- A ZIP should contain only the data needed for the intended series. It is
  checked for unsafe member paths, excessive member count, excessive expanded
  size, and suspicious compression ratios before use.

Do not rename a non-DICOM file to `.dcm`, reconstruct slice order from
filenames, or disable archive limits to make an unknown input open.

### Select the intended series

When more than one series is discovered, the selector shows a safe summary
such as modality, dimensions, and instance or frame count. Compare that summary
with the dataset documentation before continuing.

Choose only when you can identify the intended series. Cancel when series are
ambiguous, when expected instances are missing, or when the displayed modality
and dimensions do not match your task. Selecting the wrong series can still
produce a plausible-looking image, so visual appearance alone is not enough.

### Understand the Enhanced multi-frame boundary

The supported Enhanced path is monochrome CT or MR with consistent geometry in
its shared or per-frame functional groups. OpenMedVisionX rejects, rather than
guesses around, conditions such as:

- missing image orientation or position;
- inconsistent or non-uniform frame spacing;
- unsafe frame counts or decoded sizes;
- unsupported color pixel data; or
- a mixture of one multi-frame instance and separate classic image instances.

If an Enhanced object is rejected, retain the original and request a
standards-conformant export. Do not manufacture geometry from filenames or
screen order.

### Check DICOM meaning after loading

Confirm the series, modality, dimensions, spacing, orientation, frame count,
and intensity label shown by the application. DICOM metadata can describe
patient-space geometry, but only validated evidence is used:

- CT values are labelled as Hounsfield units only when valid rescale evidence
  supports that interpretation.
- PET values are labelled as SUV only when the required units, SUV type, and
  correction evidence are present.
- Missing or inconsistent geometry prevents physical volume operations rather
  than being repaired silently.

The internal volume convention is RAS+. DICOM patient coordinates are commonly
expressed in LPS; OpenMedVisionX performs the validated coordinate conversion
without changing the source file.

## NIfTI volumes

Install the optional loader in the active project environment, then restart the
application:

```bash
conda activate openmedvisionx
python -m pip install -e ".[nifti]"
```

Select `.nii` or `.nii.gz` with **Open image / DICOM ZIP**. After loading,
inspect shape, voxel spacing, orientation, affine, and intensity meaning before
measuring or comparing results.

### Three-dimensional NIfTI

A 3-D NIfTI is treated as a spatial volume and normalized to the application's
internal RAS+ convention. Its affine connects voxel indices to physical world
coordinates. A valid affine does not, by itself, prove modality, acquisition
protocol, or intensity units.

### Four-dimensional NIfTI

OpenMedVisionX does not guess whether the fourth axis represents time,
channels, echoes, or another dimension. The selection dialog therefore asks
for one time point or channel and records the chosen zero-based index in
provenance. The desktop then works with that selected 3-D volume; this is not
full 4-D playback or analysis.

Cancel if the dataset documentation does not explain the fourth axis. The same
rule applies to a 4-D NIfTI label map.

### NIfTI intensity and geometry cautions

- Intensity semantics remain unknown unless a trusted workflow explicitly
  declared them. Do not call generic NIfTI values HU.
- Check qform/sform warnings and affine orientation before treating two files
  as aligned.
- Matching array shapes do not prove matching physical geometry.
- Reorientation into the internal convention does not authorize an implicit
  resampling of a mismatched label map.

## Open raster images, TIFF pages, and large images

PNG, JPEG, and TIFF are useful for learning pixels, display mapping, histograms,
measurements, and compatible 2-D model inputs. They do not automatically carry
medical meaning.

### PNG and JPEG

- PNG can preserve lossless grayscale or color samples.
- JPEG is lossy: compression can alter edges and small structures even when an
  image looks acceptable.
- Neither format establishes modality, physical spacing, or HU.
- EXIF orientation and color/alpha handling are explicit transformations; read
  the image details before comparing coordinates with another file.

Only use **Set raster spacing…** when a trusted source gives the physical pixel
spacing. OpenMedVisionX records that value as user-provided. DPI is a printing
property and is not accepted as medical pixel spacing.

### TIFF pages are not automatically a volume

A multi-page TIFF is navigated as a page sequence unless trusted spatial
metadata establishes more. Page order, equal page size, and DPI alone do not
prove slice spacing, orientation, origin, or a patient coordinate system. Use a
validated DICOM or NIfTI volume when physical orthogonal views are required.

### Large flat images may use a bounded preview

To keep memory use bounded, a very large flat raster can be displayed from a
reduced preview rather than retaining the entire decoded page. Check the image
details for preview dimensions and recorded transforms before measuring or
using the active image as model input. Never assume that a model received
original-resolution pixels unless its recorded input proves it.

## Annotations, masks, and label layers

OpenMedVisionX has two separate annotation lifecycles. Choose the one that
matches your task.

| Need | Use | Scope and lifetime | Geometry behavior |
| --- | --- | --- | --- |
| Quickly compare a simple mask, boxes, or points with the plane now on screen | **Mask…** or **Annotations…** | Bound to one current plane or page. It reappears when you return to that plane and is cleared by **Clear marks** or by starting another study. It is not a clinical Study layer. | Exact pixel-size match only; no resize or resampling. |
| Inspect a reusable segmentation or contour object with a referenced image series | **DICOM SEG / RTSTRUCT…** or **Label map…** | Imported into the immutable Study → Series → Layer hierarchy with provenance and separate presentation state. | Identity and physical geometry are validated. A mismatch is never silently overlaid. |

Using the simple pairing path does not convert a mask or JSON file into DICOM
SEG, RTSTRUCT, or a volume label map. Conversely, importing a clinical layer
does not flatten it into a temporary current-plane overlay.

### Pair a simple current-plane mask

1. Open the intended image, page, or slice and click its view so that it is the
   active view.
2. Choose **Mask…** to pair a local mask with the current plane.
3. Select a grayscale, single-page PNG or lossless TIFF with exactly the same
   height and width as the active plane.
4. Review the pairing status. Non-zero pixels are displayed as one binary
   overlay; source values are not modified.

RGB/RGBA masks, JPEG masks, multi-page TIFF masks, and size mismatches are
rejected. The pairing does not resize the mask or infer physical coordinates.

### Pair a simple annotation JSON

Choose **Annotations…** while the intended plane is active. The JSON file is
UTF-8, self-contained, at most 4 MiB, and uses top-left pixel
coordinates. `image_size` is `[width, height]` and must match the active plane.
Boxes use `(x1, y1)` as the included top-left corner and `(x2, y2)` as the
lower-right boundary; points and boxes must remain inside the image.

Minimal example:

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

At least one box or point is required. Unknown top-level fields and external
file references are rejected; loading an annotation never follows another
path. This format is for a bounded teaching overlay, not a general annotation
exchange standard.

### Import DICOM SEG or RTSTRUCT

Open the referenced DICOM image series first, then choose **DICOM SEG /
RTSTRUCT…**. The importer checks the referenced Series Instance UID, Frame of
Reference UID, SOP instances, and geometry before adding a layer.

- Binary and fractional DICOM SEG retain their native representation. A
  fractional layer uses continuous interpolation only for an explicitly
  confirmed derived display resampling.
- RTSTRUCT closed planar contours are converted from DICOM LPS to internal RAS+
  for display. They remain vector contours and are not rasterized.
- An object that references another series is rejected rather than attached to
  whichever image happens to be active.

### Import a label map

Open and select the intended reference series, then choose **Label map…**.
Accepted maps contain finite, non-negative discrete integer labels:

- one PNG or positively identified lossless TIFF for raster label data; or
- NIfTI after installing the `nifti` extra.

JPEG, color masks, negative labels, non-finite values, and continuous-valued
maps are rejected. A 4-D NIfTI requires an explicit volume selection. Raster
maps use the reference axes you selected; NIfTI keeps its own canonical RAS+
affine for geometry comparison.

### Resolve a geometry mismatch explicitly

When a raster SEG or label map does not match the reference grid, choose one of
the options presented by the application:

- cancel the import;
- retain the original as a hidden, unmatched layer; or
- explicitly confirm a separate derived display layer when that option is
  offered.

The original import and its geometry remain unchanged. Discrete labels use
nearest-neighbour interpolation; fractional values require continuous
interpolation. Visibility, opacity, lock state, and per-label visibility are
presentation choices and do not rewrite either the source or derived array.

## Geometry, intensity, and display boundaries

Use this checklist before measuring, overlaying, reconstructing, or running a
model:

| Check | What to verify | What OpenMedVisionX does not assume |
| --- | --- | --- |
| Object type | 2-D raster, page sequence, or physical volume | Several pages do not automatically become a volume. |
| Shape | Width, height, slices/frames, channels | Equal shape does not prove equal geometry. |
| Geometry | Spacing, origin, orientation/direction, affine, coordinate convention | DPI is not medical spacing; array axes alone are not axial/coronal/sagittal. |
| Intensity | dtype, numeric range, modality, units, rescale, normalization | Grayscale is not automatically HU; NIfTI intensity is not inferred. |
| Active input | Plane, slice/page index, and RAS+ coordinate when available | Downstream tasks do not silently choose a different plane. |
| Transform | EXIF orientation, canonicalization, crop, resize, or confirmed resampling | A display transform does not change the source array. |

**Display mapping** controls how stored values become visible brightness or
color. Window, range, brightness, contrast, gamma, and inversion can make two
views look different while their source values remain identical. Conversely,
independent auto-ranging can make numerically different arrays look similar.
Compare values and geometry, not appearance alone.

## Export without changing the source

In the Images workspace:

- **Export PNG…** creates a new PNG from the full current 2-D plane after the
  active display mapping. It does not replace the source array and does not
  include viewport zoom, pan, paired overlays, annotations, or clinical
  layers.
- **Save record…** creates a pixel-free experiment JSON with image summary,
  display parameters, numeric measurements, and pairing flags. It does not
  embed image pixels, private paths, or patient metadata.

Both operations use a destination you select and refuse to overwrite an
existing file. A rendered PNG is a view for communication or teaching; it is
not a DICOM export, a reusable segmentation object, or proof that intensity and
geometry semantics were preserved.

## Privacy check before opening or sharing data

Before opening data:

- prefer synthetic, public, or institutionally approved material;
- keep source data and exports outside the repository;
- select only the needed file, series directory, or minimal archive;
- remember that DICOM metadata and filenames can contain identifiers; and
- inspect visible pixels for names, dates, accession numbers, facility labels,
  or other burned-in text.

Local image loading, navigation, measurements, annotation import, and export do
not themselves contact an AI provider. OpenMedVisionX does not automatically
upload data or download a missing model. However, the application is not an
anonymization tool: safe summaries and metadata filtering in the interface do
not prove that a source or screenshot is de-identified.

Before sharing an export, inspect the actual file again. Before using the AI
Assistant, read [AI Assistant, privacy, and cloud
images](LLM_SECURITY.md)—text can contain sensitive information, and sending a
rendered image requires a separate exact one-request review.

If a file is rejected, keep it unchanged and use
[Troubleshooting](TROUBLESHOOTING.md#opening-images-and-volumes) rather than weakening a
safety or geometry check.

## Next steps

- Operate the viewer: [User guide · Images](USER_GUIDE.md#images)
- Learn the underlying terms: [Glossary](GLOSSARY.md)
- Practise pixels and geometry: [Learning curriculum](TEACHING_CURRICULUM.md#1-pixels-dtype-and-display)
- Fix a loading problem: [Troubleshooting · Opening images and volumes](TROUBLESHOOTING.md#opening-images-and-volumes)

Next: [learn the Images workspace](USER_GUIDE.md#images).
