# Use a Custom Local Model

[简体中文](CUSTOM_MODELS.zh-CN.md) · [Documentation home](INDEX.md) · [User guide](USER_GUIDE.md#models)

This guide is for users who already have a local model and want to run it
through **Models → External manifests**. OpenMedVisionX does not train models,
search a model catalog, download weights, infer missing preprocessing, or make
an arbitrary checkpoint runnable.

> [!CAUTION]
> Load only models you trust and are licensed to use. A Python adapter executes
> third-party code. Running it in a separate process provides fault isolation,
> not a security sandbox.

## 1. Check whether the desktop can run your model

The current external-model page can supply exactly **one active 2-D image**.
Your model must accept that input after the preprocessing declared in its
manifest.

| Requirement | Current desktop support |
| --- | --- |
| One 2-D raster/image plane | Supported when its modality, spacing, dtype, channels, and preprocessing contract match. |
| ONNX model | Supported after installing `.[onnx]`. The default ONNX Runtime installation does not guarantee GPU execution. |
| TorchScript model | Supported after installing a compatible PyTorch runtime. |
| Python adapter | Supported only after explicit code trust; its own dependencies and environment remain your responsibility. |
| Multiple images/modalities | Not supported by the generic desktop flow. |
| Complete 3-D/4-D volume | Not supported by the generic desktop flow. |
| Prompt, text, k-space/mask, sinogram, temporal, or WSI input | Not supported by the generic desktop flow. |
| `.pth`, `.pt`, or `.ckpt` training checkpoint | Not directly supported unless it is a valid reviewed TorchScript artifact; ordinary pickle checkpoints need their architecture and trusted code. |

The protocol can describe more tasks than the desktop can currently collect.
A valid manifest can therefore be rejected by the visible capability gate.
That rejection protects you from a plausible-looking but scientifically invalid
run.

## 2. Prepare the runtime and local files

Activate the project environment first:

```bash
conda activate openmedvisionx
```

Install only the runtime your model needs:

```bash
python -m pip install -e ".[onnx]"     # ONNX + ONNX Runtime
```

For TorchScript, install the PyTorch build that matches your computer. Use the
CPU/GPU guidance in the [bundled model guide](MODEL_BUNDLES.md#install-pytorch)
rather than assuming one CUDA wheel works everywhere.

Keep the model outside the OpenMedVisionX repository. A practical local layout
is:

```text
my-local-model/
├── manifest.yaml
├── model.onnx                 # or a local TorchScript artifact
├── README.md                  # source, intended use, limits, and setup notes
└── LICENSE-or-NOTICE.txt
```

For a Python adapter, keep its adapter file and any dependency declaration in
the same trusted package directory. OpenMedVisionX never installs those
dependencies automatically.

Remote URLs, automatic downloads, and missing local files are rejected. Do not
put API keys, patient paths, or patient identifiers in the manifest.

## 3. Describe the model with a manifest

The manifest is a UTF-8 YAML file using schema version `"1.0"`. Start from the
maintained [example manifest](../src/workbench/inference/examples/manifest.yaml),
but replace every example value with the facts for your model. The referenced
`models/example_classifier.onnx` is intentionally absent, so the example itself
cannot run.

At minimum, the manifest must answer these questions:

| Section | What you must state |
| --- | --- |
| Identity | Stable name, version, family, description, source, and authors. |
| Task | Main task and a precise subtask; for example, classification and binary classification. |
| License | Separate terms for code, model definition, and weights. |
| Runtime | `onnx`, `torchscript`, or `python-adapter`, plus device policy. |
| Local files | Entrypoint and weight paths; add expected SHA-256 and byte size when available. |
| Input | Name, image semantic, modality, 2-D dimensionality, tensor shape/layout, color/channel order, dtype/range, spacing requirement, and ordered preprocessing. |
| Output | Tensor name, semantic meaning, shape/dtype, labels or coordinates, activation/postprocessing, threshold, and uncertainty meaning. |
| Capabilities | Only optional features the model really implements. |

This shortened example shows the shape of a single-image ONNX classifier. It
is not a universal template and is not runnable as written:

```yaml
schema_version: "1.0"
name: my-local-classifier
version: "1.0.0"
family: resnet
source:
  name: my-reviewed-source
  organization: my-organization
  repository: null
  publication: null
  model_id: classifier-v1
description: Binary classifier for one documented 2-D image domain.
tasks: [classification]
subtasks: [binary-classification]
license:
  code: MIT
  model: Apache-2.0
  weights: user-supplied
  notice: Review all terms before loading.
  urls: {}
runtime:
  kind: onnx
  device: auto
  provider: null
  options: {}
entrypoint:
  path: model.onnx
weights:
  - name: model
    path: model.onnx
    format: onnx
    sha256: REPLACE_WITH_SHA256
    size_bytes: REPLACE_WITH_BYTES
    required: true
    license: user-supplied
inputs:
  - name: image
    semantic: image
    modalities: [generic-image]
    dimensionality: 2d
    shape: [1, 3, 224, 224]
    spacing: {required: false, values: null, unit: pixel, tolerance: null}
    preprocessing:
      layout: nchw
      color_space: rgb
      channel_order: [R, G, B]
      alpha_handling: drop
      dtype: float32
      value_range: [0, 255]
      normalization:
        scale: 0.00392156862745098
        offset: 0.0
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
      spatial:
        - operation: letterbox
          size: [224, 224]
          interpolation: bilinear
          anchor: center
          pad_value: [0, 0, 0]
          allow_upscale: true
      orientation: apply-exif
    description: Canonically oriented RGB image.
outputs:
  - name: scores
    semantic: class-scores
    dtype: float32
    shape: [1, 2]
    coordinate_system: not-applicable
    labels: {"0": negative, "1": positive}
    postprocessing:
      activation: softmax
      threshold: null
      nms_iou_threshold: null
      interpolation: null
      discrete_labels: false
      parameters: {}
    uncertainty: {kind: none, description: Not calibrated, calibrated: false, units: null}
    description: Scores in manifest label order.
capabilities:
  prompts: false
  intermediate_features: false
  attention: false
  embeddings: false
  multiscale_outputs: false
  vector_fields: false
  sampling_trajectory: false
  multimodal_text: false
  uncertainty: false
authors: [Local user]
references: []
extensions: {}
```

Do not copy normalization, label order, range, resize, or channel order from
this example unless they are true for your model. A wrong but shape-compatible
preprocessing pipeline can produce convincing nonsense.

## 4. Review trust, files, and licenses

Before loading, verify all of the following:

- the model came from the source named in the manifest;
- the local file size and SHA-256 match the value supplied by that source;
- code, model, weights, and data licenses permit your intended use;
- the task, population, modality, dimensionality, input range, channel order,
  spacing, and normalization match the model's training/evaluation record;
- every output label and coordinate system is defined;
- you understand whether scores are probabilities, logits, calibrated values,
  or something else;
- no path points to a network share or untrusted executable.

For a Python adapter, read its source and dependency files before accepting the
trust dialog. Separate-process execution cannot protect credentials or readable
local files from malicious code.

## 5. Load and run the model

1. Open a permitted image in **Images**.
2. Click the exact plane or image you want to use. Confirm the **active view**
   line identifies the correct plane and slice.
3. Open **Models → External manifests**.
4. Choose **1. Choose manifest…** and select the local YAML file.
5. Read the displayed identity, source, task, input/output contract, runtime,
   licenses, and referenced files. Resolve every validation error before
   continuing.
6. Choose **2. Load model**. If the runtime is a Python adapter, accept only
   after completing the trust review above.
7. Read the capability gate. It must report that the active input satisfies the
   declared image count, semantic, dimensionality, modality, spacing, and
   preprocessing requirements.
8. If a very large source is represented by a bounded preview, do not approve a
   preview override unless using those preview pixels is scientifically valid
   for your experiment.
9. Choose **3. Run inference**.
10. Wait for completion or use **Stop** to request cancellation.

**Success check:** the status reports completion, the visualization matches a
declared output, and model name/version/runtime plus preprocessing provenance
remain visible with the result.

## 6. Interpret the result safely

Inspect the visualization together with the contract; never treat the picture
alone as the result.

- Classification: confirm label order, activation, threshold, and whether
  scores are calibrated.
- Segmentation: confirm source-plane mapping, class IDs, interpolation, and
  whether the mask is discrete or probabilistic.
- Detection/keypoints: confirm coordinates were mapped back through resize,
  crop, letterbox, and EXIF orientation.
- Reconstructed/restored images: confirm units, source geometry, numeric range,
  and whether measurement consistency was evaluated.
- Heatmaps/attention: treat them as model-derived signals, not automatically as
  anatomical explanations.

Record the input source and permission, manifest/model hash, runtime/device,
preprocessing, output mapping, warnings, and limitations. A successful run
shows technical compatibility, not generalization or clinical validity.

## 7. Common blocks and their meaning

| Message or symptom | What to do |
| --- | --- |
| Manifest is valid but **Run inference** is disabled | The current GUI cannot supply the declared input. Use a one-image 2-D model or an external task-specific workflow; do not rewrite the contract merely to pass the gate. |
| Referenced model file is missing | Correct the local path or restore the file from a trusted source. OpenMedVisionX will not download it. |
| ONNX runtime is unavailable | Activate `openmedvisionx`, install `.[onnx]`, and restart. GPU execution needs a separately compatible provider/runtime. |
| TorchScript runtime is unavailable | Install a PyTorch build compatible with your computer. |
| `.pth` or `.ckpt` is rejected | Obtain a trustworthy TorchScript/ONNX export or use a reviewed Python adapter with the original architecture. Renaming the file is not conversion. |
| Python adapter requests trust | Stop and review the code, source, hashes, environment, and licenses. Decline if any item is unclear. |
| Modality, spacing, channel, or range mismatch | Use the correct input. Do not invent metadata or change normalization just to make the run proceed. |
| Large-image preview warning | The active pixels may be a bounded preview rather than the full source. Cancel unless that exact input is valid for the task. |

For more symptoms, see [Troubleshooting · Models](TROUBLESHOOTING.md#models).

## 8. Related guides

- [Data and formats](DATA_FORMATS.md) — prepare the active image correctly.
- [User guide · Models](USER_GUIDE.md#models) — understand the complete Models workspace.
- [Bundled model guide](MODEL_BUNDLES.md) — use the three reviewed fixed workflows.
- [Learning curriculum · Local inference](TEACHING_CURRICULUM.md#6-reproducible-local-model-inference) — turn a run into a reproducible experiment.
- [Glossary](GLOSSARY.md) — look up manifest, provenance, calibration, and geometry terms.

Back to the [documentation home](INDEX.md).
