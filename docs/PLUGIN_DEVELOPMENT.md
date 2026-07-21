# Model Plugin Development

OpenMedVisionX integrates user-supplied models through a runtime-neutral
protocol. The project does not ship model implementations, checkpoints, or
weights. Compatibility means that an adapter satisfies this protocol; it does
not mean that an arbitrary `.pth`, `.ckpt`, or other checkpoint can run
without its architecture, preprocessing, and postprocessing.

## Supported import paths

The protocol recognizes three runtime kinds:

- `onnx`;
- `torchscript`;
- `python-adapter`.

ONNX and TorchScript are standardized model artifacts. A research model that
needs Python architecture code must use `python-adapter`. Every plugin
directory has exactly one `manifest.yaml`.

The manifest contract is independent from a runtime implementation: calling
`load_plugin_manifest()` validates YAML and path declarations without
importing the adapter, opening a weight, downloading a resource, installing a
dependency, or creating a model session.

## Recommended layout

```text
my-plugin/
├── manifest.yaml
├── adapter.py              # required only by a Python adapter
├── requirements.txt        # optional declaration; never auto-installed
└── README.md               # trust, license, and environment instructions

outside-the-repository/
└── user-selected-model.onnx
```

The adapter entry point and an optional requirements file must remain inside
the plugin root. Weight paths may point to another local location chosen by the
user. They are referenced in place and are never copied into the plugin or
OpenMedVisionX source tree.

URLs, URIs, UNC/network paths, automatic downloads, automatic dependency
installation, and copying weights into the project are rejected.

## Manifest schema 1.0

Start from the maintained
[example manifest](../src/dicom_viewer/inference/examples/manifest.yaml).
`manifest.yaml` (UTF-8) is the conventional filename used for directory-based
plugin discovery and examples. The desktop's explicit file picker may open a
differently named `.yaml` or `.yml` file; it is validated against exactly the
same schema and never causes neighbouring files to be scanned.

Required top-level keys:

| Key | Meaning |
| --- | --- |
| `schema_version` | Currently `"1.0"`. |
| `name`, `version`, `description` | Stable human and provenance identity. |
| `family`, `source` | Architecture family plus origin; use both to disambiguate names such as DINO. |
| `tasks`, `subtasks` | Controlled main tasks plus more specific labels. |
| `license` | Separate code, model, and weight license information. |
| `runtime` | Runtime kind, device policy, and optional Python environment declaration. |
| `entrypoint` | Local model/adapter path and, for Python, an object name. |
| `weights` | One or more local external weight references with optional size/hash. |
| `inputs` | Named tensors and all required preprocessing semantics. |
| `outputs` | Named tensor meaning, coordinates, labels, postprocessing, and uncertainty. |
| `capabilities` | Optional outputs and interaction features the plugin really supports. |

Optional top-level keys are `authors`, `references`, and `extensions`.
Unknown keys and duplicate YAML mapping keys are rejected. A manifest must have
at least one task, weight, input, and output.

### Controlled tasks

`tasks` accepts:

- `representation`;
- `classification`;
- `segmentation`;
- `detection`;
- `registration`;
- `reconstruction`;
- `restoration`;
- `generation`;
- `retrieval`;
- `vqa`;
- `report_generation`;
- `wsi_mil`;
- `tracking`;
- `anomaly_detection`;
- `multimodal`.

For an overloaded family name, record both source and task. For example, a
self-supervised DINO representation encoder, a DINO detector, and Grounding
DINO must not share an ambiguous manifest identity.

### Inputs

An input describes more than shape. For 2D image semantics, declare:

- modality and dimensionality;
- shape and tensor layout, such as `nchw` or `nhwc`;
- color space and channel order;
- alpha handling;
- dtype and numeric range;
- spacing requirements;
- scale/offset and mean/std normalization;
- ordered resize, crop, letterbox, or fit operations and interpolation;
- orientation handling, normally `apply-exif`.

OpenMedVisionX preprocessing creates a `TransformRecord` for every named 2D
input. Put those records into `InferenceRequest.transform_records`. They are
required to map masks, boxes, keypoints, probability maps, and heatmaps back to
source pixels. Discrete label masks use nearest-neighbour interpolation.

Do not silently change RGB to BGR, discard alpha, compress a 16-bit image to
8-bit, or infer medical spacing from DPI.

### Outputs

Each output declares:

- semantic type;
- dtype and shape;
- coordinate system;
- labels, where applicable;
- activation, thresholds, NMS, interpolation, and other postprocessing;
- uncertainty meaning and calibration status.

Output semantics include class scores, masks, boxes, keypoints, embeddings,
feature/attention maps, vector fields, transforms, reconstructed/restored
images, generated samples, sampling trajectories, anomaly values, tracks,
text, similarity, and uncertainty.

The typed `InferenceResult` family currently includes:

- `ClassificationResult`;
- `SegmentationResult`;
- `DetectionResult`;
- `RepresentationResult`;
- `RegistrationResult`;
- inference `ReconstructionResult`;
- `RestorationResult`;
- `GenerationResult`;
- `MultimodalResult`;
- `TrackingResult`;
- `AnomalyDetectionResult`;
- `WSIMILResult`.

Every result carries `InferenceProvenance`. Its task, model name, model
version, and runtime must agree with the manifest.

### Capabilities

Declare only implemented capabilities:

- prompts;
- intermediate features;
- attention;
- embeddings;
- multiscale outputs;
- vector fields;
- sampling trajectory;
- multimodal text;
- uncertainty.

Capability flags drive UI availability. They are not marketing metadata.

## ModelPlugin contract

A plugin implements:

```python
class ModelPlugin:
    def describe(self) -> ModelManifest: ...
    def validate(self, context=None) -> PluginValidationReport: ...
    def capabilities(self) -> CapabilitySpec: ...
    def load(self, context: PluginLoadContext) -> None: ...
    def predict(self, request: InferenceRequest) -> InferenceResult: ...
    def visualize(
        self,
        result: InferenceResult,
        context: VisualizationContext | None = None,
    ) -> Sequence[VisualizationArtifact]: ...
    def close(self) -> None: ...
```

`ManifestBackedModelPlugin` implements `describe`, `capabilities`, and
manifest file validation. Subclasses should:

1. call `authorize_load(context)` before loading;
2. load only the local files declared by the manifest;
3. reject incompatible input names, shapes, dtype, modality, and spacing;
4. return one typed result;
5. call `validate_result(result)` before returning it;
6. release runtime resources in `close()`.

`visualize` returns declarative `VisualizationArtifact` objects. A plugin
must not import Qt or mutate a widget. Rendering remains a UI responsibility.

## Validation and load sequence

A host should perform these steps in order:

1. `load_plugin_manifest(plugin_root)` to parse and validate the contract.
2. Display identity, source, tasks, runtime, and all three license categories.
3. Resolve references with `validate_manifest_references`.
4. Optionally call `validate_manifest_files(..., verify_hashes=True)`.
5. Ask for explicit consent if the runtime is `python-adapter`.
6. Start the adapter in the declared user-selected Python/Conda environment as
   a separate subprocess.
7. Build `PluginLoadContext` with
   `user_consented_python_code=True` and
   `subprocess_isolated=True` only after those facts are true.
8. Call `validate`, `load`, and then `predict` in a cancellable background
   workflow.

Manifest loading itself never imports the adapter. Validation errors should be
shown with their structured field path instead of a generic load failure.

## Python adapter warning

> [!CAUTION]
> A Python adapter executes arbitrary third-party code. A separate process and
> separate Conda environment provide fault and dependency isolation, but they
> are **not a security sandbox**. They do not make malicious code safe.

The current security contract requires explicit user consent and forbids
running a Python adapter in the GUI process. The host is still responsible for
process creation, environment selection, timeout, cancellation, resource
limits, and termination after a crash.

Never set the consent or subprocess flags merely to bypass validation. Review
the adapter source, model origin, hashes, and licenses first.

## External weights and licenses

- No weight or serialized model belongs in this repository.
- The platform does not automatically download a missing weight.
- A weight may be an absolute local path or a path relative to the plugin root.
- Adapter/requirements paths cannot escape the plugin root.
- `size_bytes` and `sha256` can detect an unexpected local file.
- Project MIT licensing covers only OpenMedVisionX source, not imported code,
  model definitions, or weights.

If a model definition or checkpoint license is missing or incompatible, do not
load or redistribute it.

## Whole-slide and temporal boundaries

The protocol can describe WSI MIL, video, tracking, and 4D inputs. Protocol
support does not imply that the base viewer has a WSI pyramid loader or tile
scheduler. Until those services exist, a WSI plugin accepts only user-provided
pre-tiles or precomputed features.

The GUI is an inference and teaching surface. A plugin must not add model
training, validation, optimizer state, or checkpoint-resume workflows.

## Testing a plugin

Tests should create minimal model bytes or mock adapters at runtime in a
temporary directory. Do not commit a model file or weight.

Cover:

- manifest round trip and duplicate/unknown key errors;
- remote reference and path traversal rejection;
- missing dependency, input mismatch, and unavailable device errors;
- explicit Python consent and subprocess requirements;
- crash, timeout, cancellation, and cleanup;
- provenance/task validation for the typed result;
- reversible overlay mapping after EXIF/resize/crop/letterbox;
- all license fields shown before load.

Run the repository policy check before submitting:

```bash
python scripts/check_repository.py
```
