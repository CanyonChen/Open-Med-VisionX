# OpenMedVisionX Architecture

This document describes the dependency rules and public contracts implemented
under `src/dicom_viewer`. It is normative for new maintained code. Legacy
top-level scripts are compatibility entry points and must not become a second
architecture.

## Goals

The architecture keeps five concerns independent:

1. what an image means;
2. how untrusted local bytes are decoded;
3. how an algorithm transforms validated data;
4. how optional model or cloud integrations are isolated;
5. how a PyQt UI presents state without owning core logic.

The base package must remain importable without NIfTI, ONNX, PyTorch, CUDA, a
cloud SDK, a model file, or a network connection.

## Seven layers and one orchestration seam

```text
                                      ┌───────────────────────┐
                                      │ 7. UI (PyQt)          │
                                      │ pages, state display, │
                                      │ teaching visuals      │
                                      └───────────┬───────────┘
                                                  │ use cases
                                      ┌───────────▼───────────┐
                                      │ Services              │
                                      │ orchestration seam    │
                                      └──┬────┬────┬────┬─────┘
                                         │    │    │    │
              ┌──────────────────────────┘    │    │    └─────────────────────┐
              ▼                               ▼    ▼                          ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ 2. IO               │  │ 3. Algorithms       │  │ 4. Inference    │  │ 5. LLM          │
│ probe/load/limits    │  │ CT reconstruction   │  │ model contracts │  │ providers       │
└──────────┬──────────┘  └──────────┬──────────┘  └────────┬────────┘  └────────┬────────┘
           │                         │                      │                    │
           └─────────────────────────┴──────────┬───────────┘                    │
                                                ▼                                │
                                      ┌─────────────────────┐                    │
                                      │ 1. Domain           │◄───────────────────┘
                                      │ meaning + geometry  │  only shared value contracts
                                      └─────────────────────┘

        6. Runtime is a cross-cutting lower-level layer: tasks, cancellation,
        session state, credential references, redaction, and common errors.
```

Services are deliberately an orchestration seam rather than an eighth domain
layer. `ImageService` selects a registered loader and commits a
`LoadedStudy`; `ReconstructionService` selects algorithms and submits work.
The UI calls services instead of composing decoders or algorithms itself.

## Dependency rules

| Layer | May depend on | Must not depend on |
| --- | --- | --- |
| Domain | NumPy and common validation errors | IO, algorithms, inference, LLM, services, Qt |
| IO | Domain contracts, common errors, optional decoder imported on demand | UI, model plugins, LLM providers |
| Algorithms | Validated request/value objects, NumPy/SciPy/scikit-image, common errors | File dialogs, loaders, Qt, cloud APIs |
| Inference | Domain transforms, its manifest/result contracts, common errors | Qt widgets, automatic downloads, in-process untrusted adapters |
| LLM | Provider value objects, transport contract, runtime credentials/redaction | Medical file loaders, Qt widgets, raw image files |
| Runtime | Common errors and Python concurrency/storage primitives | UI, concrete decoders, algorithms, model families |
| UI | Services and public domain/result/task contracts | Decoder internals, algorithm implementations, vendor request builders |

Additional invariants:

- a lower layer never imports a higher layer;
- optional dependencies are imported only inside the capability that needs
  them;
- importing `dicom_viewer.inference` does not import a model runtime or
  adapter;
- importing `dicom_viewer.llm` does not perform a network request;
- GUI availability is derived from `ImageData.capabilities`, not a filename
  extension;
- callbacks from worker threads must be marshalled onto the Qt GUI thread
  before changing a widget.

## Domain layer

### ImageData family

`ImageData` is an immutable top-level value object. It owns:

- a read-only numeric `array`;
- `SourceType`;
- `IntensitySemantics`;
- sanitized, non-sensitive `runtime_metadata`;
- derived `dtype`, `shape`, and `capabilities`.

Known patient-bearing metadata keys are removed before runtime metadata reaches
the UI or logs. This filter is defense in depth, not a DICOM anonymizer.

The concrete types have intentionally different semantics:

| Type | Array | Geometry | Important capabilities |
| --- | --- | --- | --- |
| `RasterImage2D` | `H x W` or `H x W x C` | top-left pixel coordinates; optional spacing only from the user | 2D display, histogram, pixel measurement, ROI, annotation, overlay |
| `ImageSequence2D` | `N x H x W` or `N x H x W x C` | independent page/frame transforms; no implied 3D geometry | raster tools plus frame navigation/playback |
| `ImageVolume` | `Z x Y x X` | validated RAS+ affine, spacing, origin, direction | physical measurement, orthogonal views, volume rendering, reconstruction |

Only `ImageVolume` with
`IntensitySemantics.HOUNSFIELD_UNIT` receives `HU_WINDOWING`. A raster can
gain `PHYSICAL_MEASUREMENT` only through
`RasterImage2D.with_user_spacing(x_mm, y_mm)`, which records
`SpacingSource.USER`. DPI is not accepted as medical spacing.

The current core `ImageVolume` contract is three-dimensional. A manifest may
describe 2.5D or 4D model input, but that does not silently turn a page
sequence into a 3D/4D medical volume.

### TransformRecord

`TransformRecord` is a validated invertible 3 x 3 homogeneous mapping in
`(x, y)` pixel coordinates. It records `original_shape`,
`output_shape`, a matrix, and auditable `TransformOperation` entries.

Implemented constructors and operations are:

- `identity(shape)`;
- `from_exif_orientation(orientation, shape)`;
- `resize(input_shape, output_shape)`;
- `crop(input_shape, left=..., top=..., width=..., height=...)`;
- `letterbox(input_shape, output_shape)`;
- `then(next_record)` for ordered composition;
- `forward(points)` and `inverse(points)`;
- `forward_boxes(boxes)` for axis-aligned boxes.

`forward` maps original pixels into canonical/preprocessed coordinates.
`inverse` maps results back to original pixels. Mask resampling is a separate
operation; discrete labels must use nearest-neighbour interpolation.

## IO layer

`ImageLoader` is the stable decoder extension point:

```python
class ImageLoader:
    name: str

    def can_load(self, source) -> bool: ...
    def probe(self, source) -> ProbeResult: ...
    def load(self, source, *, limits=None, cancel=None) -> ImageData: ...
```

`probe` performs non-destructive signature/metadata inspection. A loader
returns a `ProbeResult(accepted, format_name, confidence, details)`.
`ImageLoaderRegistry.probe` chooses the accepted loader with the greatest
confidence; registration order breaks ties. The UI does not choose a loader.

`default_loader_registry()` currently registers `DicomLoader`,
`NiftiLoader`, and `RasterImageLoader`. Their decoder dependencies are
imported locally, so absence of an optional backend is reported only when that
format is used.

`LoadLimits` centralizes maximum pixels, frames, decoded bytes, ZIP members,
member bytes, total ZIP bytes, and compression ratio. A loader accepts an
`Event`, callable, or no cancellation check and raises
`OperationCancelled` cooperatively.

DICOM ZIP handling validates members rather than trusting `extractall`.
DICOM and NIfTI become RAS+ `ImageVolume` objects; raster IO retains source
dtype and records explicit palette, alpha, and EXIF conversion history.

## Algorithm layer

All reconstruction implementations receive one shared geometry:

```python
ReconstructionRequest(
    sinogram=detector_by_projection_array,
    theta_degrees=angles,
    output_size=size,
    circle=True,
)
```

`ReconstructionAlgorithm.reconstruct(request, cancel=None, progress=None)`
returns an algorithm `ReconstructionResult` with a finite 2D `image`,
`algorithm` name, `intermediate` values, and `metadata`.

Built-in exports include `DirectFourierReconstruction`,
`BackProjection`, `FilteredBackProjection`, and `SARTReconstruction`.
`generate_sinogram` returns `SinogramResult` using the same theta and circle
semantics. Metrics are represented by `MetricReport` from
`compute_metrics`.

Algorithms report progress in `[0, 1]` and check cancellation inside long
loops. They do not normalize source and reconstruction independently before
metrics, access a file, or update a widget.

## Inference layer

The inference layer is runtime-neutral. It validates a fixed
`manifest.yaml`, preprocessing declarations, plugin requests, typed results,
licenses, paths, and capabilities without importing ONNX, PyTorch, CUDA, or
user adapter code.

The stable `ModelPlugin` contract is:

```python
describe() -> ModelManifest
validate(context=None) -> PluginValidationReport
capabilities() -> CapabilitySpec
load(context: PluginLoadContext) -> None
predict(request: InferenceRequest) -> InferenceResult
visualize(result, context=None) -> Sequence[VisualizationArtifact]
close() -> None
```

`VisualizationArtifact` is declarative; the plugin does not mutate GUI
state. `InferenceRequest.transform_records` carries the mappings needed to
return overlays to source pixels. Result classes cover classification,
segmentation, detection, representation, registration, reconstruction,
restoration, generation, multimodal output, tracking, anomaly detection, and
WSI MIL.

See [PLUGIN_DEVELOPMENT.md](PLUGIN_DEVELOPMENT.md) for the manifest schema and
execution boundary.

## LLM layer

`LLMProvider` exposes:

```python
chat(messages, *, preview=None) -> LLMResponse
stream(messages, *, preview=None) -> Iterator[str]
capabilities() -> ProviderCapabilities
```

The implemented adapters are `OpenAIProvider` (Responses API),
`AnthropicProvider`, `MoonshotProvider`/`KimiProvider`, `GLMProvider`,
`DeepSeekProvider`, and `OpenAICompatibleProvider`.

Providers use `DisabledTransport` unless the host explicitly injects an
enabled `Transport`. A provider requires a user-selected model ID and a
`CredentialReference`; raw keys are not configuration values. Image input
accepts only a validated `RenderedPreview` and additionally requires the
model's vision flag and provider-local authorization.

See [LLM_SECURITY.md](LLM_SECURITY.md) for the full transfer and credential
policy.

## Runtime layer

`TaskRunner.submit(operation, *args, **kwargs)` runs
`operation(context, *args, **kwargs)` in a bounded thread pool and returns a
`BackgroundTask[T]`.

The stable task surface is:

- `cancel() -> bool`;
- `progress() -> TaskProgress`;
- `result(timeout=None) -> T`;
- `status`, `done`, `running`, and `cancelled`;
- `add_progress_callback(...)` and `add_done_callback(...)`.

`TaskContext.raise_if_cancelled()` and `report_progress(...)` are the
worker-facing surface. Cancellation is cooperative, not forced thread
termination: a long operation must call the context regularly. A pending task
can be cancelled immediately; a running task enters `CANCELLING` until the
worker observes its token.

`AtomicSessionState` provides generation-checked replacement so a result from
an older load cannot overwrite a newer session. `CredentialResolver` resolves
`env:` and `keyring:` references on demand. Redaction helpers sanitize
structured values, text, and log records.

## Service and UI integration

The concrete extension factories terminate at the service boundary:

- `ModelInferenceService` parses the manifest, chooses the ONNX, TorchScript,
  or subprocess adapter, validates and loads external references, builds typed
  inference requests, delegates visualization, forwards cancellation, and
  records when a cancelled Python interpreter must be loaded again.
- `LLMProviderRegistry` is the sole mapping from user-visible provider names to
  vendor adapters and online/offline transports. `TeachingAssistantService`
  exposes those defaults and forwards provider-neutral chat requests with the
  background task's cancellation token.

Consequently `ui/main_window.py` imports neither concrete model runtimes nor
concrete LLM providers/transports. Arbitrary-code consent and destination-
scoped rendered-image consent remain explicit presentation decisions, while
their enforcement stays below the UI boundary.

The expected flow is:

```text
user event
  -> UI validates intent
  -> service starts BackgroundTask
  -> loader/algorithm/provider performs bounded work
  -> task publishes immutable progress/result
  -> UI marshals callback to Qt thread
  -> capability-driven widgets render the result
```

Starting a new study resets the prior session generation, measurement state,
derived volume, sinogram, reconstruction, and stale callbacks as one
transaction. A completed worker must compare its generation before committing.

## Review checklist

Before merging architectural code, verify:

- no lower layer imports Qt or a service;
- no concrete decoder or vendor request builder is imported by UI code;
- new data semantics are represented in Domain rather than a filename branch;
- optional libraries are imported lazily;
- arrays and runtime metadata do not leak patient identifiers;
- long operations expose progress and cooperative cancellation;
- transforms needed for output overlays are retained;
- plugin and cloud code preserves its explicit authorization boundary;
- contract tests cover both success and understandable failure modes.
