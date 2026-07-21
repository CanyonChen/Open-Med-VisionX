# LLM Provider and Cloud Image Security

The OpenMedVisionX assistant explains concepts, parameters, and experimental
results. It is not a diagnosis service. A provider receives data outside the
local application trust boundary, so network use is opt-in and auditable.

## Security invariants

- Importing `dicom_viewer.llm` does not open a network connection.
- Providers use `DisabledTransport` unless the host explicitly injects an
  enabled transport.
- Configuration stores a `CredentialReference`, never a raw API key.
- Cloud image transfer begins disabled for every provider instance.
- Image input requires a model declared as vision-capable and explicit consent
  for that provider.
- The only accepted image payload is a validated, metadata-free rendered PNG.
- Original DICOM/NIfTI bytes, a complete series, DICOM metadata, and arbitrary
  image files are not valid provider inputs.
- Every answer includes provider, model ID, UTC time, and an education-only
  disclaimer.

These controls reduce accidental disclosure. They cannot determine whether
free-form text or burned-in image pixels contain patient information.

## Provider interface

The stable interface is:

```python
class LLMProvider:
    def chat(self, messages, *, preview=None) -> LLMResponse: ...
    def stream(self, messages, *, preview=None) -> Iterator[str]: ...
    def capabilities(self) -> ProviderCapabilities: ...
```

Implemented adapters:

| Adapter | Protocol |
| --- | --- |
| `OpenAIProvider` | OpenAI Responses API |
| `AnthropicProvider` | Anthropic Messages API |
| `MoonshotProvider` / `KimiProvider` | OpenAI-compatible Chat Completions |
| `GLMProvider` | OpenAI-compatible Chat Completions |
| `DeepSeekProvider` | OpenAI-compatible Chat Completions |
| `OpenAICompatibleProvider` | User-named compatible endpoint |

The user supplies a model ID. Do not encode a vendor's “latest model” in
application logic. `supports_vision` is a configuration assertion, not an
automatically discovered fact; enable it only after checking the selected
model.

## Endpoint policy

`validate_endpoint_url` accepts:

- HTTPS endpoints with a hostname;
- plain HTTP only for `localhost`, `127.0.0.1`, or `::1`, primarily for
  tests.

Endpoint URLs cannot contain a username, password, query string, or fragment.
Production adapters must not weaken this policy to support an insecure vendor
configuration.

`Transport` is injectable:

```python
class Transport:
    def send(self, request: HttpRequest) -> Mapping[str, Any]: ...
    def stream(self, request: HttpRequest) -> Iterable[Mapping[str, Any]]: ...
```

Tests use an in-memory transport or localhost mock server. CI never contacts a
real provider.

## Credential references

`CredentialReference` supports two schemes:

- `env:VARIABLE_NAME`;
- `keyring:service/username`.

Example non-secret profile:

```toml
provider_id = "openai"
model_id = "user-selected-model-id"
credential_ref = "env:OPENAI_API_KEY"
supports_vision = false
timeout = 30
```

or:

```toml
credential_ref = "keyring:openmedvisionx/teaching-account"
```

`CredentialResolver` resolves the reference immediately before request
construction and does not cache the resolved value. `CredentialReference`,
provider configuration, and `HttpRequest` representations redact credentials.
Keyring support is optional and loaded only for a `keyring:` reference.

Default environment references are:

- `OPENAI_API_KEY`;
- `ANTHROPIC_API_KEY`;
- `MOONSHOT_API_KEY`;
- `ZHIPU_API_KEY`;
- `DEEPSEEK_API_KEY`.

Do not place a real value in TOML/YAML, source, a command example, screenshot,
traceback, log, test snapshot, chat export, or issue. Environment variables
must be set outside the repository. Prefer the operating-system credential
store on shared machines.

## Redaction

Runtime redaction helpers handle structured values, text, and log records:

- `redact(value)`;
- `redact_text(value)`;
- `install_redaction(logger)`.

Install the logging filter before recording provider configuration or request
errors. Do not log request headers or full request/response bodies. Redaction is
defense in depth, not authorization to log sensitive data.

An exception exposed to the UI should name the provider and error class without
including a key, Authorization header, patient text, or response payload.

## Per-provider image authorization

Each provider instance owns an `ImageTransferAuthorization` initialized to
false. The host exposes:

- `authorize_image_transfer()`;
- `revoke_image_transfer()`;
- `capabilities().image_transfer_authorized`.

Consent is provider-local: authorizing OpenAI must not authorize Anthropic,
Kimi, GLM, DeepSeek, or a custom endpoint. The current authorization object is
in-memory for that provider instance; a newly constructed provider starts
disabled again.

The UI must:

1. show the exact provider, endpoint host, and model ID;
2. explain that pixels leave the local computer;
3. require an affirmative user action;
4. keep a visible “cloud image transfer enabled” status while authorized;
5. offer immediate revocation;
6. re-check capabilities just before every image request.

Revocation blocks future requests. It cannot recall a request already sent to a
provider.

## RenderedPreview boundary

`LLMProvider.chat` and `stream` accept image data only as
`RenderedPreview`. Passing raw bytes, an `ImageData`, a DICOM dataset, or a
path is a contract error.

`RenderedPreview.from_png(data)` validates:

- the PNG signature and chunk integrity;
- exactly one valid header and a complete data/end sequence;
- at most 8 MiB;
- at most 16,777,216 pixels;
- only `IHDR`, `PLTE`, `IDAT`, `IEND`, and `tRNS` chunks.

EXIF, text, XMP, ICC, time, and other ancillary chunks are rejected rather than
forwarded. The host must create the preview from the exact rendered slice that
the user selected and previewed.

Safe construction workflow:

```text
selected local image/volume
  -> select one visible slice or 2D image
  -> apply the current display mapping
  -> render to a new pixel buffer
  -> inspect for burned-in identifiers
  -> encode a fresh metadata-free PNG
  -> RenderedPreview.from_png(...)
  -> verify provider vision capability and consent
  -> send
```

Do not pass through the source file's PNG chunks. Re-encode the displayed pixel
buffer so the allow-list validation has a meaningful privacy boundary.

## Burned-in text and free-form prompts

Removing metadata does not remove names, dates, accession numbers, facility
labels, or other identifiers burned into pixels. The UI must warn the user and
offer a final preview. Automatic OCR is not proof that a preview is anonymous.

Text prompts can also contain identifying information. The provider interface
cannot determine whether a user's prose is PHI. Lessons should use synthetic
cases, and users should not paste clinical reports or patient context into a
cloud assistant.

## Request and response behavior

When a preview is supplied, adapters attach it to the last user message. A
provider rejects the request when:

- `preview` is not a validated `RenderedPreview`;
- `supports_vision` is false;
- provider-local authorization is false.

`LLMResponse` stores answer text, provider, model, UTC timestamp, and the
educational disclaimer. `LLMResponse.content` appends the footer.
`stream()` yields text deltas and then a mandatory footer.

The provider calls are synchronous interfaces, so UI integrations run them
outside the Qt thread using the runtime task service. `HttpRequest` carries the
task's cancellation token. `UrllibTransport` checks it around every bounded
read and uses a watcher to close an active response socket, allowing a blocked
body read to stop promptly. DNS/connect still relies on the configured finite
network timeout because Python cannot portably interrupt every resolver call.
The loopback tests cover both active-read cancellation and timeout behavior.

## Failure handling

- Missing credentials produce `CredentialResolutionError` without printing a
  key.
- An unauthorized preview produces `CloudTransferDenied`.
- Invalid model capability, endpoint, preview, or messages produce validation
  errors before transport.
- Provider/transport failures are converted to `ProviderError` without
  embedding raw payloads.
- Timeouts must be positive and explicitly configured.

Never silently retry an image request after consent has been revoked.

## Incident response

If a key is exposed:

1. revoke or rotate it immediately;
2. inspect provider audit and billing logs;
3. remove it from the working tree, Git history, CI artifacts, logs, and
   screenshots;
4. strengthen tests and scanning without preserving the leaked value.

If patient information is sent:

1. stop further transfers and revoke provider authorization;
2. follow the provider's deletion process where available;
3. notify the responsible institutional/data owner through private channels;
4. follow applicable legal and institutional incident procedures;
5. do not reproduce the data in a public issue.

See [../SECURITY.md](../SECURITY.md) for private reporting.

## Security tests

Use local mocks to cover:

- environment and keyring reference resolution;
- redaction in configuration/request/error representations;
- HTTPS and localhost-only endpoint rules;
- no network activity with `DisabledTransport`;
- provider-local grant and revoke behavior;
- rejection of raw, oversized, malformed, metadata-bearing, or high-pixel PNG;
- rejection when vision capability or consent is absent;
- provider/model/time/disclaimer footer for chat and streaming;
- timeout and cancellation behavior;
- confirmation that no real API is called in CI.
