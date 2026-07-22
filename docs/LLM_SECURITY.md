# AI Assistant, Privacy, and Cloud Images

[简体中文](LLM_SECURITY.zh-CN.md) · [Documentation home](INDEX.md) · [User guide](USER_GUIDE.md#ai-assistant)

The **AI Assistant** can ask a provider you configure to explain concepts,
parameters, or experimental results. Provider use is optional. Local image
viewing, CT experiments, evaluation, learning modules, and bundled-model runs do
not depend on it.

> [!WARNING]
> A provider response is a teaching aid, not a diagnosis or validated image
> interpretation. Text and image pixels can contain sensitive information. Use
> only data you are authorized to send to the chosen destination.

## Decide whether you need a provider

Use the assistant when you want help explaining a visible concept or result and
you have reviewed the provider's terms, retention, location, and account
controls. Do not use it merely because the workspace is present.

Prefer the built-in **Learn** modules and local guides when:

- the question is already covered by the curriculum;
- the prompt would contain patient, institutional, or confidential context;
- you cannot verify where the provider processes or retains data;
- you do not have permission to send the exact text or pixels;
- an answer could influence a patient decision.

OpenMedVisionX starts with network access off. Importing or opening local data
does not contact an AI provider.

## What can leave your computer

There are two independent transfer paths:

| Action | What is sent | What is not sent |
| --- | --- | --- |
| Send a text question | The complete prompt and conversation message required for the request, plus provider/model request fields. | Local images unless image attachment is separately enabled and authorized. |
| Attach the active rendered plane | A newly encoded PNG of the complete active 2-D plane after display mapping, together with the prompt and request fields. | Original DICOM/NIfTI bytes, DICOM metadata, the full series/volume, source paths, viewport zoom/pan, overlays, measurements, and annotations. |

Removing metadata does **not** remove names, dates, accession numbers, facility
labels, or other identifiers burned into pixels. Free-form text can also reveal
sensitive information. OpenMedVisionX cannot prove that either payload is
anonymous; you must inspect it.

The only other network action in the desktop is opening an external HTTP(S)
link after confirmation. Local workflows do not automatically download a
dataset, model, or dependency.

## Configure a provider without storing the key here

Open **AI Assistant → Teaching chat** and expand **Provider configuration**.
Complete these fields:

1. **Provider** — select the protocol/provider account you intend to use.
2. **Model ID** — enter the provider's exact current model ID. OpenMedVisionX
   does not choose or verify a “latest” model for you.
3. **Endpoint** — review the exact destination. Remote endpoints must use
   HTTPS. Plain HTTP is accepted only for exact local loopback destinations
   such as `localhost`, `127.0.0.1`, or `::1`.
4. **Credential reference** — enter a reference, never the secret itself.
5. **Enable network** — turn this on only when the destination and prompt are
   ready for a request.

Current provider choices and default credential references are:

| Provider choice | Default protocol destination | Default credential reference |
| --- | --- | --- |
| OpenAI | `https://api.openai.com/v1/responses` | `env:OPENAI_API_KEY` |
| Anthropic | `https://api.anthropic.com/v1/messages` | `env:ANTHROPIC_API_KEY` |
| Moonshot / Kimi | `https://api.moonshot.cn/v1/chat/completions` | `env:MOONSHOT_API_KEY` |
| Zhipu GLM | `https://open.bigmodel.cn/api/paas/v4/chat/completions` | `env:ZHIPU_API_KEY` |
| DeepSeek | `https://api.deepseek.com/chat/completions` | `env:DEEPSEEK_API_KEY` |
| OpenAI-compatible | User-supplied compatible endpoint; the initial example.invalid value is a non-working placeholder | `env:OPENMEDVISIONX_API_KEY` |

An endpoint cannot contain a username, password, query string, or fragment.
Review the displayed host before every sensitive request.

### Credential references

- `env:NAME` reads a key from an environment variable set outside the
  repository, for example `env:OPENAI_API_KEY`.
- `keyring:service/username` reads from the operating-system credential store
  after installing `python -m pip install -e ".[llm]"`.
- `none` is allowed only for an intentionally unauthenticated service on an
  exact loopback destination.

Never paste a real key into the **Credential reference** field, YAML/TOML,
source code, a screenshot, log, issue, or chat transcript. The field is for the
reference string only.

The **Vision input** checkbox is your assertion that the selected model accepts
images; it is not automatic capability discovery. The current DeepSeek choice
is treated as text-only by the desktop.

## Send a text-only teaching question

Start with a question that contains no sensitive context:

1. Configure provider, exact model ID, endpoint, and credential reference.
2. Leave **Vision input** and **Attach active rendered plane** off.
3. Turn on **Enable network**.
4. Enter a focused question, such as “Why does FBP use a ramp filter?”
5. Re-read the full prompt. Remove identifiers, private paths, reports, and
   unnecessary case details.
6. Choose **Send** or press `Ctrl/Cmd+Enter`.
7. Read the response metadata and education-only notice. Verify important
   claims against the visible experiment and a trusted source.

Use **Cancel request** or `Esc` to request cancellation. Cancellation can stop
a pending read, but it cannot recall text already transmitted. Links in a
Markdown response are limited to HTTP(S) and require another confirmation
before the browser opens.

**Success check:** the answer identifies the configured provider/model and the
image-transfer status remains off.

## Share one rendered image only after final review

Image use has more gates than text use. It requires a loaded image, a model you
have verified supports vision, and three explicit choices: **Vision input**,
**Attach active rendered plane**, and **Enable network**.

Before dispatch, OpenMedVisionX creates a transfer plan and final native review
showing:

- provider, exact endpoint, destination host, and model ID;
- task and a SHA-256 fingerprint of the prompt;
- the exact PNG preview, filename, MIME type, dimensions, byte count, and
  SHA-256;
- how the preview was created and what metadata was excluded;
- residual risks, including burned-in text and provider retention.

The outgoing PNG is the **complete active 2-D plane after display mapping**.
It is not a viewport screenshot: zoom, pan, overlays, measurements, and
annotations are excluded. The review image is therefore the authoritative
payload. Inspect every corner for burned-in identifiers.

Authorization defaults to **No/Cancel** and applies to one exact
provider/endpoint/model/task/prompt/PNG plan. Changing any bound field or pixel
invalidates it. Authorization is sealed and consumed immediately before one
dispatch whether that attempt succeeds or fails; a retry needs a new review.
The validated PNG is limited to 8 MiB encoded and 16,777,216 pixels.

Choose **Yes** only if all of the following are true:

- you are authorized to send the exact pixels and prompt to that destination;
- provider/model/host are the intended values;
- the complete preview contains no prohibited visible information;
- the task genuinely needs image input;
- you accept the provider's retention and processing terms.

## Understand the Structured artifacts preview

The second tab, **Structured artifacts · API preview**, explains seven typed
result contracts: `text`, `class_scores`, `labels`, `mask_2d`, `mask_3d`,
`reconstructed_image`, and `reconstructed_volume`.

This tab is not an ordinary chat-output converter. In this release:

- only a trusted host integration can supply an already typed request and
  response;
- Teaching chat responses are **not automatically adapted into typed artifacts**;
- there is **no desktop artifact importer** and no file-picker route into the
  preview;
- an empty state is expected for a normal desktop-only session;
- **Confirm artifact** or **Reject artifact** becomes available only for a
  matching typed request/response;
- either action records a local immutable review, sends nothing, and **does not
  create a layer** or change source data.

Confirmation means only that the exact loaded artifact was reviewed locally.
It does not establish truth, clinical validity, or permission for downstream
use.

## Cancel, revoke, or change your mind

- Before sending text, turn off **Enable network** or simply do not press
  **Send**.
- Before image dispatch, choose **No/Cancel** in the final review.
- If a reviewed field changes, inspect the newly generated plan instead of
  relying on the previous decision.
- **Cancel request** requests cancellation of an active operation; it is not a
  data-recall mechanism.
- Turning off image attachment prevents a future image from being added, but
  cannot remove data already received by a provider.

Never silently retry an image request. A retry is another disclosure attempt
and needs a new exact review.

## Respond to a credential or data incident

If a credential may have been exposed:

1. revoke or rotate it immediately at the provider;
2. inspect provider audit and billing logs;
3. remove it from local files, Git history, logs, screenshots, and shared
   messages without reposting the value;
4. replace raw-key use with an environment or keyring reference.

If patient or other restricted information may have been sent:

1. stop further requests and disable network use;
2. use the provider's deletion/incident process where available;
3. notify the responsible institutional or data owner through a private
   channel;
4. follow applicable legal and institutional procedures;
5. do not reproduce the data in a public issue.

Use the [security policy](../SECURITY.md) for private vulnerability reporting.

## Pre-send checklist and next steps

Before every text request:

- [ ] The provider, endpoint host, and exact model are intentional.
- [ ] The credential field contains a reference, not a secret.
- [ ] The complete prompt is permitted and contains no unnecessary private data.
- [ ] You will verify the response rather than treat it as evidence.

Before every image request:

- [ ] All text checks above are complete.
- [ ] The vision capability is verified with the provider.
- [ ] The final PNG—not merely the viewport—has been inspected pixel by pixel.
- [ ] The one-request transfer plan matches the intended task and destination.
- [ ] You understand that a completed transfer cannot be recalled.

Next, return to the [AI Assistant user workflow](USER_GUIDE.md#ai-assistant),
review related terms in the [Glossary](GLOSSARY.md), or use [AI Assistant
troubleshooting](TROUBLESHOOTING.md#ai-assistant).

Back to the [documentation home](INDEX.md).
