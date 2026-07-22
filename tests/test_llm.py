from __future__ import annotations

import binascii
import json
import zlib
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from time import monotonic
from typing import Any

import pytest

from workbench.errors import OperationCancelled, ProviderError, ValidationError
from workbench.llm import (
    CLOUD_PREVIEW_PRIVACY_WARNING,
    AnthropicProvider,
    CloudTransferDenied,
    DeepSeekProvider,
    GLMProvider,
    KimiProvider,
    MoonshotProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    RenderedPreview,
    UrllibTransport,
)
from workbench.llm.transport import HttpRequest
from workbench.runtime import BackgroundTask, CredentialResolver, TaskRunner


class FakeTransport:
    def __init__(
        self,
        response: Mapping[str, Any] | None = None,
        events: Iterable[Mapping[str, Any]] = (),
    ) -> None:
        self.response = response or {}
        self.events = tuple(events)
        self.requests: list[HttpRequest] = []

    def send(self, request: HttpRequest) -> Mapping[str, Any]:
        self.requests.append(request)
        return self.response

    def stream(self, request: HttpRequest) -> Iterable[Mapping[str, Any]]:
        self.requests.append(request)
        return iter(self.events)


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    crc = binascii.crc32(chunk_type + payload) & 0xFFFFFFFF
    return len(payload).to_bytes(4, "big") + chunk_type + payload + crc.to_bytes(4, "big")


def _one_pixel_png(
    *,
    metadata: bool = False,
    rgba: bytes = b"\x00\x00\x00\xff",
) -> bytes:
    if len(rgba) != 4:
        raise ValueError("rgba must contain exactly four bytes")
    signature = b"\x89PNG\r\n\x1a\n"
    header = (1).to_bytes(4, "big") + (1).to_bytes(4, "big") + bytes([8, 6, 0, 0, 0])
    chunks = [_png_chunk(b"IHDR", header)]
    if metadata:
        chunks.append(_png_chunk(b"tEXt", b"PatientName\x00Alice"))
    chunks.extend(
        [
            _png_chunk(b"IDAT", zlib.compress(b"\x00" + rgba)),
            _png_chunk(b"IEND", b""),
        ]
    )
    return signature + b"".join(chunks)


def test_openai_uses_responses_api_shape_and_attributed_answer() -> None:
    transport = FakeTransport(
        response={"output": [{"content": [{"type": "output_text", "text": "Teaching answer"}]}]}
    )
    provider = OpenAIProvider(
        model_id="user-selected-model",
        transport=transport,
        credential_resolver=CredentialResolver(environment={"OPENAI_API_KEY": "openai-secret"}),
    )

    answer = provider.chat("Explain window width.")
    request = transport.requests[0]

    assert request.url == "https://api.openai.com/v1/responses"
    assert request.json_body["input"][0]["content"] == "Explain window width."
    assert request.headers["Authorization"] == "Bearer openai-secret"
    assert "openai-secret" not in repr(request)
    assert answer.text == "Teaching answer"
    assert "Provider: OpenAI" in answer.content
    assert "Model: user-selected-model" in answer.content
    assert "Time:" in answer.content
    assert "OpenMedVisionX" in answer.content
    assert "医学诊断" in answer.content


def test_default_transport_is_offline() -> None:
    provider = OpenAIProvider(
        model_id="configured-model",
        credential_resolver=CredentialResolver(environment={"OPENAI_API_KEY": "unused-secret"}),
    )
    with pytest.raises(ProviderError, match="Network transport is disabled"):
        provider.chat("Stay offline")


def test_provider_rejects_pre_cancelled_request_before_credentials_or_transport() -> None:
    transport = FakeTransport(response={"output_text": "must not run"})
    provider = OpenAIProvider(
        model_id="cancelled-model",
        transport=transport,
        credential_resolver=CredentialResolver(environment={"OPENAI_API_KEY": "unused-secret"}),
    )
    task: BackgroundTask[Any] = BackgroundTask()
    assert task.cancel()

    with pytest.raises(OperationCancelled):
        provider.chat(
            "Do not send",
            cancellation_token=task.cancellation_token,
        )

    assert transport.requests == []


def test_preview_requires_validated_png_explicit_consent_and_vision() -> None:
    preview = RenderedPreview.from_png(_one_pixel_png())
    transport = FakeTransport(response={"output_text": "Preview explanation"})
    provider = OpenAIProvider(
        model_id="vision-model-chosen-by-user",
        supports_vision=True,
        transport=transport,
        credential_resolver=CredentialResolver(environment={"OPENAI_API_KEY": "openai-secret"}),
    )

    assert not provider.capabilities().image_transfer_authorized
    with pytest.raises(CloudTransferDenied):
        provider.chat("Explain this rendered slice", preview=preview)
    with pytest.raises(ValidationError):
        provider.chat("Raw bytes are forbidden", preview=_one_pixel_png())  # type: ignore[arg-type]

    plan = provider.plan_image_transfer(
        "Explain this rendered slice",
        preview,
        task="teaching-explanation",
    )
    assert plan.endpoint_host == "api.openai.com"
    assert plan.total_bytes == len(preview.data)
    assert plan.matches_preview(preview)
    assert plan.items[0].transform
    assert plan.items[0].deidentification_actions
    assert "Original DICOM/NIfTI" in plan.items[0].deidentification_actions[0]
    assert "Not automatically assessed" in plan.items[0].burned_in_text_review
    with pytest.raises(ValidationError, match="at least one non-empty residual risk"):
        replace(plan, residual_risks=())
    provider.authorize_image_transfer(plan)
    answer = provider.chat(
        "Explain this rendered slice",
        preview=preview,
        transfer_plan=plan,
    )
    content = transport.requests[-1].json_body["input"][0]["content"]
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    assert answer.text == "Preview explanation"

    with pytest.raises(CloudTransferDenied):
        provider.chat(
            "Explain this rendered slice",
            preview=preview,
            transfer_plan=plan,
        )

    provider.revoke_image_transfer()
    with pytest.raises(CloudTransferDenied):
        provider.chat("Explain this rendered slice", preview=preview, transfer_plan=plan)
    assert "烧录" in CLOUD_PREVIEW_PRIVACY_WARNING
    assert "隐私" in CLOUD_PREVIEW_PRIVACY_WARNING


def test_changed_prompt_or_preview_invalidates_consent_without_network_dispatch() -> None:
    preview = RenderedPreview.from_png(_one_pixel_png())
    changed_preview = RenderedPreview.from_png(_one_pixel_png(rgba=b"\xff\x00\x00\xff"))
    transport = FakeTransport(response={"output_text": "must not be sent"})
    provider = OpenAIProvider(
        model_id="vision-model",
        supports_vision=True,
        transport=transport,
        credential_resolver=CredentialResolver(environment={"OPENAI_API_KEY": "secret"}),
    )
    plan = provider.plan_image_transfer(
        "Explain this preview",
        preview,
        task="teaching-explanation",
    )

    provider.authorize_image_transfer(plan)
    with pytest.raises(CloudTransferDenied, match="prompt or preview changed"):
        provider.chat("Changed prompt", preview=preview, transfer_plan=plan)
    assert not provider.capabilities().image_transfer_authorized
    assert transport.requests == []

    provider.authorize_image_transfer(plan)
    with pytest.raises(CloudTransferDenied, match="prompt or preview changed"):
        provider.chat(
            "Explain this preview",
            preview=changed_preview,
            transfer_plan=plan,
        )
    assert not provider.capabilities().image_transfer_authorized
    assert transport.requests == []


def test_changed_task_cannot_reuse_an_authorization() -> None:
    preview = RenderedPreview.from_png(_one_pixel_png())
    transport = FakeTransport(response={"output_text": "must not be sent"})
    provider = OpenAIProvider(
        model_id="vision-model",
        supports_vision=True,
        transport=transport,
        credential_resolver=CredentialResolver(environment={"OPENAI_API_KEY": "secret"}),
    )
    reviewed = provider.plan_image_transfer(
        "Inspect this preview",
        preview,
        task="teaching-explanation",
    )
    changed_task = provider.plan_image_transfer(
        "Inspect this preview",
        preview,
        task="segmentation",
    )

    provider.authorize_image_transfer(reviewed)
    with pytest.raises(CloudTransferDenied, match="different provider/model/task/payload"):
        provider.chat(
            "Inspect this preview",
            preview=preview,
            transfer_plan=changed_task,
        )

    assert not provider.capabilities().image_transfer_authorized
    assert transport.requests == []


def test_preview_rejects_png_metadata_chunks() -> None:
    with pytest.raises(ValidationError, match="forbidden metadata chunk"):
        RenderedPreview.from_png(_one_pixel_png(metadata=True))


def test_preview_decodes_pixels_and_removes_unused_palette_payload() -> None:
    signature = b"\x89PNG\r\n\x1a\n"
    header = (1).to_bytes(4, "big") + (1).to_bytes(4, "big") + bytes([8, 3, 0, 0, 0])
    paletted_png = signature + b"".join(
        [
            _png_chunk(b"IHDR", header),
            _png_chunk(b"PLTE", b"Alice!"),
            _png_chunk(b"IDAT", zlib.compress(b"\x00\x00")),
            _png_chunk(b"IEND", b""),
        ]
    )

    preview = RenderedPreview.from_png(paletted_png)

    assert b"Alice" not in preview.data
    assert b"PLTE" not in preview.data


def test_preview_rejects_nondecodable_or_invalid_filtered_pixel_data() -> None:
    signature = b"\x89PNG\r\n\x1a\n"
    header = (1).to_bytes(4, "big") + (1).to_bytes(4, "big") + bytes([8, 6, 0, 0, 0])

    def preview_with(payload: bytes) -> bytes:
        return signature + b"".join(
            [
                _png_chunk(b"IHDR", header),
                _png_chunk(b"IDAT", payload),
                _png_chunk(b"IEND", b""),
            ]
        )

    with pytest.raises(ValidationError, match="pixel data is invalid"):
        RenderedPreview.from_png(preview_with(b"raw DICOM payload"))
    with pytest.raises(ValidationError, match="invalid row filter"):
        RenderedPreview.from_png(preview_with(zlib.compress(b"\x05\x00\x00\x00\xff")))


def test_revocation_during_request_build_prevents_preview_transfer() -> None:
    class RevokingResolver:
        provider = None

        def resolve(self, _reference) -> str:
            assert self.provider is not None
            revoker = Thread(target=self.provider.revoke_image_transfer)
            revoker.start()
            revoker.join(timeout=1.0)
            assert not revoker.is_alive(), "revocation must not block behind request construction"
            return "openai-secret"

    resolver = RevokingResolver()
    transport = FakeTransport(response={"output_text": "must not be sent"})
    provider = OpenAIProvider(
        model_id="vision-model",
        supports_vision=True,
        transport=transport,
        credential_resolver=resolver,  # type: ignore[arg-type]
    )
    resolver.provider = provider
    preview = RenderedPreview.from_png(_one_pixel_png())
    plan = provider.plan_image_transfer(
        "Explain this preview",
        preview,
        task="teaching-explanation",
    )
    provider.authorize_image_transfer(plan)

    with pytest.raises(CloudTransferDenied, match="revoked before network dispatch"):
        provider.chat("Explain this preview", preview=preview, transfer_plan=plan)

    assert transport.requests == []


def test_anthropic_messages_request_and_response() -> None:
    transport = FakeTransport(response={"content": [{"type": "text", "text": "Claude answer"}]})
    provider = AnthropicProvider(
        model_id="user-claude-model",
        transport=transport,
        credential_resolver=CredentialResolver(
            environment={"ANTHROPIC_API_KEY": "anthropic-secret"}
        ),
    )

    answer = provider.chat(
        [
            {"role": "system", "content": "Teach, do not diagnose."},
            {"role": "user", "content": "What is a sinogram?"},
        ]
    )
    request = transport.requests[0]

    assert request.url == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == "anthropic-secret"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert request.json_body["system"] == "Teach, do not diagnose."
    assert request.json_body["messages"][0]["role"] == "user"
    assert answer.text == "Claude answer"


def test_stream_yields_deltas_then_mandatory_footer() -> None:
    transport = FakeTransport(
        events=(
            {"type": "response.output_text.delta", "delta": "A"},
            {"type": "response.output_text.delta", "delta": "B"},
            {"type": "response.completed"},
        )
    )
    provider = OpenAIProvider(
        model_id="stream-model",
        transport=transport,
        credential_resolver=CredentialResolver(environment={"OPENAI_API_KEY": "openai-secret"}),
    )

    chunks = list(provider.stream("Stream a teaching answer"))
    assert chunks[:2] == ["A", "B"]
    assert "Provider: OpenAI" in chunks[-1]
    assert "Model: stream-model" in chunks[-1]
    assert "OpenMedVisionX" in chunks[-1]
    assert transport.requests[0].json_body["stream"] is True


def test_empty_and_failed_streams_do_not_emit_success_footer() -> None:
    resolver = CredentialResolver(environment={"OPENAI_API_KEY": "openai-secret"})
    empty_provider = OpenAIProvider(
        model_id="stream-model",
        transport=FakeTransport(events=()),
        credential_resolver=resolver,
    )
    failed_provider = OpenAIProvider(
        model_id="stream-model",
        transport=FakeTransport(events=({"type": "response.failed"},)),
        credential_resolver=resolver,
    )

    with pytest.raises(ProviderError, match="no streamed text"):
        list(empty_provider.stream("Explain"))
    with pytest.raises(ProviderError, match="stream failed"):
        list(failed_provider.stream("Explain"))


@pytest.mark.parametrize(
    ("provider", "expected_url"),
    [
        (
            MoonshotProvider(
                model_id="kimi-user-model",
                transport=FakeTransport(),
                credential_resolver=CredentialResolver(environment={"MOONSHOT_API_KEY": "secret"}),
            ),
            "https://api.moonshot.cn/v1/chat/completions",
        ),
        (
            GLMProvider(
                model_id="glm-user-model",
                transport=FakeTransport(),
                credential_resolver=CredentialResolver(environment={"ZHIPU_API_KEY": "secret"}),
            ),
            "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        ),
        (
            DeepSeekProvider(
                model_id="deepseek-user-model",
                transport=FakeTransport(),
                credential_resolver=CredentialResolver(environment={"DEEPSEEK_API_KEY": "secret"}),
            ),
            "https://api.deepseek.com/chat/completions",
        ),
    ],
)
def test_named_openai_compatible_provider_endpoints(provider, expected_url: str) -> None:
    assert provider.endpoint == expected_url


@pytest.mark.parametrize(
    ("model_id", "expected_mode"),
    [
        ("kimi-k3", {"reasoning_effort": "low"}),
        ("kimi-k2.6", {"thinking": {"type": "disabled"}}),
    ],
)
def test_kimi_interactive_models_use_responsive_reasoning_modes(
    model_id: str,
    expected_mode: Mapping[str, Any],
) -> None:
    transport = FakeTransport(response={"choices": [{"message": {"content": "Answer"}}]})
    provider = KimiProvider(
        model_id=model_id,
        transport=transport,
        credential_resolver=CredentialResolver(environment={"MOONSHOT_API_KEY": "secret"}),
    )

    provider.chat("Explain this concept.")

    for key, value in expected_mode.items():
        assert transport.requests[0].json_body[key] == value


def test_deepseek_v4_request_is_explicitly_text_only_and_non_thinking() -> None:
    transport = FakeTransport(response={"choices": [{"message": {"content": "Answer"}}]})
    provider = DeepSeekProvider(
        model_id="deepseek-v4-flash",
        transport=transport,
        credential_resolver=CredentialResolver(environment={"DEEPSEEK_API_KEY": "secret"}),
    )

    answer = provider.chat("Explain this concept without an image.")

    assert answer.text == "Answer"
    assert transport.requests[0].json_body == {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "Explain this concept without an image."}],
        "stream": False,
        "thinking": {"type": "disabled"},
    }


def test_deepseek_rejects_unsupported_vision_configuration_locally() -> None:
    with pytest.raises(ValidationError, match="accepts text only"):
        DeepSeekProvider(model_id="deepseek-v4-flash", supports_vision=True)


def test_generic_openai_compatible_provider_uses_user_endpoint_and_model() -> None:
    transport = FakeTransport(response={"choices": [{"message": {"content": "Local answer"}}]})
    provider = OpenAICompatibleProvider(
        provider_name="Research Gateway",
        model_id="lab-model-v2",
        endpoint="http://localhost:8080/v1/chat/completions",
        credential_ref="env:LAB_API_KEY",
        transport=transport,
        credential_resolver=CredentialResolver(environment={"LAB_API_KEY": "local-secret"}),
    )

    answer = provider.chat("Explain filtered back projection.")
    request = transport.requests[0]
    assert request.json_body["model"] == "lab-model-v2"
    assert request.url == "http://localhost:8080/v1/chat/completions"
    assert answer.text == "Local answer"


def test_loopback_openai_compatible_provider_can_explicitly_use_no_credential() -> None:
    transport = FakeTransport(response={"choices": [{"message": {"content": "Local"}}]})
    provider = OpenAICompatibleProvider(
        provider_name="Local runtime",
        model_id="local-model",
        endpoint="http://127.0.0.1:8080/v1/chat/completions",
        credential_ref="none",
        transport=transport,
    )

    assert provider.chat("Explain locally.").text == "Local"
    assert "Authorization" not in transport.requests[0].headers

    with pytest.raises(ValidationError, match="loopback"):
        OpenAICompatibleProvider(
            provider_name="Remote runtime",
            model_id="remote-model",
            endpoint="https://example.com/v1/chat/completions",
            credential_ref="none",
        )


def test_non_loopback_plain_http_endpoint_is_rejected() -> None:
    with pytest.raises(ValidationError, match="HTTPS"):
        OpenAICompatibleProvider(
            provider_name="Unsafe",
            model_id="model",
            endpoint="http://example.com/v1/chat/completions",
            credential_ref="env:KEY",
        )


def test_endpoint_query_credentials_and_invalid_chat_roles_are_rejected() -> None:
    with pytest.raises(ValidationError, match="query strings"):
        OpenAICompatibleProvider(
            provider_name="Unsafe query",
            model_id="model",
            endpoint="https://example.com/v1/chat/completions?api_key=secret",
            credential_ref="env:KEY",
        )

    provider = OpenAIProvider(
        model_id="configured-model",
        credential_resolver=CredentialResolver(environment={"OPENAI_API_KEY": "unused-secret"}),
    )
    with pytest.raises(ValidationError, match="Unsupported chat role"):
        provider.chat([{"role": "doctor", "content": "Diagnose this"}])


def test_explicit_transport_works_against_loopback_json_and_sse() -> None:
    seen: list[Mapping[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            seen.append(json.loads(self.rfile.read(length)))
            if self.path == "/error":
                payload = b'{"error":"sensitive provider body"}'
                content_type = "application/json"
                self.send_response(401)
            elif self.path == "/events":
                payload = (
                    b'data: {"type":"response.output_text.delta","delta":"ok"}\n\ndata: [DONE]\n\n'
                )
                content_type = "text/event-stream"
                self.send_response(200)
            else:
                payload = b'{"output_text":"ok"}'
                content_type = "application/json"
                self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *args: Any) -> None:
            del args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    transport = UrllibTransport()
    try:
        response = transport.send(
            HttpRequest("POST", base_url + "/json", json_body={"model": "mock"})
        )
        events = list(
            transport.stream(
                HttpRequest(
                    "POST",
                    base_url + "/events",
                    json_body={"stream": True},
                )
            )
        )
        with pytest.raises(ProviderError, match="HTTP error 401") as captured:
            transport.send(
                HttpRequest("POST", base_url + "/error", json_body={"prompt": "private"})
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)

    assert response == {"output_text": "ok"}
    assert events == [{"type": "response.output_text.delta", "delta": "ok"}]
    assert "sensitive provider body" not in str(captured.value)
    assert seen == [{"model": "mock"}, {"stream": True}, {"prompt": "private"}]


def test_blocked_provider_body_read_is_cancelled_promptly() -> None:
    headers_sent = Event()
    release_handler = Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(1024 * 1024))
            self.end_headers()
            self.wfile.flush()
            headers_sent.set()
            release_handler.wait(5)
            with suppress(OSError):
                self.wfile.write(b"{}")

        def log_message(self, _format: str, *args: Any) -> None:
            del args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    provider = OpenAIProvider(
        model_id="cancel-test",
        endpoint=f"http://127.0.0.1:{server.server_port}/blocked",
        transport=UrllibTransport(),
        timeout=5,
        credential_resolver=CredentialResolver(environment={"OPENAI_API_KEY": "mock"}),
    )
    runner = TaskRunner(max_workers=1)
    try:
        task = runner.submit(
            lambda context: provider.chat(
                "Cancel this request",
                cancellation_token=context.token,
            )
        )
        assert headers_sent.wait(2), "mock provider did not begin the blocked response"
        started = monotonic()
        assert task.cancel()
        with pytest.raises(OperationCancelled):
            task.result(2)
        assert monotonic() - started < 1.0
    finally:
        release_handler.set()
        runner.shutdown(wait=True, cancel_pending=True)
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_provider_body_read_honours_network_timeout() -> None:
    headers_sent = Event()
    release_handler = Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(1024 * 1024))
            self.end_headers()
            self.wfile.flush()
            headers_sent.set()
            release_handler.wait(5)

        def log_message(self, _format: str, *args: Any) -> None:
            del args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = HttpRequest(
            "POST",
            f"http://127.0.0.1:{server.server_port}/timeout",
            json_body={"model": "mock"},
            timeout=0.1,
        )
        started = monotonic()
        with pytest.raises(ProviderError, match="failed or timed out"):
            UrllibTransport().send(request)
        assert headers_sent.is_set()
        assert monotonic() - started < 1.0
    finally:
        release_handler.set()
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_every_provider_adapter_works_with_a_loopback_mock_service() -> None:
    seen: list[tuple[str, Mapping[str, Any]]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            seen.append((self.path, body))
            if self.path == "/openai":
                response = {"output_text": "OpenAI mock"}
            elif self.path == "/anthropic":
                response = {"content": [{"type": "text", "text": "Anthropic mock"}]}
            else:
                response = {"choices": [{"message": {"content": "Compatible mock"}}]}
            payload = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *args: Any) -> None:
            del args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    resolver = CredentialResolver(
        environment={
            "OPENAI_API_KEY": "mock",
            "ANTHROPIC_API_KEY": "mock",
            "MOONSHOT_API_KEY": "mock",
            "ZHIPU_API_KEY": "mock",
            "DEEPSEEK_API_KEY": "mock",
            "LAB_API_KEY": "mock",
        }
    )
    common = {"transport": UrllibTransport(), "credential_resolver": resolver}
    providers = [
        OpenAIProvider(
            model_id="openai-user-model",
            endpoint=base_url + "/openai",
            **common,
        ),
        AnthropicProvider(
            model_id="anthropic-user-model",
            endpoint=base_url + "/anthropic",
            **common,
        ),
        MoonshotProvider(
            model_id="kimi-user-model",
            endpoint=base_url + "/moonshot",
            **common,
        ),
        GLMProvider(
            model_id="glm-user-model",
            endpoint=base_url + "/glm",
            **common,
        ),
        DeepSeekProvider(
            model_id="deepseek-user-model",
            endpoint=base_url + "/deepseek",
            **common,
        ),
        OpenAICompatibleProvider(
            provider_name="Lab Gateway",
            model_id="lab-user-model",
            endpoint=base_url + "/generic",
            credential_ref="env:LAB_API_KEY",
            **common,
        ),
    ]
    try:
        answers = [provider.chat("Explain a reconstruction concept") for provider in providers]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)

    assert [answer.text for answer in answers] == [
        "OpenAI mock",
        "Anthropic mock",
        "Compatible mock",
        "Compatible mock",
        "Compatible mock",
        "Compatible mock",
    ]
    assert [path for path, _ in seen] == [
        "/openai",
        "/anthropic",
        "/moonshot",
        "/glm",
        "/deepseek",
        "/generic",
    ]
    assert [body["model"] for _, body in seen] == [provider.model_id for provider in providers]
