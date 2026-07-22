"""OpenAI, Anthropic, and OpenAI-compatible teaching-assistant adapters."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from ipaddress import ip_address
from itertools import chain
from typing import Any
from urllib.parse import urlsplit

from ..errors import OperationCancelled, ProviderError, ValidationError
from ..runtime.credentials import CredentialReference, CredentialResolver
from ..runtime.tasks import CancellationToken
from .transport import DisabledTransport, HttpRequest, Transport, validate_endpoint_url
from .types import (
    EDUCATIONAL_DISCLAIMER,
    ChatMessage,
    ChatRole,
    CloudTransferDenied,
    ImageTransferAuthorization,
    LLMResponse,
    ProviderCapabilities,
    RenderedPreview,
    TransferPlan,
    normalize_messages,
)


@dataclass(frozen=True, slots=True, repr=False)
class _ProviderConfig:
    provider_id: str
    display_name: str
    model_id: str
    endpoint: str
    credential_ref: CredentialReference
    supports_vision: bool
    timeout: float

    def __post_init__(self) -> None:
        if not isinstance(self.provider_id, str) or not isinstance(self.display_name, str):
            raise ValidationError("Provider identifiers must be text.")
        if not self.provider_id.strip() or not self.display_name.strip():
            raise ValidationError("Provider identifiers cannot be empty.")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValidationError("A user-supplied model ID is required.")
        validate_endpoint_url(self.endpoint)
        if self.credential_ref.scheme == "none":
            host = (urlsplit(self.endpoint).hostname or "").casefold()
            loopback = host == "localhost" or host.endswith(".localhost")
            if not loopback:
                try:
                    loopback = ip_address(host).is_loopback
                except ValueError:
                    loopback = False
            if not loopback:
                raise ValidationError(
                    "credential_ref 'none' is allowed only for an explicit loopback endpoint."
                )
        try:
            timeout = float(self.timeout)
        except (TypeError, ValueError):
            raise ValidationError("Provider timeout must be a finite positive number.") from None
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValidationError("Provider timeout must be positive.")
        object.__setattr__(self, "timeout", timeout)

    def __repr__(self) -> str:
        return (
            f"ProviderConfig(provider_id={self.provider_id!r}, model_id={self.model_id!r}, "
            f"endpoint={self.endpoint!r}, credential_ref=<configured>, "
            f"supports_vision={self.supports_vision!r}, timeout={self.timeout!r})"
        )


class LLMProvider(ABC):
    """Stable provider interface used by UI and teaching services."""

    @abstractmethod
    def chat(
        self,
        messages: str | Sequence[ChatMessage | Mapping[str, Any]],
        *,
        preview: RenderedPreview | None = None,
        transfer_plan: TransferPlan | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> LLMResponse:
        """Return one complete answer with provenance and disclaimer."""

    @abstractmethod
    def stream(
        self,
        messages: str | Sequence[ChatMessage | Mapping[str, Any]],
        *,
        preview: RenderedPreview | None = None,
        transfer_plan: TransferPlan | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> Iterator[str]:
        """Yield text deltas followed by a mandatory provenance footer."""

    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Describe provider support and current image-transfer authorization."""

    @abstractmethod
    def plan_image_transfer(
        self,
        messages: str | Sequence[ChatMessage | Mapping[str, Any]],
        preview: RenderedPreview,
        *,
        task: str,
    ) -> TransferPlan:
        """Build the exact transfer plan that the presentation layer must show."""


class _BaseProvider(LLMProvider):
    def __init__(
        self,
        *,
        provider_id: str,
        display_name: str,
        model_id: str,
        endpoint: str,
        credential_ref: str | CredentialReference,
        supports_vision: bool,
        timeout: float,
        transport: Transport | None,
        credential_resolver: CredentialResolver | None,
    ) -> None:
        self._config = _ProviderConfig(
            provider_id=provider_id,
            display_name=display_name,
            model_id=model_id,
            endpoint=endpoint,
            credential_ref=CredentialReference.parse(credential_ref),
            supports_vision=bool(supports_vision),
            timeout=timeout,
        )
        self._transport: Transport = transport if transport is not None else DisabledTransport()
        self._credentials = (
            credential_resolver if credential_resolver is not None else CredentialResolver()
        )
        self._image_authorization = ImageTransferAuthorization()

    @property
    def model_id(self) -> str:
        return self._config.model_id

    @property
    def provider_id(self) -> str:
        return self._config.provider_id

    @property
    def endpoint(self) -> str:
        return self._config.endpoint

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider=self._config.display_name,
            vision=self._config.supports_vision,
            image_transfer_authorized=self._image_authorization.authorized,
        )

    def authorize_image_transfer(self, plan: TransferPlan) -> int:
        """Grant one-shot consent for one exact provider/model/task/payload plan."""

        self._validate_plan_provider(plan)
        return self._image_authorization.grant(plan.plan_id)

    def revoke_image_transfer(self) -> int:
        """Revoke consent for all future rendered-preview requests."""

        return self._image_authorization.revoke()

    def plan_image_transfer(
        self,
        messages: str | Sequence[ChatMessage | Mapping[str, Any]],
        preview: RenderedPreview,
        *,
        task: str = "teaching-explanation",
    ) -> TransferPlan:
        if not isinstance(preview, RenderedPreview):
            raise ValidationError("Only a validated rendered PNG can enter a transfer plan.")
        if not self._config.supports_vision:
            raise ValidationError(
                f"Configured model {self._config.model_id!r} has no declared vision capability."
            )
        normalized = normalize_messages(messages)
        return TransferPlan.for_preview(
            provider_id=self._config.provider_id,
            provider_name=self._config.display_name,
            endpoint=self._config.endpoint,
            model_id=self._config.model_id,
            task=task,
            messages=normalized,
            preview=preview,
        )

    def chat(
        self,
        messages: str | Sequence[ChatMessage | Mapping[str, Any]],
        *,
        preview: RenderedPreview | None = None,
        transfer_plan: TransferPlan | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> LLMResponse:
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        normalized = self._validate_input(messages, preview, transfer_plan)
        try:
            if preview is None:
                request = self._build_request(normalized, preview=None, stream=False)
                if cancellation_token is not None:
                    request = replace(request, cancellation_token=cancellation_token)
                    cancellation_token.raise_if_cancelled()
                payload = self._transport.send(request)
            else:
                assert transfer_plan is not None
                with self._image_authorization.transfer_guard(
                    transfer_plan.plan_id
                ) as authorization_generation:
                    request = self._build_request(normalized, preview=preview, stream=False)
                    if cancellation_token is not None:
                        request = replace(request, cancellation_token=cancellation_token)
                        cancellation_token.raise_if_cancelled()
                    self._image_authorization.consume_for_dispatch(
                        transfer_plan.plan_id,
                        authorization_generation,
                    )
                    payload = self._transport.send(request)
            if cancellation_token is not None:
                cancellation_token.raise_if_cancelled()
            text = self._parse_response(payload)
        except (OperationCancelled, ProviderError):
            raise
        except Exception as error:
            raise ProviderError(
                f"{self._config.display_name} request failed ({type(error).__name__})."
            ) from None
        return LLMResponse(
            text=text,
            provider=self._config.display_name,
            model=self._config.model_id,
        )

    def stream(
        self,
        messages: str | Sequence[ChatMessage | Mapping[str, Any]],
        *,
        preview: RenderedPreview | None = None,
        transfer_plan: TransferPlan | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> Iterator[str]:
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        normalized = self._validate_input(messages, preview, transfer_plan)
        try:
            first_events: tuple[Mapping[str, Any], ...] = ()
            if preview is None:
                request = self._build_request(normalized, preview=None, stream=True)
                if cancellation_token is not None:
                    request = replace(request, cancellation_token=cancellation_token)
                    cancellation_token.raise_if_cancelled()
                events = iter(self._transport.stream(request))
            else:
                assert transfer_plan is not None
                with self._image_authorization.transfer_guard(
                    transfer_plan.plan_id
                ) as authorization_generation:
                    request = self._build_request(normalized, preview=preview, stream=True)
                    if cancellation_token is not None:
                        request = replace(request, cancellation_token=cancellation_token)
                        cancellation_token.raise_if_cancelled()
                    self._image_authorization.consume_for_dispatch(
                        transfer_plan.plan_id,
                        authorization_generation,
                    )
                    events = iter(self._transport.stream(request))
                    with suppress(StopIteration):
                        first_events = (next(events),)
            has_text = False
            for event in chain(first_events, events):
                if cancellation_token is not None:
                    cancellation_token.raise_if_cancelled()
                delta = self._parse_stream_event(event)
                if delta:
                    has_text = has_text or bool(delta.strip())
                    yield delta
            if not has_text:
                raise ProviderError(f"{self._config.display_name} returned no streamed text.")
            if cancellation_token is not None:
                cancellation_token.raise_if_cancelled()
        except (OperationCancelled, ProviderError):
            raise
        except Exception as error:
            raise ProviderError(
                f"{self._config.display_name} stream failed ({type(error).__name__})."
            ) from None
        yield "\n\n" + LLMResponse.format_footer(
            provider=self._config.display_name,
            model=self._config.model_id,
            timestamp=datetime.now(timezone.utc),
            disclaimer=EDUCATIONAL_DISCLAIMER,
        )

    def _validate_input(
        self,
        messages: str | Sequence[ChatMessage | Mapping[str, Any]],
        preview: RenderedPreview | None,
        transfer_plan: TransferPlan | None,
    ) -> tuple[ChatMessage, ...]:
        normalized = normalize_messages(messages)
        if preview is None and transfer_plan is not None:
            self._image_authorization.revoke()
            raise ValidationError("A transfer plan cannot be supplied without its preview payload.")
        if preview is not None:
            if not isinstance(preview, RenderedPreview):
                raise ValidationError(
                    "Cloud image input must be a validated RenderedPreview, "
                    "never raw DICOM/NIfTI bytes."
                )
            if not self._config.supports_vision:
                raise ValidationError(
                    f"Configured model {self._config.model_id!r} has no declared vision capability."
                )
            if transfer_plan is None:
                self._image_authorization.revoke()
                raise CloudTransferDenied(
                    "Cloud image transfer requires an exact reviewed transfer plan."
                )
            self._validate_plan_provider(transfer_plan)
            expected = TransferPlan.for_preview(
                provider_id=self._config.provider_id,
                provider_name=self._config.display_name,
                endpoint=self._config.endpoint,
                model_id=self._config.model_id,
                task=transfer_plan.task,
                messages=normalized,
                preview=preview,
            )
            if transfer_plan.plan_id != expected.plan_id:
                self._image_authorization.revoke()
                raise CloudTransferDenied(
                    "The prompt or preview changed after consent; review a new transfer plan."
                )
            if not self._image_authorization.authorized:
                raise CloudTransferDenied("The reviewed transfer plan has not been authorized.")
        return normalized

    def _validate_plan_provider(self, plan: TransferPlan) -> None:
        if not isinstance(plan, TransferPlan):
            raise ValidationError("Image transfer consent requires a validated TransferPlan.")
        expected = (
            self._config.provider_id,
            self._config.endpoint,
            self._config.model_id,
        )
        observed = (plan.provider_id, plan.endpoint, plan.model_id)
        if observed != expected:
            self._image_authorization.revoke()
            raise CloudTransferDenied(
                "Transfer plan does not match the configured provider, endpoint, and model."
            )

    def _headers(self, *, stream: bool) -> dict[str, str]:
        api_key = self._credentials.resolve(self._config.credential_ref)
        headers = self._authentication_headers(api_key) if api_key else {}
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "text/event-stream" if stream else "application/json"
        return headers

    def _request(
        self,
        body: Mapping[str, Any],
        *,
        stream: bool,
    ) -> HttpRequest:
        return HttpRequest(
            method="POST",
            url=self._config.endpoint,
            headers=self._headers(stream=stream),
            json_body=body,
            timeout=self._config.timeout,
        )

    @abstractmethod
    def _authentication_headers(self, api_key: str) -> dict[str, str]: ...

    @abstractmethod
    def _build_request(
        self,
        messages: tuple[ChatMessage, ...],
        *,
        preview: RenderedPreview | None,
        stream: bool,
    ) -> HttpRequest: ...

    @abstractmethod
    def _parse_response(self, payload: Mapping[str, Any]) -> str: ...

    @abstractmethod
    def _parse_stream_event(self, event: Mapping[str, Any]) -> str: ...


class OpenAIProvider(_BaseProvider):
    """OpenAI Responses API adapter; no Chat Completions fallback is used."""

    def __init__(
        self,
        *,
        model_id: str,
        credential_ref: str | CredentialReference = "env:OPENAI_API_KEY",
        endpoint: str = "https://api.openai.com/v1/responses",
        supports_vision: bool = False,
        max_output_tokens: int | None = None,
        timeout: float = 30.0,
        transport: Transport | None = None,
        credential_resolver: CredentialResolver | None = None,
    ) -> None:
        if max_output_tokens is not None and max_output_tokens <= 0:
            raise ValidationError("max_output_tokens must be positive.")
        self._max_output_tokens = max_output_tokens
        super().__init__(
            provider_id="openai",
            display_name="OpenAI",
            model_id=model_id,
            endpoint=endpoint,
            credential_ref=credential_ref,
            supports_vision=supports_vision,
            timeout=timeout,
            transport=transport,
            credential_resolver=credential_resolver,
        )

    def _authentication_headers(self, api_key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}"}

    def _build_request(
        self,
        messages: tuple[ChatMessage, ...],
        *,
        preview: RenderedPreview | None,
        stream: bool,
    ) -> HttpRequest:
        preview_index = _last_user_index(messages) if preview else -1
        input_items: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if index == preview_index and preview is not None:
                content: Any = [
                    {"type": "input_text", "text": message.content},
                    {
                        "type": "input_image",
                        "image_url": preview.to_data_url(),
                        "detail": "auto",
                    },
                ]
            else:
                content = message.content
            input_items.append({"role": message.role.value, "content": content})
        body: dict[str, Any] = {
            "model": self._config.model_id,
            "input": input_items,
            "stream": stream,
        }
        if self._max_output_tokens is not None:
            body["max_output_tokens"] = self._max_output_tokens
        return self._request(body, stream=stream)

    def _parse_response(self, payload: Mapping[str, Any]) -> str:
        direct = payload.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct
        fragments: list[str] = []
        output = payload.get("output", ())
        if isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
            for item in output:
                if not isinstance(item, Mapping):
                    continue
                content = item.get("content", ())
                if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
                    continue
                for part in content:
                    if isinstance(part, Mapping) and part.get("type") == "output_text":
                        text = part.get("text")
                        if isinstance(text, str):
                            fragments.append(text)
        text = "".join(fragments)
        if not text.strip():
            raise ProviderError("OpenAI Responses API returned no output text.")
        return text

    def _parse_stream_event(self, event: Mapping[str, Any]) -> str:
        event_type = event.get("type")
        if event_type in {"error", "response.failed", "response.incomplete"}:
            raise ProviderError("OpenAI Responses API stream failed.")
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            return delta if isinstance(delta, str) else ""
        return ""


class AnthropicProvider(_BaseProvider):
    """Anthropic Messages API adapter."""

    def __init__(
        self,
        *,
        model_id: str,
        credential_ref: str | CredentialReference = "env:ANTHROPIC_API_KEY",
        endpoint: str = "https://api.anthropic.com/v1/messages",
        supports_vision: bool = False,
        max_tokens: int = 1024,
        timeout: float = 30.0,
        transport: Transport | None = None,
        credential_resolver: CredentialResolver | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValidationError("max_tokens must be positive.")
        self._max_tokens = int(max_tokens)
        super().__init__(
            provider_id="anthropic",
            display_name="Anthropic Claude",
            model_id=model_id,
            endpoint=endpoint,
            credential_ref=credential_ref,
            supports_vision=supports_vision,
            timeout=timeout,
            transport=transport,
            credential_resolver=credential_resolver,
        )

    def _authentication_headers(self, api_key: str) -> dict[str, str]:
        return {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }

    def _build_request(
        self,
        messages: tuple[ChatMessage, ...],
        *,
        preview: RenderedPreview | None,
        stream: bool,
    ) -> HttpRequest:
        system_parts = [
            message.content
            for message in messages
            if message.role in {ChatRole.SYSTEM, ChatRole.DEVELOPER}
        ]
        conversational = [
            message for message in messages if message.role in {ChatRole.USER, ChatRole.ASSISTANT}
        ]
        preview_index = _last_user_index(conversational) if preview else -1
        body_messages: list[dict[str, Any]] = []
        for index, message in enumerate(conversational):
            if index == preview_index and preview is not None:
                content: Any = [
                    {"type": "text", "text": message.content},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": preview.mime_type,
                            "data": preview.base64_data(),
                        },
                    },
                ]
            else:
                content = message.content
            body_messages.append({"role": message.role.value, "content": content})
        body: dict[str, Any] = {
            "model": self._config.model_id,
            "max_tokens": self._max_tokens,
            "messages": body_messages,
            "stream": stream,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        return self._request(body, stream=stream)

    def _parse_response(self, payload: Mapping[str, Any]) -> str:
        fragments: list[str] = []
        content = payload.get("content", ())
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
            for part in content:
                if isinstance(part, Mapping) and part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        fragments.append(text)
        text = "".join(fragments)
        if not text.strip():
            raise ProviderError("Anthropic Messages API returned no text content.")
        return text

    def _parse_stream_event(self, event: Mapping[str, Any]) -> str:
        event_type = event.get("type")
        if event_type == "error":
            raise ProviderError("Anthropic Messages API stream failed.")
        if event_type != "content_block_delta":
            return ""
        delta = event.get("delta")
        if isinstance(delta, Mapping) and delta.get("type") == "text_delta":
            text = delta.get("text")
            return text if isinstance(text, str) else ""
        return ""


class OpenAICompatibleProvider(_BaseProvider):
    """Configurable OpenAI-compatible Chat Completions adapter."""

    def __init__(
        self,
        *,
        provider_name: str,
        model_id: str,
        endpoint: str,
        credential_ref: str | CredentialReference,
        supports_vision: bool = False,
        max_tokens: int | None = None,
        timeout: float = 30.0,
        transport: Transport | None = None,
        credential_resolver: CredentialResolver | None = None,
    ) -> None:
        if max_tokens is not None and max_tokens <= 0:
            raise ValidationError("max_tokens must be positive.")
        self._max_tokens = max_tokens
        display_name = provider_name.strip()
        if not display_name:
            raise ValidationError("provider_name cannot be empty.")
        provider_id = "".join(
            character.lower() if character.isalnum() else "_" for character in display_name
        ).strip("_")
        super().__init__(
            provider_id=provider_id or "openai_compatible",
            display_name=display_name,
            model_id=model_id,
            endpoint=endpoint,
            credential_ref=credential_ref,
            supports_vision=supports_vision,
            timeout=timeout,
            transport=transport,
            credential_resolver=credential_resolver,
        )

    def _authentication_headers(self, api_key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}"}

    def _build_request(
        self,
        messages: tuple[ChatMessage, ...],
        *,
        preview: RenderedPreview | None,
        stream: bool,
    ) -> HttpRequest:
        normalized_messages = [
            ChatMessage(
                ChatRole.SYSTEM if item.role is ChatRole.DEVELOPER else item.role,
                item.content,
            )
            for item in messages
        ]
        preview_index = _last_user_index(normalized_messages) if preview else -1
        body_messages: list[dict[str, Any]] = []
        for index, message in enumerate(normalized_messages):
            if index == preview_index and preview is not None:
                content: Any = [
                    {"type": "text", "text": message.content},
                    {
                        "type": "image_url",
                        "image_url": {"url": preview.to_data_url()},
                    },
                ]
            else:
                content = message.content
            body_messages.append({"role": message.role.value, "content": content})
        body: dict[str, Any] = {
            "model": self._config.model_id,
            "messages": body_messages,
            "stream": stream,
        }
        if self._max_tokens is not None:
            body["max_tokens"] = self._max_tokens
        return self._request(body, stream=stream)

    def _parse_response(self, payload: Mapping[str, Any]) -> str:
        choices = payload.get("choices")
        if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)) and choices:
            first = choices[0]
            if isinstance(first, Mapping):
                message = first.get("message")
                if isinstance(message, Mapping):
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content
        raise ProviderError(f"{self._config.display_name} returned no Chat Completions text.")

    def _parse_stream_event(self, event: Mapping[str, Any]) -> str:
        if "error" in event:
            raise ProviderError(f"{self._config.display_name} stream failed.")
        choices = event.get("choices")
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, Mapping):
            return ""
        delta = first.get("delta")
        if not isinstance(delta, Mapping):
            return ""
        content = delta.get("content")
        return content if isinstance(content, str) else ""


class MoonshotProvider(OpenAICompatibleProvider):
    """Moonshot/Kimi OpenAI-compatible adapter."""

    _NON_THINKING_MODELS = frozenset({"kimi-k2.5", "kimi-k2.6"})

    def __init__(
        self,
        *,
        model_id: str,
        credential_ref: str | CredentialReference = "env:MOONSHOT_API_KEY",
        endpoint: str = "https://api.moonshot.cn/v1/chat/completions",
        supports_vision: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            provider_name="Moonshot/Kimi",
            model_id=model_id,
            endpoint=endpoint,
            credential_ref=credential_ref,
            supports_vision=supports_vision,
            **kwargs,
        )

    def _build_request(
        self,
        messages: tuple[ChatMessage, ...],
        *,
        preview: RenderedPreview | None,
        stream: bool,
    ) -> HttpRequest:
        request = super()._build_request(messages, preview=preview, stream=stream)
        body = dict(request.json_body)
        if self.model_id == "kimi-k3":
            # K3 always reasons; low effort keeps interactive requests responsive.
            body["reasoning_effort"] = "low"
        elif self.model_id in self._NON_THINKING_MODELS:
            # These models default to thinking, but the teaching-assistant UI
            # currently displays only the final answer.
            body["thinking"] = {"type": "disabled"}
        else:
            return request
        return replace(request, json_body=body)


class KimiProvider(MoonshotProvider):
    """Readable alias for :class:`MoonshotProvider`."""


class GLMProvider(OpenAICompatibleProvider):
    """Zhipu GLM OpenAI-compatible adapter."""

    def __init__(
        self,
        *,
        model_id: str,
        credential_ref: str | CredentialReference = "env:ZHIPU_API_KEY",
        endpoint: str = "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        supports_vision: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            provider_name="Zhipu GLM",
            model_id=model_id,
            endpoint=endpoint,
            credential_ref=credential_ref,
            supports_vision=supports_vision,
            **kwargs,
        )


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek OpenAI-compatible adapter with a user-selected model ID."""

    _V4_MODELS = frozenset({"deepseek-v4-flash", "deepseek-v4-pro"})

    def __init__(
        self,
        *,
        model_id: str,
        credential_ref: str | CredentialReference = "env:DEEPSEEK_API_KEY",
        endpoint: str = "https://api.deepseek.com/chat/completions",
        supports_vision: bool = False,
        thinking_enabled: bool = False,
        **kwargs: Any,
    ) -> None:
        if supports_vision:
            raise ValidationError(
                "DeepSeek Chat Completions currently accepts text only. "
                "Disable Vision input and Attach visible slice."
            )
        if not isinstance(thinking_enabled, bool):
            raise ValidationError("DeepSeek thinking_enabled must be true or false.")
        self._thinking_enabled = thinking_enabled
        super().__init__(
            provider_name="DeepSeek",
            model_id=model_id,
            endpoint=endpoint,
            credential_ref=credential_ref,
            supports_vision=False,
            **kwargs,
        )

    def _build_request(
        self,
        messages: tuple[ChatMessage, ...],
        *,
        preview: RenderedPreview | None,
        stream: bool,
    ) -> HttpRequest:
        request = super()._build_request(messages, preview=preview, stream=stream)
        if self.model_id not in self._V4_MODELS:
            return request
        body = dict(request.json_body)
        body["thinking"] = {
            "type": "enabled" if self._thinking_enabled else "disabled",
        }
        if self._thinking_enabled:
            body["reasoning_effort"] = "high"
        return replace(request, json_body=body)


GenericOpenAICompatibleProvider = OpenAICompatibleProvider


def _last_user_index(messages: Sequence[ChatMessage]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role is ChatRole.USER:
            return index
    raise ValidationError("A rendered preview requires a user message.")
