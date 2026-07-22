"""Provider registry and application facade for the teaching assistant."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..errors import ProviderError, ValidationError
from ..llm import (
    AnthropicProvider,
    ChatMessage,
    DeepSeekProvider,
    DisabledTransport,
    GLMProvider,
    KimiProvider,
    LLMProvider,
    LLMResponse,
    OpenAICompatibleProvider,
    OpenAIProvider,
    RenderedPreview,
    TransferPlan,
    Transport,
    UrllibTransport,
)
from ..runtime import CancellationToken
from ..runtime.credentials import CredentialResolver

ProviderBuilder = Callable[..., LLMProvider]
TransportFactory = Callable[[], Transport]


@dataclass(frozen=True, slots=True)
class ProviderDefaults:
    """User-visible non-secret defaults for one registered provider."""

    name: str
    endpoint: str
    credential_ref: str
    timeout: float = 30.0


@dataclass(frozen=True, slots=True)
class ProviderConfiguration:
    """Complete user selection passed from a presentation layer."""

    provider_name: str
    model_id: str
    endpoint: str
    credential_ref: str
    supports_vision: bool = False
    network_enabled: bool = False
    timeout: float = 30.0

    def __post_init__(self) -> None:
        for field_name in ("provider_name", "model_id", "endpoint", "credential_ref"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise ValidationError(f"{field_name} must be text.")
        if not self.provider_name.strip():
            raise ValidationError("Select a registered provider.")
        if not self.model_id.strip():
            raise ValidationError("Enter the exact model ID supplied by your provider.")
        if not self.endpoint.strip():
            raise ValidationError("Enter the provider endpoint.")
        if not self.credential_ref.strip():
            raise ValidationError("Enter an environment or keyring credential reference.")


_DEFAULTS = (
    ProviderDefaults(
        "OpenAI",
        "https://api.openai.com/v1/responses",
        "env:OPENAI_API_KEY",
    ),
    ProviderDefaults(
        "Anthropic",
        "https://api.anthropic.com/v1/messages",
        "env:ANTHROPIC_API_KEY",
    ),
    ProviderDefaults(
        "Moonshot/Kimi",
        "https://api.moonshot.cn/v1/chat/completions",
        "env:MOONSHOT_API_KEY",
        120.0,
    ),
    ProviderDefaults(
        "Zhipu GLM",
        "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "env:ZHIPU_API_KEY",
    ),
    ProviderDefaults(
        "DeepSeek",
        "https://api.deepseek.com/chat/completions",
        "env:DEEPSEEK_API_KEY",
    ),
    ProviderDefaults(
        "OpenAI-compatible",
        "https://example.invalid/v1/chat/completions",
        "env:OPENMEDVISIONX_API_KEY",
    ),
)


def _default_builders() -> dict[str, ProviderBuilder]:
    return {
        "OpenAI": OpenAIProvider,
        "Anthropic": AnthropicProvider,
        "Moonshot/Kimi": KimiProvider,
        "Zhipu GLM": GLMProvider,
        "DeepSeek": DeepSeekProvider,
        "OpenAI-compatible": OpenAICompatibleProvider,
    }


class LLMProviderRegistry:
    """Resolve a user-facing provider name to one public provider interface."""

    def __init__(
        self,
        *,
        defaults: Sequence[ProviderDefaults] = _DEFAULTS,
        builders: Mapping[str, ProviderBuilder] | None = None,
        online_transport_factory: TransportFactory = UrllibTransport,
        offline_transport_factory: TransportFactory = DisabledTransport,
        credential_resolver: CredentialResolver | None = None,
    ) -> None:
        self._defaults = {item.name: item for item in defaults}
        if len(self._defaults) != len(defaults):
            raise ValueError("Provider registry names must be unique.")
        self._builders = dict(builders or _default_builders())
        missing = set(self._defaults) - set(self._builders)
        if missing:
            raise ValueError(f"Provider factories are missing: {sorted(missing)}.")
        self._online_transport_factory = online_transport_factory
        self._offline_transport_factory = offline_transport_factory
        self._credential_resolver = credential_resolver

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._defaults)

    def defaults_for(self, name: str) -> ProviderDefaults:
        try:
            return self._defaults[name]
        except KeyError:
            raise ValidationError(f"Unknown provider {name!r}.") from None

    def create(self, configuration: ProviderConfiguration) -> LLMProvider:
        """Construct a provider with an explicitly enabled or disabled transport."""

        self.defaults_for(configuration.provider_name)
        transport = (
            self._online_transport_factory()
            if configuration.network_enabled
            else self._offline_transport_factory()
        )
        common: dict[str, Any] = {
            "model_id": configuration.model_id.strip(),
            "endpoint": configuration.endpoint.strip(),
            "credential_ref": configuration.credential_ref.strip(),
            "supports_vision": configuration.supports_vision,
            "timeout": configuration.timeout,
            "transport": transport,
        }
        if self._credential_resolver is not None:
            common["credential_resolver"] = self._credential_resolver
        builder = self._builders[configuration.provider_name]
        if configuration.provider_name == "OpenAI-compatible":
            common["provider_name"] = configuration.provider_name
        return builder(**common)


class TeachingAssistantService:
    """Small facade preserving provider, transport, consent, and cancellation boundaries."""

    def __init__(self, registry: LLMProviderRegistry | None = None) -> None:
        self.registry = registry or LLMProviderRegistry()

    @property
    def provider_names(self) -> tuple[str, ...]:
        return self.registry.names

    def provider_defaults(self, name: str) -> ProviderDefaults:
        return self.registry.defaults_for(name)

    def create_provider(self, configuration: ProviderConfiguration) -> LLMProvider:
        return self.registry.create(configuration)

    @staticmethod
    def plan_image_transfer(
        provider: LLMProvider,
        messages: str | Sequence[ChatMessage | Mapping[str, Any]],
        preview: RenderedPreview,
        *,
        task: str,
    ) -> TransferPlan:
        return provider.plan_image_transfer(messages, preview, task=task)

    @staticmethod
    def authorize_image_transfer(provider: LLMProvider, plan: TransferPlan) -> None:
        if not provider.capabilities().vision:
            raise ValidationError("Declare vision capability before authorizing an image preview.")
        authorize = getattr(provider, "authorize_image_transfer", None)
        if not callable(authorize):
            raise ProviderError("The configured provider cannot authorize image transfer.")
        authorize(plan)

    @staticmethod
    def revoke_image_transfer(provider: LLMProvider) -> None:
        revoke = getattr(provider, "revoke_image_transfer", None)
        if callable(revoke):
            revoke()

    @staticmethod
    def chat(
        provider: LLMProvider,
        messages: str | Sequence[ChatMessage | Mapping[str, Any]],
        *,
        preview: RenderedPreview | None = None,
        transfer_plan: TransferPlan | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> LLMResponse:
        return provider.chat(
            messages,
            preview=preview,
            transfer_plan=transfer_plan,
            cancellation_token=cancellation_token,
        )


__all__ = [
    "LLMProviderRegistry",
    "ProviderBuilder",
    "ProviderConfiguration",
    "ProviderDefaults",
    "TeachingAssistantService",
    "TransportFactory",
]
