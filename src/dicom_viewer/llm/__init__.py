"""Provider-neutral teaching-assistant interfaces.

Importing this package never opens a network connection.  Providers use a
disabled transport unless the host application explicitly injects an enabled
transport implementation.
"""

from .providers import (
    AnthropicProvider,
    DeepSeekProvider,
    GenericOpenAICompatibleProvider,
    GLMProvider,
    KimiProvider,
    LLMProvider,
    MoonshotProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
)
from .transport import DisabledTransport, HttpRequest, Transport, UrllibTransport
from .types import (
    CLOUD_PREVIEW_PRIVACY_WARNING,
    EDUCATIONAL_DISCLAIMER,
    ChatMessage,
    ChatRole,
    CloudTransferDenied,
    LLMResponse,
    ProviderCapabilities,
    RenderedPreview,
)

__all__ = [
    "AnthropicProvider",
    "ChatMessage",
    "ChatRole",
    "CLOUD_PREVIEW_PRIVACY_WARNING",
    "CloudTransferDenied",
    "DeepSeekProvider",
    "DisabledTransport",
    "EDUCATIONAL_DISCLAIMER",
    "GLMProvider",
    "GenericOpenAICompatibleProvider",
    "HttpRequest",
    "KimiProvider",
    "LLMProvider",
    "LLMResponse",
    "MoonshotProvider",
    "OpenAICompatibleProvider",
    "OpenAIProvider",
    "ProviderCapabilities",
    "RenderedPreview",
    "Transport",
    "UrllibTransport",
]
