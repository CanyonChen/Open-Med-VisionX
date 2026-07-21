"""Public value types for cloud teaching-assistant providers."""

from __future__ import annotations

import base64
import binascii
import zlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from io import BytesIO
from threading import RLock
from typing import Any

from ..errors import MissingDependencyError, ProviderError, ValidationError

EDUCATIONAL_DISCLAIMER = (
    "OpenMedVisionX 仅用于教学与概念解释，不构成医学诊断、治疗建议或临床决策依据。"
)
CLOUD_PREVIEW_PRIVACY_WARNING = (
    "仅发送用户已预览的渲染 PNG；像素中烧录的文字仍可能包含隐私信息，授权前请检查。"
)


class CloudTransferDenied(ProviderError):
    """Rendered-preview transfer has not been authorized for this provider."""


class ChatRole(str, Enum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: ChatRole
    content: str

    def __post_init__(self) -> None:
        try:
            role = ChatRole(self.role)
        except ValueError:
            raise ValidationError(f"Unsupported chat role {self.role!r}.") from None
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValidationError("Chat message content must be non-empty text.")
        object.__setattr__(self, "role", role)

    @classmethod
    def coerce(cls, value: ChatMessage | Mapping[str, Any]) -> ChatMessage:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValidationError("Messages must be ChatMessage objects or role/content mappings.")
        role = value.get("role", "")
        content = value.get("content", "")
        if not isinstance(role, (str, ChatRole)) or not isinstance(content, str):
            raise ValidationError("Message role and content must be text.")
        try:
            chat_role = ChatRole(role)
        except ValueError:
            raise ValidationError(f"Unsupported chat role {role!r}.") from None
        return cls(role=chat_role, content=content)


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    provider: str
    text: bool = True
    streaming: bool = True
    vision: bool = False
    network_required: bool = True
    image_transfer_authorized: bool = False


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """A provider answer with mandatory provenance and safety context."""

    text: str
    provider: str
    model: str
    timestamp: datetime = field(default_factory=_utc_now)
    disclaimer: str = EDUCATIONAL_DISCLAIMER

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ProviderError("The provider returned an empty text response.")
        if not self.provider.strip() or not self.model.strip():
            raise ValidationError("Provider responses require provider and model identifiers.")
        timestamp = self.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        object.__setattr__(self, "timestamp", timestamp.astimezone(timezone.utc))

    @property
    def footer(self) -> str:
        return self.format_footer(
            provider=self.provider,
            model=self.model,
            timestamp=self.timestamp,
            disclaimer=self.disclaimer,
        )

    @property
    def content(self) -> str:
        return f"{self.text.rstrip()}\n\n{self.footer}"

    @staticmethod
    def format_footer(
        *,
        provider: str,
        model: str,
        timestamp: datetime | None = None,
        disclaimer: str = EDUCATIONAL_DISCLAIMER,
    ) -> str:
        observed = timestamp or _utc_now()
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        return (
            "---\n"
            f"Provider: {provider}\n"
            f"Model: {model}\n"
            f"Time: {observed.astimezone(timezone.utc).isoformat()}\n"
            f"声明: {disclaimer}"
        )

    def __str__(self) -> str:
        return self.content


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SAFE_PNG_CHUNKS = {b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tRNS"}
_MAX_PREVIEW_BYTES = 8 * 1024 * 1024
_MAX_PREVIEW_PIXELS = 16_777_216
_MAX_DECODED_PREVIEW_BYTES = 64 * 1024 * 1024
_PNG_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_PNG_BIT_DEPTHS = {
    0: {1, 2, 4, 8, 16},
    2: {8, 16},
    3: {1, 2, 4, 8},
    4: {8, 16},
    6: {8, 16},
}


@dataclass(frozen=True, slots=True, repr=False)
class RenderedPreview:
    """A metadata-free rendered PNG preview, never an original medical file.

    Only a deliberately small PNG chunk allow-list is accepted.  EXIF, text,
    XMP, ICC, time, and other ancillary metadata chunks are rejected rather
    than silently forwarded to a cloud service.
    """

    data: bytes
    mime_type: str = "image/png"
    width: int = field(init=False)
    height: int = field(init=False)

    def __post_init__(self) -> None:
        data = bytes(self.data)
        if self.mime_type != "image/png":
            raise ValidationError("Cloud previews must be metadata-free PNG bytes.")
        width, height = self._validate_png(data)
        data = self._canonicalize_png(data, width, height)
        sanitized_width, sanitized_height = self._validate_png(data)
        if (sanitized_width, sanitized_height) != (width, height):
            raise ValidationError("Rendered preview dimensions changed during sanitization.")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)

    @classmethod
    def from_png(cls, data: bytes | bytearray | memoryview) -> RenderedPreview:
        return cls(bytes(data))

    @staticmethod
    def _validate_png(data: bytes) -> tuple[int, int]:
        if len(data) > _MAX_PREVIEW_BYTES:
            raise ValidationError("Rendered preview exceeds the 8 MiB transfer limit.")
        if not data.startswith(_PNG_SIGNATURE):
            raise ValidationError(
                "Only rendered PNG previews are accepted; raw medical data is forbidden."
            )
        offset = len(_PNG_SIGNATURE)
        width = height = 0
        bit_depth = color_type = 0
        saw_header = saw_data = saw_end = False
        saw_palette = saw_transparency = False
        palette_entries = 0
        compressed_parts: list[bytes] = []
        chunk_index = 0
        while offset < len(data):
            if offset + 12 > len(data):
                raise ValidationError("Rendered preview PNG is truncated.")
            length = int.from_bytes(data[offset : offset + 4], "big")
            chunk_type = data[offset + 4 : offset + 8]
            end = offset + 12 + length
            if end > len(data):
                raise ValidationError("Rendered preview PNG contains a truncated chunk.")
            payload = data[offset + 8 : offset + 8 + length]
            expected_crc = int.from_bytes(data[offset + 8 + length : end], "big")
            actual_crc = binascii.crc32(chunk_type + payload) & 0xFFFFFFFF
            if expected_crc != actual_crc:
                raise ValidationError("Rendered preview PNG failed its integrity check.")
            if chunk_type not in _SAFE_PNG_CHUNKS:
                raise ValidationError(
                    f"Rendered preview contains forbidden metadata chunk {chunk_type!r}."
                )
            if chunk_index == 0:
                if chunk_type != b"IHDR" or length != 13:
                    raise ValidationError("Rendered preview PNG must begin with one IHDR chunk.")
                width = int.from_bytes(payload[0:4], "big")
                height = int.from_bytes(payload[4:8], "big")
                if width <= 0 or height <= 0 or width * height > _MAX_PREVIEW_PIXELS:
                    raise ValidationError(
                        "Rendered preview pixel dimensions exceed the safety limit."
                    )
                bit_depth = payload[8]
                color_type = payload[9]
                if (
                    color_type not in _PNG_BIT_DEPTHS
                    or bit_depth not in _PNG_BIT_DEPTHS[color_type]
                ):
                    raise ValidationError("Rendered preview PNG has an invalid pixel format.")
                if payload[10:12] != b"\x00\x00" or payload[12] != 0:
                    raise ValidationError(
                        "Rendered preview PNG must use standard compression and no interlacing."
                    )
                saw_header = True
            elif chunk_type == b"IHDR":
                raise ValidationError("Rendered preview PNG contains duplicate IHDR chunks.")
            elif chunk_type == b"PLTE":
                if saw_palette or saw_data or color_type in {0, 4}:
                    raise ValidationError("Rendered preview PNG has an invalid palette.")
                if length == 0 or length > 768 or length % 3:
                    raise ValidationError("Rendered preview PNG has an invalid palette.")
                palette_entries = length // 3
                if color_type == 3 and palette_entries > 2**bit_depth:
                    raise ValidationError("Rendered preview PNG palette exceeds its bit depth.")
                saw_palette = True
            elif chunk_type == b"tRNS":
                if saw_transparency or saw_data:
                    raise ValidationError("Rendered preview PNG has invalid transparency data.")
                valid_length = (
                    (color_type == 0 and length == 2)
                    or (color_type == 2 and length == 6)
                    or (color_type == 3 and saw_palette and 0 < length <= palette_entries)
                )
                if not valid_length:
                    raise ValidationError("Rendered preview PNG has invalid transparency data.")
                saw_transparency = True
            elif chunk_type == b"IDAT":
                saw_data = True
                compressed_parts.append(payload)
            elif chunk_type == b"IEND":
                if length != 0 or end != len(data):
                    raise ValidationError("Rendered preview PNG has invalid data after IEND.")
                saw_end = True
                offset = end
                break
            offset = end
            chunk_index += 1
        if not (saw_header and saw_data and saw_end):
            raise ValidationError("Rendered preview PNG is incomplete.")
        if color_type == 3 and not saw_palette:
            raise ValidationError("Rendered preview indexed PNG is missing its palette.")

        channels = _PNG_CHANNELS[color_type]
        row_bytes = (width * channels * bit_depth + 7) // 8
        decoded_size = height * (row_bytes + 1)
        if decoded_size > _MAX_DECODED_PREVIEW_BYTES:
            raise ValidationError("Rendered preview decoded raster exceeds the safety limit.")
        try:
            decoder = zlib.decompressobj()
            decoded = decoder.decompress(b"".join(compressed_parts), decoded_size + 1)
            if len(decoded) <= decoded_size:
                decoded += decoder.flush(decoded_size + 1 - len(decoded))
        except zlib.error:
            raise ValidationError("Rendered preview PNG pixel data is invalid.") from None
        if (
            len(decoded) != decoded_size
            or not decoder.eof
            or decoder.unused_data
            or decoder.unconsumed_tail
        ):
            raise ValidationError("Rendered preview PNG pixel data is invalid.")
        if any(decoded[row * (row_bytes + 1)] > 4 for row in range(height)):
            raise ValidationError("Rendered preview PNG contains an invalid row filter.")
        return width, height

    @staticmethod
    def _canonicalize_png(data: bytes, width: int, height: int) -> bytes:
        """Decode pixels and re-encode RGBA so no unused chunk payload survives."""

        try:
            from PIL import Image
        except ImportError:
            raise MissingDependencyError(
                "Rendered preview sanitization requires the base Pillow dependency."
            ) from None
        try:
            with Image.open(BytesIO(data)) as source:
                if source.format != "PNG" or source.size != (width, height):
                    raise ValidationError("Rendered preview could not be decoded as PNG pixels.")
                source.load()
                converted = source.convert("RGBA")
                try:
                    raster = converted.tobytes()
                finally:
                    converted.close()
            clean = Image.frombytes("RGBA", (width, height), raster)
            output = BytesIO()
            try:
                clean.save(output, format="PNG", compress_level=6)
            finally:
                clean.close()
        except ValidationError:
            raise
        except (OSError, TypeError, ValueError):
            raise ValidationError("Rendered preview PNG pixel data is invalid.") from None
        return output.getvalue()

    def to_data_url(self) -> str:
        encoded = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}"

    def base64_data(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    def __repr__(self) -> str:
        return (
            f"RenderedPreview(mime_type='image/png', size={self.width}x{self.height}, "
            f"bytes={len(self.data)})"
        )


class ImageTransferAuthorization:
    """Provider-local, thread-safe and revocable image-transfer consent."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._authorized = False
        self._generation = 0

    @property
    def authorized(self) -> bool:
        with self._lock:
            return self._authorized

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def grant(self) -> int:
        with self._lock:
            self._authorized = True
            self._generation += 1
            return self._generation

    def revoke(self) -> int:
        with self._lock:
            self._authorized = False
            self._generation += 1
            return self._generation

    @contextmanager
    def transfer_guard(self) -> Iterator[int]:
        """Serialize the network transfer boundary with grant/revoke actions."""

        with self._lock:
            if not self._authorized:
                raise CloudTransferDenied(
                    "Cloud image transfer authorization was revoked before sending."
                )
            yield self._generation


def normalize_messages(
    messages: str | Sequence[ChatMessage | Mapping[str, Any]],
) -> tuple[ChatMessage, ...]:
    if isinstance(messages, str):
        normalized = (ChatMessage(ChatRole.USER, messages),)
    else:
        normalized = tuple(ChatMessage.coerce(item) for item in messages)
    if not normalized:
        raise ValidationError("At least one chat message is required.")
    if not any(item.role is ChatRole.USER for item in normalized):
        raise ValidationError("A chat request must contain at least one user message.")
    return normalized
