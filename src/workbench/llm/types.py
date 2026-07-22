"""Public value types for cloud teaching-assistant providers."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import zlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from io import BytesIO
from threading import RLock
from typing import Any
from urllib.parse import urlsplit

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

    @property
    def sha256(self) -> str:
        """Return the digest of the exact canonical PNG bytes to be transferred."""

        return hashlib.sha256(self.data).hexdigest()

    def __repr__(self) -> str:
        return (
            f"RenderedPreview(mime_type='image/png', size={self.width}x{self.height}, "
            f"bytes={len(self.data)})"
        )


@dataclass(frozen=True, slots=True)
class TransferItem:
    """One immutable file entry shown at the final cloud-transfer boundary."""

    name: str
    mime_type: str
    size_bytes: int
    sha256: str
    width: int | None = None
    height: int | None = None
    transform: str = ""
    deidentification_actions: tuple[str, ...] = ()
    burned_in_text_review: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.mime_type.strip():
            raise ValidationError("Transfer items require a name and MIME type.")
        if self.size_bytes < 0:
            raise ValidationError("Transfer item size cannot be negative.")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise ValidationError("Transfer item SHA-256 must be lowercase hexadecimal.")
        dimensions = (self.width, self.height)
        if any(value is not None and value <= 0 for value in dimensions):
            raise ValidationError("Transfer item dimensions must be positive when present.")
        transform = str(self.transform).strip()
        if not transform:
            raise ValidationError("Transfer items must state the input transformation.")
        actions = tuple(str(action).strip() for action in self.deidentification_actions)
        if not actions or any(not action for action in actions):
            raise ValidationError(
                "Transfer items must state every de-identification or payload-minimization action."
            )
        burned_in_review = str(self.burned_in_text_review).strip()
        if not burned_in_review:
            raise ValidationError("Transfer items must state the burned-in text review status.")
        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "deidentification_actions", actions)
        object.__setattr__(self, "burned_in_text_review", burned_in_review)


@dataclass(frozen=True, slots=True)
class TransferPlan:
    """Exact, one-request authorization plan for an outbound image payload.

    The plan intentionally stores only hashes of prompt text.  It binds consent
    to the provider, endpoint, model, task, and canonical file bytes without
    copying potentially sensitive prompt content into settings or logs.
    """

    provider_id: str
    provider_name: str
    endpoint: str
    model_id: str
    task: str
    prompt_sha256: str
    items: tuple[TransferItem, ...]
    residual_risks: tuple[str, ...] = (
        "Burned-in text can remain visible in rendered pixels.",
        "A remote recipient may retain data after this request is sent.",
        "Cancellation cannot recall bytes that have already left this device.",
    )
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        text_fields = {
            "provider_id": self.provider_id,
            "provider_name": self.provider_name,
            "endpoint": self.endpoint,
            "model_id": self.model_id,
            "task": self.task,
        }
        if any(not isinstance(value, str) or not value.strip() for value in text_fields.values()):
            raise ValidationError("Transfer plans require provider, endpoint, model, and task.")
        parsed = urlsplit(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValidationError("Transfer plan endpoint must be an absolute HTTP(S) URL.")
        if len(self.prompt_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.prompt_sha256
        ):
            raise ValidationError("Transfer plan prompt SHA-256 must be lowercase hexadecimal.")
        items = tuple(self.items)
        if not items:
            raise ValidationError("Transfer plans must contain at least one file.")
        residual_risks = tuple(str(risk).strip() for risk in self.residual_risks)
        if not residual_risks or any(not risk for risk in residual_risks):
            raise ValidationError("Transfer plans require at least one non-empty residual risk.")
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "residual_risks", residual_risks)
        canonical = {
            "provider_id": self.provider_id.strip(),
            "provider_name": self.provider_name.strip(),
            "endpoint": self.endpoint.strip(),
            "model_id": self.model_id.strip(),
            "task": self.task.strip(),
            "prompt_sha256": self.prompt_sha256,
            "items": [
                {
                    "name": item.name,
                    "mime_type": item.mime_type,
                    "size_bytes": item.size_bytes,
                    "sha256": item.sha256,
                    "width": item.width,
                    "height": item.height,
                    "transform": item.transform,
                    "deidentification_actions": list(item.deidentification_actions),
                    "burned_in_text_review": item.burned_in_text_review,
                }
                for item in items
            ],
            "residual_risks": list(residual_risks),
        }
        encoded = json.dumps(
            canonical,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        object.__setattr__(self, "plan_id", hashlib.sha256(encoded).hexdigest())

    @property
    def endpoint_host(self) -> str:
        parsed = urlsplit(self.endpoint)
        host = parsed.hostname or ""
        return f"{host}:{parsed.port}" if parsed.port is not None else host

    @property
    def total_bytes(self) -> int:
        return sum(item.size_bytes for item in self.items)

    @classmethod
    def for_preview(
        cls,
        *,
        provider_id: str,
        provider_name: str,
        endpoint: str,
        model_id: str,
        task: str,
        messages: Sequence[ChatMessage],
        preview: RenderedPreview,
        transform: str = "active-view rendered PNG; metadata removed",
    ) -> TransferPlan:
        canonical_messages = json.dumps(
            [{"role": message.role.value, "content": message.content} for message in messages],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return cls(
            provider_id=provider_id,
            provider_name=provider_name,
            endpoint=endpoint,
            model_id=model_id,
            task=task,
            prompt_sha256=hashlib.sha256(canonical_messages).hexdigest(),
            items=(
                TransferItem(
                    name="rendered-preview.png",
                    mime_type=preview.mime_type,
                    size_bytes=len(preview.data),
                    sha256=preview.sha256,
                    width=preview.width,
                    height=preview.height,
                    transform=transform,
                    deidentification_actions=(
                        "Original DICOM/NIfTI files and source metadata are excluded.",
                        "Rendered pixels are decoded and re-encoded as a canonical PNG with "
                        "metadata removed.",
                    ),
                    burned_in_text_review=(
                        "Not automatically assessed; inspect the exact final preview for "
                        "burned-in identifiers."
                    ),
                ),
            ),
        )

    def matches_preview(self, preview: RenderedPreview) -> bool:
        """Return whether the plan describes exactly this one canonical PNG payload."""

        if not isinstance(preview, RenderedPreview) or len(self.items) != 1:
            return False
        item = self.items[0]
        return (
            item.mime_type == preview.mime_type
            and item.size_bytes == len(preview.data)
            and item.sha256 == preview.sha256
            and item.width == preview.width
            and item.height == preview.height
        )


class ImageTransferAuthorization:
    """Provider-local, one-shot consent bound to one exact transfer plan."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._authorized_plan_id: str | None = None
        self._generation = 0

    @property
    def authorized(self) -> bool:
        with self._lock:
            return self._authorized_plan_id is not None

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def grant(self, plan_id: str) -> int:
        if (
            not isinstance(plan_id, str)
            or len(plan_id) != 64
            or any(char not in "0123456789abcdef" for char in plan_id)
        ):
            raise ValidationError("Image transfer authorization requires an exact plan ID.")
        with self._lock:
            self._authorized_plan_id = plan_id
            self._generation += 1
            return self._generation

    def revoke(self) -> int:
        with self._lock:
            self._authorized_plan_id = None
            self._generation += 1
            return self._generation

    @contextmanager
    def transfer_guard(self, plan_id: str) -> Iterator[int]:
        """Authorize exactly one transfer and consume consent after the attempt."""

        with self._lock:
            if self._authorized_plan_id != plan_id:
                self._authorized_plan_id = None
                self._generation += 1
                raise CloudTransferDenied(
                    "Cloud image transfer authorization is missing, stale, or for a different "
                    "provider/model/task/payload."
                )
            generation = self._generation
        try:
            yield generation
        finally:
            with self._lock:
                if self._authorized_plan_id == plan_id and self._generation == generation:
                    self._authorized_plan_id = None
                    self._generation += 1

    def consume_for_dispatch(self, plan_id: str, generation: int) -> None:
        """Atomically seal one current authorization immediately before dispatch."""

        with self._lock:
            if self._authorized_plan_id != plan_id or self._generation != generation:
                self._authorized_plan_id = None
                self._generation += 1
                raise CloudTransferDenied(
                    "Cloud image transfer authorization was revoked before network dispatch."
                )
            self._authorized_plan_id = None
            self._generation += 1


def normalize_messages(
    messages: str | Sequence[ChatMessage | Mapping[str, Any]],
) -> tuple[ChatMessage, ...]:
    normalized: tuple[ChatMessage, ...]
    if isinstance(messages, str):
        normalized = (ChatMessage(ChatRole.USER, messages),)
    else:
        normalized = tuple(ChatMessage.coerce(item) for item in messages)
    if not normalized:
        raise ValidationError("At least one chat message is required.")
    if not any(item.role is ChatRole.USER for item in normalized):
        raise ValidationError("A chat request must contain at least one user message.")
    return normalized
