"""Injectable JSON/SSE transports with offline-safe defaults."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from socket import SHUT_RDWR
from threading import Event, Thread
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from ..errors import OperationCancelled, ProviderError, ValidationError
from ..runtime.privacy import redact
from ..runtime.tasks import CancellationToken

_READ_CHUNK_BYTES = 64 * 1024
_MAX_JSON_RESPONSE_BYTES = 16 * 1024 * 1024


def validate_endpoint_url(url: str) -> None:
    """Accept HTTPS endpoints and loopback-only HTTP endpoints."""

    if not isinstance(url, str) or not url or url != url.strip():
        raise ValidationError("Provider endpoint is not a valid URL.")
    try:
        parsed = urlsplit(url)
        _ = parsed.port
    except (TypeError, ValueError):
        raise ValidationError("Provider endpoint is not a valid URL.") from None
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValidationError(
            "Provider endpoints cannot contain credentials, query strings, or URL fragments."
        )
    if parsed.scheme == "https" and parsed.hostname:
        return
    if parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
        return
    raise ValidationError("Provider endpoints must use HTTPS (HTTP is allowed only for localhost).")


@dataclass(frozen=True, slots=True, repr=False)
class HttpRequest:
    method: str
    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    json_body: Mapping[str, Any] = field(default_factory=dict)
    timeout: float = 30.0
    cancellation_token: CancellationToken | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.method.upper() not in {"POST", "GET"}:
            raise ValidationError(f"Unsupported HTTP method {self.method!r}.")
        validate_endpoint_url(self.url)
        try:
            timeout = float(self.timeout)
        except (TypeError, ValueError):
            raise ValidationError("Provider timeout must be a finite positive number.") from None
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValidationError("Provider timeout must be positive.")
        object.__setattr__(self, "method", self.method.upper())
        object.__setattr__(self, "headers", MappingProxyType(dict(self.headers)))
        object.__setattr__(self, "json_body", MappingProxyType(dict(self.json_body)))
        object.__setattr__(self, "timeout", timeout)
        if self.cancellation_token is not None and not isinstance(
            self.cancellation_token, CancellationToken
        ):
            raise ValidationError("cancellation_token must be a CancellationToken.")

    def __repr__(self) -> str:
        return (
            f"HttpRequest(method={self.method!r}, url={self.url!r}, "
            f"headers={redact(self.headers)!r}, json_keys={tuple(self.json_body)!r}, "
            f"timeout={self.timeout!r})"
        )


@runtime_checkable
class Transport(Protocol):
    """Minimal transport contract; tests can provide an in-memory fake.

    Implementations must observe ``HttpRequest.cancellation_token`` while
    opening and consuming a response.  A cancellation is reported as
    :class:`OperationCancelled`, never as a provider failure.
    """

    def send(self, request: HttpRequest) -> Mapping[str, Any]: ...

    def stream(self, request: HttpRequest) -> Iterable[Mapping[str, Any]]: ...


class DisabledTransport:
    """Default transport that guarantees importing/using providers stays offline."""

    _MESSAGE = (
        "Network transport is disabled. Inject an explicitly enabled Transport "
        "after the user configures and authorizes a provider."
    )

    def send(self, request: HttpRequest) -> Mapping[str, Any]:
        UrllibTransport._raise_if_cancelled(request)
        raise ProviderError(self._MESSAGE)

    def stream(self, request: HttpRequest) -> Iterable[Mapping[str, Any]]:
        UrllibTransport._raise_if_cancelled(request)
        raise ProviderError(self._MESSAGE)


class UrllibTransport:
    """Explicit opt-in standard-library HTTPS transport.

    It has no provider-specific logic and never logs headers or request bodies.
    """

    def send(self, request: HttpRequest) -> Mapping[str, Any]:
        try:
            with self._cancellable_response(request, accept="application/json") as response:
                chunks: list[bytes] = []
                total = 0
                while True:
                    self._raise_if_cancelled(request)
                    chunk = response.read(_READ_CHUNK_BYTES)
                    self._raise_if_cancelled(request)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > _MAX_JSON_RESPONSE_BYTES:
                        raise ProviderError("Provider JSON response exceeds the safety limit.")
                    chunks.append(chunk)
                raw = b"".join(chunks)
        except OperationCancelled:
            raise
        except ProviderError:
            raise
        except (OSError, TimeoutError, ValueError):
            self._raise_if_cancelled(request)
            raise ProviderError("Provider request failed or timed out.") from None
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ProviderError("Provider returned an invalid JSON response.") from None
        if not isinstance(payload, Mapping):
            raise ProviderError("Provider returned an unexpected JSON response shape.")
        return payload

    def stream(self, request: HttpRequest) -> Iterator[Mapping[str, Any]]:
        data_lines: list[str] = []
        try:
            with self._cancellable_response(request, accept="text/event-stream") as response:
                for raw_line in response:
                    self._raise_if_cancelled(request)
                    try:
                        line = raw_line.decode("utf-8").rstrip("\r\n")
                    except UnicodeDecodeError:
                        raise ProviderError(
                            "Provider returned invalid UTF-8 stream data."
                        ) from None
                    if not line:
                        yield from self._decode_sse_data(data_lines)
                        data_lines.clear()
                        continue
                    if line.startswith(":") or line.startswith("event:"):
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                self._raise_if_cancelled(request)
                yield from self._decode_sse_data(data_lines)
        except OperationCancelled:
            raise
        except ProviderError:
            raise
        except (OSError, TimeoutError, ValueError):
            self._raise_if_cancelled(request)
            raise ProviderError("Provider request failed or timed out.") from None

    @contextmanager
    def _cancellable_response(self, request: HttpRequest, *, accept: str):
        """Close an active socket when another thread cancels the request.

        ``urlopen`` itself retains the configured finite timeout for DNS and
        connection setup.  Once response headers arrive, this watcher closes
        the underlying socket so a blocked body read or SSE iterator wakes
        promptly instead of waiting for that timeout.
        """

        self._raise_if_cancelled(request)
        response = self._open(request, accept=accept)
        self._raise_if_cancelled(request, response=response)
        stop = Event()
        watcher: Thread | None = None
        token = request.cancellation_token
        if token is not None:

            def close_on_cancel() -> None:
                while not stop.wait(0.025):
                    if token.is_cancelled:
                        self._force_close(response)
                        return

            watcher = Thread(
                target=close_on_cancel,
                name="openmedvisionx-http-cancel",
                daemon=True,
            )
            watcher.start()
        try:
            yield response
        finally:
            stop.set()
            self._force_close(response)
            if watcher is not None:
                watcher.join(0.25)

    @staticmethod
    def _force_close(response: Any) -> None:
        """Best-effort socket shutdown that can interrupt a read in another thread."""

        stream = getattr(response, "fp", None)
        raw = getattr(stream, "raw", None)
        sock = getattr(raw, "_sock", None)
        if sock is not None:
            with suppress(OSError):
                sock.shutdown(SHUT_RDWR)
        with suppress(Exception):
            response.close()

    @staticmethod
    def _raise_if_cancelled(request: HttpRequest, *, response: Any | None = None) -> None:
        token = request.cancellation_token
        if token is None:
            return
        if token.is_cancelled:
            if response is not None:
                UrllibTransport._force_close(response)
            raise OperationCancelled("Provider request was cancelled.")

    @staticmethod
    def _decode_sse_data(data_lines: list[str]) -> Iterator[Mapping[str, Any]]:
        if not data_lines:
            return
        data = "\n".join(data_lines)
        if data.strip() == "[DONE]":
            return
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            raise ProviderError("Provider returned an invalid server-sent event.") from None
        if isinstance(payload, Mapping):
            yield payload

    @staticmethod
    def _open(request: HttpRequest, *, accept: str):
        UrllibTransport._raise_if_cancelled(request)
        body = json.dumps(dict(request.json_body), ensure_ascii=False).encode("utf-8")
        headers = dict(request.headers)
        headers.setdefault("Content-Type", "application/json")
        headers.setdefault("Accept", accept)
        outbound = Request(
            request.url,
            data=body,
            headers=headers,
            method=request.method,
        )
        try:
            response = urlopen(outbound, timeout=request.timeout)  # noqa: S310 - endpoint is validated
            UrllibTransport._raise_if_cancelled(request, response=response)
            return response
        except HTTPError as error:
            # Provider bodies can echo prompts or request details.  Keep them
            # out of exceptions and logs entirely.
            status_code = error.code
            error.close()
            UrllibTransport._raise_if_cancelled(request)
            raise ProviderError(f"Provider HTTP error {status_code}.") from None
        except (URLError, TimeoutError, OSError):
            UrllibTransport._raise_if_cancelled(request)
            raise ProviderError("Provider request failed or timed out.") from None
