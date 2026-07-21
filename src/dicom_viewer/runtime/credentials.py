"""Resolve credential references without persisting secret values."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ..errors import MissingDependencyError, ProviderError, ValidationError

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class CredentialResolutionError(ProviderError):
    """A credential reference is valid but cannot currently be resolved."""


@dataclass(frozen=True, slots=True, repr=False)
class CredentialReference:
    """A non-secret pointer suitable for configuration files."""

    scheme: str
    target: str

    def __post_init__(self) -> None:
        scheme = self.scheme.strip().lower()
        target = self.target.strip()
        if scheme not in {"env", "keyring"}:
            raise ValidationError("Credential references must use env: or keyring:.")
        if not target:
            raise ValidationError("Credential reference target cannot be empty.")
        if scheme == "env" and not _ENV_NAME.fullmatch(target):
            raise ValidationError(f"Invalid environment variable name {target!r}.")
        if scheme == "keyring":
            parts = target.split("/", 1)
            if len(parts) != 2 or not all(part.strip() for part in parts):
                raise ValidationError("keyring references must be keyring:<service>/<username>.")
        object.__setattr__(self, "scheme", scheme)
        object.__setattr__(self, "target", target)

    @classmethod
    def parse(cls, value: str | CredentialReference) -> CredentialReference:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str) or ":" not in value:
            raise ValidationError(
                "Store a credential reference such as env:OPENAI_API_KEY, never a raw key."
            )
        scheme, target = value.split(":", 1)
        return cls(scheme, target)

    def __str__(self) -> str:
        return f"{self.scheme}:{self.target}"

    def __repr__(self) -> str:
        return f"CredentialReference(scheme={self.scheme!r}, target=<configured>)"


class CredentialResolver:
    """Resolve ``env:`` and optional ``keyring:`` references on demand.

    Resolved values are returned directly to the request builder and are never
    cached by this object.
    """

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        keyring_getter: Callable[[str, str], str | None] | None = None,
    ) -> None:
        self._environment = os.environ if environment is None else environment
        self._keyring_getter = keyring_getter

    def resolve(self, reference: str | CredentialReference) -> str:
        parsed = CredentialReference.parse(reference)
        if parsed.scheme == "env":
            value = self._environment.get(parsed.target)
            if value is None or not str(value).strip():
                raise CredentialResolutionError(
                    f"Credential environment variable {parsed.target!r} is not set."
                )
            return str(value)

        service, username = (part.strip() for part in parsed.target.split("/", 1))
        getter = self._keyring_getter or self._load_keyring_getter()
        try:
            value = getter(service, username)
        except Exception:
            raise CredentialResolutionError(
                "The configured operating-system credential could not be read."
            ) from None
        if value is None or not str(value).strip():
            raise CredentialResolutionError(
                "The configured operating-system credential was not found."
            )
        return str(value)

    @staticmethod
    def _load_keyring_getter() -> Callable[[str, str], str | None]:
        try:
            import keyring  # type: ignore[import-not-found]
        except ImportError:
            raise MissingDependencyError(
                "keyring: credential references require the optional 'keyring' package."
            ) from None
        return keyring.get_password
