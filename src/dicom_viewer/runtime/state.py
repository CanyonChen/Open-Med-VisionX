"""Atomic, generation-tracked application session state."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Condition, RLock
from time import monotonic
from typing import Generic, TypeVar

from ..errors import ViewerError

T = TypeVar("T")


class SessionConflictError(ViewerError):
    """A compare-and-swap session update used a stale generation."""


@dataclass(frozen=True, slots=True)
class SessionSnapshot(Generic[T]):
    """One immutable observation of the session pointer."""

    value: T | None
    generation: int
    updated_at: datetime

    @property
    def is_empty(self) -> bool:
        return self.value is None


class AtomicSessionState(Generic[T]):
    """Atomically replace complete session objects instead of mutating fields.

    The contained value should itself be immutable (the domain image types are
    frozen dataclasses).  A loader can build and validate a new session off to
    the side, then commit it with one ``replace`` call.  ``expected_generation``
    prevents an older background load from overwriting a newer user action.
    """

    def __init__(self, initial: T | None = None) -> None:
        self._condition = Condition(RLock())
        self._value = initial
        self._generation = 0
        self._updated_at = datetime.now(timezone.utc)
        self._listeners: list[Callable[[SessionSnapshot[T]], None]] = []
        self._notifications: deque[
            tuple[
                SessionSnapshot[T],
                tuple[Callable[[SessionSnapshot[T]], None], ...],
            ]
        ] = deque()
        self._notification_lock = RLock()
        self._notifying = False

    def snapshot(self) -> SessionSnapshot[T]:
        with self._condition:
            return self._snapshot_unlocked()

    @property
    def value(self) -> T | None:
        return self.snapshot().value

    @property
    def generation(self) -> int:
        return self.snapshot().generation

    def replace(
        self,
        value: T,
        *,
        expected_generation: int | None = None,
    ) -> SessionSnapshot[T]:
        """Atomically install a fully validated, non-empty session value."""

        if value is None:
            raise ValueError("Use reset() to clear session state.")
        return self._commit(value, expected_generation)

    def compare_and_replace(self, expected_generation: int, value: T) -> SessionSnapshot[T]:
        return self.replace(value, expected_generation=expected_generation)

    def reset(self, *, expected_generation: int | None = None) -> SessionSnapshot[T]:
        """Atomically clear every dataset-derived state field at once."""

        return self._commit(None, expected_generation)

    def wait_for_change(
        self,
        after_generation: int,
        timeout: float | None = None,
    ) -> SessionSnapshot[T]:
        """Wait until the generation advances, useful for non-Qt observers."""

        if timeout is not None and timeout < 0:
            raise ValueError("timeout cannot be negative")
        deadline = None if timeout is None else monotonic() + timeout
        with self._condition:
            while self._generation <= after_generation:
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("Session state did not change before the timeout.")
                self._condition.wait(remaining)
            return self._snapshot_unlocked()

    def add_listener(
        self,
        listener: Callable[[SessionSnapshot[T]], None],
        *,
        replay_latest: bool = True,
    ) -> None:
        if not callable(listener):
            raise TypeError("listener must be callable")
        with self._condition:
            self._listeners.append(listener)
            if replay_latest:
                self._notifications.append((self._snapshot_unlocked(), (listener,)))
        if replay_latest:
            self._drain_notifications()

    def _commit(
        self,
        value: T | None,
        expected_generation: int | None,
    ) -> SessionSnapshot[T]:
        with self._condition:
            if expected_generation is not None and expected_generation != self._generation:
                raise SessionConflictError(
                    "Session changed while this operation was running; "
                    "stale result was not committed."
                )
            self._value = value
            self._generation += 1
            self._updated_at = datetime.now(timezone.utc)
            snapshot = self._snapshot_unlocked()
            listeners = tuple(self._listeners)
            self._notifications.append((snapshot, listeners))
            self._condition.notify_all()
        self._drain_notifications()
        return snapshot

    def _snapshot_unlocked(self) -> SessionSnapshot[T]:
        return SessionSnapshot(self._value, self._generation, self._updated_at)

    def _drain_notifications(self) -> None:
        """Deliver committed snapshots in generation order without lock callbacks."""

        with self._notification_lock:
            if self._notifying:
                return
            self._notifying = True
            try:
                while True:
                    with self._condition:
                        if not self._notifications:
                            return
                        snapshot, listeners = self._notifications.popleft()
                    for listener in listeners:
                        self._safe_notify(listener, snapshot)
            finally:
                self._notifying = False

    @staticmethod
    def _safe_notify(
        listener: Callable[[SessionSnapshot[T]], None],
        snapshot: SessionSnapshot[T],
    ) -> None:
        try:
            listener(snapshot)
        except Exception:
            # Observers are deliberately isolated from the committed state.
            return
