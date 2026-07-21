"""Thread-safe, cooperatively cancellable background tasks.

Worker functions submitted through :class:`TaskRunner` receive a
:class:`TaskContext` as their first argument.  Long-running code must call
``context.raise_if_cancelled()`` at useful boundaries and may report progress
through ``context.report_progress(...)``.  Cancellation never kills a thread
and therefore cannot leave native imaging libraries in an undefined state.
"""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from threading import Condition, Event, RLock
from time import monotonic
from typing import Any, Generic, TypeVar, cast

from ..errors import OperationCancelled, ValidationError

T = TypeVar("T")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TaskStatus(str, Enum):
    """Lifecycle states for :class:`BackgroundTask`."""

    PENDING = "pending"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TERMINAL_STATES = {
    TaskStatus.SUCCEEDED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
}


@dataclass(frozen=True, slots=True)
class TaskProgress:
    """An immutable progress snapshot.

    ``fraction`` is always in ``[0, 1]`` and is monotonic for a task.  Optional
    ``current`` and ``total`` values are useful for displaying exact counts.
    """

    fraction: float = 0.0
    message: str = ""
    current: int | None = None
    total: int | None = None
    updated_at: datetime = field(default_factory=_utc_now)

    def __init__(
        self,
        fraction: float = 0.0,
        message: str = "",
        current: int | None = None,
        total: int | None = None,
        updated_at: datetime | None = None,
    ) -> None:
        fraction = float(fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValidationError("Task progress fraction must be between 0 and 1.")
        if current is not None and current < 0:
            raise ValidationError("Task progress current value cannot be negative.")
        if total is not None and total <= 0:
            raise ValidationError("Task progress total must be positive.")
        if current is not None and total is not None and current > total:
            raise ValidationError("Task progress current value cannot exceed total.")
        object.__setattr__(self, "fraction", fraction)
        object.__setattr__(self, "message", str(message))
        object.__setattr__(self, "current", current)
        object.__setattr__(self, "total", total)
        object.__setattr__(self, "updated_at", updated_at or _utc_now())


class CancellationToken:
    """A small thread-safe cooperative cancellation primitive."""

    def __init__(self) -> None:
        self._event = Event()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait until cancellation is requested, returning whether it occurred."""

        return self._event.wait(timeout)

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise OperationCancelled("Background operation was cancelled.")

    def _cancel(self) -> None:
        self._event.set()


class TaskContext:
    """The only control surface passed from a runner into worker code."""

    def __init__(
        self,
        token: CancellationToken,
        progress_reporter: Callable[[TaskProgress], None],
    ) -> None:
        self._token = token
        self._progress_reporter = progress_reporter

    @property
    def token(self) -> CancellationToken:
        return self._token

    @property
    def cancelled(self) -> bool:
        return self._token.is_cancelled

    def raise_if_cancelled(self) -> None:
        self._token.raise_if_cancelled()

    check_cancelled = raise_if_cancelled

    def report_progress(
        self,
        fraction: float | None = None,
        *,
        message: str = "",
        current: int | None = None,
        total: int | None = None,
    ) -> None:
        """Publish progress, deriving the fraction from counts when requested."""

        self.raise_if_cancelled()
        if fraction is None:
            if current is None or total is None:
                raise ValidationError(
                    "Provide either a progress fraction or both current and total."
                )
            if total <= 0:
                raise ValidationError("Task progress total must be positive.")
            fraction = current / total
        self._progress_reporter(
            TaskProgress(
                fraction=fraction,
                message=message,
                current=current,
                total=total,
            )
        )


class BackgroundTask(Generic[T]):
    """A stable task handle safe to query from GUI and worker threads."""

    def __init__(self) -> None:
        self._condition = Condition(RLock())
        self._status = TaskStatus.PENDING
        self._progress = TaskProgress()
        self._token = CancellationToken()
        self._future: Future[Any] | None = None
        self._result: T | None = None
        self._exception: BaseException | None = None
        self._started_at: datetime | None = None
        self._finished_at: datetime | None = None
        self._progress_callbacks: list[Callable[[TaskProgress], None]] = []
        self._progress_notifications: deque[
            tuple[TaskProgress, tuple[Callable[[TaskProgress], None], ...]]
        ] = deque()
        self._progress_dispatch_lock = RLock()
        self._progress_dispatching = False
        self._done_callbacks: list[Callable[[BackgroundTask[T]], None]] = []

    @property
    def status(self) -> TaskStatus:
        with self._condition:
            return self._status

    @property
    def cancellation_token(self) -> CancellationToken:
        return self._token

    @property
    def started_at(self) -> datetime | None:
        with self._condition:
            return self._started_at

    @property
    def finished_at(self) -> datetime | None:
        with self._condition:
            return self._finished_at

    @property
    def done(self) -> bool:
        with self._condition:
            return self._status in _TERMINAL_STATES

    @property
    def running(self) -> bool:
        with self._condition:
            return self._status in {TaskStatus.RUNNING, TaskStatus.CANCELLING}

    @property
    def cancelled(self) -> bool:
        with self._condition:
            return self._status is TaskStatus.CANCELLED

    def progress(self) -> TaskProgress:
        """Return the latest immutable progress snapshot."""

        with self._condition:
            return self._progress

    @property
    def progress_snapshot(self) -> TaskProgress:
        return self.progress()

    def cancel(self) -> bool:
        """Request cooperative cancellation.

        Returns ``False`` only when the task had already reached a terminal
        state.  Pending work is cancelled immediately; running work observes
        the request through its :class:`CancellationToken`.
        """

        callbacks: tuple[Callable[[BackgroundTask[T]], None], ...] = ()
        future: Future[Any] | None
        with self._condition:
            if self._status in _TERMINAL_STATES:
                return False
            self._token._cancel()
            future = self._future
            if self._status is TaskStatus.PENDING:
                self._status = TaskStatus.CANCELLED
                self._finished_at = _utc_now()
                callbacks = tuple(self._done_callbacks)
                self._done_callbacks.clear()
                self._progress_callbacks.clear()
                self._condition.notify_all()
            elif self._status is TaskStatus.RUNNING:
                self._status = TaskStatus.CANCELLING
                self._condition.notify_all()
        if future is not None:
            future.cancel()
        self._notify_done(callbacks)
        return True

    def result(self, timeout: float | None = None) -> T:
        """Wait for completion and return the result or raise its terminal error."""

        if timeout is not None and timeout < 0:
            raise ValueError("timeout cannot be negative")
        deadline = None if timeout is None else monotonic() + timeout
        with self._condition:
            while self._status not in _TERMINAL_STATES:
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("Background task did not finish before the timeout.")
                self._condition.wait(remaining)
            if self._status is TaskStatus.CANCELLED:
                raise OperationCancelled("Background operation was cancelled.")
            if self._status is TaskStatus.FAILED:
                assert self._exception is not None
                raise self._exception
            return cast(T, self._result)

    def exception(self, timeout: float | None = None) -> BaseException | None:
        """Wait for completion and return the worker exception, if any."""

        try:
            self.result(timeout)
        except OperationCancelled:
            return OperationCancelled("Background operation was cancelled.")
        except BaseException as error:  # noqa: BLE001 - this is an inspection API
            return error
        return None

    def add_progress_callback(
        self,
        callback: Callable[[TaskProgress], None],
        *,
        replay_latest: bool = True,
    ) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._condition:
            if self._status not in _TERMINAL_STATES:
                self._progress_callbacks.append(callback)
            if replay_latest:
                self._progress_notifications.append((self._progress, (callback,)))
        if replay_latest:
            self._drain_progress_notifications()

    def add_done_callback(self, callback: Callable[[BackgroundTask[T]], None]) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._condition:
            if self._status in _TERMINAL_STATES:
                call_now = True
            else:
                self._done_callbacks.append(callback)
                call_now = False
        if call_now:
            self._safe_call(callback, self)

    def _bind_future(self, future: Future[Any]) -> None:
        with self._condition:
            self._future = future
            cancelled = self._status is TaskStatus.CANCELLED
        if cancelled:
            future.cancel()

    def _start(self) -> bool:
        with self._condition:
            if self._status is not TaskStatus.PENDING:
                return False
            if self._token.is_cancelled:
                self._status = TaskStatus.CANCELLED
                self._finished_at = _utc_now()
                self._condition.notify_all()
                return False
            self._status = TaskStatus.RUNNING
            self._started_at = _utc_now()
            self._condition.notify_all()
            return True

    def _report_progress(self, progress: TaskProgress) -> None:
        with self._condition:
            if self._status is not TaskStatus.RUNNING:
                return
            if progress.fraction < self._progress.fraction:
                progress = replace(progress, fraction=self._progress.fraction)
            self._progress = progress
            callbacks = tuple(self._progress_callbacks)
            self._progress_notifications.append((progress, callbacks))
            self._condition.notify_all()
        self._drain_progress_notifications()

    def _succeed(self, value: T) -> None:
        progress_callbacks: tuple[Callable[[TaskProgress], None], ...] = ()
        with self._condition:
            if self._status in _TERMINAL_STATES:
                return
            if self._token.is_cancelled:
                self._status = TaskStatus.CANCELLED
            else:
                self._status = TaskStatus.SUCCEEDED
                self._result = value
                previous = self._progress
                final_current = previous.total if previous.total is not None else previous.current
                self._progress = TaskProgress(
                    1.0,
                    previous.message,
                    current=final_current,
                    total=previous.total,
                )
                progress_callbacks = tuple(self._progress_callbacks)
                self._progress_notifications.append((self._progress, progress_callbacks))
            self._finished_at = _utc_now()
            callbacks = tuple(self._done_callbacks)
            self._progress_callbacks.clear()
            self._done_callbacks.clear()
            self._condition.notify_all()
        self._drain_progress_notifications()
        self._notify_done(callbacks)

    def _fail(self, error: BaseException) -> None:
        with self._condition:
            if self._status in _TERMINAL_STATES:
                return
            if isinstance(error, OperationCancelled) or self._token.is_cancelled:
                self._status = TaskStatus.CANCELLED
            else:
                self._status = TaskStatus.FAILED
                self._exception = error
            self._finished_at = _utc_now()
            callbacks = tuple(self._done_callbacks)
            self._progress_callbacks.clear()
            self._done_callbacks.clear()
            self._condition.notify_all()
        self._notify_done(callbacks)

    def _notify_done(
        self,
        callbacks: tuple[Callable[[BackgroundTask[T]], None], ...],
    ) -> None:
        for callback in callbacks:
            self._safe_call(callback, self)

    def _drain_progress_notifications(self) -> None:
        """Deliver snapshots in commit order, including reentrant reports."""

        with self._progress_dispatch_lock:
            if self._progress_dispatching:
                return
            self._progress_dispatching = True
            try:
                while True:
                    with self._condition:
                        if not self._progress_notifications:
                            return
                        progress, callbacks = self._progress_notifications.popleft()
                    for callback in callbacks:
                        self._safe_call(callback, progress)
            finally:
                self._progress_dispatching = False

    @staticmethod
    def _safe_call(callback: Callable[[Any], None], value: Any) -> None:
        try:
            callback(value)
        except Exception:
            # A UI observer must never be able to corrupt task state.
            return


class TaskRunner:
    """Own a bounded thread pool and create :class:`BackgroundTask` handles."""

    def __init__(self, max_workers: int | None = None, *, thread_name_prefix: str = "viewer"):
        if max_workers is None:
            max_workers = min(4, max(1, os.cpu_count() or 1))
        if max_workers <= 0:
            raise ValidationError("max_workers must be positive.")
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._lock = RLock()
        self._closed = False
        self._tasks: set[BackgroundTask[Any]] = set()

    def submit(
        self,
        operation: Callable[..., T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> BackgroundTask[T]:
        """Run ``operation(context, *args, **kwargs)`` on the bounded pool."""

        if not callable(operation):
            raise TypeError("operation must be callable")
        task: BackgroundTask[T] = BackgroundTask()
        with self._lock:
            if self._closed:
                raise RuntimeError("TaskRunner has been shut down.")
            self._tasks.add(task)
            future = self._executor.submit(self._execute, task, operation, args, kwargs)
            task._bind_future(future)
            task.add_done_callback(self._discard_task)
        return task

    @staticmethod
    def _execute(
        task: BackgroundTask[T],
        operation: Callable[..., T],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        if not task._start():
            return
        context = TaskContext(task.cancellation_token, task._report_progress)
        try:
            context.raise_if_cancelled()
            value = operation(context, *args, **kwargs)
            context.raise_if_cancelled()
        except BaseException as error:  # noqa: BLE001 - worker failures belong to the task
            task._fail(error)
        else:
            task._succeed(value)

    def _discard_task(self, task: BackgroundTask[Any]) -> None:
        with self._lock:
            self._tasks.discard(task)

    def shutdown(self, *, wait: bool = True, cancel_pending: bool = False) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            tasks = tuple(self._tasks)
        if cancel_pending:
            for task in tasks:
                task.cancel()
        self._executor.shutdown(wait=wait, cancel_futures=cancel_pending)

    def __enter__(self) -> TaskRunner:
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown(wait=True, cancel_pending=True)
