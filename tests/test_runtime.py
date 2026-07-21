from __future__ import annotations

import io
import logging
from threading import Event

import pytest

from dicom_viewer.errors import OperationCancelled, ValidationError
from dicom_viewer.runtime import (
    REDACTED,
    AtomicSessionState,
    CredentialReference,
    CredentialResolutionError,
    CredentialResolver,
    SessionConflictError,
    TaskRunner,
    TaskStatus,
    install_redaction,
    redact,
    redact_text,
)


def test_task_runner_reports_progress_and_returns_result() -> None:
    observations: list[float] = []

    def work(context, value: int) -> int:
        context.report_progress(current=1, total=4, message="decoded")
        context.report_progress(0.75, message="validated")
        return value * 2

    with TaskRunner(max_workers=1) as runner:
        task = runner.submit(work, 21)
        task.add_progress_callback(lambda progress: observations.append(progress.fraction))
        assert task.result(timeout=2) == 42

    assert task.status is TaskStatus.SUCCEEDED
    assert task.done
    assert task.progress().fraction == 1.0
    assert observations == sorted(observations)
    assert observations[-1] == 1.0


def test_task_progress_count_validation_preserves_public_error() -> None:
    def work(context) -> None:
        context.report_progress(current=0, total=0)

    with TaskRunner(max_workers=1) as runner:
        task = runner.submit(work)
        with pytest.raises(ValidationError, match="total must be positive"):
            task.result(timeout=2)


def test_task_runner_cancels_cooperatively() -> None:
    started = Event()

    def work(context) -> None:
        started.set()
        context.token.wait(2)
        context.raise_if_cancelled()

    with TaskRunner(max_workers=1) as runner:
        task = runner.submit(work)
        assert started.wait(1)
        assert task.cancel()
        with pytest.raises(OperationCancelled):
            task.result(timeout=2)

    assert task.status is TaskStatus.CANCELLED
    assert task.cancelled
    assert not task.cancel()


def test_task_failure_is_preserved_for_result() -> None:
    problem = RuntimeError("algorithm failed")

    def work(context) -> None:
        context.raise_if_cancelled()
        raise problem

    with TaskRunner(max_workers=1) as runner:
        task = runner.submit(work)
        with pytest.raises(RuntimeError, match="algorithm failed") as captured:
            task.result(timeout=2)

    assert captured.value is problem
    assert task.status is TaskStatus.FAILED


def test_atomic_session_replacement_reset_and_stale_write_protection() -> None:
    state = AtomicSessionState[tuple[str, ...]]()
    initial = state.snapshot()
    loaded = state.replace(("volume-a",), expected_generation=initial.generation)

    assert loaded.value == ("volume-a",)
    assert loaded.generation == 1
    with pytest.raises(SessionConflictError):
        state.replace(("stale-volume",), expected_generation=initial.generation)

    cleared = state.reset(expected_generation=loaded.generation)
    assert cleared.is_empty
    assert cleared.generation == 2


def test_progress_callbacks_are_serialized_during_reentrant_reports() -> None:
    worker_ready = Event()
    callbacks_ready = Event()
    context_box = {}
    first_observations: list[float] = []
    second_observations: list[float] = []

    def work(context) -> None:
        context_box["context"] = context
        worker_ready.set()
        assert callbacks_ready.wait(1)
        context.report_progress(0.25)

    def first_callback(progress) -> None:
        first_observations.append(progress.fraction)
        if progress.fraction == 0.25:
            context_box["context"].report_progress(0.5)

    def second_callback(progress) -> None:
        second_observations.append(progress.fraction)

    with TaskRunner(max_workers=1) as runner:
        task = runner.submit(work)
        assert worker_ready.wait(1)
        task.add_progress_callback(first_callback, replay_latest=False)
        task.add_progress_callback(second_callback, replay_latest=False)
        callbacks_ready.set()
        task.result(timeout=2)

    assert first_observations == [0.25, 0.5, 1.0]
    assert second_observations == [0.25, 0.5, 1.0]


def test_session_listeners_preserve_generation_order_during_reentrant_commit() -> None:
    state = AtomicSessionState[str]()
    observations: list[tuple[str, int]] = []

    def first_listener(snapshot) -> None:
        observations.append(("first", snapshot.generation))
        if snapshot.generation == 1:
            state.replace("nested", expected_generation=1)

    def second_listener(snapshot) -> None:
        observations.append(("second", snapshot.generation))

    state.add_listener(first_listener, replay_latest=False)
    state.add_listener(second_listener, replay_latest=False)
    state.replace("outer", expected_generation=0)

    assert observations == [
        ("first", 1),
        ("second", 1),
        ("first", 2),
        ("second", 2),
    ]


def test_credential_references_never_accept_or_represent_raw_keys() -> None:
    resolver = CredentialResolver(environment={"OPENAI_API_KEY": "secret-value"})
    reference = CredentialReference.parse("env:OPENAI_API_KEY")

    assert resolver.resolve(reference) == "secret-value"
    assert "secret-value" not in repr(reference)
    with pytest.raises(ValidationError):
        CredentialReference.parse("secret-value")
    with pytest.raises(CredentialResolutionError):
        CredentialResolver(environment={}).resolve(reference)


def test_optional_keyring_resolution_is_injectable() -> None:
    calls: list[tuple[str, str]] = []

    def getter(service: str, username: str) -> str:
        calls.append((service, username))
        return "keyring-secret"

    resolver = CredentialResolver(keyring_getter=getter)
    assert resolver.resolve("keyring:openmedvisionx/alice") == "keyring-secret"
    assert calls == [("openmedvisionx", "alice")]


def test_redacting_filter_removes_credentials_phi_and_exception_text() -> None:
    output = io.StringIO()
    logger = logging.getLogger("openmedvisionx.tests.redaction")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(output)
    logger.addHandler(handler)
    install_redaction(logger)
    second_handler = logging.StreamHandler(output)
    logger.addHandler(second_handler)
    install_redaction(logger)
    assert any(type(item).__name__ == "RedactingFilter" for item in second_handler.filters)

    try:
        raise RuntimeError("Authorization: Bearer bearer-secret")
    except RuntimeError:
        logger.exception("api_key=%s PatientName: Alice", "sk-test-secret-value")

    rendered = output.getvalue()
    assert "bearer-secret" not in rendered
    assert "sk-test-secret-value" not in rendered
    assert "Alice" not in rendered
    assert REDACTED in rendered

    structured = redact({"patient_id": "123", "max_tokens": 100, "data": b"pixels"})
    assert structured["patient_id"] == REDACTED
    assert structured["max_tokens"] == 100
    assert structured["data"] == "<binary:6 bytes>"
    assert "Alice" not in redact_text("(0010, 0010) Patient's Name PN: [Alice]")
    assert "Alice Smith" not in redact_text(r"C:\Patients\Alice Smith\scan.dcm")
