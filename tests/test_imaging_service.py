from __future__ import annotations

from pathlib import Path
from threading import Event

import numpy as np
import pytest

from workbench.domain import (
    ImageStudy,
    ImageVolume,
    IntensitySemantics,
    RasterImage2D,
    SourceType,
    VolumeLayer,
)
from workbench.errors import DecodeError, OperationCancelled
from workbench.io import ImageLoader, ImageLoaderRegistry, LoadLimits, ProbeResult
from workbench.services import ImageService


class _ControlledLoader(ImageLoader):
    name = "controlled"

    def __init__(self) -> None:
        self.slow_started = Event()
        self.release_slow = Event()

    def probe(self, source: str | Path) -> ProbeResult:
        return ProbeResult(True, "CONTROLLED", 100)

    def load(
        self,
        source: str | Path,
        *,
        limits: LoadLimits | None = None,
        cancel=None,
    ) -> RasterImage2D:
        del limits
        name = Path(source).name
        if name == "bad.controlled":
            raise DecodeError("deliberate decode failure")
        if name.startswith("slow"):
            self.slow_started.set()
            if not self.release_slow.wait(timeout=5):
                raise TimeoutError("test did not release controlled loader")
        value = {
            "committed.controlled": 1,
            "replacement.controlled": 2,
            "slow.controlled": 3,
            "slow-stale.controlled": 4,
            "listener.controlled": 5,
        }.get(name, 0)
        return RasterImage2D(
            np.full((3, 3), value, dtype=np.uint8),
            SourceType.RASTER,
            IntensitySemantics.GRAYSCALE,
        )


class _NiftiLikeLoader(ImageLoader):
    name = "nifti-like"

    def probe(self, source: str | Path) -> ProbeResult:
        return ProbeResult(True, "NIFTI", 100)

    def load(
        self,
        source: str | Path,
        *,
        limits: LoadLimits | None = None,
        cancel=None,
    ) -> ImageVolume:
        del source, limits, cancel
        return ImageVolume(
            np.arange(24, dtype=np.int16).reshape(2, 3, 4),
            SourceType.NIFTI,
            IntensitySemantics.ARBITRARY_SIGNAL,
            affine=np.eye(4),
            modality="MR",
        )


@pytest.fixture
def controlled_service():
    loader = _ControlledLoader()
    service = ImageService(registry=ImageLoaderRegistry((loader,)))
    try:
        yield service, loader
    finally:
        loader.release_slow.set()
        service.close()


def test_failed_pending_load_preserves_the_committed_study(controlled_service) -> None:
    service, _loader = controlled_service
    committed = service.begin_load("committed.controlled").result(timeout=5)
    before = service.state.snapshot()

    pending = service.begin_load("bad.controlled")

    with pytest.raises(DecodeError, match="deliberate"):
        pending.result(timeout=5)
    after = service.state.snapshot()
    assert after.value is committed
    assert after.generation == before.generation


def test_cancelled_pending_load_preserves_the_committed_study(controlled_service) -> None:
    service, loader = controlled_service
    committed = service.begin_load("committed.controlled").result(timeout=5)
    before = service.state.snapshot()

    pending = service.begin_load("slow.controlled")
    assert loader.slow_started.wait(timeout=5)
    during = service.state.snapshot()
    assert service.pending_load is pending
    assert service.committed_study is committed
    assert during.value is committed
    assert during.generation == before.generation
    assert service.cancel_active()
    loader.release_slow.set()

    with pytest.raises(OperationCancelled):
        pending.result(timeout=5)
    assert service.pending_load is None
    after = service.state.snapshot()
    assert after.value is committed
    assert after.generation == before.generation


def test_superseded_late_load_cannot_replace_a_newer_commit(controlled_service) -> None:
    service, loader = controlled_service
    service.begin_load("committed.controlled").result(timeout=5)

    stale = service.begin_load("slow-stale.controlled")
    assert loader.slow_started.wait(timeout=5)
    replacement = service.begin_load("replacement.controlled").result(timeout=5)
    loader.release_slow.set()

    with pytest.raises(OperationCancelled):
        stale.result(timeout=5)
    assert service.state.value is replacement
    assert int(replacement.image.array[0, 0]) == 2


def test_reentrant_listener_can_start_the_next_load_without_cancelling_commit(
    controlled_service,
) -> None:
    service, _loader = controlled_service
    service.begin_load("committed.controlled").result(timeout=5)
    spawned = []

    def start_next_load(snapshot) -> None:
        if int(snapshot.value.image.array[0, 0]) == 2 and not spawned:
            spawned.append(service.begin_load("listener.controlled"))

    service.state.add_listener(start_next_load, replay_latest=False)
    replacement = service.begin_load("replacement.controlled").result(timeout=5)
    listener_replacement = spawned[0].result(timeout=5)

    assert int(replacement.image.array[0, 0]) == 2
    assert int(listener_replacement.image.array[0, 0]) == 5
    assert service.state.value is listener_replacement


def test_nifti_volume_receives_a_path_free_study_and_base_layer() -> None:
    service = ImageService(registry=ImageLoaderRegistry((_NiftiLikeLoader(),)))
    try:
        loaded = service.begin_load("volume.fake", prepare_axis_aligned_volume=True).result(
            timeout=5
        )
    finally:
        service.close()

    assert isinstance(loaded.domain_study, ImageStudy)
    series = loaded.reference_series
    assert series is not None
    assert series.series_id.startswith("sha256:")
    assert series.series_instance_uid is None
    assert series.source.content_sha256 is not None
    assert len(series.layers) == 1
    assert isinstance(series.layers[0], VolumeLayer)
    assert series.layers[0].volume is loaded.display_image
    assert "volume.fake" not in repr(loaded.domain_study)


def _block_replacement_notification(service: ImageService) -> tuple[Event, Event]:
    committed = Event()
    release = Event()

    def listener(snapshot) -> None:
        if int(snapshot.value.image.array[0, 0]) == 2:
            committed.set()
            release.wait(timeout=5)

    service.state.add_listener(listener, replay_latest=False)
    return committed, release


def test_direct_cancel_is_too_late_after_atomic_commit(controlled_service) -> None:
    service, _loader = controlled_service
    old = service.begin_load("committed.controlled").result(timeout=5)
    before = service.state.snapshot()
    committed, release = _block_replacement_notification(service)

    task = service.begin_load("replacement.controlled")
    assert committed.wait(timeout=5)
    assert not task.done
    assert service.state.value is not old
    try:
        assert not task.cancel()
    finally:
        release.set()

    replacement = task.result(timeout=5)
    after = service.state.snapshot()
    assert after.value is replacement
    assert after.generation == before.generation + 1


def test_close_does_not_cancel_a_load_past_the_atomic_commit_boundary(
    controlled_service,
) -> None:
    service, _loader = controlled_service
    old = service.begin_load("committed.controlled").result(timeout=5)
    committed, release = _block_replacement_notification(service)

    task = service.begin_load("replacement.controlled")
    assert committed.wait(timeout=5)
    try:
        service.close()
    finally:
        release.set()

    replacement = task.result(timeout=5)
    assert replacement is service.state.value
    assert replacement is not old
