from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

import numpy as np
import pytest
from PIL import Image

from workbench.domain import AlphaSemantics, ColorSpace
from workbench.errors import (
    DecodeError,
    FormatMismatchError,
    OperationCancelled,
    ResourceLimitError,
)
from workbench.io import LoadLimits, RasterImageLoader, RasterTileCache, RasterTileSource
from workbench.services import ImageService


def _write_multipage_tiff(path: Path, arrays: list[np.ndarray]) -> None:
    pages = [Image.fromarray(array) for array in arrays]
    pages[0].save(path, save_all=True, append_images=pages[1:])


def test_reads_one_tiff_page_and_canonical_tile_without_retaining_a_path() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "runtime-pages.tiff"
        first = np.arange(80, dtype=np.uint16).reshape(8, 10)
        second = (first + 1_000).astype(np.uint16)
        _write_multipage_tiff(path, [first, second])

        source = RasterTileSource(path)
        tile = source.read_tile(1, (3, 2, 8, 6))
        cached = source.read_tile(1, (3, 2, 8, 6))

        assert source.info.frame_count == 2
        assert source.info.access_model == "flat_pages"
        assert source.info.wsi_pyramid is False
        assert tile.dtype == np.dtype("uint16")
        np.testing.assert_array_equal(tile.array, second[2:6, 3:8])
        np.testing.assert_allclose(tile.transform_record.forward([3.0, 2.0]), [0.0, 0.0])
        np.testing.assert_allclose(tile.transform_record.inverse([0.0, 0.0]), [3.0, 2.0])
        assert cached is tile
        assert source.cache_info.hits == 1
        metadata_text = repr(dict(tile.runtime_metadata))
        assert str(path) not in metadata_text
        assert "runtime-pages.tiff" not in metadata_text
        assert not any(isinstance(value, np.ndarray) for value in tile.runtime_metadata.values())


def test_thumbnail_applies_exif_and_tracks_rgba_semantics() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "oriented-rgba.tiff"
        pixels = np.zeros((3, 5, 4), dtype=np.uint8)
        pixels[:, :, 0] = np.arange(5, dtype=np.uint8)
        pixels[:, :, 3] = 127
        image = Image.fromarray(pixels, "RGBA")
        exif = Image.Exif()
        exif[274] = 6
        image.save(path, exif=exif)

        source = RasterTileSource(path)
        preview = source.read_thumbnail(max_size=(2, 2))

        page = source.info.pages[0]
        assert page.raw_shape == (3, 5)
        assert page.canonical_shape == (5, 3)
        assert page.color_space is ColorSpace.RGBA
        assert page.alpha_semantics is AlphaSemantics.STRAIGHT
        assert preview.shape == (2, 1, 4)
        assert preview.color_space is ColorSpace.RGBA
        assert preview.alpha_semantics is AlphaSemantics.STRAIGHT
        steps = preview.runtime_metadata["conversion_records"][0]["steps"]
        assert any(step["operation"] == "exif_orientation" for step in steps)
        assert any(step["operation"] == "thumbnail" for step in steps)
        raw_points = np.array([[0.0, 0.0], [4.0, 2.0]])
        np.testing.assert_allclose(
            preview.transform_record.inverse(preview.transform_record.forward(raw_points)),
            raw_points,
        )


@pytest.mark.parametrize(("format_name", "suffix"), [("TIFF", ".tiff"), ("JPEG", ".jpg")])
@pytest.mark.parametrize(("orientation", "quarter_turns"), [(6, 3), (8, 1)])
def test_square_exif_orientation_moves_pixels_not_just_shape(
    format_name: str,
    suffix: str,
    orientation: int,
    quarter_turns: int,
) -> None:
    """Square inputs catch accidental double/no-op orientation transforms."""

    with TemporaryDirectory() as directory:
        path = Path(directory) / f"square{suffix}"
        raw = np.zeros((16, 16, 3), dtype=np.uint8)
        raw[:8, :8] = (255, 0, 0)
        raw[:8, 8:] = (0, 255, 0)
        raw[8:, :8] = (0, 0, 255)
        raw[8:, 8:] = (255, 255, 0)
        exif = Image.Exif()
        exif[274] = orientation
        save_options: dict[str, object] = {"format": format_name, "exif": exif}
        if format_name == "JPEG":
            save_options.update(quality=100, subsampling=0)
        Image.fromarray(raw, "RGB").save(path, **save_options)

        expected = np.rot90(raw, quarter_turns)
        tiled = RasterTileSource(path).read_page()
        loaded = RasterImageLoader().load(path)

        tolerance = 2 if format_name == "JPEG" else 0
        np.testing.assert_allclose(tiled.array, expected, atol=tolerance)
        np.testing.assert_allclose(loaded.array, expected, atol=tolerance)
        assert tiled.transform_record.operations[0].parameters["orientation"] == orientation
        assert loaded.transform_record.operations[0].parameters["orientation"] == orientation


def test_tile_cache_is_thread_safe_and_strictly_bounded() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "cache.tiff"
        pixels = np.arange(144, dtype=np.uint8).reshape(12, 12)
        Image.fromarray(pixels).save(path)
        cache = RasterTileCache(max_entries=2, max_decoded_bytes=32)
        source = RasterTileSource(path, cache=cache)

        boxes = [(0, 0, 4, 4), (4, 0, 8, 4), (8, 0, 12, 4)]
        for box in boxes:
            source.read_tile(0, box)
        assert cache.info.entries == 2
        assert cache.info.decoded_bytes == 32

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(
                    lambda box: source.read_tile(0, box).array.copy(),
                    [boxes[1], boxes[2]] * 8,
                )
            )
        np.testing.assert_array_equal(results[0], pixels[0:4, 4:8])
        np.testing.assert_array_equal(results[1], pixels[0:4, 8:12])
        assert cache.info.entries <= 2
        assert cache.info.decoded_bytes <= 32
        assert cache.info.hits > 0

        source.close()
        assert cache.info.entries == 0
        assert cache.info.decoded_bytes == 0


def test_signature_dimensions_frames_and_decoded_bytes_are_limited() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "limits.tiff"
        pages = [np.zeros((8, 10), dtype=np.uint8), np.ones((8, 10), dtype=np.uint8)]
        _write_multipage_tiff(path, pages)

        with pytest.raises(ResourceLimitError, match="frames"):
            RasterTileSource(path, limits=LoadLimits(max_frames=1))
        with pytest.raises(ResourceLimitError, match="axis limit"):
            RasterTileSource(path, max_dimension=9)
        with pytest.raises(ResourceLimitError, match="pixels"):
            RasterTileSource(path, limits=LoadLimits(max_pixels=79))

        source = RasterTileSource(path, limits=LoadLimits(max_decoded_bytes=15))
        with pytest.raises(ResourceLimitError, match="decoded raster"):
            source.read_tile(0, (0, 0, 4, 4))
        # The Pillow backend decodes the whole 80-byte page before cropping,
        # so even a 2-by-2 output must respect the full-page peak allocation.
        with pytest.raises(ResourceLimitError, match="decoded raster"):
            source.read_tile(0, (0, 0, 2, 2))
        with pytest.raises(ResourceLimitError, match="decoded raster"):
            source.read_page(0)

        wrong_extension = Path(directory) / "wrong.jpg"
        Image.fromarray(pages[0]).save(wrong_extension, format="TIFF")
        with pytest.raises(FormatMismatchError):
            RasterTileSource(wrong_extension)


def test_page_scan_and_reads_are_cooperatively_cancellable() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "cancel.tiff"
        pages = [np.full((8, 8), value, dtype=np.uint8) for value in range(3)]
        _write_multipage_tiff(path, pages)
        calls = 0

        def cancel_during_scan() -> bool:
            nonlocal calls
            calls += 1
            return calls > 1

        with pytest.raises(OperationCancelled):
            RasterTileSource(path, cancel=cancel_during_scan)

        source = RasterTileSource(path)
        cancelled = Event()
        cancelled.set()
        with pytest.raises(OperationCancelled):
            source.read_thumbnail(cancel=cancelled)


def test_source_replacement_is_detected_before_a_cached_miss() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "mutable.tiff"
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(path)
        source = RasterTileSource(path)
        Image.fromarray(np.zeros((9, 9), dtype=np.uint8)).save(path)

        with pytest.raises(DecodeError, match="changed"):
            source.read_tile(0, (0, 0, 2, 2))


def test_image_service_uses_bounded_thumbnail_for_large_flat_raster() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "large-for-policy.png"
        pixels = np.arange(400, dtype=np.uint16).reshape(20, 20)
        Image.fromarray(pixels).save(path)
        service = ImageService(
            raster_preview_threshold_bytes=64,
            raster_preview_max_size=(5, 5),
        )
        try:
            study = service.begin_load(path).result(timeout=5)
        finally:
            service.close()

        assert study.image.shape == (5, 5)
        assert study.image.runtime_metadata["access"] == "thumbnail"
        assert study.image.runtime_metadata["source_shape"] == (20, 20)
        assert study.image.transform_record.original_shape == (20, 20)
