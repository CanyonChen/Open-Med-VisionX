from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image

from dicom_viewer.domain import ColorSpace, ImageSequence2D, RasterImage2D
from dicom_viewer.errors import DecodeError, FormatMismatchError, ResourceLimitError
from dicom_viewer.io import LoadLimits, RasterImageLoader


class RasterLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loader = RasterImageLoader()

    def test_preserves_16_bit_png_dtype_and_dynamic_range(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "gray16.png"
            source = np.array([[0, 257], [4096, 65535]], dtype=np.uint16)
            Image.fromarray(source).save(path)
            loaded = self.loader.load(path)
        self.assertIsInstance(loaded, RasterImage2D)
        self.assertEqual(loaded.dtype, np.dtype("uint16"))
        np.testing.assert_array_equal(loaded.array, source)

    def test_palette_and_alpha_conversion_is_explicit(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "palette.png"
            image = Image.new("P", (3, 2))
            palette = [0, 0, 0, 255, 0, 0] + [0] * (768 - 6)
            image.putpalette(palette)
            image.putdata([0, 1, 0, 1, 0, 1])
            image.info["transparency"] = 0
            image.save(path, transparency=0)
            loaded = self.loader.load(path)
        self.assertEqual(loaded.color_space, ColorSpace.RGBA)
        self.assertEqual(loaded.shape, (2, 3, 4))
        steps = loaded.runtime_metadata["conversion_records"][0]["steps"]
        self.assertTrue(any(item["operation"] == "palette_expand" for item in steps))

    def test_jpeg_exif_orientation_is_applied_and_reversible(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "oriented.jpg"
            array = np.zeros((2, 4, 3), dtype=np.uint8)
            array[:, :2, 0] = 255
            image = Image.fromarray(array, "RGB")
            exif = Image.Exif()
            exif[274] = 6
            image.save(path, exif=exif, quality=100)
            loaded = self.loader.load(path)
        self.assertEqual(loaded.shape, (4, 2, 3))
        self.assertTrue(loaded.runtime_metadata["lossy_compression"])
        points = np.array([[0.0, 0.0], [3.0, 1.0]])
        np.testing.assert_allclose(
            loaded.transform_record.inverse(loaded.transform_record.forward(points)),
            points,
        )

    def test_multipage_tiff_is_a_sequence_not_volume(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "pages.tiff"
            first = Image.fromarray(np.zeros((3, 4), dtype=np.uint8))
            second = Image.fromarray(np.ones((3, 4), dtype=np.uint8) * 10)
            first.save(path, save_all=True, append_images=[second])
            loaded = self.loader.load(path)
        self.assertIsInstance(loaded, ImageSequence2D)
        self.assertEqual(loaded.shape, (2, 3, 4))

    def test_extension_signature_mismatch_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.jpg"
            Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(path, format="PNG")
            with self.assertRaises(FormatMismatchError):
                self.loader.load(path)

    def test_truncated_supported_file_has_clear_decode_error(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "broken.png"
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"broken")
            with self.assertRaises(DecodeError):
                self.loader.load(path)

    def test_pixel_and_frame_limits_are_enforced(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "large.png"
            Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(path)
            with self.assertRaises(ResourceLimitError):
                self.loader.load(path, limits=LoadLimits(max_pixels=50))

    def test_decoded_byte_budget_is_checked_before_pixel_materialization(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "rgb.png"
            Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8), "RGB").save(path)

            with (
                patch(
                    "dicom_viewer.io.raster.np.asarray",
                    side_effect=AssertionError("array materialization must not occur"),
                ) as asarray,
                patch.object(
                    Image.Image,
                    "copy",
                    side_effect=AssertionError("Pillow decode must not occur"),
                ) as image_copy,
                self.assertRaises(ResourceLimitError),
            ):
                self.loader.load(path, limits=LoadLimits(max_decoded_bytes=35))

            asarray.assert_not_called()
            image_copy.assert_not_called()

    def test_decoded_byte_preflight_includes_all_frames(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "pages.tiff"
            first = Image.fromarray(np.zeros((4, 4), dtype=np.uint8))
            second = Image.fromarray(np.ones((4, 4), dtype=np.uint8))
            first.save(path, save_all=True, append_images=[second])

            with (
                patch(
                    "dicom_viewer.io.raster.np.asarray",
                    side_effect=AssertionError("array materialization must not occur"),
                ) as asarray,
                patch.object(
                    Image.Image,
                    "copy",
                    side_effect=AssertionError("Pillow decode must not occur"),
                ) as image_copy,
                self.assertRaises(ResourceLimitError),
            ):
                self.loader.load(path, limits=LoadLimits(max_decoded_bytes=31))

            asarray.assert_not_called()
            image_copy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
