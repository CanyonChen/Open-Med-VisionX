from __future__ import annotations

import unittest

import numpy as np

from workbench.domain import (
    Capability,
    ImageSequence2D,
    ImageVolume,
    IntensitySemantics,
    RasterImage2D,
    SourceType,
    SpacingSource,
    TransformRecord,
)
from workbench.errors import ValidationError


class ImageDomainTests(unittest.TestCase):
    def test_raster_has_no_fake_medical_geometry(self) -> None:
        image = RasterImage2D(
            np.arange(20, dtype=np.uint16).reshape(4, 5),
            SourceType.RASTER,
            IntensitySemantics.GRAYSCALE,
            runtime_metadata={"PatientName": "must disappear", "format": "PNG"},
            bit_depth=16,
        )
        self.assertEqual(image.dtype, np.dtype("uint16"))
        self.assertNotIn("PatientName", image.runtime_metadata)
        self.assertNotIn(Capability.HU_WINDOWING, image.capabilities)
        self.assertNotIn(Capability.ORTHOGONAL_VIEWS, image.capabilities)
        self.assertNotIn(Capability.PHYSICAL_MEASUREMENT, image.capabilities)
        self.assertIsNone(image.pixel_spacing)

    def test_image_pixels_and_runtime_metadata_are_deeply_immutable(self) -> None:
        source = np.arange(12, dtype=np.uint16).reshape(3, 4)
        image = RasterImage2D(
            source,
            SourceType.RASTER,
            IntensitySemantics.GRAYSCALE,
            runtime_metadata={
                "nested": [{"format": "PNG", "StudyDate": "must disappear"}],
                "source_path": "C:/private/patient/image.png",
            },
            bit_depth=16,
        )
        source[0, 0] = 999
        self.assertEqual(int(image.array[0, 0]), 0)
        self.assertFalse(image.array.flags.writeable)
        self.assertNotIn("source_path", image.runtime_metadata)
        nested = image.runtime_metadata["nested"][0]
        self.assertNotIn("StudyDate", nested)
        with self.assertRaises(TypeError):
            nested["format"] = "JPEG"

    def test_only_user_spacing_enables_raster_physical_measurement(self) -> None:
        image = RasterImage2D(
            np.zeros((4, 5), dtype=np.uint8),
            SourceType.RASTER,
            IntensitySemantics.GRAYSCALE,
        ).with_user_spacing(0.2, 0.3)
        self.assertEqual(image.spacing_source, SpacingSource.USER)
        self.assertEqual(image.pixel_spacing, (0.2, 0.3))
        self.assertIn(Capability.PHYSICAL_MEASUREMENT, image.capabilities)

    def test_sequence_is_not_a_volume(self) -> None:
        sequence = ImageSequence2D(
            np.zeros((3, 4, 5), dtype=np.uint8),
            SourceType.RASTER,
            IntensitySemantics.GRAYSCALE,
        )
        self.assertEqual(sequence.frame_count, 3)
        self.assertIn(Capability.FRAME_PLAYBACK, sequence.capabilities)
        self.assertNotIn(Capability.ORTHOGONAL_VIEWS, sequence.capabilities)
        self.assertNotIn(Capability.PHYSICAL_MEASUREMENT, sequence.capabilities)

    def test_volume_affine_and_hu_capabilities(self) -> None:
        affine = np.diag([0.5, 0.75, 2.0, 1.0])
        affine[:3, 3] = (10.0, 20.0, -30.0)
        volume = ImageVolume(
            np.zeros((2, 3, 4), dtype=np.float32),
            SourceType.DICOM,
            IntensitySemantics.HOUNSFIELD_UNIT,
            affine=affine,
            spacing=(0.5, 0.75, 2.0),
            origin=(10.0, 20.0, -30.0),
            direction=np.eye(3),
            modality="ct",
        )
        world = volume.voxel_xyz_to_world_ras(np.array([2.0, 1.0, 1.0]))
        np.testing.assert_allclose(world, [11.0, 20.75, -28.0])
        np.testing.assert_allclose(volume.world_ras_to_voxel_xyz(world), [2.0, 1.0, 1.0])
        self.assertIn(Capability.HU_WINDOWING, volume.capabilities)
        self.assertIn(Capability.ORTHOGONAL_VIEWS, volume.capabilities)
        self.assertNotIn(Capability.VOLUME_RENDERING, volume.capabilities)

    def test_unknown_and_arbitrary_signals_are_not_quantitative(self) -> None:
        for semantics in (
            IntensitySemantics.UNKNOWN,
            IntensitySemantics.ARBITRARY_SIGNAL,
        ):
            with self.subTest(semantics=semantics):
                image = ImageVolume(
                    np.zeros((2, 3, 4), dtype=np.float32),
                    SourceType.NIFTI,
                    semantics,
                )
                self.assertFalse(image.has_explicit_quantitative_semantics)

        probability = ImageVolume(
            np.zeros((2, 3, 4), dtype=np.float32),
            SourceType.NIFTI,
            IntensitySemantics.PROBABILITY,
        )
        self.assertTrue(probability.has_explicit_quantitative_semantics)

    def test_specific_intensity_semantics_validate_their_value_domain(self) -> None:
        with self.assertRaisesRegex(ValidationError, "Probability images"):
            ImageVolume(
                np.full((2, 3, 4), 1.1 + 0.2j, dtype=np.complex64),
                SourceType.NIFTI,
                IntensitySemantics.PROBABILITY,
            )
        with self.assertRaisesRegex(ValidationError, "Discrete-label images"):
            ImageVolume(
                np.full((2, 3, 4), 0.5, dtype=np.float32),
                SourceType.NIFTI,
                IntensitySemantics.DISCRETE_LABEL,
            )
        with self.assertRaisesRegex(ValidationError, "SUV images"):
            ImageVolume(
                np.full((2, 3, 4), -0.1, dtype=np.float32),
                SourceType.DICOM,
                IntensitySemantics.SUV,
            )
        with self.assertRaisesRegex(ValidationError, "finite real-valued"):
            ImageVolume(
                np.full((2, 3, 4), np.nan, dtype=np.float32),
                SourceType.DICOM,
                IntensitySemantics.HOUNSFIELD_UNIT,
            )


class TransformRecordTests(unittest.TestCase):
    def test_every_exif_orientation_is_reversible(self) -> None:
        points = np.array([[0.0, 0.0], [3.0, 2.0], [1.25, 0.75]])
        for orientation in range(1, 9):
            with self.subTest(orientation=orientation):
                transform = TransformRecord.from_exif_orientation(orientation, (3, 4))
                np.testing.assert_allclose(transform.inverse(transform.forward(points)), points)

    def test_resize_crop_letterbox_composition_round_trip(self) -> None:
        transform = (
            TransformRecord.crop((100, 200), left=20, top=10, width=120, height=80)
            .then(TransformRecord.resize((80, 120), (40, 60)))
            .then(TransformRecord.letterbox((40, 60), (64, 64)))
        )
        points = np.array([[20.0, 10.0], [139.0, 89.0], [75.0, 55.0]])
        np.testing.assert_allclose(transform.inverse(transform.forward(points)), points)
        self.assertEqual(
            [item.name for item in transform.operations], ["crop", "resize", "letterbox"]
        )

    def test_boxes_remain_axis_aligned_after_orientation(self) -> None:
        transform = TransformRecord.from_exif_orientation(6, (10, 20))
        mapped = transform.forward_boxes(np.array([2.0, 3.0, 7.0, 8.0]))
        np.testing.assert_allclose(mapped, [1.0, 2.0, 6.0, 7.0])


if __name__ == "__main__":
    unittest.main()
