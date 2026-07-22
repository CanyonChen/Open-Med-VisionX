from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from workbench.algorithms import (
    CTAttenuationPhantom,
    DirectFourierReconstruction,
    FilteredBackProjection,
    ReconstructionRequest,
    ReconstructionSourceKind,
    SARTReconstruction,
    SinogramResult,
    compute_metrics,
    generate_angles,
    generate_ct_phantom,
    generate_sinogram,
)
from workbench.errors import ValidationError

HAS_SKIMAGE = importlib.util.find_spec("skimage") is not None


class AngleTests(unittest.TestCase):
    def test_angle_ranges_are_endpoint_exclusive(self) -> None:
        np.testing.assert_allclose(generate_angles(4, 180), [0, 45, 90, 135])
        np.testing.assert_allclose(generate_angles(4, 360), [0, 90, 180, 270])


class ReconstructionValueTests(unittest.TestCase):
    def test_sinogram_result_copies_and_freezes_geometry_and_intermediates(self) -> None:
        sinogram = np.arange(12, dtype=np.float64).reshape(3, 4)
        theta = np.arange(4, dtype=np.float64)
        partial = sinogram[:, :2]

        result = SinogramResult(
            sinogram,
            theta,
            True,
            {"partial": partial},
            ReconstructionSourceKind.IMAGE_DERIVED_SIMULATION,
        )
        sinogram[0, 0] = 999.0
        theta[0] = 999.0

        self.assertEqual(result.sinogram[0, 0], 0.0)
        self.assertEqual(result.theta_degrees[0], 0.0)
        self.assertFalse(result.sinogram.flags.writeable)
        self.assertFalse(result.theta_degrees.flags.writeable)
        self.assertFalse(result.intermediate["partial"].flags.writeable)

    def test_unknown_or_cross_domain_source_kinds_are_rejected_explicitly(self) -> None:
        sinogram = np.zeros((3, 4))
        theta = np.arange(4)
        with self.assertRaisesRegex(ValidationError, "Unknown reconstruction source_kind"):
            ReconstructionRequest(
                sinogram,
                theta,
                3,
                source_kind={"not": "hashable"},  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValidationError, "cannot carry a k-space"):
            SinogramResult(
                sinogram,
                theta,
                True,
                source_kind=ReconstructionSourceKind.SIMULATED_KSPACE,
            )

    def test_phantom_contract_rejects_nonzero_values_outside_circle(self) -> None:
        invalid = np.zeros((16, 16), dtype=np.float64)
        invalid[0, 0] = 0.01
        with self.assertRaisesRegex(ValidationError, "outside circular support"):
            CTAttenuationPhantom(invalid, (0.0, 0.03))

    def test_metric_contract_rejects_malformed_fixed_ranges_and_fractional_rois(self) -> None:
        reference = np.arange(64, dtype=np.float64).reshape(8, 8)
        with self.assertRaisesRegex(ValidationError, "exactly two"):
            compute_metrics(reference, reference, intensity_range=(0.0, 1.0, 2.0))
        with self.assertRaisesRegex(ValidationError, "integer coordinates"):
            compute_metrics(reference, reference, rois={"bad": (0.5, 0, 3, 3)})


@unittest.skipUnless(HAS_SKIMAGE, "scikit-image is an optional test dependency")
class ReconstructionTests(unittest.TestCase):
    @staticmethod
    def phantom(size: int = 24) -> np.ndarray:
        image = np.zeros((size, size), dtype=np.float64)
        image[6:18, 8:16] = 1.0
        image[9:13, 4:20] = 0.5
        y, x = np.ogrid[:size, :size]
        center = (size - 1) / 2.0
        image[(x - center) ** 2 + (y - center) ** 2 > center**2] = 0.0
        return image

    def test_all_algorithms_share_output_circle_and_size(self) -> None:
        phantom = self.phantom()
        sino = generate_sinogram(phantom, projection_count=30, circle=True)
        request = ReconstructionRequest(sino.sinogram, sino.theta_degrees, 24, circle=True)
        for algorithm in (
            DirectFourierReconstruction("linear"),
            FilteredBackProjection("ramp"),
            SARTReconstruction(iterations=1),
        ):
            with self.subTest(algorithm=algorithm.name):
                result = algorithm.reconstruct(request)
                self.assertEqual(result.image.shape, (24, 24))
                self.assertEqual(result.metadata["circle"], True)

    def test_dfr_interpolation_option_controls_execution(self) -> None:
        phantom = self.phantom(16)
        sino = generate_sinogram(phantom, projection_count=18)
        request = ReconstructionRequest(sino.sinogram, sino.theta_degrees, 16)
        nearest = DirectFourierReconstruction("nearest").reconstruct(request)
        linear = DirectFourierReconstruction("linear").reconstruct(request)
        self.assertEqual(nearest.metadata["interpolation"], "nearest")
        self.assertEqual(linear.metadata["interpolation"], "linear")
        self.assertFalse(np.allclose(nearest.image, linear.image))

    def test_360_degree_parallel_data_is_folded_as_redundant(self) -> None:
        phantom = self.phantom(16)
        sino = generate_sinogram(phantom, projection_count=36, angle_range=360)
        request = ReconstructionRequest(sino.sinogram, sino.theta_degrees, 16)
        result = FilteredBackProjection().reconstruct(request)
        self.assertTrue(result.metadata["folded_360_redundancy"])
        self.assertEqual(result.metadata["unique_angles"], 18)

    def test_metrics_use_reference_range_raw_errors_and_roi(self) -> None:
        reference = np.arange(64, dtype=float).reshape(8, 8) - 1024.0
        reconstruction = reference + 2.0
        report = compute_metrics(
            reference,
            reconstruction,
            rois={"center": (2, 2, 4, 4)},
            unit="HU",
        )
        self.assertEqual(report.intensity_range, (-1024.0, -961.0))
        self.assertEqual(report.evaluation_range, report.intensity_range)
        self.assertEqual(report.evaluation_range_source, "reference")
        self.assertIn("center", report.roi)
        self.assertIn("center", report.raw_roi)
        self.assertGreater(report.values.mse, 0.0)
        np.testing.assert_allclose(report.difference, 2.0 / 63.0)
        self.assertIsNotNone(report.raw_values)
        self.assertEqual(report.raw_values.mae, 2.0)
        self.assertEqual(report.raw_values.rmse, 2.0)
        self.assertEqual(report.raw_values.bias, 2.0)
        self.assertEqual(report.raw_values.unit, "HU")
        self.assertFalse(report.normalized_reference.flags.writeable)
        self.assertFalse(report.difference.flags.writeable)

    def test_reconstruction_outlier_cannot_expand_implicit_evaluation_range(self) -> None:
        reference = np.linspace(0.0, 1.0, 64).reshape(8, 8)
        reconstruction = reference.copy()
        reconstruction[0, 0] = 100.0

        report = compute_metrics(reference, reconstruction)

        self.assertEqual(report.intensity_range, (0.0, 1.0))
        self.assertEqual(report.normalized_reconstruction[0, 0], 100.0)
        self.assertGreater(report.values.mse, 100.0)
        self.assertGreater(report.raw_values.rmse, 1.0)

    def test_default_ct_phantom_is_valid_attenuation_with_zero_background(self) -> None:
        phantom = generate_ct_phantom(32)

        self.assertEqual(phantom.array.shape, (32, 32))
        self.assertEqual(phantom.unit, "mm^-1")
        self.assertEqual(phantom.evaluation_range, (0.0, 0.03))
        self.assertGreater(float(phantom.array.max()), 0.0)
        self.assertGreaterEqual(float(phantom.array.min()), 0.0)
        self.assertEqual(float(phantom.array[0, 0]), 0.0)
        self.assertFalse(phantom.array.flags.writeable)

    def test_ct_radon_rejects_negative_or_nonzero_circle_background(self) -> None:
        with self.assertRaisesRegex(ValidationError, "complex-valued"):
            generate_sinogram(self.phantom().astype(np.complex128) * (1.0 + 1.0j))

        negative = self.phantom()
        negative[12, 12] = -1.0
        with self.assertRaisesRegex(ValidationError, "nonnegative attenuation"):
            generate_sinogram(negative)

        outside = self.phantom()
        outside[0, 0] = 1.0
        with self.assertRaisesRegex(ValidationError, "exact zero attenuation"):
            generate_sinogram(outside, circle=True)

    def test_source_kind_is_preserved_from_sinogram_through_reconstruction(self) -> None:
        phantom = generate_ct_phantom(24)
        sino = generate_sinogram(
            phantom.array,
            projection_count=24,
            source_kind=ReconstructionSourceKind.IMAGE_DERIVED_SIMULATION,
        )
        request = ReconstructionRequest.from_sinogram_result(sino, output_size=24)

        self.assertTrue(request.requires_image_derived_warning)
        result = FilteredBackProjection().reconstruct(request)
        self.assertEqual(result.metadata["source_kind"], "image-derived-simulation")
        self.assertTrue(result.metadata["image_derived_simulation"])

        with self.assertRaisesRegex(ValidationError, "cannot use a k-space"):
            ReconstructionRequest(
                sino.sinogram,
                sino.theta_degrees,
                24,
                source_kind=ReconstructionSourceKind.RAW_KSPACE,
            )


if __name__ == "__main__":
    unittest.main()
