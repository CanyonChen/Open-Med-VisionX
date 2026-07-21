from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from dicom_viewer.algorithms import (
    DirectFourierReconstruction,
    FilteredBackProjection,
    ReconstructionRequest,
    SARTReconstruction,
    compute_metrics,
    generate_angles,
    generate_sinogram,
)

HAS_SKIMAGE = importlib.util.find_spec("skimage") is not None


class AngleTests(unittest.TestCase):
    def test_angle_ranges_are_endpoint_exclusive(self) -> None:
        np.testing.assert_allclose(generate_angles(4, 180), [0, 45, 90, 135])
        np.testing.assert_allclose(generate_angles(4, 360), [0, 90, 180, 270])


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

    def test_metrics_use_one_joint_range_and_include_roi(self) -> None:
        reference = np.arange(64, dtype=float).reshape(8, 8) - 1024.0
        reconstruction = reference + 2.0
        report = compute_metrics(reference, reconstruction, rois={"center": (2, 2, 4, 4)})
        self.assertEqual(report.intensity_range, (-1024.0, -959.0))
        self.assertIn("center", report.roi)
        self.assertGreater(report.values.mse, 0.0)
        np.testing.assert_allclose(report.difference, 2.0 / 65.0)


if __name__ == "__main__":
    unittest.main()
