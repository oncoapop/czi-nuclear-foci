from __future__ import annotations

import unittest

import numpy as np

from czi_foci.config import FocusConfig, NucleiConfig
from czi_foci.segmentation import segment_nuclei, segment_spots


class SegmentationTests(unittest.TestCase):
    def test_segment_nuclei_and_spots_on_synthetic_image(self) -> None:
        yy, xx = np.ogrid[:96, :96]
        nuclear = np.zeros((96, 96), dtype=np.uint8)
        nucleus_a = (yy - 32) ** 2 + (xx - 32) ** 2 <= 13**2
        nucleus_b = (yy - 65) ** 2 + (xx - 64) ** 2 <= 12**2
        nuclear[nucleus_a | nucleus_b] = 120

        nuclei_config = NucleiConfig(
            channel_index=2,
            label="DAPI",
            gaussian_sigma_px=0.8,
            min_area_um2=10.0,
            hole_area_um2=5.0,
            watershed_min_distance_px=8,
        )
        labels, diagnostics = segment_nuclei(nuclear, 1.0, 1.0, nuclei_config)
        self.assertEqual(diagnostics["nuclei_count"], 2)
        self.assertEqual(labels.max(), 2)

        focus = np.zeros_like(nuclear)
        focus[32, 32] = 180
        focus[64, 64] = 200
        focus_config = FocusConfig(
            name="focus_a",
            channel_index=1,
            label="AF488",
            tophat_radius_px=3,
            min_area_px=1,
            threshold_mad_multiplier=3.0,
            min_intensity_above_local_background=5.0,
        )
        focus_labels, focus_diagnostics = segment_spots(focus, labels, focus_config)
        self.assertEqual(focus_diagnostics["focus_a_spots_count"], 2)
        self.assertEqual(focus_labels.max(), 2)


if __name__ == "__main__":
    unittest.main()
