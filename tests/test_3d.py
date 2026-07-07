from __future__ import annotations
import warnings

import unittest
import numpy as np
from czi_foci.config import NucleiConfig, FocusConfig
from czi_foci.segmentation import segment_nuclei_3d, segment_spots_3d
from czi_foci.colocalization import colocalization_calls_3d

class ThreeDTests(unittest.TestCase):
    def test_3d_segmentation_and_colocalization(self):
        warnings.filterwarnings('ignore', category=UserWarning)
        # Create small anisotropic synthetic array
        # Z = 5, Y = 64, X = 64
        # px_um_z = 0.5, px_um_y = 0.2, px_um_x = 0.2
        zz, yy, xx = np.ogrid[:5, :64, :64]

        # Nucleus 1 at (2, 32, 32)
        nucleus = np.zeros((5, 64, 64), dtype=np.uint8)
        mask = (zz - 2)**2 / 2**2 + (yy - 32)**2 / 10**2 + (xx - 32)**2 / 10**2 <= 1
        nucleus[mask] = 120

        nuclei_config = NucleiConfig(
            channel_index=2,
            label="DAPI",
            gaussian_sigma_px=0.8,
            min_area_um2=1.0,
            hole_area_um2=0.5,
            watershed_min_distance_px=2,
        )

        # Volume 3D strategy
        labels_3d, diag = segment_nuclei_3d(nucleus, 0.2, 0.2, 0.5, nuclei_config, strategy="volume_3d")
        self.assertGreaterEqual(labels_3d.max(), 1)

        # Focus A at (2, 32, 32)
        focus_a = np.zeros_like(nucleus)
        focus_a[2:4, 30:35, 30:35] = 200
        labels_3d[2:4, 30:35, 30:35] = 1
        focus_a[2, 32, 32] = 200
        focus_a[2, 32, 33] = 200
        focus_a[2, 33, 32] = 200
        focus_a[2, 33, 33] = 200
        labels_3d[2, 32, 32] = 1
        focus_a_config = FocusConfig(
            name="focus_a", channel_index=1, label="AF488",
            tophat_radius_px=2, min_area_px=1, threshold_mad_multiplier=0.0, min_intensity_above_local_background=0.0
        )
        a_labels, a_diag = segment_spots_3d(focus_a, labels_3d, 0.2, 0.2, 0.5, focus_a_config)
        self.assertGreaterEqual(a_labels.max(), 1)

        # Focus B at (2, 34, 34)
        focus_b = np.zeros_like(nucleus)
        focus_b[2:4, 33:38, 33:38] = 200
        labels_3d[2:4, 33:38, 33:38] = 1
        focus_b[2, 34, 34] = 200
        focus_b[2, 34, 35] = 200
        focus_b[2, 35, 34] = 200
        focus_b[2, 35, 35] = 200
        labels_3d[2, 34, 34] = 1
        focus_b_config = FocusConfig(
            name="focus_b", channel_index=0, label="AF647",
            tophat_radius_px=2, min_area_px=1, threshold_mad_multiplier=0.0, min_intensity_above_local_background=0.0
        )
        b_labels, b_diag = segment_spots_3d(focus_b, labels_3d, 0.2, 0.2, 0.5, focus_b_config)
        self.assertGreaterEqual(b_labels.max(), 1)

        # Colocalization (physical distance):
        # A is at (2, 32, 32) -> physical: 2*0.5, 32*0.2, 32*0.2 = (1.0, 6.4, 6.4)
        # B is at (2, 34, 34) -> physical: 2*0.5, 34*0.2, 34*0.2 = (1.0, 6.8, 6.8)
        # distance squared = (0.4)^2 + (0.4)^2 = 0.16 + 0.16 = 0.32
        # distance = sqrt(0.32) ~ 0.565

        # Distance = 0.9 um -> should colocalize
        b_to_a, coloc_b, coloc_a, _ = colocalization_calls_3d(a_labels, b_labels, 0.9, 0.2, 0.2, 0.5)
        self.assertTrue(1 in coloc_b)

        # Distance = 0.8 um -> should not colocalize
        b_to_a, coloc_b, coloc_a, _ = colocalization_calls_3d(a_labels, b_labels, 0.8, 0.2, 0.2, 0.5)
        self.assertFalse(1 in coloc_b)

if __name__ == "__main__":
    unittest.main()
