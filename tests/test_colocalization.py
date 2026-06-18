from __future__ import annotations

import unittest

import numpy as np

from czi_foci.colocalization import colocalization_calls


class ColocalizationTests(unittest.TestCase):
    def test_colocalization_requires_configured_pixel_dilation(self) -> None:
        focus_a = np.zeros((12, 12), dtype=np.uint16)
        focus_b = np.zeros((12, 12), dtype=np.uint16)
        focus_a[5, 5] = 1
        focus_b[5, 7] = 1

        b_to_a_1px, coloc_b_1px, coloc_a_1px, _ = colocalization_calls(focus_a, focus_b, dilation_px=1)
        b_to_a_2px, coloc_b_2px, coloc_a_2px, _ = colocalization_calls(focus_a, focus_b, dilation_px=2)

        self.assertEqual(b_to_a_1px, {})
        self.assertEqual(coloc_b_1px, set())
        self.assertEqual(coloc_a_1px, set())
        self.assertEqual(b_to_a_2px, {1: {1}})
        self.assertEqual(coloc_b_2px, {1})
        self.assertEqual(coloc_a_2px, {1})


if __name__ == "__main__":
    unittest.main()
