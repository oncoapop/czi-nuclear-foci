from __future__ import annotations

import unittest

import pandas as pd

from czi_foci.reports import threshold_performance


class ThresholdTests(unittest.TestCase):
    def test_threshold_performance_reports_sensitivity_specificity_and_selectivity(self) -> None:
        nuclei = pd.DataFrame(
            {
                "condition": ["Control", "Control", "Treated", "Treated"],
                "focus_a_count": [0, 2, 1, 4],
            }
        )
        table = threshold_performance(nuclei, "focus_a_count", {"Control"}, thresholds=[2])
        row = table.iloc[0]

        self.assertEqual(int(row["true_positive_treated_nuclei"]), 1)
        self.assertEqual(int(row["false_positive_control_nuclei"]), 1)
        self.assertEqual(int(row["true_negative_control_nuclei"]), 1)
        self.assertEqual(int(row["false_negative_treated_nuclei"]), 1)
        self.assertEqual(row["sensitivity"], 0.5)
        self.assertEqual(row["specificity"], 0.5)
        self.assertEqual(row["selectivity"], 0.5)


if __name__ == "__main__":
    unittest.main()
