from __future__ import annotations

import unittest

import pandas as pd

from czi_foci.current_dataset import add_experiment_classifications, best_thresholds_by_experiment, field_summary


class CurrentDatasetReportTests(unittest.TestCase):
    def test_thresholds_are_fit_by_experiment_not_timepoint(self) -> None:
        nuclei = pd.DataFrame(
            {
                "experiment": ["TS A"] * 8 + ["TS B"] * 8,
                "timepoint_hr": [2, 2, 6, 6, 2, 2, 6, 6] * 2,
                "condition": ["Control", "Control", "Drug", "Drug", "Control", "Control", "Drug", "Drug"] * 2,
                "known_control": [True, True, False, False, True, True, False, False] * 2,
                "af488_spots_count": [0, 1, 3, 4, 0, 1, 3, 4, 10, 11, 14, 16, 10, 11, 14, 16],
            }
        )
        thresholds = best_thresholds_by_experiment(nuclei, ["af488_spots_count"])

        self.assertEqual(set(thresholds["experiment"]), {"TS A", "TS B"})
        self.assertEqual(len(thresholds), 2)
        self.assertNotIn("timepoint_hr", thresholds.columns)
        by_experiment = dict(zip(thresholds["experiment"], thresholds["positive_if_ge"]))
        self.assertNotEqual(by_experiment["TS A"], by_experiment["TS B"])

    def test_batch_classification_and_field_summary(self) -> None:
        nuclei = pd.DataFrame(
            {
                "experiment": ["TS A"] * 4,
                "timepoint_hr": [2] * 4,
                "condition": ["Control", "Control", "Drug", "Drug"],
                "sample_id": ["s1", "s1", "s1", "s1"],
                "file_name": ["f.czi"] * 4,
                "replicate": ["01"] * 4,
                "known_control": [True, True, False, False],
                "af488_spots_count": [0, 2, 3, 4],
                "rhrex_spots_count": [0, 0, 1, 2],
                "colocalized_rhrex_spots_count": [0, 0, 0, 1],
            }
        )
        thresholds = pd.DataFrame(
            {
                "experiment": ["TS A", "TS A", "TS A"],
                "metric": ["af488_spots_count", "rhrex_spots_count", "colocalized_rhrex_spots_count"],
                "positive_if_ge": [3, 1, 1],
            }
        )
        classified = add_experiment_classifications(nuclei, thresholds)

        self.assertEqual(classified["af488_positive_batch"].tolist(), [False, False, True, True])
        self.assertEqual(classified["rhrex_positive_batch"].tolist(), [False, False, True, True])
        self.assertEqual(classified["dual_channel_positive_batch"].tolist(), [False, False, True, True])
        self.assertEqual(classified["coloc_positive_batch"].tolist(), [False, False, False, True])

        fields = field_summary(classified)
        drug = fields[fields["condition"].eq("Drug")].iloc[0]
        self.assertEqual(float(drug["fraction_dual_channel_positive_batch"]), 1.0)
        self.assertEqual(float(drug["fraction_coloc_positive_batch"]), 0.5)


if __name__ == "__main__":
    unittest.main()
