from __future__ import annotations

import unittest

import pandas as pd

from czi_foci.batch import batch_aware_summary


class BatchAwareTests(unittest.TestCase):
    def test_batch_normalisation_and_thresholds_are_per_experiment_timepoint(self) -> None:
        nuclei = pd.DataFrame(
            {
                "experiment": ["TS1"] * 6 + ["TS2"] * 6,
                "timepoint_hr": [2] * 6 + [2] * 6,
                "condition": ["Control", "Control", "Control", "Drug", "Drug", "Drug"] * 2,
                "focus_a_count": [0, 0, 1, 2, 3, 4, 10, 10, 11, 12, 15, 18],
            }
        )

        outputs = batch_aware_summary(
            nuclei,
            metrics=["focus_a_count"],
            control_conditions={"Control"},
            batch_cols=["experiment", "timepoint_hr"],
        )

        stats = outputs["control_stats"].sort_values("experiment").reset_index(drop=True)
        self.assertEqual(stats.loc[0, "control_median"], 0.0)
        self.assertEqual(stats.loc[1, "control_median"], 10.0)

        raw_best = outputs["raw_best_thresholds_by_batch"].sort_values("experiment").reset_index(drop=True)
        self.assertNotEqual(raw_best.loc[0, "positive_if_ge"], raw_best.loc[1, "positive_if_ge"])

        raw_stability = outputs["raw_threshold_stability"].iloc[0]
        self.assertFalse(bool(raw_stability["is_stable"]))

        effects = outputs["effect_sizes"]
        drug_effects = effects[effects["condition"].eq("Drug")].sort_values("experiment")
        self.assertTrue((drug_effects["mean_control_z"] > 0).all())


if __name__ == "__main__":
    unittest.main()
