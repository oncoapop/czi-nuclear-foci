from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from czi_foci.config import AnalysisConfig, NucleiConfig, FocusConfig, FilenameConfig, OutputConfig
from czi_foci.analysis import analyse_files

class MockSampleMetadata:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class E2E3DTests(unittest.TestCase):
    @patch("czi_foci.analysis.inspect_czi")
    @patch("czi_foci.analysis.read_channel_arrays")
    @patch("czi_foci.analysis.parse_sample_metadata")
    def test_e2e_analyse_files_3d(self, mock_parse, mock_read, mock_inspect):
        # Mock inspection
        mock_inspect.return_value = {
            "metadata": {
                "scaling": [
                    {"Id": "X", "Value": 0.0000002},
                    {"Id": "Y", "Value": 0.0000002},
                    {"Id": "Z", "Value": 0.0000005},
                ],
                "channels": [], "image": {}
            },
            "subblocks": []
        }

        # Mock reading
        mock_read.return_value = {
            2: np.zeros((3, 32, 32), dtype=np.uint8), # nuclear
            1: np.zeros((3, 32, 32), dtype=np.uint8), # focus A
            0: np.zeros((3, 32, 32), dtype=np.uint8), # focus B
        }

        # Make dummy spots
        mock_read.return_value[2][1, 16, 16] = 200 # nucleus
        mock_read.return_value[1][1, 16, 16] = 200 # A
        mock_read.return_value[0][1, 16, 16] = 200 # B

        mock_parse.return_value = MockSampleMetadata(
            sample_id="test", file_name="test.czi", source_path="test.czi",
            condition="control", condition_code="ctrl", timepoint_hr="24",
            replicate="1", acquisition_date_folder="2026-01-01"
        )

        config = AnalysisConfig(
            experiment_name="test",
            nuclei=NucleiConfig(channel_index=2, min_area_um2=0.01, hole_area_um2=0.01),
            focus_a=FocusConfig(name="A", channel_index=1, label="A", tophat_radius_px=1, min_area_px=1, threshold_mad_multiplier=0.0, min_intensity_above_local_background=0.0),
            focus_b=FocusConfig(name="B", channel_index=0, label="B", tophat_radius_px=1, min_area_px=1, threshold_mad_multiplier=0.0, min_intensity_above_local_background=0.0),
            mode="3d",
            nucleus_strategy="volume_3d"
        )

        import tempfile
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            cfg_path = Path(td) / "cfg.json"
            cfg_path.touch()

            analyse_files([Path("test.czi")], config, cfg_path, out, "QC")

            # Check outputs
            self.assertTrue((out / "resolved_config.json").exists())
            self.assertTrue((out / "image_summary.csv").exists())
            self.assertTrue((out / "nucleus_measurements.csv").exists())

if __name__ == "__main__":
    unittest.main()
