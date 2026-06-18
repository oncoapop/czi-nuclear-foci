from __future__ import annotations

import json
import unittest

from czi_foci.config import load_config


class ConfigTests(unittest.TestCase):
    def test_config_rejects_reused_channel_index(self) -> None:
        tmp_path = self._testMethodName
        path_name = f"/tmp/{tmp_path}.json"
        config = {
            "experiment_name": "bad",
            "nuclei": {"channel_index": 2, "label": "DAPI"},
            "focus_a": {
                "name": "focus_a",
                "channel_index": 1,
                "label": "A",
                "tophat_radius_px": 4,
                "min_area_px": 3,
            },
            "focus_b": {
                "name": "focus_b",
                "channel_index": 1,
                "label": "B",
                "tophat_radius_px": 8,
                "min_area_px": 8,
            },
        }
        from pathlib import Path

        path = Path(path_name)
        path.write_text(json.dumps(config), encoding="utf-8")
        try:
            with self.assertRaisesRegex(ValueError, "Focus channel indices must differ"):
                load_config(path)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
