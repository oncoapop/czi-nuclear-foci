"""Configuration loading for generic 3-channel CZI foci analysis."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NucleiConfig:
    channel_index: int = 2
    label: str = "DAPI"
    gaussian_sigma_px: float = 1.2
    min_area_um2: float = 35.0
    hole_area_um2: float = 20.0
    watershed_min_distance_px: int = 10


@dataclass(frozen=True)
class FocusConfig:
    name: str
    channel_index: int
    label: str
    tophat_radius_px: int
    min_area_px: int
    threshold_mad_multiplier: float = 4.0
    min_intensity_above_local_background: float = 8.0


@dataclass(frozen=True)
class FilenameConfig:
    regex: str = r"^(?P<timepoint>\d+)\s*hr\s+20x\s+(?P<condition>.+)-(?P<replicate>\d+)\.czi$"
    condition_group: str = "condition"
    timepoint_group: str = "timepoint"
    replicate_group: str = "replicate"


@dataclass(frozen=True)
class OutputConfig:
    qc_display_percentile_low: float = 1.0
    qc_display_percentile_high: float = 99.8
    positive_thresholds: dict[str, list[int]] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisConfig:
    experiment_name: str
    nuclei: NucleiConfig
    focus_a: FocusConfig
    focus_b: FocusConfig
    filename: FilenameConfig = field(default_factory=FilenameConfig)
    control_conditions: list[str] = field(default_factory=list)
    condition_order: list[str] = field(default_factory=list)
    colocalization_dilation_px: int = 1
    colocalization_distance_um: float = 1.0
    condition_map_csv: str | None = None
    output: OutputConfig = field(default_factory=OutputConfig)
    mode: str = "2d"
    z_projection: str = "first"  # Explicitly use the first Z plane in 2D mode
    nucleus_strategy: str = "mip_2d_propagate"


def _require_dict(data: Any, key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config key {key!r} must be an object")
    return value


def load_config(path: Path) -> AnalysisConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    nuclei = NucleiConfig(**_require_dict(data, "nuclei"))
    focus_a = FocusConfig(**_require_dict(data, "focus_a"))
    focus_b = FocusConfig(**_require_dict(data, "focus_b"))
    filename = FilenameConfig(**data.get("filename", {}))
    output = OutputConfig(**data.get("output", {}))
    config = AnalysisConfig(
        experiment_name=data["experiment_name"],
        nuclei=nuclei,
        focus_a=focus_a,
        focus_b=focus_b,
        filename=filename,
        control_conditions=list(data.get("control_conditions", [])),
        condition_order=list(data.get("condition_order", [])),
        colocalization_dilation_px=int(data.get("colocalization_dilation_px", 1)),
        colocalization_distance_um=float(data.get("colocalization_distance_um", 1.0)),
        condition_map_csv=data.get("condition_map_csv"),
        output=output,
        mode=data.get("mode", "2d"),
        z_projection=data.get("z_projection", "first"),
        nucleus_strategy=data.get("nucleus_strategy", "mip_2d_propagate"),
    )
    re.compile(config.filename.regex)
    if config.focus_a.name == config.focus_b.name:
        raise ValueError("focus_a.name and focus_b.name must differ")
    if config.nuclei.channel_index in {config.focus_a.channel_index, config.focus_b.channel_index}:
        raise ValueError("Nuclear channel index must differ from focus channel indices")
    if config.focus_a.channel_index == config.focus_b.channel_index:
        raise ValueError("Focus channel indices must differ")
    return config


def config_to_dict(config: AnalysisConfig) -> dict[str, Any]:
    return asdict(config)
