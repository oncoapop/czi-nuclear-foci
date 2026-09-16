"""Read Imaris (.ims) metadata and pixel data without changing the source.

.ims files are plain HDF5. Three traps documented here are load-bearing:

1. HDF5 string attributes are stored as fixed-length char arrays, not Python
   strings — they must be joined with ``"".join(attrs[key].astype(str))``.
2. ``DataSet/ResolutionLevel N/TimePoint 0/Channel C/Data`` is Z-padded to a
   multiple of 8. The true Z comes from ``DataSetInfo/Image`` attr ``Z`` and
   the array must always be sliced down to it.
3. Resolution levels 0-5 halve Y/X at each level but never touch Z. Level 2
   (512x512) is fast enough for QC; level 0 is required for real analysis.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def _decode_attr(value) -> str:
    """Decode an HDF5 char-array attribute into a plain string."""
    return "".join(np.asarray(value).astype(str))


def _group_attrs(group: h5py.Group) -> dict[str, str]:
    return {key: _decode_attr(value) for key, value in group.attrs.items()}


def inspect_ims(path: Path) -> dict:
    """Inspect an .ims file's structure and metadata without loading pixel data.

    Returns a dict matching the shape expected by the rest of the pipeline
    (a ``metadata`` key plus enough structure to drive ``read_channel_arrays``).
    Files with missing groups (no ``DataSetInfo``, no ``Data``, or no
    ``DataSet`` at all) are reported rather than raised on, so a batch QC pass
    can flag them instead of crashing.
    """
    result: dict[str, object] = {
        "source": str(path),
        "source_size_bytes": path.stat().st_size if path.exists() else 0,
        "format": "ims",
        "metadata": None,
        "channels": {},
        "true_z": None,
        "resolution_levels": [],
        "errors": [],
    }

    try:
        with h5py.File(path, "r") as handle:
            if "DataSet" not in handle:
                result["errors"].append("missing DataSet group")
                return result

            dataset = handle["DataSet"]
            levels = sorted(
                (name for name in dataset.keys() if name.startswith("ResolutionLevel")),
                key=lambda name: int(name.rsplit(" ", 1)[-1]),
            )
            result["resolution_levels"] = levels
            if not levels:
                result["errors"].append("no resolution levels in DataSet")

            has_info = "DataSetInfo" in handle
            image_attrs: dict[str, str] = {}
            channel_attrs: dict[int, dict[str, str]] = {}
            custom_attrs: dict[str, str] = {}

            if has_info:
                info = handle["DataSetInfo"]
                if "Image" in info:
                    image_attrs = _group_attrs(info["Image"])
                else:
                    result["errors"].append("DataSetInfo present but missing Image group")
                if "CustomData" in info:
                    custom_attrs = _group_attrs(info["CustomData"])
                for name in info.keys():
                    if name.startswith("Channel "):
                        index = int(name.rsplit(" ", 1)[-1])
                        channel_attrs[index] = _group_attrs(info[name])
            else:
                result["errors"].append("missing DataSetInfo group")

            result["metadata"] = {
                "image": image_attrs,
                "channels": channel_attrs,
                "custom": custom_attrs,
                "has_dataset_info": has_info,
            }

            if image_attrs.get("Z"):
                result["true_z"] = int(image_attrs["Z"])

            if levels:
                first_level = dataset[levels[0]]
                if "TimePoint 0" not in first_level:
                    result["errors"].append("missing TimePoint 0 group")
                else:
                    tp0 = first_level["TimePoint 0"]
                    for channel_name in tp0.keys():
                        index = int(channel_name.rsplit(" ", 1)[-1])
                        has_data = "Data" in tp0[channel_name]
                        result["channels"][index] = {
                            "name": channel_name,
                            "has_data": has_data,
                        }
                        if not has_data:
                            result["errors"].append(f"missing Data object for {channel_name}")
    except OSError as exc:
        result["errors"].append(f"could not open file: {exc}")

    return result


def _resolve_true_z(inspection: dict, sibling_metadata: dict | None) -> int | None:
    true_z = inspection.get("true_z")
    if true_z is not None:
        return true_z
    if sibling_metadata is not None:
        z = sibling_metadata.get("image", {}).get("Z")
        if z:
            return int(z)
    return None


def recover_metadata_from_sibling(path: Path) -> dict | None:
    """Try to recover voxel/image metadata from a sibling .ims file in the same folder.

    Some tiles (e.g. whole missing-DataSetInfo fields) have no metadata of
    their own, but every tile in a folder was acquired with the same voxel
    geometry, so a sibling's ``DataSetInfo/Image`` group is a valid stand-in.
    """
    for sibling in sorted(path.parent.glob("*.ims")):
        if sibling == path:
            continue
        sibling_inspection = inspect_ims(sibling)
        metadata = sibling_inspection.get("metadata")
        if metadata and metadata.get("has_dataset_info") and metadata.get("image"):
            return metadata
    return None


def read_channel_arrays_ims(
    path: Path,
    inspection: dict,
    mode: str = "3d",
    resolution_level: int = 0,
) -> dict[int, np.ndarray]:
    """Read channel arrays from an .ims file, truncated to the true (unpadded) Z.

    ``mode="2d"`` returns the first true-Z plane per channel (Y, X).
    ``mode="3d"`` returns the full truncated stack per channel (Z, Y, X).
    """
    if inspection.get("errors") and not inspection.get("channels"):
        raise ValueError(f"Cannot read channel arrays for {path}: {inspection['errors']}")

    true_z = inspection.get("true_z")
    if true_z is None:
        sibling_metadata = recover_metadata_from_sibling(path)
        true_z = _resolve_true_z(inspection, sibling_metadata)
    if true_z is None:
        raise ValueError(f"Cannot determine true Z for {path}: no DataSetInfo/Image and no usable sibling")

    level_name = f"ResolutionLevel {resolution_level}"
    arrays: dict[int, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        dataset = handle["DataSet"]
        if level_name not in dataset:
            raise ValueError(f"{level_name} not present in {path}")
        tp0 = dataset[level_name]["TimePoint 0"]
        for channel_name in tp0.keys():
            index = int(channel_name.rsplit(" ", 1)[-1])
            group = tp0[channel_name]
            if "Data" not in group:
                continue
            padded = group["Data"][()]
            stack = padded[:true_z]
            if mode == "2d":
                arrays[index] = stack[0]
            else:
                arrays[index] = stack
    return arrays


def pixel_size_um_ims(metadata: dict, mode: str = "3d") -> tuple[float, float, float]:
    """Compute (z, y, x) voxel size in microns from .ims Image attrs.

    z is None (nan) when Z <= 1 (single-plane acquisitions have no z-step).
    Never hardcode this — z-step varies between folders on this instrument.
    """
    image = metadata.get("image", {})

    def _f(key: str) -> float:
        value = image.get(key)
        if not value:
            raise ValueError(f"Missing required .ims Image attribute: {key}")
        return float(value)

    x_px = _f("X")
    y_px = _f("Y")
    xy_x = (_f("ExtMax0") - _f("ExtMin0")) / x_px
    xy_y = (_f("ExtMax1") - _f("ExtMin1")) / y_px

    z_px = int(_f("Z"))
    if z_px > 1:
        z_step = (_f("ExtMax2") - _f("ExtMin2")) / (z_px - 1)
    else:
        z_step = float("nan")

    return z_step, xy_y, xy_x


def stage_position_mm(metadata: dict) -> tuple[float, float]:
    """Stage x/y position in mm, from ExtMin0/ExtMin1 (recorded in microns)."""
    image = metadata.get("image", {})
    x_um = float(image["ExtMin0"]) if image.get("ExtMin0") else float("nan")
    y_um = float(image["ExtMin1"]) if image.get("ExtMin1") else float("nan")
    return x_um / 1000.0, y_um / 1000.0


def channel_metadata_ims(metadata: dict, index: int) -> str:
    channels = metadata.get("channels", {})
    channel = channels.get(index)
    if not channel:
        return f"C{index}"
    name = channel.get("Name", "")
    em = channel.get("LSMEmissionWavelength", "")
    ex = channel.get("LSMExcitationWavelength", "")
    parts = [f"C{index}", name]
    if ex and em:
        parts.append(f"Ex {ex} / Em {em}")
    return " | ".join(p for p in parts if p)
