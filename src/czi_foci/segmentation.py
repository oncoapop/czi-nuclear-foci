"""Image segmentation primitives for nuclear foci analysis."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation
from skimage.feature import peak_local_max

from .config import FocusConfig, NucleiConfig


def area_um2_to_px(area_um2: float, px_um_x: float, px_um_y: float) -> int:
    if not np.isfinite(px_um_x) or not np.isfinite(px_um_y) or px_um_x <= 0 or px_um_y <= 0:
        return max(1, int(round(area_um2)))
    return max(1, int(round(area_um2 / (px_um_x * px_um_y))))


def segment_nuclei(image: np.ndarray, px_um_x: float, px_um_y: float, config: NucleiConfig) -> tuple[np.ndarray, dict]:
    smoothed = filters.gaussian(image, sigma=config.gaussian_sigma_px, preserve_range=True)
    threshold = float(filters.threshold_otsu(smoothed)) if np.unique(smoothed).size > 1 else float(smoothed.mean())
    min_area = area_um2_to_px(config.min_area_um2, px_um_x, px_um_y)
    hole_area = area_um2_to_px(config.hole_area_um2, px_um_x, px_um_y)

    binary = smoothed > threshold
    binary = morphology.remove_small_objects(binary, min_size=min_area)
    binary = morphology.remove_small_holes(binary, area_threshold=hole_area)
    binary = ndi.binary_fill_holes(binary)

    distance = ndi.distance_transform_edt(binary)
    coords = peak_local_max(
        distance,
        min_distance=config.watershed_min_distance_px,
        labels=binary,
        exclude_border=False,
    )
    markers = np.zeros(binary.shape, dtype=np.int32)
    if coords.size:
        markers[tuple(coords.T)] = np.arange(1, coords.shape[0] + 1)
    markers = measure.label(markers > 0)
    labels = segmentation.watershed(-distance, markers, mask=binary) if markers.max() else measure.label(binary)
    labels = morphology.remove_small_objects(labels, min_size=min_area)
    labels = measure.label(labels > 0).astype(np.uint16)
    return labels, {
        "nuclei_threshold": threshold,
        "nuclei_min_area_px": int(min_area),
        "nuclei_hole_area_px": int(hole_area),
        "nuclei_watershed_markers": int(markers.max()),
        "nuclei_count": int(labels.max()),
    }


def robust_threshold(values: np.ndarray, multiplier: float) -> float:
    if values.size == 0:
        return float("inf")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    mad_sigma = 1.4826 * mad
    otsu = float(filters.threshold_otsu(values)) if np.unique(values).size > 1 else median
    return max(otsu, median + multiplier * mad_sigma)


def segment_spots(image: np.ndarray, nuclei_labels: np.ndarray, config: FocusConfig) -> tuple[np.ndarray, dict]:
    nuclear_mask = nuclei_labels > 0
    footprint = morphology.disk(config.tophat_radius_px)
    enhanced = morphology.white_tophat(image, footprint=footprint)
    threshold = robust_threshold(enhanced[nuclear_mask], config.threshold_mad_multiplier)
    local_background = filters.gaussian(image, sigma=config.tophat_radius_px * 2, preserve_range=True)
    above_background = image.astype(np.float32) - local_background.astype(np.float32)
    binary = (
        nuclear_mask
        & (enhanced >= threshold)
        & (above_background >= config.min_intensity_above_local_background)
    )
    binary = morphology.remove_small_objects(binary, min_size=config.min_area_px)
    labels = measure.label(binary).astype(np.uint16)
    return labels, {
        f"{config.name}_threshold": float(threshold),
        f"{config.name}_min_area_px": int(config.min_area_px),
        f"{config.name}_tophat_radius_px": int(config.tophat_radius_px),
        f"{config.name}_spots_count": int(labels.max()),
    }


def display_u8(image: np.ndarray, low_percentile: float, high_percentile: float) -> np.ndarray:
    low, high = np.percentile(image, [low_percentile, high_percentile])
    if high <= low:
        low, high = float(image.min()), float(image.max())
    if high <= low:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image.astype(np.float32) - low) * (255.0 / (high - low))
    return np.clip(scaled, 0, 255).astype(np.uint8)
