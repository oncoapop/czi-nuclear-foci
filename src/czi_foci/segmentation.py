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



def segment_nuclei_3d(
    image: np.ndarray,
    px_um_x: float,
    px_um_y: float,
    px_um_z: float,
    config: NucleiConfig,
    strategy: str = "mip_2d_propagate",
) -> tuple[np.ndarray, dict]:
    if strategy == "mip_2d_propagate":
        # Create MIP for segmentation
        mip = image.max(axis=0)
        labels_2d, diag = segment_nuclei(mip, px_um_x, px_um_y, config)

        # Propagate 2D labels through all Z slices
        labels_3d = np.repeat(labels_2d[np.newaxis, :, :], image.shape[0], axis=0)
        return labels_3d, diag

    elif strategy == "volume_3d":
        # Native 3D thresholding/morphology/watershed
        # Anisotropic gaussian based on voxel spacing
        sigma_z = config.gaussian_sigma_px * (px_um_x / px_um_z) if px_um_z > 0 else config.gaussian_sigma_px
        smoothed = filters.gaussian(image, sigma=(sigma_z, config.gaussian_sigma_px, config.gaussian_sigma_px), preserve_range=True)

        threshold = float(filters.threshold_otsu(smoothed)) if np.unique(smoothed).size > 1 else float(smoothed.mean())

        # Estimate volume based on 2D area (treat it like a sphere/cylinder)
        area_px = area_um2_to_px(config.min_area_um2, px_um_x, px_um_y)
        hole_px = area_um2_to_px(config.hole_area_um2, px_um_x, px_um_y)
        # simplistic volume estimation: A * (sqrt(A/pi) / (z_spacing / x_spacing))?
        # For simplicity, we just use arbitrary 3D volume limits or just 3/2 power
        r_px = np.sqrt(area_px / np.pi)
        vol_px = (4/3) * np.pi * (r_px**2) * (r_px * px_um_x / px_um_z)
        min_vol = max(1, int(vol_px))

        binary = smoothed > threshold
        binary = morphology.remove_small_objects(binary, min_size=min_vol)
        # remove_small_holes in 3D might be slow, but it's supported
        binary = morphology.remove_small_holes(binary, area_threshold=min_vol)
        # binary_fill_holes works in N-D
        binary = ndi.binary_fill_holes(binary)

        # Anisotropic distance transform can be done by providing sampling
        distance = ndi.distance_transform_edt(binary, sampling=[px_um_z, px_um_y, px_um_x])
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
        labels = morphology.remove_small_objects(labels, min_size=min_vol)
        labels = measure.label(labels > 0).astype(np.uint16)

        return labels, {
            "nuclei_threshold": threshold,
            "nuclei_min_vol_px": int(min_vol),
            "nuclei_watershed_markers": int(markers.max()),
            "nuclei_count": int(labels.max()),
        }
    else:
        raise ValueError(f"Unknown 3D nucleus strategy: {strategy}")

def segment_spots_3d(image: np.ndarray, nuclei_labels: np.ndarray, px_um_x: float, px_um_y: float, px_um_z: float, config: FocusConfig) -> tuple[np.ndarray, dict]:
    nuclear_mask = nuclei_labels > 0

    # Anisotropic footprint
    r_xy = config.tophat_radius_px
    r_z = max(1, int(round(r_xy * (px_um_x / px_um_z)))) if px_um_z > 0 else 1

    # Generate an ellipsoidal footprint for 3D top-hat
    # Create a grid
    z, y, x = np.ogrid[-r_z:r_z+1, -r_xy:r_xy+1, -r_xy:r_xy+1]
    footprint = (x**2 / r_xy**2 + y**2 / r_xy**2 + z**2 / r_z**2) <= 1

    enhanced = morphology.white_tophat(image, footprint=footprint)
    threshold = robust_threshold(enhanced[nuclear_mask], config.threshold_mad_multiplier)

    sigma_z = (config.tophat_radius_px * 2) * (px_um_x / px_um_z) if px_um_z > 0 else (config.tophat_radius_px * 2)
    local_background = filters.gaussian(image, sigma=(sigma_z, config.tophat_radius_px * 2, config.tophat_radius_px * 2), preserve_range=True)
    above_background = image.astype(np.float32) - local_background.astype(np.float32)

    binary = (
        nuclear_mask
        & (enhanced >= threshold)
        & (above_background >= config.min_intensity_above_local_background)
    )
    # min_area_px could be interpreted as a proxy for min volume in 3D.
    # Let's scale min_area_px by the same factor used for radius.
    min_vol = max(1, config.min_area_px * max(1, int(px_um_x / px_um_z)))

    binary = morphology.remove_small_objects(binary, min_size=min_vol)
    labels = measure.label(binary).astype(np.uint16)

    return labels, {
        f"{config.name}_threshold": float(threshold),
        f"{config.name}_min_vol_px": int(min_vol),
        f"{config.name}_tophat_radius_z_px": int(r_z),
        f"{config.name}_tophat_radius_xy_px": int(r_xy),
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
