"""Co-localisation calls between two segmented nuclear focus channels."""

from __future__ import annotations

import numpy as np
from skimage import measure, morphology


def spot_to_nucleus_labels(spot_labels: np.ndarray, nuclei_labels: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for region in measure.regionprops(spot_labels):
        coords = region.coords
        values = nuclei_labels[coords[:, 0], coords[:, 1]]
        values = values[values > 0]
        mapping[int(region.label)] = int(np.bincount(values.astype(np.int64)).argmax()) if values.size else 0
    return mapping


def colocalization_calls(
    focus_a_labels: np.ndarray,
    focus_b_labels: np.ndarray,
    dilation_px: int,
) -> tuple[dict[int, set[int]], set[int], set[int], np.ndarray]:
    """Return focus B to focus A overlap calls after binary dilation.

    A focus B object is called co-localised when any of its pixels intersects a
    focus A object after dilating the focus A mask by ``dilation_px`` pixels.
    A matching focus A label is then recovered by dilating the focus B object by
    the same radius and collecting overlapping focus A labels.
    """

    structure = morphology.disk(dilation_px)
    focus_a_binary_dilated = morphology.dilation(focus_a_labels > 0, footprint=structure)
    b_to_a: dict[int, set[int]] = {}
    coloc_b: set[int] = set()
    coloc_a: set[int] = set()

    for b_region in measure.regionprops(focus_b_labels):
        coords = b_region.coords
        if not focus_a_binary_dilated[coords[:, 0], coords[:, 1]].any():
            continue
        b_label = int(b_region.label)
        nearby_mask = np.zeros(focus_b_labels.shape, dtype=bool)
        nearby_mask[coords[:, 0], coords[:, 1]] = True
        nearby_mask = morphology.dilation(nearby_mask, footprint=structure)
        a_labels = set(int(v) for v in np.unique(focus_a_labels[nearby_mask]) if v > 0)
        if a_labels:
            b_to_a[b_label] = a_labels
            coloc_b.add(b_label)
            coloc_a.update(a_labels)

    coloc_b_labels = np.where(np.isin(focus_b_labels, list(coloc_b)), focus_b_labels, 0).astype(np.uint16)
    return b_to_a, coloc_b, coloc_a, coloc_b_labels

def colocalization_calls_3d(
    focus_a_labels: np.ndarray,
    focus_b_labels: np.ndarray,
    distance_um: float,
    px_um_x: float,
    px_um_y: float,
    px_um_z: float,
) -> tuple[dict[int, set[int]], set[int], set[int], np.ndarray]:
    from scipy.spatial import cKDTree

    b_to_a: dict[int, set[int]] = {}
    coloc_b: set[int] = set()
    coloc_a: set[int] = set()

    props_a = measure.regionprops(focus_a_labels)
    props_b = measure.regionprops(focus_b_labels)

    if not props_a or not props_b:
        return b_to_a, coloc_b, coloc_a, np.zeros_like(focus_b_labels)

    centroids_a = np.array([p.centroid for p in props_a])
    centroids_b = np.array([p.centroid for p in props_b])

    labels_a = np.array([p.label for p in props_a])
    labels_b = np.array([p.label for p in props_b])

    # Scale centroids to physical space (z, y, x)
    physical_a = centroids_a * np.array([px_um_z, px_um_y, px_um_x])
    physical_b = centroids_b * np.array([px_um_z, px_um_y, px_um_x])

    tree_a = cKDTree(physical_a)
    # find pairs within distance_um
    pairs = tree_a.query_ball_point(physical_b, r=distance_um)

    for b_idx, a_indices in enumerate(pairs):
        if a_indices:
            b_label = int(labels_b[b_idx])
            a_lbls = {int(labels_a[a_idx]) for a_idx in a_indices}
            b_to_a[b_label] = a_lbls
            coloc_b.add(b_label)
            coloc_a.update(a_lbls)

    coloc_b_labels = np.where(np.isin(focus_b_labels, list(coloc_b)), focus_b_labels, 0).astype(np.uint16)
    return b_to_a, coloc_b, coloc_a, coloc_b_labels
