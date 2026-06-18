"""End-to-end generic CZI nuclear foci analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import measure

from .colocalization import colocalization_calls, spot_to_nucleus_labels
from .config import AnalysisConfig, config_to_dict
from .io import (
    channel_metadata,
    inspect_czi,
    load_condition_map,
    parse_sample_metadata,
    pixel_size_um,
    read_channel_arrays,
)
from .reports import (
    make_qc_pdf,
    save_individual_channel_qc,
    save_label_tiff,
    save_qc_overlay,
    threshold_performance,
    write_csv,
)
from .segmentation import segment_nuclei, segment_spots


def _area_scale(px_um_x: float, px_um_y: float) -> float:
    scale = px_um_x * px_um_y
    return scale if np.isfinite(scale) else float("nan")


def _focus_rows(
    labels: np.ndarray,
    intensity: np.ndarray,
    nuclei_labels: np.ndarray,
    sample: dict,
    focus_name: str,
    coloc_labels: set[int],
    overlap_map: dict[int, set[int]] | None = None,
    area_scale: float = float("nan"),
) -> list[dict]:
    focus_to_nucleus = spot_to_nucleus_labels(labels, nuclei_labels)
    rows = []
    for region in measure.regionprops(labels, intensity_image=intensity):
        label = int(region.label)
        row = {
            **sample,
            "focus_channel": focus_name,
            "focus_label": label,
            "nucleus_label": focus_to_nucleus.get(label, 0),
            "is_colocalized": "TRUE" if label in coloc_labels else "FALSE",
            "focus_area_px": int(region.area),
            "focus_area_um2": float(region.area * area_scale),
            "focus_mean_intensity": float(region.mean_intensity),
            "focus_max_intensity": float(region.max_intensity),
            "focus_centroid_y_px": float(region.centroid[0]),
            "focus_centroid_x_px": float(region.centroid[1]),
        }
        if overlap_map is not None:
            row["overlapping_focus_labels"] = ";".join(str(v) for v in sorted(overlap_map.get(label, [])))
        rows.append(row)
    return rows


def _nucleus_rows(
    nuclei_labels: np.ndarray,
    nuclear: np.ndarray,
    focus_a: np.ndarray,
    focus_b: np.ndarray,
    focus_a_labels: np.ndarray,
    focus_b_labels: np.ndarray,
    coloc_focus_b: set[int],
    sample: dict,
    focus_a_name: str,
    focus_b_name: str,
    area_scale: float,
) -> list[dict]:
    a_to_nucleus = spot_to_nucleus_labels(focus_a_labels, nuclei_labels)
    b_to_nucleus = spot_to_nucleus_labels(focus_b_labels, nuclei_labels)
    a_by_nucleus: dict[int, list[int]] = {}
    b_by_nucleus: dict[int, list[int]] = {}
    coloc_b_by_nucleus: dict[int, list[int]] = {}
    for label, nucleus in a_to_nucleus.items():
        a_by_nucleus.setdefault(nucleus, []).append(label)
    for label, nucleus in b_to_nucleus.items():
        b_by_nucleus.setdefault(nucleus, []).append(label)
        if label in coloc_focus_b:
            coloc_b_by_nucleus.setdefault(nucleus, []).append(label)

    rows = []
    for region in measure.regionprops(nuclei_labels, intensity_image=nuclear):
        label = int(region.label)
        mask = nuclei_labels == label
        rows.append(
            {
                **sample,
                "nucleus_label": label,
                "nucleus_area_px": int(region.area),
                "nucleus_area_um2": float(region.area * area_scale),
                "nucleus_mean_intensity": float(region.mean_intensity),
                f"nucleus_mean_{focus_a_name}_intensity": float(focus_a[mask].mean()),
                f"nucleus_mean_{focus_b_name}_intensity": float(focus_b[mask].mean()),
                f"{focus_a_name}_count": len(a_by_nucleus.get(label, [])),
                f"{focus_b_name}_count": len(b_by_nucleus.get(label, [])),
                f"colocalized_{focus_b_name}_count": len(coloc_b_by_nucleus.get(label, [])),
                "nucleus_centroid_y_px": float(region.centroid[0]),
                "nucleus_centroid_x_px": float(region.centroid[1]),
            }
        )
    return rows


def _image_summary(
    sample: dict,
    inspection: dict,
    nuclei_rows: list[dict],
    focus_a_rows: list[dict],
    focus_b_rows: list[dict],
    pair_rows: list[dict],
    nuclei_diag: dict,
    focus_a_diag: dict,
    focus_b_diag: dict,
    config: AnalysisConfig,
) -> dict:
    n_nuclei = len(nuclei_rows)
    return {
        **sample,
        "acquisition_datetime": inspection["metadata"]["image"].get("AcquisitionDateAndTime", ""),
        "nuclear_channel_metadata": channel_metadata(inspection, config.nuclei.channel_index),
        f"{config.focus_a.name}_channel_metadata": channel_metadata(inspection, config.focus_a.channel_index),
        f"{config.focus_b.name}_channel_metadata": channel_metadata(inspection, config.focus_b.channel_index),
        "nuclei_count": n_nuclei,
        f"{config.focus_a.name}_total_count": len(focus_a_rows),
        f"{config.focus_b.name}_total_count": len(focus_b_rows),
        f"colocalized_{config.focus_b.name}_total_count": len({row[f"{config.focus_b.name}_label"] for row in pair_rows}),
        "colocalized_pairs_count": len(pair_rows),
        f"mean_{config.focus_a.name}_per_nucleus": float(np.mean([row[f"{config.focus_a.name}_count"] for row in nuclei_rows])) if nuclei_rows else 0.0,
        f"mean_{config.focus_b.name}_per_nucleus": float(np.mean([row[f"{config.focus_b.name}_count"] for row in nuclei_rows])) if nuclei_rows else 0.0,
        f"mean_colocalized_{config.focus_b.name}_per_nucleus": float(np.mean([row[f"colocalized_{config.focus_b.name}_count"] for row in nuclei_rows])) if nuclei_rows else 0.0,
        **nuclei_diag,
        **focus_a_diag,
        **focus_b_diag,
    }


def analyse_files(files: list[Path], config: AnalysisConfig, config_path: Path, output_dir: Path, qc_title: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_map_path = (config_path.parent / config.condition_map_csv).resolve() if config.condition_map_csv else None
    condition_map = load_condition_map(condition_map_path)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config_to_dict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    image_rows: list[dict] = []
    nucleus_rows_all: list[dict] = []
    focus_a_rows_all: list[dict] = []
    focus_b_rows_all: list[dict] = []
    pair_rows_all: list[dict] = []
    overlay_paths: list[Path] = []

    for path in files:
        inspection = inspect_czi(path)
        arrays = read_channel_arrays(path, inspection)
        nuclear = arrays[config.nuclei.channel_index]
        focus_a = arrays[config.focus_a.channel_index]
        focus_b = arrays[config.focus_b.channel_index]
        px_um_x, px_um_y = pixel_size_um(inspection["metadata"])
        scale = _area_scale(px_um_x, px_um_y)
        metadata = parse_sample_metadata(path, config, condition_map)
        sample = metadata.__dict__

        nuclei_labels, nuclei_diag = segment_nuclei(nuclear, px_um_x, px_um_y, config.nuclei)
        focus_a_labels, focus_a_diag = segment_spots(focus_a, nuclei_labels, config.focus_a)
        focus_b_labels, focus_b_diag = segment_spots(focus_b, nuclei_labels, config.focus_b)
        b_to_a, coloc_b, coloc_a, coloc_b_labels = colocalization_calls(
            focus_a_labels,
            focus_b_labels,
            config.colocalization_dilation_px,
        )

        nucleus_rows = _nucleus_rows(
            nuclei_labels,
            nuclear,
            focus_a,
            focus_b,
            focus_a_labels,
            focus_b_labels,
            coloc_b,
            sample,
            config.focus_a.name,
            config.focus_b.name,
            scale,
        )
        focus_a_rows = _focus_rows(
            focus_a_labels,
            focus_a,
            nuclei_labels,
            sample,
            config.focus_a.name,
            coloc_a,
            area_scale=scale,
        )
        focus_b_rows = _focus_rows(
            focus_b_labels,
            focus_b,
            nuclei_labels,
            sample,
            config.focus_b.name,
            coloc_b,
            overlap_map=b_to_a,
            area_scale=scale,
        )
        pair_rows = [
            {
                **sample,
                "nucleus_label": spot_to_nucleus_labels(focus_b_labels, nuclei_labels).get(b_label, 0),
                f"{config.focus_b.name}_label": b_label,
                f"{config.focus_a.name}_label": a_label,
            }
            for b_label, a_labels in b_to_a.items()
            for a_label in sorted(a_labels)
        ]

        sid = metadata.sample_id
        save_label_tiff(nuclei_labels, output_dir / "masks" / f"{sid}_nuclei_labels.tif")
        save_label_tiff(focus_a_labels, output_dir / "masks" / f"{sid}_{config.focus_a.name}_labels.tif")
        save_label_tiff(focus_b_labels, output_dir / "masks" / f"{sid}_{config.focus_b.name}_labels.tif")
        save_label_tiff(coloc_b_labels, output_dir / "masks" / f"{sid}_colocalized_{config.focus_b.name}_labels.tif")
        overlay_path = output_dir / "qc_overlays" / f"{sid}_qc_overlay.png"
        save_qc_overlay(nuclear, focus_a, focus_b, nuclei_labels, focus_a_labels, focus_b_labels, coloc_b_labels, overlay_path, config)
        save_individual_channel_qc(nuclear, nuclei_labels, output_dir / "qc_channels" / f"{sid}_nuclear_qc.png", config.nuclei.label, "nuclei", config)
        save_individual_channel_qc(focus_a, focus_a_labels, output_dir / "qc_channels" / f"{sid}_{config.focus_a.name}_qc.png", config.focus_a.label, config.focus_a.name, config)
        save_individual_channel_qc(focus_b, focus_b_labels, output_dir / "qc_channels" / f"{sid}_{config.focus_b.name}_qc.png", config.focus_b.label, config.focus_b.name, config)
        overlay_paths.append(overlay_path)

        image_rows.append(_image_summary(sample, inspection, nucleus_rows, focus_a_rows, focus_b_rows, pair_rows, nuclei_diag, focus_a_diag, focus_b_diag, config))
        nucleus_rows_all.extend(nucleus_rows)
        focus_a_rows_all.extend(focus_a_rows)
        focus_b_rows_all.extend(focus_b_rows)
        pair_rows_all.extend(pair_rows)
        print(
            f"{path.name}: nuclei={len(nucleus_rows)} "
            f"{config.focus_a.name}={len(focus_a_rows)} {config.focus_b.name}={len(focus_b_rows)} "
            f"coloc_{config.focus_b.name}={len(set(row[f'{config.focus_b.name}_label'] for row in pair_rows))}"
        )

    write_csv(output_dir / "image_summary.csv", image_rows)
    write_csv(output_dir / "nucleus_measurements.csv", nucleus_rows_all)
    write_csv(output_dir / f"{config.focus_a.name}_measurements.csv", focus_a_rows_all)
    write_csv(output_dir / f"{config.focus_b.name}_measurements.csv", focus_b_rows_all)
    write_csv(output_dir / "colocalized_focus_pairs.csv", pair_rows_all)
    make_qc_pdf(output_dir / "qc_contact_sheet.pdf", overlay_paths, qc_title)

    if nucleus_rows_all:
        nuclei = pd.DataFrame(nucleus_rows_all)
        summary_rows = []
        for (timepoint, condition), group in nuclei.groupby(["timepoint_hr", "condition"], sort=False):
            summary_rows.append(
                {
                    "timepoint_hr": timepoint,
                    "condition": condition,
                    "n_nuclei": int(len(group)),
                    f"mean_{config.focus_a.name}_per_nucleus": float(group[f"{config.focus_a.name}_count"].mean()),
                    f"mean_{config.focus_b.name}_per_nucleus": float(group[f"{config.focus_b.name}_count"].mean()),
                    f"mean_colocalized_{config.focus_b.name}_per_nucleus": float(group[f"colocalized_{config.focus_b.name}_count"].mean()),
                    f"fraction_nuclei_ge1_colocalized_{config.focus_b.name}": float((group[f"colocalized_{config.focus_b.name}_count"] >= 1).mean()),
                }
            )
        pd.DataFrame(summary_rows).to_csv(output_dir / "condition_summary.csv", index=False)

        control_conditions = set(config.control_conditions)
        if control_conditions:
            threshold_frames = []
            best_rows = []
            metrics = [
                f"{config.focus_a.name}_count",
                f"{config.focus_b.name}_count",
                f"colocalized_{config.focus_b.name}_count",
            ]
            for metric in metrics:
                explicit_thresholds = config.output.positive_thresholds.get(metric)
                table = threshold_performance(nuclei, metric, control_conditions, explicit_thresholds)
                table.to_csv(output_dir / f"threshold_performance_{metric}.csv", index=False)
                threshold_frames.append(table)
                if len(table):
                    best_rows.append(table.sort_values(["youden_j", "specificity", "sensitivity"], ascending=False).iloc[0].to_dict())
            pd.concat(threshold_frames, ignore_index=True).to_csv(output_dir / "threshold_performance_all_metrics.csv", index=False)
            pd.DataFrame(best_rows).to_csv(output_dir / "best_thresholds.csv", index=False)
