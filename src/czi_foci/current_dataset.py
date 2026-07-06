"""Batch-aware reports for the existing TS III/IV/V dual-channel dataset."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from skimage import segmentation

from .batch import (
    effect_sizes,
    robust_control_stats,
    threshold_stability,
    threshold_table,
)
from .io import inspect_czi, read_channel_arrays
from .segmentation import display_u8


METRICS = ["af488_spots_count", "rhrex_spots_count", "colocalized_rhrex_spots_count"]
NORMALISED_METRICS = [f"{metric}_control_z" for metric in METRICS]
EXPERIMENT_DIRS = {
    "TS III": Path("output/segmentation/time_series_III_dual_channel_coloc"),
    "TS IV": Path("output/segmentation/time_series_IV_dual_channel_coloc"),
    "TS V": Path("output/segmentation/time_series_V_dual_channel_coloc"),
}
QC_NAMES = {"TS III": "TSIII", "TS IV": "TSIV", "TS V": "TSV"}


def _ensure_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def load_current_dataset(nucleus_csv: Path, image_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    nuclei = pd.read_csv(nucleus_csv)
    images = pd.read_csv(image_csv)
    _ensure_columns(
        nuclei,
        {
            "experiment",
            "timepoint_hr",
            "condition",
            "known_control",
            "sample_id",
            "file_name",
            "replicate",
            *METRICS,
        },
        str(nucleus_csv),
    )
    _ensure_columns(
        images,
        {
            "experiment",
            "timepoint_hr",
            "condition",
            "known_control",
            "sample_id",
            "file_name",
            "replicate",
            "source_path",
            "nuclei_count",
            "af488_threshold",
            "rhrex_threshold",
        },
        str(image_csv),
    )
    nuclei["known_control"] = nuclei["known_control"].astype(bool)
    images["known_control"] = images["known_control"].astype(bool)
    return nuclei, images


def threshold_table_from_control_mask(frame: pd.DataFrame, metric: str, control_col: str) -> pd.DataFrame:
    data = frame.copy()
    data["condition"] = np.where(data[control_col].astype(bool), "control", "treated")
    return threshold_table(data, metric, {"control"})


def best_thresholds_by_experiment(nuclei: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for experiment, batch in nuclei.groupby("experiment", sort=False, dropna=False):
        for metric in metrics:
            table = threshold_table_from_control_mask(batch, metric, "known_control")
            best = table.sort_values(["youden_j", "specificity", "sensitivity"], ascending=False).iloc[0].to_dict()
            rows.append({"experiment": experiment, **best, "n_nuclei": int(len(batch))})
    return pd.DataFrame(rows)


def add_experiment_classifications(nuclei: pd.DataFrame, raw_thresholds: pd.DataFrame) -> pd.DataFrame:
    classified = nuclei.copy()
    threshold_map = {
        (row.experiment, row.metric): float(row.positive_if_ge)
        for row in raw_thresholds.itertuples(index=False)
    }
    for metric in METRICS:
        threshold_col = f"{metric}_batch_threshold"
        positive_col = {
            "af488_spots_count": "af488_positive_batch",
            "rhrex_spots_count": "rhrex_positive_batch",
            "colocalized_rhrex_spots_count": "coloc_positive_batch",
        }[metric]
        classified[threshold_col] = classified["experiment"].map(
            {experiment: threshold_map[(experiment, metric)] for experiment in classified["experiment"].unique()}
        )
        classified[positive_col] = classified[metric].astype(float) >= classified[threshold_col].astype(float)
    classified["dual_channel_positive_batch"] = classified["af488_positive_batch"] & classified["rhrex_positive_batch"]
    return classified


def field_summary(classified: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["experiment", "timepoint_hr", "condition", "sample_id", "file_name", "replicate", "known_control"]
    for key, group in classified.groupby(group_cols, sort=False, dropna=False):
        row = dict(zip(group_cols, key))
        row.update(
            {
                "n_nuclei": int(len(group)),
                "mean_af488_spots_per_nucleus": float(group["af488_spots_count"].mean()),
                "mean_rhrex_spots_per_nucleus": float(group["rhrex_spots_count"].mean()),
                "mean_colocalized_rhrex_spots_per_nucleus": float(group["colocalized_rhrex_spots_count"].mean()),
                "fraction_af488_positive_batch": float(group["af488_positive_batch"].mean()),
                "fraction_rhrex_positive_batch": float(group["rhrex_positive_batch"].mean()),
                "fraction_dual_channel_positive_batch": float(group["dual_channel_positive_batch"].mean()),
                "fraction_coloc_positive_batch": float(group["coloc_positive_batch"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def condition_timepoint_summary(fields: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["experiment", "timepoint_hr", "condition", "known_control"]
    metrics = [
        "mean_af488_spots_per_nucleus",
        "mean_rhrex_spots_per_nucleus",
        "mean_colocalized_rhrex_spots_per_nucleus",
        "fraction_af488_positive_batch",
        "fraction_rhrex_positive_batch",
        "fraction_dual_channel_positive_batch",
        "fraction_coloc_positive_batch",
    ]
    for key, group in fields.groupby(group_cols, sort=False, dropna=False):
        row = dict(zip(group_cols, key))
        row["n_fields"] = int(len(group))
        row["total_nuclei"] = int(group["n_nuclei"].sum())
        for metric in metrics:
            row[f"mean_field_{metric}"] = float(group[metric].mean())
            row[f"sem_field_{metric}"] = float(group[metric].std(ddof=1) / np.sqrt(len(group))) if len(group) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _colour_panel(image: np.ndarray, labels: np.ndarray, boundary_colour: tuple[int, int, int]) -> Image.Image:
    grey = display_u8(image, 1.0, 99.8)
    rgb = np.dstack([grey, grey, grey])
    if labels.max() > 0:
        rgb[segmentation.find_boundaries(labels, mode="outer")] = np.array(boundary_colour, dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path), dtype=np.uint16)


def _mask_paths(experiment: str, sample_id: str) -> dict[str, Path]:
    base = EXPERIMENT_DIRS[experiment] / "masks"
    return {
        "nuclei": base / f"{sample_id}_nuclei_labels.tif",
        "af488": base / f"{sample_id}_af488_spot_labels.tif",
        "rhrex": base / f"{sample_id}_rhrex_putative_gH2AX_labels.tif",
        "coloc": base / f"{sample_id}_colocalized_rhrex_labels.tif",
        "overlay": EXPERIMENT_DIRS[experiment] / "qc_overlays" / f"{sample_id}_dual_qc_overlay.png",
    }


def _draw_qc_page(pdf: canvas.Canvas, row, threshold_text: str, tmpdir: Path) -> None:
    width, height = landscape(A4)
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(32, height - 28, f"{row.experiment} control QC: {row.condition}, {row.timepoint_hr} hr, replicate {str(row.replicate).zfill(2)}")
    pdf.setFont("Helvetica", 7)
    pdf.drawString(32, height - 42, f"File: {row.file_name}")
    pdf.drawString(32, height - 53, f"Sample: {row.sample_id}")
    pdf.drawString(32, height - 64, f"Nuclei: {int(row.nuclei_count)} | AF488 segmentation threshold: {float(row.af488_threshold):.3f} | RhReX segmentation threshold: {float(row.rhrex_threshold):.3f}")
    pdf.drawString(32, height - 75, threshold_text)

    inspection = inspect_czi(Path(row.source_path))
    arrays = read_channel_arrays(Path(row.source_path), inspection)
    masks = _mask_paths(row.experiment, row.sample_id)
    nuclei_labels = _load_mask(masks["nuclei"])
    af488_labels = _load_mask(masks["af488"])
    rhrex_labels = _load_mask(masks["rhrex"])

    panels = [
        ("DAPI + nuclei", _colour_panel(arrays[2], nuclei_labels, (255, 255, 0))),
        ("AF488 + spots", _colour_panel(arrays[1], af488_labels, (0, 255, 255))),
        ("RhReX + foci", _colour_panel(arrays[0], rhrex_labels, (255, 128, 0))),
        ("Composite co-localisation", Image.open(masks["overlay"]).convert("RGB")),
    ]
    panel_w, panel_h = 180, 180
    for index, (title, image) in enumerate(panels):
        col = index % 4
        x = 32 + col * 198
        y = height - 285
        thumb = image.copy()
        thumb.thumbnail((panel_w, panel_h))
        tmp = tmpdir / f"{row.sample_id}_{index}.png"
        thumb.save(tmp)
        pdf.drawImage(ImageReader(str(tmp)), x, y, panel_w, panel_h, preserveAspectRatio=True, anchor="c")
        pdf.setFont("Helvetica-Bold", 8)
        pdf.drawString(x, y - 12, title)
    pdf.showPage()


def write_control_qc_pdf(images: pd.DataFrame, thresholds: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    threshold_lookup = {
        (row.experiment, row.metric): float(row.positive_if_ge)
        for row in thresholds.itertuples(index=False)
    }
    controls = images[images["known_control"]].copy()
    controls = controls.sort_values(["experiment", "timepoint_hr", "condition", "replicate", "sample_id"])
    with tempfile.TemporaryDirectory() as temp:
        tmpdir = Path(temp)
        pdf = canvas.Canvas(str(output), pagesize=landscape(A4))
        for row in controls.itertuples(index=False):
            threshold_text = (
                "Experiment-specific positivity cut-offs: "
                f"AF488 >= {threshold_lookup[(row.experiment, 'af488_spots_count')]:g}; "
                f"RhReX >= {threshold_lookup[(row.experiment, 'rhrex_spots_count')]:g}; "
                f"co-localised RhReX >= {threshold_lookup[(row.experiment, 'colocalized_rhrex_spots_count')]:g}"
            )
            _draw_qc_page(pdf, row, threshold_text, tmpdir)
        pdf.save()


def write_qc_pdfs(images: pd.DataFrame, thresholds: pd.DataFrame, output_dir: Path) -> None:
    write_control_qc_pdf(images, thresholds, output_dir / "control_qc_all_batches.pdf")
    for experiment, label in QC_NAMES.items():
        subset = images[images["experiment"].eq(experiment)]
        write_control_qc_pdf(subset, thresholds[thresholds["experiment"].eq(experiment)], output_dir / f"control_qc_{label}.pdf")


def run_current_dataset_report(
    nucleus_csv: Path,
    image_csv: Path,
    output_dir: Path,
    overwrite: bool = False,
    write_qc: bool = True,
) -> dict[str, Path]:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    nuclei, images = load_current_dataset(nucleus_csv, image_csv)
    normalised = nuclei.copy()
    normalised["is_control"] = normalised["known_control"].astype(bool)
    control_rows = []
    for experiment, batch in normalised.groupby("experiment", sort=False):
        controls = batch[batch["is_control"]]
        batch_mask = normalised["experiment"].eq(experiment)
        for metric in METRICS:
            median, robust_sd = robust_control_stats(controls[metric])
            normalised.loc[batch_mask, f"{metric}_control_z"] = (
                normalised.loc[batch_mask, metric].astype(float) - median
            ) / robust_sd
            control_rows.append(
                {
                    "experiment": experiment,
                    "metric": metric,
                    "control_n": int(len(controls)),
                    "control_median": median,
                    "control_robust_sd": robust_sd,
                }
            )
    control_stats = pd.DataFrame(control_rows)

    raw_thresholds = best_thresholds_by_experiment(normalised, METRICS)
    z_thresholds = best_thresholds_by_experiment(normalised, NORMALISED_METRICS)
    stability = pd.concat(
        [
            threshold_stability(raw_thresholds, ["experiment"]).assign(threshold_type="raw_count"),
            threshold_stability(z_thresholds, ["experiment"]).assign(threshold_type="control_z"),
        ],
        ignore_index=True,
    )
    classified = add_experiment_classifications(normalised, raw_thresholds)
    fields = field_summary(classified)
    condition_summary = condition_timepoint_summary(fields)
    effects = effect_sizes(normalised, NORMALISED_METRICS, set(), ["experiment", "timepoint_hr"])
    # Recompute effect control flags using known_control rather than condition names.
    known_by_condition = normalised[["experiment", "timepoint_hr", "condition", "known_control"]].drop_duplicates()
    effects = effects.drop(columns=["is_control"], errors="ignore").merge(
        known_by_condition,
        on=["experiment", "timepoint_hr", "condition"],
        how="left",
    )

    outputs = {
        "batch_control_stats": output_dir / "batch_control_stats.csv",
        "batch_thresholds_raw_counts": output_dir / "batch_thresholds_raw_counts.csv",
        "batch_thresholds_control_z": output_dir / "batch_thresholds_control_z.csv",
        "threshold_stability": output_dir / "threshold_stability.csv",
        "nucleus_batch_classifications": output_dir / "nucleus_batch_classifications.csv",
        "field_batch_summary": output_dir / "field_batch_summary.csv",
        "condition_timepoint_batch_summary": output_dir / "condition_timepoint_batch_summary.csv",
        "condition_batch_effect_sizes": output_dir / "condition_batch_effect_sizes.csv",
    }
    control_stats.to_csv(outputs["batch_control_stats"], index=False)
    raw_thresholds.to_csv(outputs["batch_thresholds_raw_counts"], index=False)
    z_thresholds.to_csv(outputs["batch_thresholds_control_z"], index=False)
    stability.to_csv(outputs["threshold_stability"], index=False)
    classified.to_csv(outputs["nucleus_batch_classifications"], index=False)
    fields.to_csv(outputs["field_batch_summary"], index=False)
    condition_summary.to_csv(outputs["condition_timepoint_batch_summary"], index=False)
    effects.to_csv(outputs["condition_batch_effect_sizes"], index=False)

    if write_qc:
        write_qc_pdfs(images, raw_thresholds, output_dir)
    return outputs
