"""Export representative QC assets for segmentation optimisation.

This script intentionally writes derived examples only:

- labelled QC PNG/PDF assets from local analysis outputs;
- compact 8-bit cropped z-slice PNGs and multipage TIFF stacks;
- metadata with source file names only, not local absolute paths.

It does not copy raw CZI files.
"""

from __future__ import annotations

import csv
import json
import shutil
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import tifffile

from czi_foci.io import inspect_czi, pixel_size_um, read_channel_arrays
from czi_foci.segmentation import display_u8


REPO = Path(__file__).resolve().parents[1]
ASSET_ROOT = REPO / "examples" / "representative_images" / "trf2_yh2ax_qc_baseline"

RUNS = {
    "20x_2d": REPO / "output" / "trf2_yh2ax_20x_2d_representative",
    "63x_2d": REPO / "output" / "trf2_yh2ax_63x_2d_representative",
    "20x_3d": REPO / "output" / "trf2_yh2ax_20x_3d_firstpass",
    "63x_3d": REPO / "output" / "trf2_yh2ax_63x_3d_firstpass",
}

REPRESENTATIVE_FILES = {
    "Well1.czi": "well1_20x_wt_palbociclib_typical",
    "Well2.czi": "well2_20x_wt_palbociclib_baseline",
    "Well4.czi": "well4_20x_oversegmentation_case",
    "Well5.czi": "well5_20x_brca1ko_palbociclib",
    "Well6.czi": "well6_20x_brca1ko_palbociclib",
    "Well6 63x.czi": "well6_63x_parameter_test",
}

CHANNELS = {
    0: ("C0_RhReX_gH2AX", "RhReX / gamma-H2AX candidate channel"),
    1: ("C1_FITC_TRF2", "FITC / TRF2 telomere candidate channel"),
    2: ("C2_DAPI", "DAPI nuclear channel"),
}

README_TEXT = """# TRF2/yH2AX representative QC baseline assets

These derived assets are for Jules/Codex segmentation optimisation. They are not raw CZI files.

## Purpose

Use these examples to optimise DAPI nuclear segmentation, FITC/TRF2 spot segmentation, RhReX/gamma-H2AX spot segmentation, and 2D per-plane behaviour versus 3D all-Z behaviour.

The optimisation target is biology-constrained: do not make the images look cleaner by suppressing expected TRF2/telomere signal.

## Channel mapping

- `C0_RhReX_gH2AX`: RhReX / gamma-H2AX candidate channel.
- `C1_FITC_TRF2`: FITC / TRF2 telomere candidate channel.
- `C2_DAPI`: DAPI nuclear channel.

## Representative samples

- `Well1.czi`: WT palbociclib, typical 20x field.
- `Well2.czi`: WT palbociclib, second 20x baseline field.
- `Well4.czi`: 20x oversegmentation/failure case.
- `Well5.czi`: BRCA1-/- palbociclib, likely higher gamma-H2AX.
- `Well6.czi`: BRCA1-/- palbociclib, likely higher gamma-H2AX.
- `Well6 63x.czi`: separate 63x parameter-test case.

## Files

- `counts_2d_vs_3d_representative.csv`: per-image 2D and 3D first-pass counts.
- `derived_qc/*/qc_contact_sheet.pdf`: compact composite QC contact sheets.
- `derived_qc/*/qc_channels_contact_sheet.pdf`: compact single-channel QC contact sheets.
- `derived_qc/*/single_channel_pngs/`: labelled single-channel segmentation PNGs.
- `derived_qc/*/composite_pngs/`: labelled composite QC PNGs.
- `optimisation_crops/*/z*.png`: per-Z 2D crop slices.
- `optimisation_crops/*/stack.tif`: compact 8-bit display-scaled 3D crop stack.
- `optimisation_crops/*/metadata.json`: crop metadata without local absolute paths.

## Baseline standard for Jules

Treat 2D QC/contact sheets as the minimum baseline standard. The optimised 3D pipeline should not perform worse than the existing 2D scoring/segmentation standard unless the difference is explicitly justified as removal of clear background or artefact.

Important caveat: the current repo 2D mode uses the first Z plane for z-stacks. If an older validated 2D workflow used a different plane or projection, use the older validated counts as the stronger scoring guardrail. `Well2.czi` has very poor current first-Z 2D nuclear detection, so it should be treated as a warning case rather than a good nuclear baseline.

## Biology-constrained optimisation rules

- Preserve TRF2/telomere signal in WT palbociclib Wells 1 and 2.
- Do not optimise by simply minimising spot counts.
- Do not force BRCA1-/- Wells 5 and 6 to resemble WT fields.
- Do not tune gamma-H2AX downward only because BRCA1-/- has more signal.
- Produce separate 20x and 63x parameter presets.
- Stop after two optimisation rounds unless human review approves more.

## Safety notes

- No raw `.czi` files are included.
- Metadata contains source file names only, not local absolute paths.
- PNG/TIFF crop assets are 8-bit display-scaled derivatives for optimisation, not raw microscopy data.
"""


@dataclass(frozen=True)
class CropSpec:
    source_file: str
    slug: str
    crop_size_px: int


CROP_SPECS = [
    CropSpec("Well1.czi", "well1_20x_typical_crop", 192),
    CropSpec("Well4.czi", "well4_20x_oversegmentation_crop", 192),
    CropSpec("Well6 63x.czi", "well6_63x_parameter_crop", 256),
]


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_resized_png(src: Path, dst: Path, max_side: int = 768) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(src).convert("RGB")
    image.thumbnail((max_side, max_side))
    image.save(dst, optimize=True)


def make_compact_contact_sheet(image_paths: list[Path], output_pdf: Path, title: str) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    thumb_w, thumb_h = 220, 220
    cols = 3
    label_h = 34
    title_h = 46
    rows = max(1, int(np.ceil(len(image_paths) / cols)))
    sheet = Image.new("RGB", (cols * thumb_w, title_h + rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 14), title, fill=(0, 0, 0))
    for index, path in enumerate(image_paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h))
        x = (index % cols) * thumb_w + (thumb_w - image.width) // 2
        y = title_h + (index // cols) * (thumb_h + label_h)
        sheet.paste(image, (x, y))
        label = path.parent.name.replace("_", " ") + " / " + path.stem[-48:]
        draw.text(((index % cols) * thumb_w + 8, y + thumb_h + 4), label[:58], fill=(0, 0, 0))
    sheet.save(output_pdf, "PDF", resolution=120.0)


def representative_rows(run_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(run_dir / "image_summary.csv")
    return summary[summary["file_name"].isin(REPRESENTATIVE_FILES)].copy()


def validate_required_inputs(data_root: Path) -> None:
    """Validate all local-only inputs before clearing committed assets."""
    errors: list[str] = []
    for run_name, run_dir in RUNS.items():
        summary_path = run_dir / "image_summary.csv"
        qc_channels_dir = run_dir / "qc_channels"
        qc_overlays_dir = run_dir / "qc_overlays"
        if not summary_path.is_file():
            errors.append(f"{run_name}: missing {summary_path}")
            continue
        if not qc_channels_dir.is_dir():
            errors.append(f"{run_name}: missing {qc_channels_dir}")
        if not qc_overlays_dir.is_dir():
            errors.append(f"{run_name}: missing {qc_overlays_dir}")
        try:
            rows = representative_rows(run_dir)
        except Exception as exc:
            errors.append(f"{run_name}: could not read representative rows from {summary_path}: {exc}")
            continue
        present_files = set(rows["file_name"].astype(str))
        expected_files = {
            file_name
            for file_name in REPRESENTATIVE_FILES
            if ("63x" in run_name) == ("63x" in file_name)
        }
        missing_files = sorted(expected_files - present_files)
        if missing_files:
            errors.append(f"{run_name}: missing representative rows for {', '.join(missing_files)}")
        for row in rows.itertuples(index=False):
            sample_id = row.sample_id
            if not list(qc_channels_dir.glob(f"{sample_id}_*.png")):
                errors.append(f"{run_name}: missing single-channel QC PNGs for {sample_id}")
            if not list(qc_overlays_dir.glob(f"{sample_id}_*.png")):
                errors.append(f"{run_name}: missing composite QC PNGs for {sample_id}")
    for spec in CROP_SPECS:
        czi_path = data_root / spec.source_file
        if not czi_path.is_file():
            errors.append(f"data-root: missing representative CZI source {czi_path}")
    if errors:
        joined = "\n- ".join(errors)
        raise FileNotFoundError(f"Cannot export representative QC assets; required inputs are missing:\n- {joined}")


def copy_qc_assets() -> None:
    for run_name, run_dir in RUNS.items():
        dst_root = ASSET_ROOT / "derived_qc" / run_name
        dst_root.mkdir(parents=True, exist_ok=True)
        rows = representative_rows(run_dir)
        copied: list[dict[str, str]] = []
        channel_assets: list[Path] = []
        composite_assets: list[Path] = []
        for row in rows.itertuples(index=False):
            slug = REPRESENTATIVE_FILES[row.file_name]
            sample_id = row.sample_id
            for src in sorted((run_dir / "qc_channels").glob(f"{sample_id}_*.png")):
                dst = dst_root / "single_channel_pngs" / slug / src.name
                copy_resized_png(src, dst)
                channel_assets.append(dst)
                copied.append({"run": run_name, "file_name": row.file_name, "asset": str(dst.relative_to(ASSET_ROOT))})
            for src in sorted((run_dir / "qc_overlays").glob(f"{sample_id}_*.png")):
                dst = dst_root / "composite_pngs" / slug / src.name
                copy_resized_png(src, dst)
                composite_assets.append(dst)
                copied.append({"run": run_name, "file_name": row.file_name, "asset": str(dst.relative_to(ASSET_ROOT))})
        make_compact_contact_sheet(composite_assets, dst_root / "qc_contact_sheet.pdf", f"{run_name} composite QC")
        make_compact_contact_sheet(channel_assets, dst_root / "qc_channels_contact_sheet.pdf", f"{run_name} single-channel QC")
        with (dst_root / "asset_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["run", "file_name", "asset"])
            writer.writeheader()
            writer.writerows(copied)


def write_comparison_csv() -> None:
    frames = []
    for run_name, run_dir in RUNS.items():
        frame = representative_rows(run_dir)
        frame.insert(0, "run", run_name)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["TRF2_per_nucleus"] = combined["FITC_TRF2_candidate_total_count"] / combined["nuclei_count"]
    combined["gH2AX_per_nucleus"] = combined["RhReX_gH2AX_candidate_total_count"] / combined["nuclei_count"]
    combined["coloc_gH2AX_per_nucleus"] = combined["colocalized_RhReX_gH2AX_candidate_total_count"] / combined["nuclei_count"]
    keep = [
        "run",
        "file_name",
        "sample_id",
        "nuclei_count",
        "FITC_TRF2_candidate_total_count",
        "RhReX_gH2AX_candidate_total_count",
        "colocalized_RhReX_gH2AX_candidate_total_count",
        "colocalized_pairs_count",
        "TRF2_per_nucleus",
        "gH2AX_per_nucleus",
        "coloc_gH2AX_per_nucleus",
        "voxel_size_z_um",
        "voxel_size_y_um",
        "voxel_size_x_um",
    ]
    comparison = combined[[col for col in keep if col in combined.columns]].copy()
    comparison.to_csv(ASSET_ROOT / "counts_2d_vs_3d_representative.csv", index=False)


def crop_bounds(stack: np.ndarray, crop_size: int) -> tuple[int, int, int, int]:
    mip = stack.max(axis=0)
    threshold = np.percentile(mip, 99.0)
    ys, xs = np.nonzero(mip >= threshold)
    if len(xs) == 0:
        cy, cx = np.array(mip.shape) // 2
    else:
        cy = int(np.median(ys))
        cx = int(np.median(xs))
    half = crop_size // 2
    y0 = max(0, min(cy - half, mip.shape[0] - crop_size))
    x0 = max(0, min(cx - half, mip.shape[1] - crop_size))
    y1 = min(mip.shape[0], y0 + crop_size)
    x1 = min(mip.shape[1], x0 + crop_size)
    return y0, y1, x0, x1


def save_stack_assets(crop: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_stack: list[np.ndarray] = []
    for z_index, plane in enumerate(crop):
        rendered = display_u8(plane, 1.0, 99.8)
        Image.fromarray(rendered, mode="L").save(output_dir / f"z{z_index:03d}.png")
        rendered_stack.append(rendered)
    if rendered_stack:
        tifffile.imwrite(output_dir / "stack.tif", np.stack(rendered_stack), photometric="minisblack")


def export_crops(data_root: Path) -> None:
    for spec in CROP_SPECS:
        czi_path = data_root / spec.source_file
        inspection = inspect_czi(czi_path)
        arrays = read_channel_arrays(czi_path, inspection, mode="3d")
        pz, py, px = pixel_size_um(inspection["metadata"], mode="3d")
        y0, y1, x0, x1 = crop_bounds(arrays[2], spec.crop_size_px)
        crop_root = ASSET_ROOT / "optimisation_crops" / spec.slug
        crop_root.mkdir(parents=True, exist_ok=True)
        metadata = {
            "source_file": spec.source_file,
            "purpose": "small derived crop for 2D per-Z and 3D segmentation optimisation",
            "crop_box_px": {"y0": y0, "y1": y1, "x0": x0, "x1": x1},
            "voxel_size_um": {"z": pz, "y": py, "x": px},
            "channels": {str(index): {"folder": folder, "label": label} for index, (folder, label) in CHANNELS.items()},
            "contains_raw_czi": False,
            "notes": "PNG/TIFF assets are 8-bit display-scaled crops, not raw CZI files.",
        }
        (crop_root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for channel_index, (folder, _label) in CHANNELS.items():
            crop = arrays[channel_index][:, y0:y1, x0:x1]
            save_stack_assets(crop, crop_root / folder)


def write_root_manifest() -> None:
    rows = []
    for path in sorted(ASSET_ROOT.rglob("*")):
        if path.is_file():
            rows.append({"path": str(path.relative_to(ASSET_ROOT)), "bytes": path.stat().st_size})
    with (ASSET_ROOT / "asset_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes"])
        writer.writeheader()
        writer.writerows(rows)


def write_readme() -> None:
    (ASSET_ROOT / "README.md").write_text(README_TEXT, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Local folder containing representative CZI files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_required_inputs(args.data_root)
    reset_dir(ASSET_ROOT)
    copy_qc_assets()
    write_comparison_csv()
    export_crops(args.data_root)
    write_readme()
    write_root_manifest()
    print(f"Wrote representative QC assets to {ASSET_ROOT}")


if __name__ == "__main__":
    main()
