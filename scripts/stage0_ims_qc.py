"""Stage 0: Imaris (.ims) reader QC/triage pass.

Walks every tile under the TRF2 root, computes a per-tile QC row (focus,
sharpness, saturation, nucleus count, etc.) at ResolutionLevel 2 for speed,
and writes:

  stage0_tile_qc.csv   one row per tile
  stage0_summary.csv   one row per folder ("round")
  stage0_qc.pdf        one reviewable page per successfully-read tile

This is QC/triage only: it does not assign wells, does not run real
segmentation, and does not tune thresholds. See the module docstring in
czi_foci/ims_reader.py for the three .ims traps (char-array attrs, Z padding,
resolution pyramid) this script relies on.
"""

from __future__ import annotations

import argparse
import os
import traceback
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import ndimage as ndi
from skimage import feature, filters, morphology, segmentation

from czi_foci.ims_reader import (
    inspect_ims,
    pixel_size_um_ims,
    read_channel_arrays_ims,
    recover_metadata_from_sibling,
    stage_position_mm,
)

QC_RESOLUTION_LEVEL = 2
CHANNEL_NAMES = {0: "RFP_561", 1: "DAPI_405", 2: "GFP_488"}
DAPI_CHANNEL = 1
SATURATION_THRESHOLD = 65000
SHARPNESS_CONTAINMENT_THRESHOLD = 0.35
NUCLEUS_MIN_DISTANCE_PX = 18
NUCLEUS_MIN_SIZE_PX = 800


def common_prefix_suffix(names: list[str]) -> tuple[str, str]:
    prefix = os.path.commonprefix(names)
    suffix = os.path.commonprefix([n[::-1] for n in names])[::-1]
    return prefix, suffix


def derive_round(folder_name: str, prefix: str, suffix: str) -> str:
    """Derive a short 'round' label from a folder name, matching the
    ``round`` column convention in stage1_per_tile.csv (a leading-underscore
    numeric suffix, or "(unnumbered)"). Deconvolved status is kept in the
    round text so each folder still maps to its own unique round — the
    ``deconvolved`` column also flags it per-row.
    """
    middle = folder_name[len(prefix):]
    if suffix and middle.endswith(suffix):
        middle = middle[: -len(suffix)]
    middle = middle.strip("_")
    if not middle:
        return "(unnumbered)"
    return f"_{middle}"


@dataclass
class TileResult:
    row: dict
    thumbnail: np.ndarray | None = None
    panels: dict | None = None  # {"DAPI_405": plane, "RFP_561": plane, "GFP_488": plane}
    sharpness: np.ndarray | None = None


def per_slice_sharpness(dapi_stack: np.ndarray) -> np.ndarray:
    n_z = dapi_stack.shape[0]
    sharp = np.zeros(n_z, dtype=np.float64)
    for z in range(n_z):
        a = dapi_stack[z].astype(np.float32)
        sharp[z] = ndi.laplace(ndi.gaussian_filter(a, 1.0)).var() / max(float(a.mean()) ** 2, 1e-9)
    peak = sharp.max()
    if peak > 0:
        sharp = sharp / peak
    return sharp


def count_nuclei(dapi_stack: np.ndarray, peak_idx: int) -> int:
    lo = max(peak_idx - 2, 0)
    proj = dapi_stack[lo : peak_idx + 3].astype(np.float32).max(0)
    proj = ndi.gaussian_filter(proj, 2)
    if np.unique(proj).size <= 1:
        return 0
    m = proj > filters.threshold_otsu(proj)
    m = morphology.remove_small_objects(m, NUCLEUS_MIN_SIZE_PX)
    m = ndi.binary_fill_holes(m)
    if not m.any():
        return 0
    dist = ndi.distance_transform_edt(m)
    pks = feature.peak_local_max(dist, min_distance=NUCLEUS_MIN_DISTANCE_PX, labels=m)
    if len(pks) == 0:
        return 0
    mk = np.zeros(m.shape, int)
    mk[tuple(pks.T)] = np.arange(1, len(pks) + 1)
    lab = segmentation.watershed(-dist, mk, mask=m)
    return int(lab.max())


def channel_percentiles(stack: np.ndarray) -> dict:
    flat = stack.astype(np.float32)
    p5, p50, p999 = np.percentile(flat, [5, 50, 99.9])
    return {
        "p5": float(p5),
        "p50": float(p50),
        "p999": float(p999),
        "max": float(stack.max()),
        "saturated_frac": float(np.count_nonzero(stack >= SATURATION_THRESHOLD) / stack.size),
    }


def bleach_slope(stack: np.ndarray) -> float:
    n_z = stack.shape[0]
    if n_z < 2:
        return float("nan")
    p995 = np.percentile(stack.astype(np.float32).reshape(n_z, -1), 99.5, axis=1)
    slope, _ = np.polyfit(np.arange(n_z), p995, 1)
    return float(slope)


def process_tile(args: tuple[Path, str]) -> TileResult:
    path, folder_name = args
    deconvolved = "deconvolved" in path.name.lower() or "deconvolved" in folder_name.lower()
    row: dict = {
        "round": folder_name,
        "tile": path.name,
        "deconvolved": deconvolved,
        "read_ok": False,
        "error": "",
        "recovered_metadata_from_sibling": False,
    }

    try:
        inspection = inspect_ims(path)

        if "missing DataSet group" in inspection.get("errors", []):
            # No pixel data at all (e.g. a truncated/empty .ims) - metadata
            # recovery would be pointless since there is nothing to read.
            row["error"] = "; ".join(inspection["errors"])
            return TileResult(row=row)

        metadata = inspection["metadata"]
        true_z = inspection.get("true_z")

        if metadata is None or not metadata.get("has_dataset_info") or true_z is None:
            sibling_metadata = recover_metadata_from_sibling(path)
            if sibling_metadata is not None:
                metadata = sibling_metadata
                true_z = int(metadata["image"]["Z"])
                row["recovered_metadata_from_sibling"] = True
            else:
                row["error"] = "; ".join(inspection.get("errors") or ["missing DataSetInfo; no sibling to recover from"])
                return TileResult(row=row)

        if any("missing Data object" in e for e in inspection.get("errors", [])) and len(inspection.get("channels", {})) == 0:
            row["error"] = "no channel Data objects present"
            return TileResult(row=row)

        z_step_um, y_um, x_um = pixel_size_um_ims(metadata, mode="3d")
        stage_x_mm, stage_y_mm = stage_position_mm(metadata)
        acquisition = metadata.get("custom", {}).get("DateAndTime", "")

        arrays = read_channel_arrays_ims(path, {**inspection, "true_z": true_z}, mode="3d", resolution_level=QC_RESOLUTION_LEVEL)
        if DAPI_CHANNEL not in arrays:
            row["error"] = f"missing DAPI (channel {DAPI_CHANNEL}) Data object"
            return TileResult(row=row)

        dapi = arrays[DAPI_CHANNEL]
        sharp = per_slice_sharpness(dapi)
        focus_peak_idx = int(sharp.argmax())
        axially_contained = bool(sharp[0] < SHARPNESS_CONTAINMENT_THRESHOLD and sharp[-1] < SHARPNESS_CONTAINMENT_THRESHOLD)
        n_nuclei = count_nuclei(dapi, focus_peak_idx)

        row.update(
            {
                "read_ok": True,
                "true_z": int(true_z),
                "z_step_um": z_step_um,
                "z_span_um": z_step_um * (true_z - 1) if true_z > 1 else 0.0,
                "pixel_size_um": x_um,
                "stage_x_mm": stage_x_mm,
                "stage_y_mm": stage_y_mm,
                "acquisition_datetime": acquisition,
                "focus_peak_idx": focus_peak_idx,
                "sharpness_at_first": float(sharp[0]),
                "sharpness_at_last": float(sharp[-1]),
                "axially_contained": axially_contained,
                "n_nuclei": n_nuclei,
            }
        )

        panels = {}
        for index, name in CHANNEL_NAMES.items():
            if index not in arrays:
                continue
            stack = arrays[index]
            stats = channel_percentiles(stack)
            for key, value in stats.items():
                row[f"{name}_{key}"] = value
            row[f"bleach_slope_{name}"] = bleach_slope(stack)
            plane_idx = min(focus_peak_idx, stack.shape[0] - 1)
            panels[name] = stack[plane_idx]

        return TileResult(row=row, thumbnail=panels.get("DAPI_405"), panels=panels, sharpness=sharp)

    except Exception as exc:  # noqa: BLE001 - QC pass must not crash on a bad tile
        row["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
        return TileResult(row=row)


def discover_tiles(root: Path, output_dir: Path) -> tuple[list[tuple[Path, str]], list[str], dict[str, int]]:
    output_dir = output_dir.resolve()
    folders = sorted(p for p in root.iterdir() if p.is_dir() and p.resolve() != output_dir)
    tiles: list[tuple[Path, str]] = []
    tile_counts: dict[str, int] = {}
    for folder in folders:
        folder_tiles = sorted(folder.glob("*.ims"))
        tile_counts[folder.name] = len(folder_tiles)
        for tile in folder_tiles:
            tiles.append((tile, folder.name))
    return tiles, [f.name for f in folders], tile_counts


def display_u16_to_u8(plane: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.9) -> np.ndarray:
    lo, hi = np.percentile(plane, [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((plane.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return (scaled * 255).astype(np.uint8)


def render_tile_page(pdf: PdfPages, row: dict, panels: dict, sharpness: np.ndarray) -> None:
    fig = plt.figure(figsize=(11, 8.5), dpi=300)
    grid = fig.add_gridspec(2, 3, height_ratios=[3, 1.4])

    for col, name in enumerate(["DAPI_405", "RFP_561", "GFP_488"]):
        ax = fig.add_subplot(grid[0, col])
        if name in panels:
            ax.imshow(display_u16_to_u8(panels[name]), cmap="gray", interpolation="nearest")
        ax.set_title(name, fontsize=9)
        ax.axis("off")

    ax_curve = fig.add_subplot(grid[1, :])
    ax_curve.plot(np.arange(len(sharpness)), sharpness, marker="o", markersize=3)
    ax_curve.axvline(row["focus_peak_idx"], color="red", linestyle="--", linewidth=0.8)
    ax_curve.set_xlabel("Z slice")
    ax_curve.set_ylabel("normalised sharpness")
    ax_curve.set_ylim(0, 1.05)

    caption = (
        f"round={row['round']}  tile={row['tile']}  deconvolved={row['deconvolved']}\n"
        f"Z={row.get('true_z')}  z_step={row.get('z_step_um'):.3f} um  "
        f"z_span={row.get('z_span_um'):.2f} um  xy_pixel={row.get('pixel_size_um'):.4f} um\n"
        f"focus_peak_idx={row['focus_peak_idx']}  axially_contained={row['axially_contained']}  "
        f"n_nuclei={row['n_nuclei']}  acquired={row.get('acquisition_datetime', '')}"
    )
    fig.suptitle(caption, fontsize=8, y=0.99, ha="center")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def render_summary_page(pdf: PdfPages, summary_df: pd.DataFrame, failures: list[dict], thumbnails: dict) -> None:
    fig = plt.figure(figsize=(11, 8.5), dpi=300)
    fig.suptitle("Stage 0 QC: per-folder summary", fontsize=13, y=0.98)

    ax_table = fig.add_axes((0.03, 0.55, 0.94, 0.4))
    ax_table.axis("off")
    cols = ["round", "n_tiles", "true_z", "z_step_um", "focus_range", "verdict"]
    table_data = summary_df[cols].astype(str).values if len(summary_df) else [[""] * len(cols)]
    table = ax_table.table(cellText=table_data, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1, 1.3)

    n_thumbs = min(8, len(thumbnails))
    if n_thumbs:
        thumb_items = list(thumbnails.items())[:n_thumbs]
        for i, (folder, thumb) in enumerate(thumb_items):
            ax = fig.add_axes((0.03 + (i % 4) * 0.235, 0.05 + (i // 4) * 0.22, 0.21, 0.2))
            ax.imshow(display_u16_to_u8(thumb), cmap="gray")
            ax.set_title(folder[-28:], fontsize=5)
            ax.axis("off")

    if failures:
        text = "Read failures:\n" + "\n".join(f"- {f['round']}/{f['tile']}: {f['error'].splitlines()[0]}" for f in failures[:20])
        fig.text(0.03, 0.02, text, fontsize=6, va="bottom", family="monospace")

    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def summarize_folders(tile_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for folder, group in tile_df.groupby("round", sort=False):
        ok = group[group["read_ok"]]
        n_tiles = len(group)
        n_ok = len(ok)
        if n_ok == 0:
            rows.append(
                {
                    "round": folder,
                    "n_tiles": n_tiles,
                    "n_read_ok": 0,
                    "true_z": float("nan"),
                    "z_step_um": float("nan"),
                    "focus_range": "",
                    "frac_axially_contained": float("nan"),
                    "verdict": "UNUSABLE: no tiles could be read",
                }
            )
            continue
        z_vals = ok["true_z"].dropna().unique()
        z_step_vals = ok["z_step_um"].dropna().unique()
        focus_lo, focus_hi = ok["focus_peak_idx"].min(), ok["focus_peak_idx"].max()
        frac_contained = float(ok["axially_contained"].mean())
        verdict = "OK"
        if n_ok < n_tiles:
            verdict = f"PARTIAL: {n_tiles - n_ok}/{n_tiles} tiles failed to read"
        elif frac_contained < 0.5:
            verdict = "REVIEW: many tiles not axially contained"
        rows.append(
            {
                "round": folder,
                "n_tiles": n_tiles,
                "n_read_ok": n_ok,
                "true_z": z_vals[0] if len(z_vals) == 1 else "/".join(str(int(v)) for v in sorted(z_vals)),
                "z_step_um": round(float(z_step_vals[0]), 3) if len(z_step_vals) == 1 else "/".join(f"{v:.3f}" for v in sorted(z_step_vals)),
                "focus_range": f"{int(focus_lo)}-{int(focus_hi)}",
                "frac_axially_contained": frac_contained,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def vignetting_check(tile_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for folder, group in tile_df.groupby("round", sort=False):
        ok = group[group["read_ok"]].dropna(subset=["stage_x_mm", "stage_y_mm", "n_nuclei"])
        if len(ok) < 3:
            rows.append({"round": folder, "vignetting_slope": float("nan"), "n_tiles_used": len(ok)})
            continue
        cx, cy = ok["stage_x_mm"].mean(), ok["stage_y_mm"].mean()
        radius = np.hypot(ok["stage_x_mm"] - cx, ok["stage_y_mm"] - cy)
        if radius.std() == 0:
            rows.append({"round": folder, "vignetting_slope": float("nan"), "n_tiles_used": len(ok)})
            continue
        slope, _ = np.polyfit(radius, ok["n_nuclei"], 1)
        rows.append({"round": folder, "vignetting_slope": float(slope), "n_tiles_used": len(ok)})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/Volumes/Backup/Andor imaging/TRF2"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N tiles (debugging)")
    args = parser.parse_args()

    output_dir = args.output_dir or (args.root / "_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    tiles, folder_names, tile_counts = discover_tiles(args.root, output_dir)
    if args.limit:
        tiles = tiles[: args.limit]
    print(f"Discovered {len(tiles)} tiles across {len(folder_names)} folders under {args.root}")

    prefix, suffix = common_prefix_suffix(folder_names)
    tile_args = [(path, derive_round(folder, prefix, suffix)) for path, folder in tiles]

    tile_rows: list[dict] = []
    failures: list[dict] = []
    thumbnails: dict[str, np.ndarray] = {}
    tile_results: list[TileResult] = []

    with Pool(processes=args.workers) as pool:
        for result in pool.imap(process_tile, tile_args, chunksize=1):
            tile_rows.append(result.row)
            if not result.row["read_ok"]:
                failures.append(result.row)
                continue
            tile_results.append(result)
            if result.row["round"] not in thumbnails and result.thumbnail is not None:
                thumbnails[result.row["round"]] = result.thumbnail

    tile_df = pd.DataFrame(tile_rows)
    tile_csv = output_dir / "stage0_tile_qc.csv"
    if tile_csv.exists():
        tile_csv.unlink()
    tile_df.to_csv(tile_csv, index=False)

    summary_df = summarize_folders(tile_df)
    vignette_df = vignetting_check(tile_df)
    summary_df = summary_df.merge(vignette_df, on="round", how="left")

    empty_folders = [name for name, count in tile_counts.items() if count == 0]
    if empty_folders:
        empty_rows = [
            {
                "round": derive_round(name, prefix, suffix),
                "n_tiles": 0,
                "n_read_ok": 0,
                "true_z": float("nan"),
                "z_step_um": float("nan"),
                "focus_range": "",
                "frac_axially_contained": float("nan"),
                "verdict": "EMPTY: folder contains no .ims files",
                "vignetting_slope": float("nan"),
                "n_tiles_used": 0,
            }
            for name in empty_folders
        ]
        summary_df = pd.concat([summary_df, pd.DataFrame(empty_rows)], ignore_index=True)

    summary_csv = output_dir / "stage0_summary.csv"
    if summary_csv.exists():
        summary_csv.unlink()
    summary_df.to_csv(summary_csv, index=False)

    pdf_path = output_dir / "stage0_qc.pdf"
    if pdf_path.exists():
        pdf_path.unlink()
    with PdfPages(pdf_path) as pdf:
        render_summary_page(pdf, summary_df, failures, thumbnails)
        for result in tile_results:
            render_tile_page(pdf, result.row, result.panels, result.sharpness)

    print(f"Wrote {tile_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {pdf_path}")
    print(f"Tiles: {len(tile_df)}  read_ok: {int(tile_df['read_ok'].sum())}  failed: {len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
