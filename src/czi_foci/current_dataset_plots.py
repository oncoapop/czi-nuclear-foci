"""Plot current dataset summaries in the same multi-panel format as the first iteration."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TIMEPOINT_ORDER = [2, 6, 24, 48]
NAMED_CONDITION_ORDER = [
    "Aqueous",
    "DMSO",
    "NAH2PO4",
    "Media",
    "CX-5461",
    "Cisplatin",
    "Etoposide",
    "PDS",
    "Palbociclib",
]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

PALETTE = [
    "#464C55",
    "#C5CAD3",
    "#7A828F",
    "#A3BEFA",
    "#F0986E",
    "#F390CA",
    "#A3D576",
    "#FFE15B",
]
EDGES = [
    "#1F2430",
    "#7A828F",
    "#464C55",
    "#2E4780",
    "#804126",
    "#8A3A6F",
    "#386411",
    "#736422",
]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.titlecolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "font.size": 9,
        }
    )


def condition_order(conditions: pd.Series) -> list[str]:
    present = set(conditions.dropna().astype(str))
    ordered = [condition for condition in NAMED_CONDITION_ORDER if condition in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(TOKENS["axis"])
    ax.tick_params(axis="both", labelsize=8)


def add_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.035, 0.985, textwrap.fill(title, width=100), ha="left", va="top", fontsize=14, fontweight="semibold", color=TOKENS["ink"])
    fig.text(0.035, 0.945, textwrap.fill(subtitle, width=135), ha="left", va="top", fontsize=9, color=TOKENS["muted"])


def plot_metric(
    field: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    subtitle: str,
    output_stem: Path,
) -> None:
    plot_df = field[field[metric].notna()].copy()
    experiments = ["TS III", "TS IV", "TS V"]
    rng = np.random.default_rng(20260618)
    fig, axes = plt.subplots(len(experiments), len(TIMEPOINT_ORDER), figsize=(5.0 * len(TIMEPOINT_ORDER), 3.8 * len(experiments)), sharey=True, squeeze=False)
    add_header(fig, title, subtitle)
    metric_max = plot_df[metric].max()
    if pd.notna(metric_max) and "fraction" in metric:
        y_limit = (0, min(1.02, max(0.35, float(metric_max) * 1.18)))
    elif pd.notna(metric_max):
        y_limit = (0, max(0.2, float(metric_max) * 1.18))
    else:
        y_limit = None

    for row_idx, experiment in enumerate(experiments):
        for col_idx, timepoint in enumerate(TIMEPOINT_ORDER):
            ax = axes[row_idx, col_idx]
            subset = plot_df[(plot_df["experiment"] == experiment) & (plot_df["timepoint_hr"] == timepoint)]
            ax.set_title(f"{experiment}, {timepoint} hr", fontsize=10)
            if subset.empty:
                ax.text(0.5, 0.5, "not acquired", ha="center", va="center", transform=ax.transAxes, color=TOKENS["muted"])
                ax.set_xticks([])
                style_axis(ax)
                continue
            conditions = condition_order(subset["condition"])
            for idx, condition in enumerate(conditions):
                values = subset.loc[subset["condition"] == condition, metric].to_numpy(dtype=float)
                x = np.full(values.shape, idx, dtype=float) + rng.uniform(-0.13, 0.13, size=values.shape)
                colour = PALETTE[idx % len(PALETTE)]
                edge = EDGES[idx % len(EDGES)]
                ax.scatter(x, values, s=40, color=colour, edgecolor=edge, linewidth=0.8, alpha=0.9, zorder=3)
                ax.plot([idx - 0.24, idx + 0.24], [np.nanmean(values), np.nanmean(values)], color=TOKENS["ink"], linewidth=1.4)
            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels(conditions, rotation=45, ha="right")
            ax.set_xlim(-0.6, max(len(conditions) - 0.4, 0.6))
            if y_limit:
                ax.set_ylim(*y_limit)
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            style_axis(ax)

    fig.text(0.035, 0.025, "Each dot is one CZI image field. Horizontal black ticks are condition/time field means.", fontsize=8, color=TOKENS["muted"])
    fig.tight_layout(rect=(0.02, 0.05, 0.995, 0.90))
    fig.savefig(output_stem.with_suffix(".png"), dpi=220)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def write_current_dataset_plots(field_csv: Path, output_dir: Path) -> list[Path]:
    configure_matplotlib()
    field = pd.read_csv(field_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    configs = [
        (
            "mean_af488_spots_per_nucleus",
            "Mean AF488 spots per nucleus",
            "AF488 foci burden across all CZI experiments",
            "Same field-replicate format as the first iteration. Values are unchanged segmentation outputs; comparison is against the experiment-specific batch-aware thresholds in the companion tables.",
            "batch_aware_AF488_mean_spots_per_nucleus_field_replicates",
        ),
        (
            "fraction_single_af488_positive_batch",
            "Fraction nuclei AF488-positive",
            "AF488-positive nuclei across all CZI experiments",
            "Positivity is defined by the experiment-specific AF488 threshold: TS III >=3, TS IV >=5, TS V >=5 spots per nucleus.",
            "batch_aware_AF488_fraction_positive_field_replicates",
        ),
        (
            "mean_rhrex_spots_per_nucleus",
            "Mean RhReX foci per nucleus",
            "RhReX putative gH2AX burden across all CZI experiments",
            "Same field-replicate format as the first iteration. RhReX threshold is stable at >=1 focus per nucleus across TS III, TS IV and TS V.",
            "batch_aware_RhReX_mean_spots_per_nucleus_field_replicates",
        ),
        (
            "fraction_coloc_positive_batch",
            "Fraction nuclei co-localisation-positive",
            "Co-localised RhReX/AF488 nuclei across all CZI experiments",
            "Co-localised positivity is defined by the experiment-specific threshold, which remains >=1 co-localised RhReX focus per nucleus in all three experiments.",
            "batch_aware_colocalized_fraction_positive_field_replicates",
        ),
    ]
    for metric, ylabel, title, subtitle, stem in configs:
        output_stem = output_dir / stem
        plot_metric(field, metric, ylabel, title, subtitle, output_stem)
        outputs.extend([output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")])
    return outputs
