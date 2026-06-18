"""Batch-aware threshold and effect-size summaries for nuclear foci outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BatchMetric:
    metric: str
    normalised_metric: str


def robust_control_stats(values: pd.Series) -> tuple[float, float]:
    values = values.dropna().astype(float)
    if len(values) == 0:
        return float("nan"), float("nan")
    median = float(values.median())
    mad = float((values - median).abs().median())
    robust_sd = 1.4826 * mad
    if robust_sd == 0.0:
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        robust_sd = std if std > 0 else 1.0
    return median, robust_sd


def add_control_normalised_metrics(
    nuclei: pd.DataFrame,
    metrics: list[str],
    control_conditions: set[str],
    batch_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = nuclei.copy()
    frame["is_control"] = frame["condition"].isin(control_conditions)
    stats_rows: list[dict] = []
    for batch_key, batch in frame.groupby(batch_cols, dropna=False, sort=False):
        key_values = batch_key if isinstance(batch_key, tuple) else (batch_key,)
        control = batch[batch["is_control"]]
        batch_mask = pd.Series(True, index=frame.index)
        for col, value in zip(batch_cols, key_values):
            batch_mask &= frame[col].eq(value)
        for metric in metrics:
            median, robust_sd = robust_control_stats(control[metric])
            norm_col = f"{metric}_control_z"
            frame.loc[batch_mask, norm_col] = (frame.loc[batch_mask, metric].astype(float) - median) / robust_sd
            stats_rows.append(
                {
                    **dict(zip(batch_cols, key_values)),
                    "metric": metric,
                    "control_n": int(len(control)),
                    "control_median": median,
                    "control_robust_sd": robust_sd,
                }
            )
    return frame, pd.DataFrame(stats_rows)


def threshold_table(
    frame: pd.DataFrame,
    metric: str,
    control_conditions: set[str],
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    controls = frame["condition"].isin(control_conditions)
    treated = ~controls
    values = frame[metric].astype(float)
    if thresholds is None:
        if metric.endswith("_control_z"):
            thresholds = [round(v, 2) for v in np.arange(-2.0, 8.01, 0.25)]
        else:
            max_count = int(np.nanmax(values)) if len(values) else 0
            thresholds = list(range(max_count + 2))
    rows = []
    for threshold in thresholds:
        positive = values >= threshold
        tp = int((positive & treated).sum())
        fp = int((positive & controls).sum())
        tn = int((~positive & controls).sum())
        fn = int((~positive & treated).sum())
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        rows.append(
            {
                "metric": metric,
                "positive_if_ge": float(threshold),
                "true_positive_treated_nuclei": tp,
                "false_positive_control_nuclei": fp,
                "true_negative_control_nuclei": tn,
                "false_negative_treated_nuclei": fn,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "selectivity": specificity,
                "youden_j": sensitivity + specificity - 1.0,
                "control_positive_fraction": fp / int(controls.sum()) if controls.sum() else 0.0,
                "treated_positive_fraction": tp / int(treated.sum()) if treated.sum() else 0.0,
            }
        )
    return pd.DataFrame(rows)


def best_thresholds_by_batch(
    frame: pd.DataFrame,
    metrics: list[str],
    control_conditions: set[str],
    batch_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for batch_key, batch in frame.groupby(batch_cols, dropna=False, sort=False):
        key_values = batch_key if isinstance(batch_key, tuple) else (batch_key,)
        for metric in metrics:
            table = threshold_table(batch, metric, control_conditions)
            best = table.sort_values(["youden_j", "specificity", "sensitivity"], ascending=False).iloc[0].to_dict()
            rows.append({**dict(zip(batch_cols, key_values)), **best, "n_nuclei": int(len(batch))})
    return pd.DataFrame(rows)


def threshold_stability(best_thresholds: pd.DataFrame, batch_cols: list[str]) -> pd.DataFrame:
    rows = []
    for metric, group in best_thresholds.groupby("metric", sort=False):
        thresholds = group["positive_if_ge"].astype(float)
        stable = bool(thresholds.nunique(dropna=True) == 1)
        rows.append(
            {
                "metric": metric,
                "n_batches": int(len(group)),
                "is_stable": stable,
                "min_best_threshold": float(thresholds.min()),
                "max_best_threshold": float(thresholds.max()),
                "median_best_threshold": float(thresholds.median()),
                "unique_best_thresholds": ";".join(str(v) for v in sorted(thresholds.dropna().unique())),
                "batch_definition": "+".join(batch_cols),
            }
        )
    return pd.DataFrame(rows)


def effect_sizes(
    normalised: pd.DataFrame,
    normalised_metrics: list[str],
    control_conditions: set[str],
    batch_cols: list[str],
) -> pd.DataFrame:
    rows = []
    normalised = normalised.copy()
    normalised["is_control"] = normalised["condition"].isin(control_conditions)
    for key, group in normalised.groupby(batch_cols + ["condition"], dropna=False, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        base = dict(zip(batch_cols + ["condition"], key_values))
        is_control = bool(group["condition"].isin(control_conditions).iloc[0])
        for metric in normalised_metrics:
            values = group[metric].dropna().astype(float)
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "is_control": is_control,
                    "n_nuclei": int(len(values)),
                    "mean_control_z": float(values.mean()) if len(values) else float("nan"),
                    "median_control_z": float(values.median()) if len(values) else float("nan"),
                    "fraction_ge_1_control_z": float((values >= 1.0).mean()) if len(values) else float("nan"),
                    "fraction_ge_2_control_z": float((values >= 2.0).mean()) if len(values) else float("nan"),
                    "fraction_ge_3_control_z": float((values >= 3.0).mean()) if len(values) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def batch_aware_summary(
    nuclei: pd.DataFrame,
    metrics: list[str],
    control_conditions: set[str],
    batch_cols: list[str],
) -> dict[str, pd.DataFrame]:
    normalised, control_stats = add_control_normalised_metrics(nuclei, metrics, control_conditions, batch_cols)
    normalised_metrics = [f"{metric}_control_z" for metric in metrics]
    raw_best = best_thresholds_by_batch(normalised, metrics, control_conditions, batch_cols)
    normalised_best = best_thresholds_by_batch(normalised, normalised_metrics, control_conditions, batch_cols)
    raw_stability = threshold_stability(raw_best, batch_cols)
    normalised_stability = threshold_stability(normalised_best, batch_cols)
    effects = effect_sizes(normalised, normalised_metrics, control_conditions, batch_cols)
    return {
        "normalised_nuclei": normalised,
        "control_stats": control_stats,
        "raw_best_thresholds_by_batch": raw_best,
        "normalised_best_thresholds_by_batch": normalised_best,
        "raw_threshold_stability": raw_stability,
        "normalised_threshold_stability": normalised_stability,
        "effect_sizes": effects,
    }
