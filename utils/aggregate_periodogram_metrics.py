"""
Aggregate periodogram peak metrics across all experiments.

For each experiment under <experiments_root>, this script loads
postprocessing/data/periodograms.npz and, for each RV series
('v_apparent', 'v_encode', 'v_traditionnal', 'v_correct', 'v_ref'),
computes:
    - P_peak (period of highest peak)
    - power_peak (height of highest peak)
    - n_peaks (number of local peaks)
    - n_peaks_rel (peaks above rel_height_frac * power_peak)
    - top1..top3 peak periods and powers

Outputs (written under experiments_root by default):
    - combined_periodogram_peaks.csv (long format; one row per experiment x series)
    - combined_periodogram_peaks_pivot_power.csv (pivot of power_peak per series)
    - combined_periodogram_peaks_pivot_period.csv (pivot of P_peak per series)

Usage (from repo root or anywhere with Python path):
    python -m utils.aggregate_periodogram_metrics \
            --root /path/to/experiments \
            --top-n 3 \
            --rel-height-frac 0.5

Notes:
    - If a periodograms.npz is missing for an experiment, it's skipped.
    - If a series is not present in the NPZ, it's skipped.
    - Latent coordinates (s_i) are ignored by default. Use --include-latent to include them.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


RV_SERIES_DEFAULT = [
    "v_apparent",
    "v_encode",
    "v_traditionnal",
    "v_correct",
    "v_ref",
]


def list_experiments(root: str) -> List[str]:
    return [
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]


def load_periodograms_npz(exp_dir: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load periodogram arrays from an experiment's NPZ.

    Returns a mapping series_name -> (periods, power).
    Only series with both *_periods and *_power present are returned.
    """
    path = os.path.join(exp_dir, "postprocessing", "data", "periodograms.npz")
    if not os.path.exists(path):
        return {}
    try:
        npz = np.load(path, allow_pickle=True)
    except Exception:
        return {}

    keys = list(npz.keys())
    series_map: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for k in keys:
        if not k.endswith("_periods"):
            continue
        series = k[: -len("_periods")]
        pkey = f"{series}_periods"
        powkey = f"{series}_power"
        if pkey in npz and powkey in npz:
            try:
                periods = np.asarray(npz[pkey])
                power = np.asarray(npz[powkey])
                if periods.ndim != 1 or power.ndim != 1 or periods.size != power.size:
                    continue
                series_map[series] = (periods, power)
            except Exception:
                continue
    return series_map


def load_metrics_df(exp_dir: str) -> pd.DataFrame | None:
    """Load metrics.csv for an experiment, normalize names and numeric fields.

    Returns a DataFrame or None if not available.
    """
    path = os.path.join(exp_dir, "postprocessing", "data", "metrics.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    # Normalize metric name typos and numeric conversions
    if "metric" in df.columns:
        df["metric"] = df["metric"].astype(str).replace({"fap_at_PNj": "fap_at_Pinj"})
    if "P_inj" in df.columns:
        df["P_inj_num"] = pd.to_numeric(df["P_inj"], errors="coerce")
    else:
        df["P_inj_num"] = np.nan
    if "value" in df.columns:
        df["value_num"] = pd.to_numeric(df["value"], errors="coerce")
    else:
        df["value_num"] = np.nan
    return df


def nearest_periodogram_metrics_for_series(
    metrics_df: pd.DataFrame | None, series: str, P_peak: float
) -> Dict[str, Any]:
    """Pick periodogram metrics at the P_inj closest to P_peak for a given series.

    Returns dict with keys: P_inj_nearest, fap_at_P_nearest, power_ratio_nearest,
    n_sig_nearest, delta_P_nearest. Missing values will be np.nan.
    """
    out = {
        "P_inj_nearest": np.nan,
        "fap_at_P_nearest": np.nan,
        "power_ratio_nearest": np.nan,
        "n_sig_nearest": np.nan,
        "delta_P_nearest": np.nan,
    }
    if metrics_df is None or not np.isfinite(P_peak):
        return out
    try:
        sub = metrics_df[
            (metrics_df.get("row_type") == "periodogram")
            & (metrics_df.get("series") == series)
        ].copy()
    except Exception:
        return out
    if sub.empty or "P_inj_num" not in sub.columns:
        return out
    pvals = sub["P_inj_num"].dropna().unique()
    if pvals.size == 0:
        return out
    try:
        nearest_p = float(pvals[np.argmin(np.abs(pvals - P_peak))])
    except Exception:
        return out
    rows_for_p = sub[sub["P_inj_num"] == nearest_p]
    if rows_for_p.empty:
        return out

    def _get_metric(rows: pd.DataFrame, name: str) -> float:
        try:
            rv = rows.loc[rows["metric"] == name, "value_num"]
            if not rv.empty:
                return float(rv.iloc[0])
        except Exception:
            pass
        return np.nan

    out["P_inj_nearest"] = nearest_p
    out["fap_at_P_nearest"] = _get_metric(rows_for_p, "fap_at_Pinj")
    out["power_ratio_nearest"] = _get_metric(rows_for_p, "power_ratio")
    out["n_sig_nearest"] = _get_metric(rows_for_p, "n_sig_peaks_outside")
    out["delta_P_nearest"] = _get_metric(rows_for_p, "delta_P")
    return out


def compute_peak_metrics(
    periods: np.ndarray,
    power: np.ndarray,
    top_n: int = 3,
    rel_height_frac: float = 0.5,
) -> Dict[str, Any]:
    """Compute peak-related metrics for a periodogram.

    Returns a dict with P_peak, power_peak, n_peaks, n_peaks_rel,
    and top1..topN peak period/power pairs.
    """
    metrics: Dict[str, Any] = {
        "P_peak": np.nan,
        "power_peak": np.nan,
        "n_peaks": 0,
        "n_peaks_rel": 0,
    }
    if periods is None or power is None:
        return metrics
    if periods.size == 0 or power.size == 0:
        return metrics

    # Handle NaNs in power: replace with -inf for argmax and peak finding
    power_safe = np.array(power, dtype=float)
    power_safe[~np.isfinite(power_safe)] = -np.inf

    try:
        idx_max = int(np.nanargmax(power_safe))
    except ValueError:
        return metrics

    P_peak = float(periods[idx_max])
    power_peak = float(power_safe[idx_max])
    metrics["P_peak"] = P_peak
    metrics["power_peak"] = power_peak

    # Find local peaks
    try:
        peaks, _ = find_peaks(power_safe)
        heights = power_safe[peaks] if peaks.size > 0 else np.array([])
        metrics["n_peaks"] = int(peaks.size)
        if peaks.size > 0 and np.isfinite(power_peak):
            thr = rel_height_frac * power_peak
            metrics["n_peaks_rel"] = int(np.sum(heights >= thr))
        # Top-N by height
        order = np.argsort(heights)[::-1] if peaks.size > 0 else np.array([], dtype=int)
        for i in range(min(top_n, order.size)):
            j = order[i]
            metrics[f"top{i + 1}_period"] = float(periods[peaks[j]])
            metrics[f"top{i + 1}_power"] = float(heights[j])
        # For missing top slots, fill NaN
        for i in range(order.size + 1, top_n + 1):
            metrics[f"top{i}_period"] = np.nan
            metrics[f"top{i}_power"] = np.nan
    except Exception:
        # If peak finding fails, at least keep top-1
        metrics["top1_period"] = P_peak
        metrics["top1_power"] = power_peak
        for i in range(2, top_n + 1):
            metrics[f"top{i}_period"] = np.nan
            metrics[f"top{i}_power"] = np.nan

    return metrics


def aggregate(
    experiments_root: str,
    include_latent: bool = False,
    series: List[str] | None = None,
    top_n: int = 3,
    rel_height_frac: float = 0.5,
) -> pd.DataFrame:
    """Aggregate peak metrics across experiments and series into a DataFrame."""
    if series is None:
        series = RV_SERIES_DEFAULT.copy()

    rows: List[Dict[str, Any]] = []
    exps = list_experiments(experiments_root)
    for exp_dir in exps:
        series_map = load_periodograms_npz(exp_dir)
        if not series_map:
            continue
        mdf = load_metrics_df(exp_dir)

        # Determine which series to process
        wanted_series = set(series)
        if include_latent:
            # include all s_* present
            wanted_series |= {s for s in series_map.keys() if s.startswith("s_")}

        for sname, (per, powr) in series_map.items():
            if sname not in wanted_series:
                continue
            m = compute_peak_metrics(
                per, powr, top_n=top_n, rel_height_frac=rel_height_frac
            )
            # Attach nearest periodogram metrics from metrics.csv (if available)
            attach = nearest_periodogram_metrics_for_series(
                mdf, sname, m.get("P_peak", np.nan)
            )
            row = {
                "experiment_path": exp_dir,
                "experiment_name": os.path.basename(exp_dir),
                "series": sname,
                "rel_height_frac": rel_height_frac,
                **m,
                **attach,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    # Stable sort: by series then power_peak desc
    if not df.empty:
        df.sort_values(["series", "power_peak"], ascending=[True, False], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def save_outputs(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    out_long = os.path.join(out_dir, "combined_periodogram_peaks.csv")
    out_pivot_power = os.path.join(
        out_dir, "combined_periodogram_peaks_pivot_power.csv"
    )
    out_pivot_period = os.path.join(
        out_dir, "combined_periodogram_peaks_pivot_period.csv"
    )

    df.to_csv(out_long, index=False)

    # Pivot: rows=experiment, columns=series, values=power_peak
    try:
        pivot_power = df.pivot_table(
            index=["experiment_name", "experiment_path"],
            columns="series",
            values="power_peak",
            aggfunc="max",
        ).reset_index()
        pivot_power.to_csv(out_pivot_power, index=False)
    except Exception:
        out_pivot_power = ""

    # Pivot: rows=experiment, columns=series, values=P_peak
    try:
        pivot_period = df.pivot_table(
            index=["experiment_name", "experiment_path"],
            columns="series",
            values="P_peak",
            aggfunc="first",
        ).reset_index()
        pivot_period.to_csv(out_pivot_period, index=False)
    except Exception:
        out_pivot_period = ""

    return out_long, out_pivot_power, out_pivot_period


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate periodogram peak metrics across experiments."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/home/tliopis/Codes/exoplanets_llopis_mary_2025/experiments",
        help="Path to experiments root directory",
    )
    parser.add_argument(
        "--include-latent",
        action="store_true",
        help="Include latent coordinates s_i in aggregation",
    )
    parser.add_argument(
        "--series",
        type=str,
        nargs="*",
        default=None,
        help="Specific series to include (default: common RV series)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top peaks to record per series",
    )
    parser.add_argument(
        "--rel-height-frac",
        type=float,
        default=0.5,
        help="Relative height fraction for counting significant peaks (0-1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = aggregate(
        experiments_root=args.root,
        include_latent=args.include_latent,
        series=args.series,
        top_n=args.top_n,
        rel_height_frac=args.rel_height_frac,
    )
    if df.empty:
        print("No periodograms found; no output written.")
        return
    out_long, out_pivot_power, out_pivot_period = save_outputs(df, args.root)
    print(f"Saved long format CSV: {out_long}")
    if out_pivot_power:
        print(f"Saved pivot (power_peak) CSV: {out_pivot_power}")
    if out_pivot_period:
        print(f"Saved pivot (P_peak) CSV: {out_pivot_period}")


if __name__ == "__main__":
    main()
