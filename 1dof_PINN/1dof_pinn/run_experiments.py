"""
run_experiments.py
──────────────────
Automated sweep over all parameter combinations, compiling every result
into a single CSV for analysis and plotting.

Parameter grid
──────────────
  size  : small | medium | large          (3 values)
  wind  : light | moderate | severe       (3 values)
  gusts : 0 | 1 | 2 | 3 | 4 | 5          (6 values)
  ─────────────────────────────────────────────────
  Total combinations : 3 × 3 × 6 = 54 runs
  Conditions per run : 4 (no_control, pid_only, pinn_only, pid_pinn)
  Total result rows  : 216

Output
──────
  results/simulation_sweep.csv

CSV columns
───────────
  size, wind, gusts, condition,
  rms_error_deg, max_error_deg, p95_error_deg,
  control_effort_nm, dist_est_rmse, gust_recovery_s

Usage
─────
  python run_experiments.py                        # full sweep
  python run_experiments.py --size medium          # one size only
  python run_experiments.py --size small --wind light --gusts 0 2 4
"""

import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from sim.dynamics     import BeamParams
from sim.wind         import prebuild_wind_profile, make_gust_events
from sim.motor        import MotorParams
from control.pid      import PIDParams
from simulate         import (
    CONDITIONS, DT, N, DS, NF,
    run_condition, compute_metrics,
)

# ── Parameter grid ─────────────────────────────────────────────────────────────
ALL_SIZES  = ["small", "medium", "large"]
ALL_WINDS  = ["light", "moderate", "severe"]
ALL_GUSTS  = [0, 1, 2, 3, 4, 5]

OUTPUT_CSV = "results/simulation_sweep.csv"

# Friendly condition labels for the CSV
CONDITION_LABELS = {
    "no_control": "No Control",
    "pid_only":   "PID Only",
    "pinn_only":  "PINN Only",
    "pid_pinn":   "PID + PINN",
}


def run_single(size: str, wind: str, gusts: int) -> list[dict]:
    """
    Run all four conditions for one (size, wind, gusts) combination.
    Returns a list of result dicts, one per condition.
    """
    params       = getattr(BeamParams,  size)()
    pid_params   = getattr(PIDParams,   f"for_{size}")()
    motor_params = getattr(MotorParams, f"for_{size}")()

    wind_profile     = prebuild_wind_profile(
        DT, N, intensity=wind, seed=42, n_gusts=gusts
    )
    gust_onset_times = [g.onset_time for g in make_gust_events(gusts)]

    rows = []
    for i, cond in enumerate(CONDITIONS):
        log     = run_condition(
            cond, wind_profile, params,
            pid_params=pid_params,
            motor_params=motor_params,
            imu_seed=i * 7,
        )
        metrics = compute_metrics(log, wind_profile,
                                  gust_onset_times=gust_onset_times)

        rows.append({
            "size":              size,
            "wind":              wind,
            "gusts":             gusts,
            "condition":         CONDITION_LABELS[cond],
            "rms_error_deg":     metrics["RMS Error (°)"],
            "max_error_deg":     metrics["Max Error (°)"],
            "p95_error_deg":     metrics["P95 Error (°)"],
            "control_effort_nm": metrics["Control Effort (N·m)"],
            "dist_est_rmse":     metrics["Dist. Est. RMSE"],
            "gust_recovery_s":   metrics["Gust Recovery (s)"],
            "hit_limit":         bool(log.get("hit_limit", False)),
        })

    return rows


def run_sweep(sizes: list, winds: list, gusts_list: list) -> pd.DataFrame:
    combinations = list(itertools.product(sizes, winds, gusts_list))
    total        = len(combinations)

    print(f"\nRunning {total} parameter combinations × 4 conditions = {total * 4} simulations\n")

    all_rows = []
    errors   = []

    for size, wind, gusts in tqdm(combinations, desc="Sweep", unit="combo"):
        try:
            rows = run_single(size, wind, gusts)
            all_rows.extend(rows)
        except Exception as exc:
            errors.append((size, wind, gusts, str(exc)))
            tqdm.write(f"  [ERROR] size={size} wind={wind} gusts={gusts}: {exc}")

    df = pd.DataFrame(all_rows, columns=[
        "size", "wind", "gusts", "condition",
        "rms_error_deg", "max_error_deg", "p95_error_deg",
        "control_effort_nm", "dist_est_rmse", "gust_recovery_s", "hit_limit",
    ])

    if errors:
        print(f"\n{len(errors)} combination(s) failed:")
        for size, wind, gusts, msg in errors:
            print(f"  size={size} wind={wind} gusts={gusts} → {msg}")

    return df


def print_summary(df: pd.DataFrame):
    """Print per-condition summary, split by hit_limit to avoid failed runs skewing results."""
    order       = list(CONDITION_LABELS.values())
    metric_cols = ["rms_error_deg", "max_error_deg", "p95_error_deg",
                   "control_effort_nm", "dist_est_rmse", "gust_recovery_s"]

    # ── Failure rate per arm size & condition ─────────────────────────────────
    total      = len(df)
    n_failed   = df["hit_limit"].sum()
    fail_rate  = df.groupby(["size", "condition"])["hit_limit"].mean().mul(100).round(1)

    print(f"\n── Failure rate (hit_limit=True) — {n_failed}/{total} runs ──────────────\n")
    print(fail_rate.rename("fail_%").to_string())

    # ── Clean runs only ───────────────────────────────────────────────────────
    clean   = df[df["hit_limit"] == False].copy()
    n_clean = len(clean)
    print(f"\n── Summary: clean runs only ({n_clean}/{total} rows, hit_limit=False) ──\n")
    # Replace -1.0 sentinel (no gusts defined) with NaN so it is excluded from mean
    clean.loc[clean["gust_recovery_s"] == -1.0, "gust_recovery_s"] = float("nan")
    summary = (
        clean.groupby("condition")[metric_cols]
        .mean()
        .round(4)
    )
    summary = summary.reindex([c for c in order if c in summary.index])
    print(summary.to_string())

    # ── Failed runs count by arm size ─────────────────────────────────────────
    if n_failed > 0:
        print(f"\n── Failed runs by arm size (hit_limit=True) ────────────────────────────\n")
        failed_summary = (
            df[df["hit_limit"] == True]
            .groupby(["size", "condition"])
            .size()
            .rename("failed_runs")
        )
        print(failed_summary.to_string())
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep all simulation parameters and compile results to CSV."
    )
    parser.add_argument("--size",  nargs="+", choices=ALL_SIZES,  default=ALL_SIZES,
                        help="Arm sizes to include (default: all)")
    parser.add_argument("--wind",  nargs="+", choices=ALL_WINDS,  default=ALL_WINDS,
                        help="Wind intensities to include (default: all)")
    parser.add_argument("--gusts", nargs="+", type=int,           default=ALL_GUSTS,
                        metavar="{0-5}",
                        help="Gust counts to include (default: 0 1 2 3 4 5)")
    args = parser.parse_args()

    t0 = time.time()
    df = run_sweep(args.size, args.wind, args.gusts)

    os.makedirs("results", exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    elapsed = time.time() - t0
    print(f"\nSaved {len(df)} rows → {OUTPUT_CSV}  ({elapsed:.1f}s)")
    print_summary(df)
