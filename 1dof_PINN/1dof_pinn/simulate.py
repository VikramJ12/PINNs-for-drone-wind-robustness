"""
simulate.py
───────────
Main simulation runner for the 1-DoF PINN Disturbance Observer experiment.

Runs all four experimental conditions against the same wind profile:
  A) No Control          — free response, no motor torque
  B) PID Only            — classical PID, no observer
  C) PINN Only           — PD control + PINN feedforward, 10ms latency
  D) PID + PINN Observer — proposed method (feedforward disturbance cancellation)

Outputs saved to results/plots/:
  simulation_results.png   — all trajectory plots
  metrics_table.png        — publication-ready metrics table image
  metrics_table.csv        — raw numbers, importable into Excel / pandas
"""
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import pandas as pd
import os, sys

sys.path.insert(0, os.path.dirname(__file__))

from sim.dynamics     import BeamParams, rk4_step
from sim.wind         import prebuild_wind_profile, make_gust_events
from sim.imu          import IMU
from sim.motor        import Motor,MotorParams
from control.pid      import PIDController,PIDParams
from control.observer import PhysicsResidualObserver

# ─── Simulation constants ──────────────────────────────────────────────────────
DT             = 0.001
T_END          = 15.0
N              = int(T_END / DT)
DS             = 10
NF             = N // DS
SP_DEG         = 10.0
SP_RAD         = np.deg2rad(SP_DEG)
SETPOINT_ONSET = 1.0

# ─── Condition metadata ────────────────────────────────────────────────────────
CONDITIONS = {
    "no_control": {"label": "No Control",                "color": "#ef4444", "style": "--",  "lw": 1.4},
    "pid_only":   {"label": "PID Only",                  "color": "#f59e0b", "style": "-.",  "lw": 1.4},
    "pinn_only":  {"label": "PINN Only (10ms latency)",  "color": "#38bdf8", "style": ":",   "lw": 1.6},
    "pid_pinn":   {"label": "PID + PINN Observer ★",     "color": "#4ade80", "style": "-",   "lw": 2.0},
}

RECOVERY_THRESHOLD_DEG = 0.5   # tightened from 2° to show real differentiation


# ─── Single-condition simulation ───────────────────────────────────────────────
def run_condition(condition: str,
                  wind_profile: np.ndarray,
                  params: BeamParams,
                  pid_params: Optional[PIDParams] = None,
                  motor_params: Optional[MotorParams] = None,
                  imu_seed: int = 0) -> dict:
    if pid_params   is None: pid_params   = PIDParams()
    if motor_params is None: motor_params = MotorParams()
    latency = 10 if condition == "pinn_only" else 2

    imu     = IMU(dt=DT, seed=imu_seed)
    pid     = PIDController(dt=DT, params=pid_params, tau_max=params.tau_max)
    # Fault 8: pinn_only uses size-aware PD (Ki=0) via PIDController — no more hardcoded gains
    pinn_pd = PIDController(dt=DT,
                            params=PIDParams(Kp=pid_params.Kp, Ki=0.0, Kd=pid_params.Kd),
                            tau_max=params.tau_max)
    motor   = Motor(dt=DT, params=motor_params)
    obs     = PhysicsResidualObserver(dt=DT, latency_steps=latency,
                                      I=params.I, b=params.b)

    state     = np.array([0.0, 0.0])
    tau_motor = 0.0
    hit_limit = False

    log = {k: np.zeros(NF) for k in
           ["t", "theta", "theta_dot", "tau_motor",
            "tau_wind", "tau_wind_hat", "setpoint", "error"]}
    fi = 0

    for k in range(N):
        t        = k * DT
        tau_wind = wind_profile[k]
        setpoint = SP_RAD if t >= SETPOINT_ONSET else 0.0

        theta, theta_dot = state
        theta_ddot = (tau_motor - params.b * theta_dot - tau_wind) / params.I

        gyro, _  = imu.measure(theta, theta_dot, theta_ddot)
        tau_hat  = obs.update(gyro, tau_motor)

        if condition == "no_control":
            tau_cmd = 0.0

        elif condition == "pid_only":
            tau_cmd = pid.compute(setpoint, theta, theta_dot, feedforward=0.0)

        elif condition == "pinn_only":
            tau_cmd = pinn_pd.compute(setpoint, theta, theta_dot, feedforward=tau_hat)

        elif condition == "pid_pinn":
            tau_cmd = pid.compute(setpoint, theta, theta_dot, feedforward=tau_hat)

        tau_motor = motor.step(tau_cmd)

        if k % DS == 0 and fi < NF:
            log["t"][fi]            = t
            log["theta"][fi]        = np.rad2deg(theta)
            log["theta_dot"][fi]    = theta_dot
            log["tau_motor"][fi]    = tau_motor
            log["tau_wind"][fi]     = tau_wind
            log["tau_wind_hat"][fi] = tau_hat
            log["setpoint"][fi]     = np.rad2deg(setpoint)
            log["error"][fi]        = np.rad2deg(setpoint - theta)
            fi += 1

        state = rk4_step(state, tau_motor, tau_wind, params, DT)

        # Fault 7: hard stop at ±180° — arm cannot spin past physical joint limit
        theta_new, theta_dot_new = state
        if theta_new > np.pi:
            state = np.array([np.pi, min(0.0, theta_dot_new)])
            hit_limit = True
        elif theta_new < -np.pi:
            state = np.array([-np.pi, max(0.0, theta_dot_new)])
            hit_limit = True

    log["hit_limit"] = hit_limit
    return log


# ─── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(log: dict, wind_profile: np.ndarray,
                    gust_onset_times: list = None) -> dict:
    t           = log["t"]
    stable_mask = t >= 3.0

    err_stable  = log["error"][stable_mask]
    rms_error   = np.sqrt(np.mean(err_stable ** 2))
    max_error   = np.max(np.abs(err_stable))
    p95_error   = np.percentile(np.abs(err_stable), 95)
    ctrl_effort = np.sqrt(np.mean(log["tau_motor"][stable_mask] ** 2))

    # Fault 6: evaluate observer quality over full stable period, not hardcoded gust window
    tw_true   = wind_profile[::DS][:NF][stable_mask]
    tw_hat    = log["tau_wind_hat"][stable_mask]
    dist_rmse = np.sqrt(np.mean((tw_hat - tw_true) ** 2))

    # Faults 3+4+5: per-gust recovery, worst case, with unambiguous 0.0 meaning
    recovery_t = _compute_recovery(log["error"], t, gust_onset_times)

    return {
        "RMS Error (°)":        round(rms_error,   4),
        "Max Error (°)":        round(max_error,   4),
        "P95 Error (°)":        round(p95_error,   4),
        "Control Effort (N·m)": round(ctrl_effort, 4),
        "Dist. Est. RMSE":      round(dist_rmse,   4),
        "Gust Recovery (s)":    recovery_t,
    }


def _compute_recovery(error: np.ndarray, t: np.ndarray,
                      gust_onset_times: list) -> float:
    """
    Compute worst-case gust recovery time across all gust events.

    For each gust at t_gust:
      - Scan forward from t_gust
      - If error never exceeds threshold: recovery = 0.0  (gust rejected, never disturbed)
      - If error exceeds threshold: measure time from breach back to below threshold
      - If never recovers: recovery = remaining simulation time from breach

    Returns -1.0 when no gusts are defined (metric not applicable).
    Returns the maximum recovery time across all gusts otherwise.
    """
    if not gust_onset_times:
        return -1.0

    worst = 0.0
    abs_err = np.abs(error)

    for t_gust in gust_onset_times:
        onset_idx = int(np.searchsorted(t, t_gust))
        if onset_idx >= len(t):
            continue

        # find first breach above threshold after gust onset
        breach_idx = None
        for i in range(onset_idx, len(t)):
            if abs_err[i] >= RECOVERY_THRESHOLD_DEG:
                breach_idx = i
                break

        if breach_idx is None:
            # threshold never breached — gust was rejected outright
            recovery = 0.0
        else:
            # find first sample below threshold after breach
            recovery = float(t[-1] - t[breach_idx])   # default: timeout
            for i in range(breach_idx + 1, len(t)):
                if abs_err[i] < RECOVERY_THRESHOLD_DEG:
                    recovery = float(t[i] - t[breach_idx])
                    break

        worst = max(worst, recovery)

    return round(worst, 4)


# ─── Save metrics as CSV ───────────────────────────────────────────────────────
def save_metrics_csv(metrics: dict, size: str = "medium", wind: str = "moderate", gusts: int = 2):
    path = f"results/plots/metrics_table_{size}_{wind}_{gusts}g.csv"
    """
    Save metrics to a CSV file.
    Rows = metrics, Columns = conditions.
    Easily imported into Excel or pandas later.
    """
    rows = []
    metric_keys = list(next(iter(metrics.values())).keys())
    for mk in metric_keys:
        row = {"Metric": mk}
        for cond, cfg in CONDITIONS.items():
            row[cfg["label"]] = metrics[cond][mk]
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Metric")
    df.to_csv(path)
    print(f"CSV  saved → {path}")
    return df


# ─── Save metrics as PNG table image ──────────────────────────────────────────
def save_metrics_png(metrics: dict, size: str = "medium", wind: str = "moderate", gusts: int = 2):
    path = f"results/plots/metrics_table_{size}_{wind}_{gusts}g.png"
    """
    Render the metrics table as a styled PNG image.

    Each cell is coloured by rank within its row:
      green  = best value
      yellow = second best
      orange = third
      red    = worst
    """
    metric_keys  = list(next(iter(metrics.values())).keys())
    cond_keys    = list(CONDITIONS.keys())
    cond_labels  = [CONDITIONS[c]["label"] for c in cond_keys]
    cond_colors  = [CONDITIONS[c]["color"] for c in cond_keys]

    # Build data matrix: rows = metrics, cols = conditions
    data = np.zeros((len(metric_keys), len(cond_keys)))
    for r, mk in enumerate(metric_keys):
        for c, ck in enumerate(cond_keys):
            data[r, c] = metrics[ck][mk]

    # Rank each row (0 = best/lowest, 3 = worst/highest)
    rank_colors = ["#166534", "#854d0e", "#9a3412", "#7f1d1d"]  # green→red
    cell_bg = np.empty_like(data, dtype=object)
    for r in range(len(metric_keys)):
        order = np.argsort(data[r])   # ascending = best first (lower is better)
        for rank, col_idx in enumerate(order):
            cell_bg[r, col_idx] = rank_colors[rank]

    # ── Figure setup ──────────────────────────────────────────────────────
    n_rows = len(metric_keys)
    n_cols = len(cond_keys)
    cell_h = 0.55
    cell_w = 2.2
    header_h = 0.9

    fig_w = cell_w * (n_cols + 1)
    fig_h = header_h + cell_h * n_rows + 0.6

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#0a0f1e")
    ax.set_facecolor("#0a0f1e")
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    # ── Title ──────────────────────────────────────────────────────────────
    ax.text(fig_w / 2, fig_h - 0.25,
            "Performance Metrics — 1-DoF PINN Disturbance Observer",
            color="white", fontsize=11, fontweight="bold",
            ha="center", va="center")
    ax.text(fig_w / 2, fig_h - 0.58,
            f"Gust recovery threshold: {RECOVERY_THRESHOLD_DEG}°  |  "
            f"Stable window: t ≥ 3s  |  Dryden moderate turbulence",
            color="#94a3b8", fontsize=7.5, ha="center", va="center")

    y0 = fig_h - header_h   # top of table body

    # ── Column headers ─────────────────────────────────────────────────────
    # Row label header
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.05, y0 - 0.02), cell_w - 0.1, header_h - 0.1,
        boxstyle="round,pad=0.05", facecolor="#1e293b", edgecolor="#334155", lw=0.8
    ))
    ax.text(cell_w / 2, y0 + (header_h - 0.1) / 2,
            "Metric", color="white", fontsize=9, fontweight="bold",
            ha="center", va="center")

    # Condition headers
    for c, (label, color) in enumerate(zip(cond_labels, cond_colors)):
        x = cell_w * (c + 1)
        ax.add_patch(mpatches.FancyBboxPatch(
            (x + 0.05, y0 - 0.02), cell_w - 0.1, header_h - 0.1,
            boxstyle="round,pad=0.05", facecolor="#1e293b",
            edgecolor=color, lw=1.5
        ))
        # Colour swatch
        ax.add_patch(mpatches.FancyBboxPatch(
            (x + 0.15, y0 + 0.38), 0.25, 0.18,
            boxstyle="round,pad=0.02", facecolor=color, edgecolor="none"
        ))
        # Label (split long labels across two lines)
        parts = label.split(" (")
        line1 = parts[0]
        line2 = f"({parts[1]}" if len(parts) > 1 else ""
        ax.text(x + cell_w / 2, y0 + 0.45,
                line1, color=color, fontsize=8, fontweight="bold",
                ha="center", va="center")
        if line2:
            ax.text(x + cell_w / 2, y0 + 0.20,
                    line2, color="#94a3b8", fontsize=7,
                    ha="center", va="center")

    # ── Data rows ──────────────────────────────────────────────────────────
    for r, mk in enumerate(metric_keys):
        y = y0 - (r + 1) * cell_h
        row_bg = "#111827" if r % 2 == 0 else "#0f172a"

        # Metric name cell
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.05, y + 0.05), cell_w - 0.1, cell_h - 0.1,
            boxstyle="round,pad=0.05", facecolor=row_bg,
            edgecolor="#1e293b", lw=0.5
        ))
        ax.text(0.2, y + cell_h / 2,
                mk, color="#e2e8f0", fontsize=8.5,
                ha="left", va="center")

        # Value cells
        for c, ck in enumerate(cond_keys):
            x     = cell_w * (c + 1)
            val   = metrics[ck][mk]
            bg    = cell_bg[r, c]
            is_best = (np.argsort(data[r])[0] == c)

            ax.add_patch(mpatches.FancyBboxPatch(
                (x + 0.05, y + 0.05), cell_w - 0.1, cell_h - 0.1,
                boxstyle="round,pad=0.05", facecolor=bg,
                edgecolor="#1e293b", lw=0.5
            ))

            # Value text
            ax.text(x + cell_w / 2, y + cell_h / 2 + 0.05,
                    f"{val:.4f}",
                    color="white", fontsize=9,
                    fontweight="bold" if is_best else "normal",
                    ha="center", va="center")

            # Best marker
            if is_best:
                ax.text(x + cell_w - 0.25, y + cell_h / 2 + 0.05,
                        "✓", color="#4ade80", fontsize=9,
                        ha="center", va="center")

    # ── Legend ─────────────────────────────────────────────────────────────
    legend_y = y0 - (n_rows + 0.5) * cell_h
    legend_items = [
        ("#166534", "Best"),
        ("#854d0e", "2nd"),
        ("#9a3412", "3rd"),
        ("#7f1d1d", "Worst"),
    ]
    ax.text(0.2, legend_y, "Rank:", color="#94a3b8", fontsize=7.5, va="center")
    for i, (color, label) in enumerate(legend_items):
        xp = 1.0 + i * 1.5
        ax.add_patch(mpatches.FancyBboxPatch(
            (xp, legend_y - 0.12), 0.5, 0.28,
            boxstyle="round,pad=0.03", facecolor=color, edgecolor="none"
        ))
        ax.text(xp + 0.65, legend_y, label,
                color="#94a3b8", fontsize=7.5, va="center")

    plt.tight_layout(pad=0.2)
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="#0a0f1e")
    plt.close()
    print(f"Table saved → {path}")


# ─── Terminal table printer ────────────────────────────────────────────────────
def print_metrics_table(metrics: dict):
    conds  = list(CONDITIONS.keys())
    m_keys = list(next(iter(metrics.values())).keys())
    cw, nw = 22, 28
    sep    = "═" * (nw + cw * len(conds))

    print(f"\n{sep}")
    print("  PERFORMANCE METRICS")
    print(sep)
    print(f"{'Metric':<{nw}}" + "".join(
        f"{CONDITIONS[c]['label'][:cw]:^{cw}}" for c in conds
    ))
    print("─" * (nw + cw * len(conds)))

    for mk in m_keys:
        vals = {c: metrics[c][mk] for c in conds}
        best = min(vals, key=vals.get)
        row  = f"{mk:<{nw}}"
        for c in conds:
            cell = f"{vals[c]:.4f}" + (" ✓" if c == best else "  ")
            row += f"{cell:^{cw}}"
        print(row)

    print(sep)
    print(f"  ✓ = best | Recovery threshold = {RECOVERY_THRESHOLD_DEG}°\n")


# ─── Trajectory plots ──────────────────────────────────────────────────────────
def plot_results(results: dict, wind_profile: np.ndarray, metrics: dict,
                 size: str = "medium", wind: str = "moderate", gusts: int = 2):
    t = results["pid_pinn"]["t"]

    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor("#0a0f1e")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32)

    ax_track  = fig.add_subplot(gs[0, :])
    ax_error  = fig.add_subplot(gs[1, :])
    ax_wind   = fig.add_subplot(gs[2, 0])
    ax_metric = fig.add_subplot(gs[2, 1])

    def style(ax, title):
        ax.set_facecolor("#111827")
        ax.tick_params(colors="white", labelsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor("#374151")
        ax.set_title(title, color="white", fontsize=10, fontweight="bold")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.grid(True, color="#1f2937", linewidth=0.6, linestyle="--", alpha=0.8)

    def add_event_lines(ax):
        ax.axvspan(4, 9, color="#ef4444", alpha=0.06)
        ax.axvline(4.0, color="#ef4444", lw=0.8, ls="--", alpha=0.5)
        ax.axvline(9.0, color="#f59e0b", lw=0.8, ls="--", alpha=0.5)

    # ── Angle tracking ─────────────────────────────────────────────────────
    style(ax_track, "Angle Tracking  θ(t)")
    ax_track.plot(t, results["pid_pinn"]["setpoint"],
                  "w--", lw=1.0, alpha=0.4, label="Setpoint 10°")
    for cond, cfg in CONDITIONS.items():
        theta = np.clip(results[cond]["theta"], -40, 40)
        ax_track.plot(t, theta, color=cfg["color"], ls=cfg["style"],
                      lw=cfg["lw"], label=cfg["label"])
    add_event_lines(ax_track)
    ax_track.set_ylabel("θ (degrees)", fontsize=9)
    ax_track.set_xlabel("Time (s)", fontsize=9)
    ax_track.set_ylim(-35, 25)
    ax_track.legend(fontsize=8, facecolor="#111827",
                    edgecolor="#374151", labelcolor="white")

    # ── Tracking error ─────────────────────────────────────────────────────
    style(ax_error, "Absolute Tracking Error  |θ_ref − θ|")
    for cond, cfg in CONDITIONS.items():
        err = np.clip(np.abs(results[cond]["error"]), 0, 30)
        ax_error.plot(t, err, color=cfg["color"], ls=cfg["style"],
                      lw=cfg["lw"], label=cfg["label"])
    add_event_lines(ax_error)
    ax_error.axhline(RECOVERY_THRESHOLD_DEG, color="white", lw=0.6,
                     ls=":", alpha=0.35,
                     label=f"{RECOVERY_THRESHOLD_DEG}° recovery threshold")
    ax_error.set_ylabel("|Error| (degrees)", fontsize=9)
    ax_error.set_xlabel("Time (s)", fontsize=9)
    ax_error.set_ylim(0, 20)
    ax_error.legend(fontsize=8, facecolor="#111827",
                    edgecolor="#374151", labelcolor="white")

    # ── Disturbance estimation ─────────────────────────────────────────────
    style(ax_wind, "Wind Disturbance & Estimation")
    ax_wind.plot(t, wind_profile[::DS][:NF],
                 color="white", lw=1.2, alpha=0.6, ls="--", label="True τ_wind")
    for cond in ["pid_pinn", "pinn_only"]:
        cfg = CONDITIONS[cond]
        ax_wind.plot(t, results[cond]["tau_wind_hat"],
                     color=cfg["color"], lw=1.2, ls=cfg["style"],
                     label=f"Est. ({cfg['label'][:18]})")
    add_event_lines(ax_wind)
    ax_wind.set_ylabel("Torque (N·m)", fontsize=9)
    ax_wind.set_xlabel("Time (s)", fontsize=9)
    ax_wind.legend(fontsize=7, facecolor="#111827",
                   edgecolor="#374151", labelcolor="white")

    # ── Metrics bar chart ──────────────────────────────────────────────────
    style(ax_metric, "Key Metrics Comparison")
    mk_plot = ["RMS Error (°)", "Max Error (°)", "P95 Error (°)"]
    x = np.arange(len(mk_plot))
    w = 0.20
    for i, (cond, cfg) in enumerate(CONDITIONS.items()):
        vals = [min(metrics[cond][mk], 10) for mk in mk_plot]
        ax_metric.bar(x + i * w, vals, w,
                      color=cfg["color"], alpha=0.82,
                      label=cfg["label"][:20], edgecolor="#0a0f1e")
    ax_metric.set_xticks(x + 1.5 * w)
    ax_metric.set_xticklabels(mk_plot, fontsize=8)
    ax_metric.set_ylabel("Degrees (capped at 10°)", fontsize=8)
    ax_metric.yaxis.label.set_color("white")
    ax_metric.legend(fontsize=7, facecolor="#111827",
                     edgecolor="#374151", labelcolor="white")

    fig.suptitle(
        "1-DoF PINN Disturbance Observer — Simulation Results\n"
        "I=0.008 kg·m²  ·  b=0.004 N·m·s/rad  ·  Dryden turbulence  ·  RK4 @ 1kHz",
        color="white", fontsize=11, fontweight="bold"
    )

    os.makedirs("results/plots", exist_ok=True)
    out = f"results/plots/simulation_results_{size}_{wind}_{gusts}g.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="#0a0f1e")
    plt.close()
    print(f"Plot saved → {out}")


# ─── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--size",  choices=["small", "medium", "large"],
                        default="medium")
    parser.add_argument("--wind",  choices=["light", "moderate", "severe"],
                        default="moderate")
    parser.add_argument("--gusts", type=int, default=2,
                        choices=range(0, 6), metavar="{0-5}",
                        help="Number of deterministic gust events (0–5, default 2)")
    args = parser.parse_args()

    print(f"Arm size : {args.size}")
    print(f"Wind     : {args.wind}")
    print(f"Gusts    : {args.gusts}")

    params       = getattr(BeamParams, args.size)()
    pid_params   = getattr(PIDParams,  f"for_{args.size}")()
    motor_params = getattr(MotorParams, f"for_{args.size}")()
    wind_profile      = prebuild_wind_profile(DT, N,
                                              intensity=args.wind, seed=42,
                                              n_gusts=args.gusts)
    gust_onset_times  = [g.onset_time for g in make_gust_events(args.gusts)]

    results = {}
    metrics = {}
    for i, cond in enumerate(CONDITIONS):
        print(f"Running: {CONDITIONS[cond]['label']} ...")
        results[cond] = run_condition(cond, wind_profile, params,
                                      pid_params=pid_params,
                                      motor_params=motor_params,
                                      imu_seed=i * 7)
        metrics[cond] = compute_metrics(results[cond], wind_profile,
                                        gust_onset_times=gust_onset_times)

    print_metrics_table(metrics)

    print("Saving outputs...")
    os.makedirs("results/plots", exist_ok=True)
    plot_results(results, wind_profile, metrics,
                 size=args.size, wind=args.wind, gusts=args.gusts)

    save_metrics_csv(metrics,
                     size=args.size, wind=args.wind, gusts=args.gusts)

    save_metrics_png(metrics,
                     size=args.size, wind=args.wind, gusts=args.gusts)

    print("\nAll outputs saved to results/plots/")
    print("  simulation_results.png  — trajectory plots")
    print("  metrics_table.png       — styled metrics table image")
    print("  metrics_table.csv       — raw numbers for Excel / pandas")