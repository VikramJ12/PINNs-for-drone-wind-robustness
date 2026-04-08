# 1-DoF PINN Disturbance Observer — Development Timeline

> **How to read this document:**
> Each `[Dn]` entry is a spine node — a major development milestone.
> Each `[Bn]` entry is a rib — a bug or fault discovered, with the steps taken to resolve it.
> Observations are things found but intentionally left unchanged, noted for research context.
> This document is intended to be converted into a fishbone (Ishikawa) diagram where the spine is the development timeline and each rib is a fault branch.

---

## [D1] Initial Commit — Core Simulation Engine

**What was built:**
- 1-DoF rotating arm physics (ODE: `I·θ̈ = τ_motor − b·θ̇ − τ_wind`) with RK4 integrator
- Motor first-order lag model (BLDC response simulation)
- IMU sensor noise model (ICM-42688-P: white noise, bias, bias walk)
- Dryden atmospheric turbulence wind model (MIL-SPEC)
- Discrete-time PID controller with anti-windup and feedforward hook
- `PhysicsResidualObserver` — analytical disturbance estimator
- Four experimental conditions: No Control, PID Only, PINN Only, PID+PINN
- Main simulation runner (`simulate.py`) with metrics and plots
- Output: per-run PNG plots and CSV

**Fixed parameters at this stage:**
- One arm size (medium: 5-inch, `I=0.008 kg·m²`)
- One wind intensity (moderate, `σ=0.06 N·m`)
- Two hardcoded gust events (t=4s, t=9s)

---

## [D2] Interactive Browser Simulation

**What was built:**
- React/Vite visual simulator (`visual_sim/`)
- RK4 physics duplicated in JavaScript for real-time browser execution
- Configurable: control architecture, wind intensity (light/moderate/severe), gust count (0–5)
- Live visualisation: rotating arm SVG, real-time angle/wind/error charts, propeller blur
- Playback speed control (1×, 2×, 4×, 8×)
- Deterministic PRNG for reproducible wind across sessions

**Note:** Browser simulator and Python simulator are independent. Parameters added to one were not always mirrored in the other during later development.

---

## [D3] Arm Size Parameter Added to Python Simulation

**What was built:**
- `BeamParams.small()` — 3-inch micro, `I=8×10⁻⁵ kg·m²`, `τ_max=0.05 N·m`
- `BeamParams.medium()` — 5-inch racing (default), `I=0.008 kg·m²`, `τ_max=0.50 N·m`
- `BeamParams.large()` — 7-inch long-range, `I=0.080 kg·m²`, `τ_max=3.00 N·m`
- `PIDParams.for_small/medium/large()` — gains tuned per arm
- `MotorParams.for_small/medium/large()` — time constant and torque limit per arm
- `--size` CLI argument in `simulate.py`

### [B1] Bug: `pid_params` and `motor_params` computed but never passed into simulation loop

**Discovered:** After arm sizes were added, `simulate.py` resolved the correct params from `--size` but the `run_condition()` call still used `None` (defaulting to medium arm values for all sizes).

**Root cause:** A TODO comment was left in place of the actual fix — `run_condition` call was never updated to pass the new keyword arguments.

**Resolution steps:**
1. Identified that `run_condition(cond, wind_profile, params, imu_seed=i*7)` was missing `pid_params` and `motor_params`
2. Updated call to pass `pid_params=pid_params, motor_params=motor_params`
3. Removed the stale TODO comment

---

## [D4] Wind Intensity and Gust Count Parameters Added

**What was built:**
- `--wind {light|moderate|severe}` CLI argument
- `--gusts {0–5}` CLI argument
- `make_gust_events(n_gusts)` in `wind.py` — evenly spaces N gust events between t=2s and t=13s with alternating +0.10 / −0.08 N·m magnitudes
- Output filenames now include all three parameters: `simulation_results_{size}_{wind}_{gusts}g.png`

### [B2] Bug: `MotorParams.for_{size}()` called but not implemented

**Discovered:** `simulate.py` called `getattr(MotorParams, f"for_{args.size}")()` but `MotorParams` in `motor.py` had no such classmethods — would crash at runtime with `AttributeError`.

**Root cause:** `for_small/medium/large` classmethods were added to `BeamParams` and `PIDParams` but the same pattern was forgotten for `MotorParams`.

**Resolution steps:**
1. Confirmed `MotorParams` had no size classmethods via grep
2. Added `MotorParams.for_small()`, `.for_medium()`, `.for_large()` with realistic BLDC time constants per arm class
3. Verified consistent `tau_max` values match corresponding `BeamParams`

---

## [D5] Automated Experiment Sweep (`run_experiments.py`)

**What was built:**
- Full parameter sweep: 3 sizes × 3 winds × 6 gust counts = 54 combinations × 4 conditions = 216 rows
- Single output CSV: `results/simulation_sweep.csv`
- Columns: `size, wind, gusts, condition, rms_error_deg, max_error_deg, p95_error_deg, control_effort_nm, dist_est_rmse, gust_recovery_s`
- `tqdm` progress bar, per-combo error catching, end-of-sweep summary table
- Supports partial sweeps via `--size`, `--wind`, `--gusts` flags

---

## [D6] Bug Audit and Systematic Fixes

Full sweep results were analysed to identify faults introduced during parameterisation. Six bugs were fixed and two physics-level observations were documented.

---

### [B3] Bug: Gust recovery metric hardcoded to t=4s

**Discovered:** `compute_metrics` always scanned from `gust_start = int(4.0 / (DT * DS))` regardless of actual gust placement. With `--gusts 3`, events land at t≈2s, 5.5s, 9s — none at t=4s. The metric was measuring recovery from a moment with no gust event.

**Root cause:** The metric was designed for the original hardcoded 2-gust setup (t=4s, t=9s) and was never updated when gust timing became dynamic.

**Resolution steps:**
1. Added `gust_onset_times` parameter to `compute_metrics` signature
2. Extracted actual gust times via `make_gust_events(n_gusts)` in entry point and `run_experiments.py`
3. Replaced hardcoded index with `np.searchsorted(t, t_gust)` per gust onset
4. Factored recovery logic into `_compute_recovery()` helper

---

### [B4] Bug: Recovery metric only tracked first threshold crossing

**Discovered:** The loop `break`ed on the first time `|error| < 0.5°`. With multiple gusts, recovery from gust 1 was captured but recovery from gust 2 was ignored entirely.

**Root cause:** The original single-gust design used a `break` that was not revisited when multi-gust support was added.

**Resolution steps:**
1. Changed `_compute_recovery()` to iterate over all gust onset times independently
2. For each gust: measure time from error breach to error return below threshold
3. Return `max()` across all gusts — worst-case recovery is the meaningful metric

---

### [B5] Bug: Recovery time of 0.0s was ambiguous

**Discovered:** `gust_recovery_s = 0.0` could mean either "the controller recovered instantly" or "the error threshold was never breached at all (perfect rejection)." These are meaningfully different — one is reactive recovery, the other is proactive rejection.

**Root cause:** The original loop started scanning at `gust_start` without first checking whether error ever exceeded the threshold.

**Resolution steps:**
1. In `_compute_recovery()`, scan for breach above threshold first
2. If no breach found: return `0.0` (threshold never crossed — perfect rejection)
3. If breach found: measure time from breach to recovery
4. Semantics documented: `0.0 = never breached`, `>0.0 = time to recover`, `-1.0 = no gusts defined`

---

### [B6] Bug: `dist_est_rmse` evaluation window hardcoded to t=4–9s

**Discovered:** `gust_mask = (t >= 4.0) & (t <= 9.0)` was fixed regardless of gust count or timing. With `--gusts 0` there are no events in this window; with `--gusts 5` events extend past t=9s. The metric was evaluating observer quality over an arbitrary window unrelated to actual disturbances.

**Root cause:** Same origin as B3 — designed for the original 2-gust setup, never updated.

**Resolution steps:**
1. Replaced `gust_mask` with `stable_mask = (t >= 3.0)` — full stable period
2. `dist_est_rmse` now measures observer accuracy over the entire post-transient window
3. More informative and independent of gust count/timing

---

### [B7] Bug: No physical angle constraint — arm spun freely past ±180°

**Discovered:** Failed runs (e.g., small arm under moderate wind) accumulated RMS errors of 10,000–45,000° because the arm kept spinning with no joint limit. Metrics were numerically meaningless and incomparable between arm sizes.

**Root cause:** `rk4_step` integrates θ freely with no bound. Real testbench arms have physical stops.

**Resolution steps:**
1. Added hard-stop clamping in `run_condition` after each `rk4_step`
2. At θ > π: clamp to π, zero out positive velocity (inelastic stop)
3. At θ < −π: clamp to −π, zero out negative velocity
4. Added `hit_limit` boolean flag to log and CSV output
5. Rows with `hit_limit = True` are identifiable as physically failed runs

---

### [B8] Bug: PINN Only condition used hardcoded PD gains not tied to arm size

**Discovered:** The `pinn_only` branch computed torque as `1.5·e + 0.10·ė + τ̂_wind` with fixed gains, while `pid_only` and `pid_pinn` used `PIDParams.for_{size}()`. This made cross-arm comparison of PINN Only unfair and uninterpretable.

**Root cause:** Originally written as a placeholder for one arm size. When `PIDParams` per-size classmethods were added, the `pinn_only` branch was never updated.

**Resolution steps:**
1. Removed hardcoded gain arithmetic and `pinn_prev_err` state variable
2. Instantiated a second `PIDController` named `pinn_pd` with `Ki=0.0` and gains from `pid_params`
3. `pinn_only` now calls `pinn_pd.compute(setpoint, theta, theta_dot, feedforward=tau_hat)`
4. All four conditions now use the same `PIDController` infrastructure, differing only in `Ki` and feedforward

---

### [B9] Bug: Terminal summary skewed by failed runs and negative sentinel values

**Discovered:** After the full 216-run sweep, the terminal summary showed PID Only, PINN Only, and PID+PINN all with near-identical RMS errors (~42–44°), making results look indistinguishable. On closer inspection, the mean was dominated by the 105 failed small arm runs (each clamped at ~170° error). Additionally, `gust_recovery_s` was showing negative values (e.g., −0.37s) in the summary because `−1.0` sentinel values (meaning "no gusts defined") from `gusts=0` runs were being averaged together with real recovery times from `gusts=2` runs.

**Root cause:** `print_summary` in `run_experiments.py` computed mean metrics across all 216 rows unconditionally, with no awareness of `hit_limit` or the `−1.0` sentinel convention for `gust_recovery_s`.

**Resolution steps:**
1. Split summary into three sections: failure rate table, clean-runs-only metrics, failed run counts
2. Filtered to `hit_limit == False` before computing condition means
3. Replaced `gust_recovery_s == −1.0` with `NaN` before averaging so pandas excludes no-gust runs from the recovery mean
4. Added failure rate per `(size, condition)` as a percentage — immediately surfaces the small arm saturation finding
5. Added failed run counts by arm size for transparency on excluded data

---

## Observations (Not Fixed — Research Context)

### [O1] Wind not scaled per arm size — intentional experimental design

The Dryden turbulence σ values (0.02 / 0.06 / 0.12 N·m) are identical for all arm sizes. For the small arm, moderate wind (`σ=0.06 N·m`) exceeds its motor limit (`τ_max=0.05 N·m`). This is intentional: the experiment places all arms in the same real-world wind scenario to test physical limits, not relative difficulty. Small arm saturation under outdoor wind is a result, not a bug.

### [O2] Physics-residual observer has fundamental SNR limitation on high-inertia arms

The observer estimates `τ̂_wind = τ_motor − b·θ̇ − I·θ̈`. The `θ̈` term is a finite difference of gyro samples. Noise in `θ̈` is amplified by `I` before the low-pass filter. For the large arm (`I=0.080`), the noise floor in the estimate is ~10× higher than for the medium arm, and exceeds the actual wind signal magnitude. The fixed `alpha=0.15` filter was tuned for the medium arm and is insufficient for the large arm.

More fundamentally: the large arm's high inertia means wind produces very small angular accelerations (`θ̈ = τ_wind/I = 0.06/0.080 ≈ 0.75 rad/s²`), which are close to the gyro noise floor. The observer has almost no signal to work with, regardless of filter tuning. This suggests physics-residual observers of this form are most effective in a specific inertia range, and alternative observer architectures (e.g., momentum-based, longer-window integration) would be needed for high-inertia arms.
