"""
control/observer.py
───────────────────
Disturbance observer implementations.

Two classes with identical interfaces:

  PhysicsResidualObserver
      Analytical surrogate used before the PINN is trained.
      Rearranges the beam ODE to directly solve for τ_wind.
      Equivalent to what a perfectly trained PINN converges to
      for this linear 1-DoF system.

  TrainedPINNObserver
      Wraps a loaded PyTorch model. Drop-in replacement for
      PhysicsResidualObserver once training is complete.

Both expose the same interface:
    observer.update(gyro_meas, tau_motor) → τ̂_wind
    observer.reset()

Latency model
─────────────
Both observers maintain a delay buffer that simulates compute latency:
  PID + PINN : 2 steps (2ms)   — short USB round-trip
  PINN Only  : 10 steps (10ms) — full inference round-trip
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

_I_DEFAULT = 0.008
_B_DEFAULT = 0.004
_DT        = 0.001


@dataclass
class PhysicsResidualObserver:
    """
    Analytical disturbance observer.

    Derivation:
        I·θ̈ = τ_motor − b·θ̇ − τ_wind
        ⟹  τ_wind = τ_motor − b·θ̇ − I·θ̈

    τ̂_wind is estimated from:
      - τ_motor : known (we commanded it)
      - θ̇       : from gyro measurement
      - θ̈       : finite difference of two consecutive gyro samples

    A low-pass filter (alpha) smooths amplified finite-difference noise.

    Parameters
    ----------
    dt            : timestep [s]
    window        : gyro history buffer length (samples)
    alpha         : low-pass weight (higher = more responsive, noisier)
    I, b          : physical constants — must match BeamParams
    latency_steps : delay buffer depth (simulates compute latency)
    """
    dt:            float = _DT
    window:        int   = 20
    alpha:         float = 0.15
    I:             float = _I_DEFAULT
    b:             float = _B_DEFAULT
    latency_steps: int   = 2

    _gyro_buf:  object = field(init=False, repr=False)
    _delay_buf: object = field(init=False, repr=False)
    _estimate:  float  = field(init=False, repr=False, default=0.0)

    def __post_init__(self):
        self._gyro_buf  = deque([0.0] * self.window,
                                maxlen=self.window)
        self._delay_buf = deque([0.0] * max(self.latency_steps, 1),
                                maxlen=max(self.latency_steps, 1))
        self._estimate  = 0.0

    def update(self, gyro_meas: float, tau_motor: float) -> float:
        """
        Update with latest sensor reading and control command.

        Returns τ̂_wind delayed by latency_steps timesteps.
        """
        self._gyro_buf.append(gyro_meas)

        # Finite-difference angular acceleration estimate
        theta_dot_now  = self._gyro_buf[-1]
        theta_dot_prev = self._gyro_buf[-2]
        theta_ddot_est = (theta_dot_now - theta_dot_prev) / self.dt

        # Physics residual
        raw = tau_motor - self.b * theta_dot_now - self.I * theta_ddot_est

        # Low-pass filter
        self._estimate = self.alpha * raw + (1.0 - self.alpha) * self._estimate

        # Push into delay buffer, return oldest (= delayed by latency_steps)
        self._delay_buf.append(self._estimate)
        return self._delay_buf[0]

    def reset(self):
        """Reset all state — call between trajectories."""
        self._gyro_buf  = deque([0.0] * self.window,  maxlen=self.window)
        self._delay_buf = deque([0.0] * max(self.latency_steps, 1),
                                maxlen=max(self.latency_steps, 1))
        self._estimate  = 0.0


@dataclass
class TrainedPINNObserver:
    """
    Wrapper for a trained PyTorch PINN model.

    Drop-in replacement for PhysicsResidualObserver.
    Requires pinn/model.py and a checkpoint from pinn/trainer.py.

    Parameters
    ----------
    checkpoint_path : path to .pt checkpoint file
    window          : gyro history window (must match training config)
    latency_steps   : delay buffer depth
    device          : 'cpu' or 'cuda'
    """
    checkpoint_path: str
    window:          int  = 20
    latency_steps:   int  = 2
    device:          str  = "cpu"

    _model:     object = field(init=False, repr=False)
    _gyro_buf:  object = field(init=False, repr=False)
    _delay_buf: object = field(init=False, repr=False)
    _gyro_mean: float  = field(init=False, repr=False, default=0.0)
    _gyro_std:  float  = field(init=False, repr=False, default=1.0)
    _tm_mean:   float  = field(init=False, repr=False, default=0.0)
    _tm_std:    float  = field(init=False, repr=False, default=1.0)

    def __post_init__(self):
        import torch
        from pinn.model import PINNObserverNet

        ck = torch.load(self.checkpoint_path, map_location=self.device)
        self._model = PINNObserverNet(window=self.window).to(self.device)
        self._model.load_state_dict(ck["model_state"])
        self._model.eval()

        self._gyro_mean = float(ck["gyro_mean"])
        self._gyro_std  = float(ck["gyro_std"])
        self._tm_mean   = float(ck["tm_mean"])
        self._tm_std    = float(ck["tm_std"])

        self._gyro_buf  = deque([0.0] * self.window, maxlen=self.window)
        self._delay_buf = deque([0.0] * max(self.latency_steps, 1),
                                maxlen=max(self.latency_steps, 1))

    def update(self, gyro_meas: float, tau_motor: float) -> float:
        """Run PINN inference for one timestep, return delayed estimate."""
        import torch

        self._gyro_buf.append(gyro_meas)

        gw = np.array(list(self._gyro_buf), dtype=np.float32)
        gw = (gw - self._gyro_mean) / (self._gyro_std + 1e-8)

        tm = np.array([[tau_motor]], dtype=np.float32)
        tm = (tm - self._tm_mean) / (self._tm_std + 1e-8)

        gw_t = torch.from_numpy(gw).unsqueeze(0).to(self.device)
        tm_t = torch.from_numpy(tm).to(self.device)

        with torch.no_grad():
            tau_hat = self._model(gw_t, tm_t).item()

        self._delay_buf.append(tau_hat)
        return self._delay_buf[0]

    def reset(self):
        """Reset buffers — call between trajectories."""
        self._gyro_buf  = deque([0.0] * self.window,  maxlen=self.window)
        self._delay_buf = deque([0.0] * max(self.latency_steps, 1),
                                maxlen=max(self.latency_steps, 1))


# ─── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    import matplotlib.pyplot as plt
    from sim.dynamics import BeamParams, rk4_step
    from sim.wind     import prebuild_wind_profile
    from sim.imu      import IMU
    from control.pid  import PIDController

    DT = 0.001; T = 10.0; N = int(T / DT)
    t  = np.arange(N) * DT

    params  = BeamParams()
    wind    = prebuild_wind_profile(DT, N, seed=42)
    imu     = IMU(dt=DT, seed=1)
    pid     = PIDController(dt=DT)
    obs     = PhysicsResidualObserver(dt=DT, latency_steps=2)
    state   = np.array([0.0, 0.0])
    tau_cmd = 0.0
    SP      = np.deg2rad(10.0)

    true_wind = []
    est_wind  = []

    for k in range(N):
        theta, theta_dot = state
        theta_ddot = (tau_cmd - params.b * theta_dot - wind[k]) / params.I
        gyro, _    = imu.measure(theta, theta_dot, theta_ddot)
        tau_hat    = obs.update(gyro, tau_cmd)
        tau_cmd    = pid.compute(SP, theta, theta_dot, feedforward=tau_hat)
        state      = rk4_step(state, tau_cmd, wind[k], params, DT)
        true_wind.append(wind[k])
        est_wind.append(tau_hat)

    true_wind = np.array(true_wind)
    est_wind  = np.array(est_wind)
    rmse = np.sqrt(np.mean((true_wind[500:] - est_wind[500:]) ** 2))
    print(f"Disturbance estimation RMSE: {rmse:.5f} N·m")

    plt.figure(figsize=(12, 4))
    plt.plot(t, true_wind, color="white",   lw=0.8, alpha=0.7, label="True τ_wind")
    plt.plot(t, est_wind,  color="#4ade80", lw=1.2, alpha=0.9, label="Estimated τ̂_wind")
    plt.xlabel("Time (s)"); plt.ylabel("Torque (N·m)")
    plt.title(f"Physics residual observer — RMSE = {rmse:.4f} N·m")
    plt.legend(); plt.grid(alpha=0.2); plt.tight_layout()
    plt.savefig("results/plots/observer_estimation.png", dpi=150, bbox_inches="tight")
    print("Saved: results/plots/observer_estimation.png")
    print("observer.py: OK")