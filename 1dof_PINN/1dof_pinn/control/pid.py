"""
control/pid.py
──────────────
Discrete-time PID controller with feedforward input.

Standard PID control law:
    τ = Kp·e  +  Ki·∫e dt  +  Kd·ė  +  feedforward

where e = θ_ref − θ  (angle tracking error)

The feedforward parameter is the architectural hook for the PINN observer:
  - PID Only condition:    feedforward = 0.0
  - PID + PINN condition:  feedforward = τ̂_wind (PINN estimate)

Integral anti-windup via conditional integration: if the output is
saturated, the integral is not updated. This prevents the integral from
accumulating during large disturbances and causing overshoot on release.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PIDParams:
    """
    PID gain parameters tuned for BeamParams (I=0.008, b=0.004).

    Kp : proportional gain
    Ki : integral gain     — set high enough to eliminate steady-state error
    Kd : derivative gain   — damps oscillations
    """
    Kp: float = 1.5
    Ki: float = 0.80
    Kd: float = 0.10
    @classmethod
    def for_small(cls):
        """Gains tuned for small arm. Higher Kp needed due to low inertia."""
        return cls(Kp=2.0, Ki=1.2, Kd=0.05)

    @classmethod
    def for_medium(cls):
        """Gains tuned for medium arm. Current default."""
        return cls(Kp=1.5, Ki=0.80, Kd=0.10)

    @classmethod
    def for_large(cls):
        """Gains tuned for large arm. Higher Kd needed to damp slow oscillations."""
        return cls(Kp=1.2, Ki=0.50, Kd=0.15)

@dataclass
class PIDController:
    """
    Stateful discrete-time PID controller.

    Parameters
    ----------
    dt      : control loop timestep [s]
    params  : PIDParams gain configuration
    tau_max : output saturation limit [N·m]
    """
    dt:      float
    params:  PIDParams = field(default_factory=PIDParams)
    tau_max: float = 0.5

    _integral:   float = field(init=False, repr=False, default=0.0)
    _prev_error: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self):
        self._integral   = 0.0
        self._prev_error = 0.0

    def compute(self,
                setpoint:    float,
                theta:       float,
                theta_dot:   float,
                feedforward: float = 0.0) -> float:
        """
        Compute control torque for this timestep.

        Parameters
        ----------
        setpoint    : desired angle [rad]
        theta       : current measured angle [rad]
        theta_dot   : current angular velocity [rad/s]
        feedforward : PINN disturbance estimate [N·m], zero for PID-Only

        Returns
        -------
        tau_cmd : clipped torque command [N·m]
        """
        error      = setpoint - theta
        derivative = (error - self._prev_error) / self.dt

        # Tentative integral update
        integral_candidate = self._integral + error * self.dt

        # Full tentative output
        tau_raw = (
            self.params.Kp * error
            + self.params.Ki * integral_candidate
            + self.params.Kd * derivative
            + feedforward
        )

        # Anti-windup: only commit integral if output is not saturated
        if abs(tau_raw) <= self.tau_max:
            self._integral = integral_candidate

        self._prev_error = error
        return float(np.clip(tau_raw, -self.tau_max, self.tau_max))

    def reset(self):
        """Reset integrator and error memory — call between trajectories."""
        self._integral   = 0.0
        self._prev_error = 0.0


# ─── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    import matplotlib.pyplot as plt
    from sim.dynamics import BeamParams, rk4_step

    DT     = 0.001
    T      = 5.0
    N      = int(T / DT)
    SP_RAD = np.deg2rad(10.0)

    params  = BeamParams()
    pid     = PIDController(dt=DT)
    state   = np.array([0.0, 0.0])
    tau_cmd = 0.0

    log_theta = np.zeros(N)
    log_tau   = np.zeros(N)
    t         = np.arange(N) * DT

    for k in range(N):
        theta, theta_dot = state
        tau_cmd       = pid.compute(SP_RAD, theta, theta_dot)
        log_theta[k]  = np.rad2deg(theta)
        log_tau[k]    = tau_cmd
        state         = rk4_step(state, tau_cmd, 0.0, params, DT)

    final_err = abs(np.rad2deg(SP_RAD) - log_theta[-1])
    print(f"Setpoint           : {np.rad2deg(SP_RAD):.1f}°")
    print(f"Final angle        : {log_theta[-1]:.4f}°")
    print(f"Steady-state error : {final_err:.5f}°  (expected < 0.01°)")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(t, log_theta, color="#4ade80", lw=1.5, label="θ actual")
    ax1.axhline(10.0, color="white", lw=1.0, ls="--", label="Setpoint 10°")
    ax1.set_ylabel("θ (degrees)"); ax1.legend(); ax1.grid(alpha=0.2)
    ax1.set_title("PID step response — no wind")

    ax2.plot(t, log_tau, color="#f59e0b", lw=1.2, label="τ_cmd")
    ax2.set_ylabel("τ (N·m)"); ax2.set_xlabel("Time (s)")
    ax2.legend(); ax2.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig("results/plots/pid_step_response.png", dpi=150, bbox_inches="tight")
    print("Saved: results/plots/pid_step_response.png")
    print("pid.py: OK")