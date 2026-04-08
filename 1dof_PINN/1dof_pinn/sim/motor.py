"""
sim/motor.py
────────────
First-order lag model for BLDC motor + ESC dynamics.

A motor cannot change speed instantaneously. When the controller commands
a new torque, the motor takes some time to reach it. This is modelled as
a first-order low-pass filter (exponential lag):

    τ_actual[k] = α · τ_commanded[k]  +  (1 − α) · τ_actual[k-1]

    α = 1 − exp(−dt / τ_motor)

where τ_motor (the time constant) is approximately 20ms for a typical
2205 BLDC motor at mid-throttle.

This model also enforces actuator saturation: the motor cannot produce
torque beyond ±tau_max regardless of the command.

Why this matters: without motor lag, the simulation assumes the arm
responds to control commands instantaneously. In reality there is a
lag between "PID says push harder" and "motor actually pushes harder".
This lag is small (20ms) but measurable — especially during fast gusts.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class MotorParams:
    """
    Parameters for the motor + ESC lag model.

    tau_motor : motor time constant [s]
                Time for motor to reach 63% of a step command change.
                Typical 2205 BLDC at 50% throttle: ~15–25ms.

    tau_max   : maximum producible torque [N·m]
                Hard saturation — commands beyond this are clipped.
                Should match BeamParams.tau_max.
    """
    tau_motor: float = 0.020   # 20ms time constant
    tau_max:   float = 0.5     # N·m
    @classmethod
    def for_small(cls):
        """Small ESC — faster response, lower torque ceiling."""
        return cls(tau_motor=0.015, tau_max=0.05)

    @classmethod
    def for_medium(cls):
        """Medium ESC — current default."""
        return cls(tau_motor=0.020, tau_max=0.50)

    @classmethod
    def for_large(cls):
        """Large ESC — slower response, higher torque ceiling."""
        return cls(tau_motor=0.030, tau_max=3.00)

@dataclass
class Motor:
    """
    Stateful first-order motor lag model.

    Parameters
    ----------
    dt     : simulation timestep [s]
    params : MotorParams

    Internal state
    --------------
    _actual : current actual motor torque output [N·m]
              Starts at zero (motor off at simulation start).
    """
    dt:     float
    params: MotorParams = field(default_factory=MotorParams)

    _actual: float = field(init=False, repr=False, default=0.0)
    _alpha:  float = field(init=False, repr=False)

    def __post_init__(self):
        # α controls how fast the motor responds
        # α close to 1 → fast response (small time constant)
        # α close to 0 → slow response (large time constant)
        self._alpha = 1.0 - np.exp(-self.dt / self.params.tau_motor)

    def step(self, tau_commanded: float) -> float:
        """
        Advance motor state by one timestep.

        Parameters
        ----------
        tau_commanded : torque requested by the controller [N·m]
                        Will be clipped to ±tau_max before filtering.

        Returns
        -------
        tau_actual : torque actually delivered to the arm this step [N·m]
        """
        # Clip command to physical actuator limits
        tau_cmd_clipped = np.clip(
            tau_commanded, -self.params.tau_max, self.params.tau_max
        )

        # First-order lag: blend toward commanded value
        self._actual = (
            self._alpha * tau_cmd_clipped
            + (1.0 - self._alpha) * self._actual
        )

        return self._actual

    def reset(self):
        """Reset motor to zero torque — call between trajectories."""
        self._actual = 0.0


# ─── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Verifies motor lag with a step command:
      - Command jumps from 0 to 0.3 N·m at t=0.1s
      - Actual output should follow with ~20ms lag
    Run with: python -m sim.motor
    """
    import matplotlib.pyplot as plt

    DT    = 0.001
    N     = 500
    motor = Motor(dt=DT)
    t     = np.arange(N) * DT

    commanded = np.zeros(N)
    commanded[100:] = 0.3          # step at t=0.1s

    actual = np.zeros(N)
    for k in range(N):
        actual[k] = motor.step(commanded[k])

    # Time to reach 63% of 0.3 N·m should be ~20ms after step
    target_63pct = 0.3 * 0.632
    idx_63 = np.argmax(actual >= target_63pct)
    lag_measured = (idx_63 - 100) * DT * 1000   # ms

    print(f"Commanded step : 0 → 0.3 N·m at t=0.1s")
    print(f"63% rise time  : {lag_measured:.1f} ms  (expected ≈ {MotorParams().tau_motor*1000:.0f} ms)")
    print(f"Final actual   : {actual[-1]:.4f} N·m  (expected ≈ 0.3000)")

    plt.figure(figsize=(10, 4))
    plt.plot(t * 1000, commanded, "w--", lw=1.2, label="Commanded τ")
    plt.plot(t * 1000, actual,    color="#4ade80", lw=1.5, label="Actual τ")
    plt.axvline(100 + lag_measured, color="#f59e0b", lw=0.8,
                ls=":", label=f"63% point ({lag_measured:.0f}ms lag)")
    plt.xlabel("Time (ms)"); plt.ylabel("Torque (N·m)")
    plt.title("Motor first-order lag model — step response")
    plt.legend(); plt.grid(alpha=0.2); plt.tight_layout()
    plt.savefig("results/plots/motor_step_response.png", dpi=150, bbox_inches="tight")
    print("Saved: results/plots/motor_step_response.png")
    print("motor.py: OK")