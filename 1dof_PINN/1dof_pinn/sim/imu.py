"""
sim/imu.py
──────────
IMU (Inertial Measurement Unit) sensor noise model.

Models a MEMS gyroscope + accelerometer with three realistic imperfections:
  1. White noise       — random measurement error every timestep
  2. Constant bias     — fixed offset that shifts all readings
  3. Bias random walk  — the bias drifts slowly over time

The PINN receives these noisy measurements, not the true state.
This directly tests the sparse/degraded sensing claim of the paper.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class IMUParams:
    """
    Noise parameters for the IMU model.

    gyro_noise_density : white noise spectral density [rad/s / sqrt(Hz)]
                         Scales with 1/sqrt(dt) — standard shot noise model.
                         Typical MEMS value: 0.003–0.010 rad/s/sqrt(Hz)

    gyro_bias          : fixed offset on all gyro readings [rad/s]

    gyro_bias_walk     : std dev of bias random walk per step [rad/s]
                         Models temperature-induced drift over time.

    accel_noise        : white noise on accelerometer [m/s²]
    """
    gyro_noise_density: float = 0.003
    gyro_bias:          float = 0.001
    gyro_bias_walk:     float = 5e-6
    accel_noise:        float = 0.008


@dataclass
class IMU:
    """
    Stateful IMU sensor model.

    Simulates the ICM-42688-P MEMS IMU planned for the physical testbench.
    Each call to .measure() returns noisy gyro and accelerometer readings.

    Parameters
    ----------
    dt     : timestep [s] — used to scale noise density correctly
    params : IMUParams noise configuration
    seed   : random seed for reproducibility
    """
    dt:     float
    params: IMUParams = field(default_factory=IMUParams)
    seed:   Optional[int] = None

    _bias_walk_state: float  = field(init=False, repr=False, default=0.0)
    _rng:             object = field(init=False, repr=False)
    _noise_scaled:    float  = field(init=False, repr=False)

    def __post_init__(self):
        self._rng             = np.random.default_rng(self.seed)
        self._noise_scaled    = self.params.gyro_noise_density / np.sqrt(self.dt)
        self._bias_walk_state = 0.0

    def measure(self,
                theta:      float,
                theta_dot:  float,
                theta_ddot: float) -> Tuple[float, float]:
        """
        Apply sensor noise model to true state values.

        Parameters
        ----------
        theta      : true angle [rad]
        theta_dot  : true angular velocity [rad/s]
        theta_ddot : true angular acceleration [rad/s²]

        Returns
        -------
        (gyro_meas, accel_meas) : noisy sensor readings
        """
        # Bias random walk — small drift each timestep
        self._bias_walk_state += (
            self.params.gyro_bias_walk * self._rng.standard_normal()
        )

        # Gyroscope: true rate + bias + drift + white noise
        gyro_meas = (
            theta_dot
            + self.params.gyro_bias
            + self._bias_walk_state
            + self._noise_scaled * self._rng.standard_normal()
        )

        # Accelerometer: true acceleration + white noise
        accel_meas = (
            theta_ddot
            + self.params.accel_noise * self._rng.standard_normal()
        )

        return gyro_meas, accel_meas

    def reset(self):
        """Reset bias walk state — call between trajectories."""
        self._bias_walk_state = 0.0
        self._rng = np.random.default_rng(self.seed)


# ─── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    DT  = 0.001
    N   = 15000
    imu = IMU(dt=DT, seed=0)

    true_theta_dot  = 0.5
    true_theta_ddot = 0.0

    gyro_readings = []
    for _ in range(N):
        g, _ = imu.measure(0.0, true_theta_dot, true_theta_ddot)
        gyro_readings.append(g)

    gyro_readings = np.array(gyro_readings)
    t = np.arange(N) * DT

    expected_mean = true_theta_dot + IMUParams().gyro_bias
    expected_std  = IMUParams().gyro_noise_density / np.sqrt(DT)

    print(f"True θ̇       : {true_theta_dot:.4f} rad/s")
    print(f"Gyro mean    : {gyro_readings.mean():.4f}  (expected ≈ {expected_mean:.4f})")
    print(f"Gyro std     : {gyro_readings.std():.5f}  (expected ≈ {expected_std:.5f})")

    plt.figure(figsize=(12, 4))
    plt.plot(t, gyro_readings, lw=0.6, color="#38bdf8", label="Gyro measurement")
    plt.axhline(true_theta_dot, color="white",   lw=1.2, ls="--", label="True θ̇")
    plt.axhline(expected_mean,  color="#f59e0b", lw=1.0, ls=":",  label="True + bias")
    plt.xlabel("Time (s)"); plt.ylabel("θ̇ (rad/s)")
    plt.title("IMU gyroscope noise model")
    plt.legend(); plt.grid(alpha=0.2); plt.tight_layout()
    plt.savefig("results/plots/imu_noise.png", dpi=150, bbox_inches="tight")
    print("Saved: results/plots/imu_noise.png")
    print("imu.py: OK")