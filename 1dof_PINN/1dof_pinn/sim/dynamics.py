"""
sim/dynamics.py
───────────────
Physical model of the 1-DoF rotating arm testbench.

System equation (Newton's second law for rotation):
    I · θ̈  =  τ_motor  -  b · θ̇  -  τ_wind(t)

State vector: s = [θ, θ̇]
    θ   = angle from horizontal (radians)
    θ̇   = angular velocity (rad/s)
    θ̈   = angular acceleration (rad/s²)  ← computed, not stored

The second-order ODE is rewritten as two coupled first-order ODEs:
    ds/dt = [ θ̇,  (τ_motor - b·θ̇ - τ_wind) / I ]

This is the only physics in the simulator. Everything else (wind, IMU,
controller) feeds τ values into this equation or reads the resulting state.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class BeamParams:
    """
    Physical constants for the 1-DoF arm testbench.

    I     : Moment of inertia [kg·m²]
            Estimated from: 25cm carbon arm + 2205 motor + counterweight.
            I = Σ(m_i · r_i²) ≈ 0.008 kg·m²

    b     : Viscous damping coefficient [N·m·s/rad]
            Models bearing friction + aerodynamic drag on the arm itself
            (not propeller thrust — that is τ_motor).

    tau_max : Actuator saturation limit [N·m]
              The motor cannot exceed this torque regardless of command.
              Prevents physically unrealistic control signals.
    """
    I:       float = 0.008   # kg·m²
    b:       float = 0.004   # N·m·s/rad
    tau_max: float = 0.5     # N·m
    @classmethod
    def small(cls):
        """3-inch micro class. I=8e-5 kg·m², tau_max=0.05 N·m"""
        return cls(I=0.00008, b=0.0008, tau_max=0.05)

    @classmethod
    def medium(cls):
        """5-inch racing class. Default. I=0.008 kg·m², tau_max=0.50 N·m"""
        return cls(I=0.008, b=0.004, tau_max=0.5)

    @classmethod
    def large(cls):
        """7-inch long-range class. I=0.080 kg·m², tau_max=3.00 N·m"""
        return cls(I=0.080, b=0.018, tau_max=3.0)


def beam_ode(state: np.ndarray,
             tau_motor: float,
             tau_wind: float,
             params: BeamParams) -> np.ndarray:
    """
    The ODE function f such that ds/dt = f(state, inputs).

    Parameters
    ----------
    state     : [θ (rad), θ̇ (rad/s)]
    tau_motor : control torque from motor [N·m]
    tau_wind  : external wind disturbance torque [N·m]
    params    : BeamParams physical constants

    Returns
    -------
    dstate_dt : [θ̇,  θ̈]  — the time derivatives of the state
    """
    theta, theta_dot = state

    # Rearranged from  I·θ̈ = τ_motor − b·θ̇ − τ_wind
    theta_ddot = (tau_motor - params.b * theta_dot - tau_wind) / params.I

    return np.array([theta_dot, theta_ddot])


def rk4_step(state: np.ndarray,
             tau_motor: float,
             tau_wind: float,
             params: BeamParams,
             dt: float) -> np.ndarray:
    """
    Advance the state by one timestep using 4th-order Runge-Kutta.

    RK4 evaluates the ODE at four points within [t, t+dt] and combines
    them with a 1-2-2-1 weighted average. The local truncation error is
    O(dt⁵), which at dt=0.001s is numerically negligible.

    Parameters
    ----------
    state     : current [θ, θ̇]
    tau_motor : motor torque command at this step [N·m]
    tau_wind  : wind disturbance torque at this step [N·m]
    params    : BeamParams
    dt        : timestep [s]

    Returns
    -------
    next_state : [θ, θ̇] at time t + dt
    """
    # Evaluate ODE at start, two midpoints, and end
    k1 = beam_ode(state,                   tau_motor, tau_wind, params)
    k2 = beam_ode(state + dt / 2.0 * k1,   tau_motor, tau_wind, params)
    k3 = beam_ode(state + dt / 2.0 * k2,   tau_motor, tau_wind, params)
    k4 = beam_ode(state + dt        * k3,   tau_motor, tau_wind, params)

    # Weighted average: midpoint estimates (k2, k3) count double
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ─── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Sanity check: arm starts horizontal (θ=0), motor applies +0.1 N·m,
    no wind. Should rotate in the positive direction.
    Run with:  python -m sim.dynamics
    """
    params = BeamParams()
    state  = np.array([0.0, 0.0])   # [θ=0, θ̇=0]
    dt     = 0.001
    t_end  = 2.0
    steps  = int(t_end / dt)

    for k in range(steps):
        state = rk4_step(state, tau_motor=0.1, tau_wind=0.0, params=params, dt=dt)

    theta_deg = np.rad2deg(state[0])
    print(f"After {t_end}s with τ_motor=0.1 N·m, no wind:")
    print(f"  θ      = {theta_deg:.4f}°  (expected: large positive angle)")
    print(f"  θ̇      = {state[1]:.4f} rad/s")
    print(f"  θ̈_end  ≈ {(0.1 - params.b * state[1]) / params.I:.4f} rad/s²")
    print("dynamics.py: OK")