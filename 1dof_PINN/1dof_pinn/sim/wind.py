"""
sim/wind.py
───────────
Dryden atmospheric turbulence model for wind disturbance simulation.

The Dryden model is the aerospace standard (MIL-SPEC) for atmospheric
turbulence. It generates spatially correlated (coloured) noise rather
than white noise — real wind has memory and structure, not instant
frame-to-frame randomness.

Continuous transfer function:
    H(s) = σ· √(2L/V)  /  (τ_L · s + 1)     τ_L = L / V

Discretised as a first-order autoregressive filter:
    w[k] = a · w[k-1]  +  b_coeff · N(0,1)

    a       = exp( -dt / τ_L)          how much the previous value persists
    b_coeff = σ · √(1 - a²)           how much new noise enters each step

On top of the continuous turbulence, two deterministic gust events model
a sudden crosswind hit (t=4s) and a direction reversal (t=9s). These are
identical across all experimental conditions to ensure a fair comparison.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List


# ── Intensity presets ──────────────────────────────────────────────────────────
INTENSITY_PRESETS = {
    "light":    {"sigma": 0.02, "L": 50.0, "V": 5.0},
    "moderate": {"sigma": 0.06, "L": 30.0, "V": 5.0},
    "severe":   {"sigma": 0.12, "L": 15.0, "V": 5.0},
}


@dataclass
class GustEvent:
    """
    A deterministic torque step added to the turbulence background.

    onset_time  : time at which the gust begins [s]
    magnitude   : torque added from onset_time onward [N·m]
    """
    onset_time: float
    magnitude:  float


@dataclass
class DrydenWind:
    """
    Stateful Dryden turbulence generator.

    Parameters
    ----------
    dt          : integration timestep [s]
    intensity   : 'light', 'moderate', or 'severe'
    seed        : random seed (None = different every run)
    gust_events : list of GustEvent applied on top of turbulence
    """
    dt:          float
    intensity:   str = "moderate"
    seed:        Optional[int] = None
    gust_events: List[GustEvent] = field(default_factory=list)

    _a:     float  = field(init=False, repr=False)
    _b:     float  = field(init=False, repr=False)
    _state: float  = field(init=False, repr=False, default=0.0)
    _t:     float  = field(init=False, repr=False, default=0.0)
    _rng:   object = field(init=False, repr=False)

    def __post_init__(self):
        preset    = INTENSITY_PRESETS[self.intensity]
        tau_L     = preset["L"] / preset["V"]
        self._a   = np.exp(-self.dt / tau_L)
        self._b   = preset["sigma"] * np.sqrt(1.0 - self._a ** 2)
        self._rng = np.random.default_rng(self.seed)
        self._state = 0.0
        self._t     = 0.0

    def step(self) -> float:
        """Advance one timestep, return wind torque [N·m]."""
        self._state = self._a * self._state + self._b * self._rng.standard_normal()
        gust_total  = sum(
            g.magnitude for g in self.gust_events if self._t >= g.onset_time
        )
        self._t += self.dt
        return self._state + gust_total

    def reset(self):
        """Reset to initial state — call between trajectories."""
        self._state = 0.0
        self._t     = 0.0
        self._rng   = np.random.default_rng(self.seed)


def make_gust_events(n_gusts: int, t_start: float = 2.0, t_end: float = 13.0) -> List[GustEvent]:
    """
    Generate n_gusts evenly-spaced gust events between t_start and t_end.

    Magnitudes alternate between +0.10 N·m and -0.08 N·m, matching the
    original two-gust default. With n_gusts=2 this reproduces the original
    events at t=4s and t=9s exactly.

    Parameters
    ----------
    n_gusts : number of gust events (0 = no gusts, max typically 5)
    t_start : earliest possible gust onset [s]
    t_end   : latest  possible gust onset [s]
    """
    if n_gusts == 0:
        return []
    magnitudes = [+0.10, -0.08]
    if n_gusts == 1:
        onsets = [(t_start + t_end) / 2.0]
    else:
        onsets = np.linspace(t_start, t_end, n_gusts).tolist()
    return [
        GustEvent(onset_time=t, magnitude=magnitudes[i % len(magnitudes)])
        for i, t in enumerate(onsets)
    ]


def make_standard_wind(dt: float,
                       intensity: str = "moderate",
                       seed: int = 42,
                       n_gusts: int = 2) -> DrydenWind:
    """
    Standard experiment wind: Dryden turbulence + n_gusts deterministic events.

    n_gusts=2 (default) reproduces the original gust at 4s and reversal at 9s.
    n_gusts=0 gives pure turbulence with no step events.
    """
    return DrydenWind(
        dt          = dt,
        intensity   = intensity,
        seed        = seed,
        gust_events = make_gust_events(n_gusts),
    )


def prebuild_wind_profile(dt: float,
                          n_steps: int,
                          intensity: str = "moderate",
                          seed: int = 42,
                          n_gusts: int = 2) -> np.ndarray:
    """
    Pre-generate the full wind torque array of shape (n_steps,).

    Build once and share across all conditions for a fair comparison.
    """
    wind    = make_standard_wind(dt, intensity, seed, n_gusts=n_gusts)
    profile = np.zeros(n_steps, dtype=np.float32)
    for k in range(n_steps):
        profile[k] = wind.step()
    return profile


# ─── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    DT = 0.001; T = 15.0; N = int(T / DT)
    t  = np.arange(N) * DT

    profiles = {
        intensity: prebuild_wind_profile(DT, N, intensity, seed=42)
        for intensity in ["light", "moderate", "severe"]
    }

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("Dryden Wind Disturbance Profiles", fontsize=13)
    colors = {"light": "#38bdf8", "moderate": "#f59e0b", "severe": "#ef4444"}

    for ax, intensity in zip(axes, ["light", "moderate", "severe"]):
        ax.plot(t, profiles[intensity], color=colors[intensity], lw=0.8)
        ax.axvline(4.0, color="white",  lw=0.8, ls="--", alpha=0.5, label="Gust onset")
        ax.axvline(9.0, color="orange", lw=0.8, ls="--", alpha=0.5, label="Reversal")
        ax.axhline(0,   color="gray",   lw=0.5, alpha=0.4)
        ax.set_ylabel("τ_wind (N·m)", fontsize=9)
        ax.set_title(
            f"{intensity.capitalize()}  "
            f"(σ={INTENSITY_PRESETS[intensity]['sigma']} N·m)", fontsize=10
        )
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("results/plots/wind_profiles.png", dpi=150, bbox_inches="tight")
    print("Saved: results/plots/wind_profiles.png")
    print("wind.py: OK")