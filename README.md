# PINN Disturbance Observer for 1-DoF Drone Arm

A research project demonstrating **Physics-Informed Neural Networks (PINNs)** as real-time wind disturbance observers for a 1-DoF drone arm testbench. The system augments a classical PID controller with a PINN-based feedforward disturbance estimate, achieving improved tracking accuracy and faster gust recovery compared to PID-only approaches.

---

## Project Overview

This project validates the hypothesis that **PINNs can estimate unmeasured external disturbances (wind) online**, using only gyroscope measurements and known motor torque — no dedicated wind sensors required. The estimated disturbance is fed forward into a PID controller to cancel wind effects before they degrade tracking performance.

### Key Idea

A 1-DoF rotating arm obeys the rigid-body ODE:

```
I·θ̈ = τ_motor − b·θ̇ − τ_wind
```

Rearranging gives the physics residual observer:

```
τ̂_wind = τ_motor − b·θ̇ − I·θ̈
```

A PINN learns to replicate this residual from raw IMU data alone, embedding the physics as a soft constraint in its loss function. This makes it more data-efficient and better generalizing than a black-box neural network, while being more robust to sensor noise than a purely analytical observer.

---

## Outcomes

The simulation compares **four control architectures** under identical Dryden turbulence and step gust conditions:

| Condition | Description |
|---|---|
| **No Control** | Open-loop baseline — no actuation |
| **PID Only** | Classical PID feedback, no disturbance compensation |
| **PINN Only** | PD + feedforward disturbance estimate (10 ms latency) |
| **PID + PINN (Proposed)** | Full PID + physics-residual feedforward (2 ms latency) |

**Expected outcomes** (based on simulation design):
- **PID + PINN** achieves the lowest RMS tracking error and fastest gust recovery time
- **PINN Only** outperforms PID alone in steady-state accuracy due to feedforward cancellation
- **Physics-residual observer** provides ground-truth disturbance estimation for PINN training targets
- **Gust recovery** (time to return within 0.5° after a step gust) improves by 40–60% with disturbance feedforward

Simulation outputs — trajectory plots, metrics table (PNG + CSV) — are written to `results/plots/`.

---

## Project Structure

```
PINNS for Drone/
├── 1dof_PINN/
│   ├── 1dof_pinn/                  # Python simulation package
│   │   ├── sim/
│   │   │   ├── dynamics.py         # 1-DoF beam ODE, RK4 integrator
│   │   │   ├── motor.py            # First-order motor lag model
│   │   │   ├── imu.py              # MEMS IMU noise model (ICM-42688-P)
│   │   │   └── wind.py             # Dryden atmospheric turbulence
│   │   ├── control/
│   │   │   ├── pid.py              # Discrete-time PID with anti-windup
│   │   │   └── observer.py         # Physics residual & PINN observers
│   │   ├── pinn/                   # PINN training modules (stubs — not yet implemented)
│   │   │   ├── model.py            # PyTorch PINNObserverNet architecture
│   │   │   ├── dataset.py          # Data generation and loading
│   │   │   ├── loss.py             # Physics-informed loss function
│   │   │   └── trainer.py          # Training loop with checkpointing
│   │   ├── simulate.py             # Main experiment runner and metrics
│   │   ├── requirements.txt        # Python dependencies
│   │   └── run_system.sh           # Sequential component test + full run script
│   ├── visual_sim/                 # Interactive browser simulator
│   │   ├── src/main.jsx            # React entry point
│   │   ├── arm_sim.jsx             # Main ArmSim component (full physics in JS)
│   │   ├── index.html
│   │   ├── vite.config.js
│   │   └── package.json
│   └── Documentation/
│       └── Current_vs_Future_PINNs_comparison.tex
├── Literature Review.md            # Research context and related work
├── LICENSE
└── README.md
```

---

## Physics Models

### Beam Dynamics
- **ODE:** `I·θ̈ = τ_motor − b·θ̇ − τ_wind`
- **Integrator:** 4th-order Runge-Kutta at Δt = 1 ms (1 kHz)
- **Arm presets:** small (3-inch), medium (5-inch, default), large (7-inch)

### Motor
- First-order lag: `τ[k] = α·τ_cmd[k] + (1−α)·τ[k−1]`, τ ≈ 20 ms
- Hard actuator saturation at ±τ_max

### IMU (ICM-42688-P)
- White noise (0.003 rad/s/√Hz), fixed bias, bias random walk

### Wind (Dryden Turbulence)
- MIL-SPEC autoregressive filter with intensity presets: light, moderate, severe
- Deterministic step gusts at t = 4 s and t = 9 s

---

## How to Run

### Prerequisites

- Python 3.9+
- Node.js 18+ (for the browser simulator)

---

### Python Simulation

#### 1. Install dependencies

```bash
cd "1dof_PINN/1dof_pinn"
pip install -r requirements.txt
```

#### 2. Run the full experiment

```bash
python simulate.py
```

With optional arguments:

```bash
python simulate.py --size medium --wind moderate
# --size: small | medium | large   (default: medium)
# --wind: light | moderate | severe (default: moderate)
```

Outputs are saved to `results/plots/`:
- `simulation_results_<size>_<wind>.png` — 6-panel trajectory & metric plots
- `metrics_table_<size>_<wind>.png` — styled comparison table
- `metrics_table_<size>_<wind>.csv` — raw numbers

#### 3. Run individual component diagnostics (optional)

Each module has a self-test that saves a diagnostic plot:

```bash
python -m sim.imu          # IMU noise characterization
python -m sim.motor        # Motor lag step response
python -m control.pid      # PID step response
python -m control.observer # Disturbance observer estimation accuracy
```

#### 4. Run all tests sequentially

```bash
bash run_system.sh
```

---

### Interactive Browser Simulator

A fully self-contained browser simulation that mirrors the Python physics model in JavaScript, running at 60 FPS with configurable conditions.

#### 1. Install and start

```bash
cd 1dof_PINN/visual_sim
npm install
npm run dev
```

Open the URL printed in the terminal (default: `http://localhost:5173`).

#### 2. Build for production

```bash
npm run build
npm run preview
```

#### Features

- **Control mode selector:** No Control / PID / PINN / PID + PINN
- **Wind intensity:** Light / Moderate / Severe
- **Gust count:** 0–5 step disturbances via slider
- **Playback controls:** Play / Pause, speed × 1 / 2 / 4 / 8
- **Live visualization:** Rotating arm, wind arrows, propeller blur, real-time charts

---

## Implementation Status

| Component | Status |
|---|---|
| Beam dynamics (RK4) | Done |
| Motor lag model | Done |
| IMU noise model | Done |
| Dryden wind turbulence | Done |
| PID controller | Done |
| Physics residual observer | Done |
| Main simulation + metrics | Done |
| Interactive browser simulator | Done |
| PINN model architecture | Not yet implemented |
| PINN training pipeline | Not yet implemented |
| Hardware data collection | Not yet implemented |

The PINN modules in `pinn/` are stubs. The current simulation uses a physics-residual observer (analytical) as a proxy for what a trained PINN would produce. PINN training is the next development phase.

---

## Dependencies

**Python (`requirements.txt`)**

```
numpy>=1.24.0
scipy>=1.10.0
torch>=2.0.0
matplotlib>=3.7.0
tqdm>=4.65.0
pandas>=2.0.0
```

**Node.js (`package.json`)**

```json
{
  "react": "^18.3.1",
  "react-dom": "^18.3.1",
  "vite": "^6.3.0"
}
```

---

## Related Work

This project builds on research in:

- Physics-informed neural networks for UAV state estimation (Wang et al., 2025)
- Physics-informed system identification (Bianchi et al., 2024)
- Neural adaptive control for quadrotors (Bisheban & Lee, 2019)
- LSTM-based wind estimation (Valente et al., 2022)
- Active disturbance rejection control (Shao et al., 2023)

See [Literature Review.md](Literature Review.md) for full context and citations.

---

## License

See [LICENSE](LICENSE).
