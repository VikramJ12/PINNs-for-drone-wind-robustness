#1/bin/bash

python -m sim.imu       # → saves results/plots/imu_noise.png
python -m sim.motor     # → saves results/plots/motor_step_response.png
python -m control.pid   # → saves results/plots/pid_step_response.png
python -m control.observer  # → saves results/plots/observer_estimation.png

# Then run the full experiment:
python simulate.py
# → prints metrics table
# → saves results/plots/simulation_results.png
