#!/usr/bin/env python
"""
Minimal driver for protein energy optimization.
This script:
  1. Accepts the number of beads and spatial dimension.
  2. Calls our Cython energy optimization function (which uses our own BFGS routine).
  3. Retrieves the optimized configuration and energy.
  4. Uses the returned trajectory (if any) to animate the process.
  5. Logs runtime information using utilities from utils.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import time
import logging

# Import timing and logging utilities from utils.py
from utils import time_function, setup_logger, log_runtime

# Import functions from the Cython module.
# Adjust the import below depending on your package structure.
from .full_optimization import initialize_protein, bfgs, total_energy, total_energy_grad

def energy_optimization(positions, n_beads, tol=1e-6, max_iter=1000,
                        epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    """
    Optimize the protein energy function using our custom BFGS routine.
    
    Parameters:
        positions: Initial protein configuration (2D NumPy array of shape (n_beads, dim)).
        n_beads: Number of beads.
        tol: Tolerance for the gradient norm.
        max_iter: Maximum iterations.
        epsilon, sigma, b, k_b: Energy parameters.
    
    Returns:
        A tuple (opt_config, opt_energy) where:
          - opt_config is the optimized configuration as a flattened 1D array.
          - opt_energy is the final energy value.
    """
    dim = positions.shape[1]
    x0 = positions.flatten()

    # Define the objective function: reshape x to (n_beads, dim) and compute total energy.
    def objective(x):
        return total_energy(x.reshape((n_beads, dim)), n_beads, epsilon, sigma, b, k_b)

    # Define the gradient function: compute analytic gradient and flatten.
    def gradient(x):
        return total_energy_grad(x.reshape((n_beads, dim)), n_beads, epsilon, sigma, b, k_b).flatten()

    return bfgs(np.array(x0, dtype=np.float64), objective, gradient, tol, max_iter)

def animate_trajectory(trajectory, interval=200):
    """
    Animate the optimization trajectory in 3D.
    
    Parameters:
        trajectory: List of 2D NumPy arrays (each with shape (n_beads, dimension)).
        interval: Delay between frames in milliseconds.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame):
        ax.clear()
        pos = trajectory[frame]
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'o-', markersize=6)
        ax.set_title(f"Iteration {frame+1}/{len(trajectory)}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        return ax,
    
    ani = FuncAnimation(fig, update, frames=len(trajectory), interval=interval, blit=False)
    plt.show()

def run_simulation(n_beads, dimension, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    """
    Run the simulation to optimize the protein configuration and energy.
    """
    # For benchmarking, we force n_beads and dimension here.
    n_beads = 100  # Change to 10, 100, or 500 as needed.
    dimension = 3

    # 1. Initialize the protein configuration.
    positions = initialize_protein(n_beads, dimension)
    print("Initial Energy:", total_energy(positions, n_beads))
    
    # 2. Set up the logger.
    runtime_logger = setup_logger("runtime", "logs/full_cython.log", level=logging.INFO)
    
    # 3. Benchmark the energy optimization using our custom BFGS optimizer.
    start_time = time.perf_counter()
    opt_config, opt_energy = energy_optimization(positions, n_beads, tol=1e-6, max_iter=5000,
                                                 epsilon=epsilon, sigma=sigma, b=b, k_b=k_b)
    elapsed = time.perf_counter() - start_time
    log_runtime(runtime_logger, "OptimizeProtein", elapsed, extra=f"n_beads={n_beads}")
    print(f"Optimization completed in {elapsed:.4f} seconds.")
    print("Optimized Protein Energy:", opt_energy)
    
    # Create a dummy trajectory (initial and final configurations).
    trajectory = [positions, opt_config.reshape((n_beads, dimension))]
    return opt_config, opt_energy, trajectory

def main():
    # Set parameters.
    n_beads = 10  # Change to 10, 100, or 500 for benchmarking.
    dimension = 3

    # Run multiple simulations (for example, 10 runs)
    for i in range(2):
        print(f"\n--- Run {i+1} ---")
        opt_config, opt_energy, trajectory = run_simulation(n_beads, dimension)
    
    # Animate the trajectory from the last run.
    animate_trajectory(trajectory)
    return opt_config, opt_energy, trajectory

if __name__ == "__main__":
    main()
