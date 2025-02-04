import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import time

# Import timing and logging functions
from utils import time_function, log_runtime, setup_logger, logging

# Use an absolute import (ensure protein_cython is in your PYTHONPATH or same folder)
from protein_cython import initialize_protein, total_energy, lennard_jones_potential, bond_potential

#######################################
# Wrapper Function for total_energy
#######################################
def total_energy_wrapper(flat_positions, n_beads, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    """
    Reshape the flattened array (1D) into a 2D array (n_beads x 3) and call the Cython total_energy.
    """
    dimension = 3  # Set the dimension for your protein (must match your initialization)
    positions = flat_positions.reshape((n_beads, dimension))
    return total_energy(positions, n_beads, epsilon, sigma, b, k_b)

#######################################
# Optimization Function
#######################################
def optimize_protein(positions, n_beads, write_csv=False, maxiter=1000, tol=1e-6):
    """
    Optimize the positions of the protein to minimize total energy.
    Uses SciPy's BFGS optimizer and the wrapper for our Cython total_energy.
    """
    trajectory = []

    def callback(x):
        # Record the trajectory by reshaping the flat array back to 2D for logging/visualization
        trajectory.append(x.reshape((n_beads, -1)))
        if len(trajectory) % 20 == 0:
            print(f"Trajectory steps recorded: {len(trajectory)}")

    result = minimize(
        fun=total_energy_wrapper,      # use the wrapper function here
        x0=positions.flatten(),        # optimizer works on a flat array
        args=(n_beads,),               # pass n_beads as additional argument
        method='BFGS',
        callback=callback,
        tol=tol,
        options={'maxiter': maxiter, 'disp': True}
    )
    if write_csv:
        csv_filepath = f'protein{n_beads}.csv'
        print(f'Writing data to file {csv_filepath}')
        np.savetxt(csv_filepath, trajectory[-1], delimiter=",")
    return result, trajectory

#######################################
# 3D Visualization Function
#######################################
def plot_protein_3d(positions, title="Protein Conformation", ax=None):
    """
    Plot the 3D positions of the protein.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    positions = positions.reshape((-1, 3))
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o', markersize=6)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

#######################################
# Animation Function
#######################################
def animate_optimization(trajectory, interval=100):
    """
    Animate the protein folding process in 3D.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    line, = ax.plot([], [], [], '-o', markersize=6)

    def update(frame):
        positions = trajectory[frame]
        line.set_data(positions[:, 0], positions[:, 1])
        line.set_3d_properties(positions[:, 2])

        # Autoscale the axes
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        z_min, z_max = positions[:, 2].min(), positions[:, 2].max()
        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylim(y_min - 1, y_max + 1)
        ax.set_zlim(z_min - 1, z_max + 1)

        ax.set_title(f"Step {frame + 1}/{len(trajectory)}")
        return line,

    ani = FuncAnimation(fig, update, frames=len(trajectory), interval=interval, blit=False)
    plt.show()

#######################################
# Main Execution Block
#######################################
if __name__ == "__main__":
    # Set test parameters
    n_beads = 10      # Adjust as needed for benchmarking
    dimension = 3

    # Initialize protein configuration
    initial_positions = initialize_protein(n_beads, dimension)
    
    # Compute and print initial energy
    initial_energy = total_energy(initial_positions, n_beads)
    print("Initial Energy:", initial_energy)
    plot_protein_3d(initial_positions, title="Initial Configuration")

    # Benchmark and optimize the protein configuration
    (result, trajectory), elapsed = time_function(optimize_protein, initial_positions, n_beads, write_csv=True)
    runtime_logger = setup_logger("runtime", "logs/cython_scipy.log", level=logging.INFO)
    log_runtime(runtime_logger, "OptimizeProtein", elapsed, extra=f"n_beads={n_beads}")
    print(f"Optimization completed in {elapsed:.4f} seconds.")

    # Process and visualize the optimized configuration
    optimized_positions = result.x.reshape((n_beads, dimension))
    optimized_energy = total_energy(optimized_positions, n_beads)
    print("Optimized Energy:", optimized_energy)
    plot_protein_3d(optimized_positions, title="Optimized Configuration")

    # Optionally, animate the optimization process
    animate_optimization(trajectory)
