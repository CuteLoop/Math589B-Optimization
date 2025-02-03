import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import time
# Import the timing function from utils.py
from utils import time_function, log_runtime, setup_logger, logging



# Import our Cython functions
from .protein_cython import initialize_protein, total_energy, lennard_jones_potential, bond_potential

# If needed, you can still use the pure Python optimization routine:
def optimize_protein(positions, n_beads, write_csv=False):
    """
    Optimize the positions of the protein to minimize total energy.
    Uses SciPy's BFGS optimizer and our Cython total_energy function.
    """
    trajectory = []

    def callback(x):
        trajectory.append(x.reshape((n_beads, -1)))
        if len(trajectory) % 20 == 0:
            print(len(trajectory))

    result = minimize(
        fun=lambda x: total_energy(x.reshape((n_beads, 3)), n_beads),
        x0=positions.flatten(),
        args=(),
        method='BFGS',
        callback=callback,
        options={'disp': True}
    )
    if write_csv:
        csv_filepath = f'protein{n_beads}.csv'
        print(f'Writing data to file {csv_filepath}')
        np.savetxt(csv_filepath, trajectory[-1], delimiter=",")
    return result, trajectory

# 3D visualization function
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

# Animation function
# Animation function with autoscaling
def animate_optimization(trajectory, interval=100):
    """
    Animate the protein folding process in 3D with autoscaling.
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

    ani = FuncAnimation(
        fig, update, frames=len(trajectory), interval=interval, blit=False
    )
    plt.show()


# ------------------------------
# Main block with runtime benchmarking using time_function
# ------------------------------
# Main block with runtime benchmarking using time_function
# ------------------------------

if __name__ == "__main__":
    # Set test parameters
    n_beads = 100      # Try 10, 100, or 500 for benchmarking.
    dimension = 3

    # Initialize protein configuration
    initial_positions = initialize_protein(n_beads, dimension)
    
    # Print and plot initial configuration
    initial_energy = total_energy(initial_positions, n_beads)  # <-- Remove .flatten()
    print("Initial Energy:", initial_energy)
    plot_protein_3d(initial_positions, title="Initial Configuration")

    # Benchmark the optimization using the time_function wrapper
    (result, trajectory), elapsed = time_function(optimize_protein, initial_positions, n_beads, write_csv=True)
    runtime_logger = setup_logger("runtime", "logs/cython_scipy.log", level=logging.INFO)
    log_runtime(runtime_logger, "OptimizeProtein", elapsed, extra=f"n_beads={n_beads}")
    print(f"Optimization completed in {elapsed:.4f} seconds.")
    
    # Process and visualize the optimized configuration
    optimized_positions = result.x.reshape((n_beads, dimension))
    optimized_energy = total_energy(optimized_positions, n_beads)  # Also pass the 2D array here.
    print("Optimized Energy:", optimized_energy)
    plot_protein_3d(optimized_positions, title="Optimized Configuration")

    # Optionally, animate the optimization process
    animate_optimization(trajectory)


# ------------------------------
# Testing Benchmarking: Repeated runs
# ------------------------------

# Set up a single runtime logger for the benchmarking section.
  #  runtime_logger = setup_logger("runtime", "logs/cython_scipy.log", level=logging.INFO)


'''
    for i in range(10):
        # Initialize protein configuration (2D array)
        initial_positions = initialize_protein(n_beads, dimension)
        
        # Compute the initial energy using the Cython total_energy function
        # (Pass the 2D array directly)
        initial_energy = total_energy(initial_positions, n_beads)
        print(f"Run {i+1}: Initial Energy: {initial_energy}")
        
        # Benchmark the optimization using the time_function wrapper.
        # Note: The lambda in optimize_protein uses the 2D array as needed.
        (result, trajectory), elapsed = time_function(optimize_protein, initial_positions, n_beads, write_csv=True)
        
        # Log the runtime information
        log_runtime(runtime_logger, "OptimizeProtein", elapsed, extra=f"n_beads={n_beads}")
        print(f"Run {i+1}: Optimization completed in {elapsed:.4f} seconds.\n")
'''