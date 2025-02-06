import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from energy_wrapper import optimize_protein_c, compute_total_energy

def optimize_protein(positions, n_beads, write_csv=False, maxiter=1000, tol=1e-6):
    """
    Optimize the positions of the protein using the C implementation of BFGS.
    """
    trajectory = []  # Empty for now, unless modified in C
    optimized_positions = optimize_protein_c(positions, n_beads, maxiter, tol)

    if write_csv:
        np.savetxt(f'protein{n_beads}.csv', optimized_positions, delimiter=",")

    return optimized_positions, trajectory

# Initialize protein positions
def initialize_protein(n_beads, dimension=3, fudge=1e-5):
    """
    Initialize a protein with `n_beads` arranged almost linearly in `dimension`-dimensional space.
    The `fudge` is a factor that, if non-zero, adds a spiral structure to the configuration.
    """
    positions = np.zeros((n_beads, dimension))
    for i in range(1, n_beads):
        positions[i, 0] = positions[i-1, 0] + 1  # Fixed bond length of 1 unit
        positions[i, 1] = fudge * np.sin(i)       # Small perturbation in y
        positions[i, 2] = fudge * np.sin(i*i)       # Small perturbation in z               
    return positions

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

# Main function
if __name__ == "__main__":
    # Record the start time.
    

    n_beads = 10
    dimension = 3
    maxiter = 1000
    tol = 1e-6

    # Initialize positions
    initial_positions = initialize_protein(n_beads, dimension)
    
    # Compute and print initial energy
    initial_energy = compute_total_energy(initial_positions.flatten())
    print(f"Initial Energy: {initial_energy}")

    # Plot initial configuration
    plot_protein_3d(initial_positions, title="Initial Configuration")

    # Optimize
    start_time = time.time()
    optimized_positions, trajectory = optimize_protein(initial_positions, n_beads, write_csv=True, maxiter=maxiter, tol=tol)

    # Compute and print optimized energy
    optimized_energy = compute_total_energy(optimized_positions.flatten())
    end_time = time.time()
    print(f"Optimized Energy: {optimized_energy}")

    # Plot optimized configuration
    plot_protein_3d(optimized_positions, title="Optimized Configuration")

    # Animate optimization (only if trajectory tracking is implemented)
    if trajectory:
        animate_optimization(trajectory)

    # Record the end time and compute elapsed time.
    
    elapsed = end_time - start_time
    print(f"Elapsed time: {elapsed:.4f} seconds")
