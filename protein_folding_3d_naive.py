import time
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Import C version functions from your energy wrapper.
from energy_wrapper import optimize_protein_c, compute_total_energy as c_compute_total_energy

# ------------------------------
# Utility Class: OptimizeResult
# ------------------------------
class OptimizeResult:
    def __init__(self, x, fun=None, nit=None, success=None, message=None, elapsed_time=None):
        self.x = x
        self.fun = fun
        self.nit = nit
        self.success = success
        self.message = message
        self.elapsed_time = elapsed_time

# ------------------------------
# Naïve Python Implementation
# ------------------------------
def initialize_protein(n_beads, dimension=3, fudge=1e-5):
    """
    Initialize a protein with `n_beads` arranged almost linearly in `dimension`-dimensional space.
    The `fudge` parameter adds a spiral structure to the configuration.
    """
    positions = np.zeros((n_beads, dimension))
    for i in range(1, n_beads):
        positions[i, 0] = positions[i - 1, 0] + 1  # Fixed bond length of 1 unit
        positions[i, 1] = fudge * np.sin(i)          # Small perturbation in y
        positions[i, 2] = fudge * np.sin(i * i)        # Small perturbation in z
    return positions

def lennard_jones_potential(r, epsilon=1.0, sigma=1.0):
    """Compute Lennard-Jones potential between two beads."""
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

def bond_potential(r, b=1.0, k_b=100.0):
    """Compute harmonic bond potential between two bonded beads."""
    return k_b * (r - b)**2

def total_energy(positions, n_beads, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    """
    Compute the total energy of the protein conformation using the Python implementation.
    """
    positions = positions.reshape((n_beads, -1))
    energy = 0.0

    # Bond energy: over consecutive beads.
    for i in range(n_beads - 1):
        r = np.linalg.norm(positions[i + 1] - positions[i])
        energy += bond_potential(r, b, k_b)

    # Lennard-Jones potential for non-bonded interactions.
    for i in range(n_beads):
        for j in range(i + 1, n_beads):
            r = np.linalg.norm(positions[i] - positions[j])
            if r > 1e-2:  # Avoid division by zero.
                energy += lennard_jones_potential(r, epsilon, sigma)

    return energy

def optimize_protein_naive(positions, n_beads, write_csv=False, maxiter=1000, tol=1e-6):
    """
    Optimize the positions of the protein using SciPy's BFGS.
    
    Returns:
      naive_result: An OptimizeResult object (with attribute x for optimized positions,
                    and elapsed_time for the run time).
      trajectory: A list of intermediate configurations (each is an (n_beads, d) array).
    """
    trajectory = []

    def callback(x):
        trajectory.append(x.reshape((n_beads, -1)))
        if len(trajectory) % 20 == 0:
            print(f"Naïve optimizer - trajectory steps: {len(trajectory)}")

    t0 = time.time()
    result = minimize(
        fun=total_energy,
        x0=positions.flatten(),
        args=(n_beads,),
        method='BFGS',
        callback=callback,
        tol=tol,
        options={'maxiter': maxiter, 'disp': True}
    )
    t1 = time.time()
    elapsed = t1 - t0

    if write_csv and trajectory:
        csv_filepath = f'protein{n_beads}_naive.csv'
        print(f"Naïve optimizer - Writing data to file {csv_filepath}")
        np.savetxt(csv_filepath, trajectory[-1], delimiter=",")

    naive_result = OptimizeResult(
        x=result.x,
        fun=result.fun,
        nit=result.nit,
        success=result.success,
        message=result.message,
        elapsed_time=elapsed
    )
    return naive_result, trajectory

# ------------------------------
# Visualization Functions
# ------------------------------
def plot_protein_3d(positions, title="Protein Conformation", ax=None):
    """Plot the 3D positions of the protein."""
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

def animate_optimization(trajectory, interval=100):
    """Animate the protein folding process in 3D with autoscaling."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    line, = ax.plot([], [], [], '-o', markersize=6)

    def update(frame):
        positions = trajectory[frame]
        line.set_data(positions[:, 0], positions[:, 1])
        line.set_3d_properties(positions[:, 2])
        # Autoscale the axes.
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

# ------------------------------
# Main Block: Compare Naïve and C Versions
# ------------------------------
if __name__ == "__main__":
    n_beads = 10
    dimension = 3
    maxiter = 1000
    tol = 1e-6

    # Initialize protein configuration.
    initial_positions = initialize_protein(n_beads, dimension)

    # Compute and display initial energy (Python version).
    initial_energy = total_energy(initial_positions.flatten(), n_beads)
    print(f"Initial Energy (Python): {initial_energy}")

    # Plot the initial configuration.
    plot_protein_3d(initial_positions, title="Initial Configuration")

    # ------------------------------
    # Naïve Optimization using SciPy
    # ------------------------------
    print("Running naïve (SciPy) optimization...")
    naive_result, naive_traj = optimize_protein_naive(initial_positions, n_beads, write_csv=True, maxiter=maxiter, tol=tol)
    naive_optimized_positions = naive_result.x.reshape((n_beads, dimension))
    naive_energy = total_energy(naive_optimized_positions.flatten(), n_beads)
    print(f"Naïve Optimized Energy: {naive_energy}")
    print(f"Naïve optimizer elapsed time: {naive_result.elapsed_time:.4f} seconds")
    plot_protein_3d(naive_optimized_positions, title="Naïve Optimized Configuration")
    if naive_traj:
        animate_optimization(naive_traj)

    # ------------------------------
    # C-Based Optimization using our Wrapper
    # ------------------------------
    print("Running C-based optimization...")
    t0 = time.time()
    c_optimized_positions = optimize_protein_c(initial_positions, n_beads, maxiter=maxiter, tol=tol)
    t1 = time.time()
    c_elapsed = t1 - t0
    # The C function returns a numpy array (which we wrap if needed); assume it has the same shape.
    c_energy = c_compute_total_energy(c_optimized_positions.flatten())
    print(f"C-based Optimized Energy: {c_energy}")
    print(f"C-based optimizer elapsed time: {c_elapsed:.4f} seconds")
    plot_protein_3d(c_optimized_positions, title="C-Based Optimized Configuration")
    # (Note: If trajectory tracking is added in the C code, you could also animate it here.)

    # ------------------------------
    # Final Comparison Summary
    # ------------------------------
    print("\nComparison Summary:")
    print(f"Naïve Energy: {naive_energy}, Time: {naive_result.elapsed_time:.4f} seconds")
    print(f"C-based Energy: {c_energy}, Time: {c_elapsed:.4f} seconds")
