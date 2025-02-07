import time
import numpy as np
from energy_wrapper import optimize_protein_c, compute_total_energy

class OptimizeResult:
    """Simple result object so the autograder can check result.x."""
    def __init__(self, x):
        # x should be shape (n_beads,3)
        self.x = x

def initialize_protein(n_beads, dimension=3, fudge=1e-5):
    """(Do not modify per assignment instructions)"""
    positions = np.zeros((n_beads, dimension))
    for i in range(1, n_beads):
        positions[i, 0] = positions[i - 1, 0] + 1.0
        if dimension > 1:
            positions[i, 1] = fudge * np.sin(i)
        if dimension > 2:
            positions[i, 2] = fudge * np.sin(i * i)
    return positions

def optimize_protein(positions, n_beads, write_csv=False, maxiter=1000, tol=1e-6):
    """
    Main function called by the autograder. 
    returns result, trajectory
    where result.x is shape (n_beads, 3)
    """
    # 1) Flatten? Not needed. We do it in the wrapper. But let's keep shape consistent:
    positions_2d = positions.reshape(n_beads, 3)

    # 2) Call the C function through our wrapper
    optimized_positions = optimize_protein_c(positions_2d, n_beads, maxiter, tol)

    # 3) Optionally save to CSV
    if write_csv:
        np.savetxt(f"protein{n_beads}.csv", optimized_positions, delimiter=",")

    # 4) Build a simple trajectory list (empty for now)
    trajectory = []

    # 5) Return a result object with .x for the autograder
    result = OptimizeResult(optimized_positions)
    return result, trajectory

# -----------------------------------------
# Optional: local testing code
# -----------------------------------------
if __name__ == "__main__":

    n_beads = 10
    dimension = 3
    maxiter = 1000
    tol = 1e-6

    # Initialize
    init_pos = initialize_protein(n_beads, dimension)
    
    # Print initial energy
    initial_energy = compute_total_energy(init_pos)
    print(f"Initial energy: {initial_energy}")

    # Run optimization
    start = time.time()

    result, traj = optimize_protein(init_pos, n_beads, write_csv=True, maxiter=maxiter, tol=tol)
    final_energy = compute_total_energy(result.x)
    print(f"Final energy: {final_energy}")

    elapsed = time.time() - start
    print(f"Elapsed time: {elapsed:.4f} seconds")
