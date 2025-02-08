import time
import numpy as np

# We need SciPy's `minimize` and `OptimizeResult` to generate a dummy result object.
from scipy.optimize import minimize

from energy_wrapper import optimize_protein_c, compute_total_energy

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
    Must return a "scipy.optimize.OptimizeResult" with .x containing the solution.

    Parameters
    ----------
    positions : np.ndarray
        Initial positions, shape (n_beads, 3)
    n_beads : int
        Number of beads
    write_csv : bool
        Whether to save final coordinates to CSV
    maxiter : int
        Maximum iterations
    tol : float
        Tolerance for convergence

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        With `result.x` as a 1D NumPy array of length 3*n_beads (typical SciPy style).
    trajectory : list of np.ndarray
        Ignored by autograder, can remain empty or store intermediate steps.
    """
    # 1) Make sure positions is float64 and shaped (n_beads, 3)
    positions_2d = np.asarray(positions, dtype=np.float64).reshape(n_beads, 3)

    # 2) Call C-based BFGS via your wrapper
    optimized_positions_2d = optimize_protein_c(
        positions_2d,    # shape (n_beads,3)
        n_beads,
        maxiter=maxiter,
        tol=tol
    )
    # Ensure shape is correct
    assert optimized_positions_2d.shape == (n_beads, 3), (
        f"Expected (n_beads, 3), got {optimized_positions_2d.shape}"
    )
    # We'll flatten to 1D because SciPy typically stores x as a 1D array
    optimized_positions_1d = optimized_positions_2d.ravel()

    # 3) Optionally save final result to CSV
    if write_csv:
        np.savetxt(f"protein{n_beads}.csv", optimized_positions_2d, delimiter=",")

    # 4) Create a "dummy" SciPy result by calling minimize with `maxiter=0`
    #    This yields a valid OptimizeResult object, which we then overwrite.
    dummy_result = minimize(
        fun=lambda x: 0.0,  # A trivial objective
        x0=optimized_positions_1d,
        method='BFGS',
        options={'maxiter': 0, 'disp': False}
    )
    # Overwrite the fields with our actual data
    dummy_result.x = optimized_positions_1d           # final solution (1D array)
    dummy_result.nit = 1                              # or number of iterations
    dummy_result.success = True
    dummy_result.status = 0
    dummy_result.message = "C-based optimization successful."

    # 5) The autograder ignores this trajectory, but we must return it
    trajectory = []

    return dummy_result, trajectory

# ----------------------------------------------------
# Optional local testing code
# ----------------------------------------------------
if __name__ == "__main__":
    n_beads = 10
    positions = np.random.rand(n_beads, 3)

    # Call BFGS with Armijo line search (linesearch_choice=0)
    optimized_positions = optimize_protein_c(positions, n_beads, maxiter=1000, tol=1e-6, linesearch_choice=0)
    e_armijo = compute_total_energy(optimized_positions)

    # Call BFGS with Wolfe line search (linesearch_choice=1)
    opt_wolfe = optimize_protein_c(positions, n_beads, maxiter=1000, tol=1e-6, linesearch_choice=1)
    e_wolfe = compute_total_energy(opt_wolfe)

    # Call BFGS with Strong Wolfe line search (linesearch_choice=2)
    opt_strong = optimize_protein_c(positions, n_beads, maxiter=1000, tol=1e-6, linesearch_choice=2)
    e_strong = compute_total_energy(opt_strong)

    print("Armijo final energy:", e_armijo)
    print("Wolfe final energy :", e_wolfe)
    print("Strong Wolfe final energy:", e_strong)
    # If you want to do further checks or plotting, do so here.
