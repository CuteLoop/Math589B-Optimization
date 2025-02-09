import ctypes
import numpy as np
import platform

# ----------------------------------------------------------------------
# STEP 1: Detect the platform and load the correct shared library name.
#         (Windows often uses .dll, Linux uses .so.)
# ----------------------------------------------------------------------
system_name = platform.system().lower()
if 'windows' in system_name:
    lib_filename = './libbfgs.dll'
else:
    lib_filename = './libbfgs.so'

try:
    lib = ctypes.CDLL(lib_filename)
except OSError:
    raise OSError(f"Could not load the shared library: {lib_filename}")

# ----------------------------------------------------------------------
# STEP 2: Define function prototypes
# ----------------------------------------------------------------------
#
# We assume your updated C function signature is now:
#   void bfgs_optimize(double *x, int n_beads, int maxiter, double tol,
#                      double epsilon, double sigma, double b, double k_b,
#                      int linesearch_choice);
#

lib.bfgs_optimize.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # positions
    ctypes.c_int,                     # n_beads
    ctypes.c_int,                     # maxiter
    ctypes.c_double,                  # tol
    ctypes.c_double, ctypes.c_double, # epsilon, sigma
    ctypes.c_double, ctypes.c_double, # b, k_b
    ctypes.c_int                      # linesearch_choice
]
lib.bfgs_optimize.restype = None

def optimize_protein_c(positions,
                       n_beads,
                       maxiter=1000,
                       tol=1e-6,
                       epsilon=1.0,
                       sigma=1.0,
                       b=1.0,
                       k_b=100.0,
                       # linesearch_choice is now determined automatically
                       ):
    """
    Call the (updated) C implementation of BFGS to optimize the protein positions.
    
    The function chooses the type of line search based on the problem size and tolerance:
      - If n_beads is high (>=50) and tol is small (<1e-5), Wolfe line search is used.
      - Otherwise, Armijo line search is used.
    
    Parameters
    ----------
    positions : array-like
        Initial coordinates of shape (n_beads, 3) or flattened (3*n_beads,).
    n_beads : int
        Number of beads.
    maxiter : int, optional
        Maximum BFGS iterations (default=1000).
    tol : float, optional
        Tolerance for gradient norm (default=1e-6).
    epsilon, sigma, b, k_b : floats, optional
        Energy parameters (default=1.0, 1.0, 1.0, 100.0).
    
    Returns
    -------
    optimized_positions : np.ndarray
        A 2D NumPy array of shape (n_beads, 3) with the optimized coordinates.
    """

    

    # Determine line search type based on the number of beads and tolerance.
    # For high particle count (>=50) and small tolerance (<1e-5), use Wolfe (choice=1).
    # Otherwise, use Armijo (choice=0) for speed.
    if n_beads >= 50 and tol < 1e-5:
        chosen_ls = 1  # Wolfe line search
    else:
        chosen_ls = 0  # Armijo line search

    # Print out which line search is chosen (for debugging/confirmation)
    print(f"Using line search type {chosen_ls} for n_beads={n_beads} and tol={tol}")

    # 1) Flatten if needed, ensure double precision
    positions_flat = np.asarray(positions, dtype=np.float64).ravel()

    # 2) Create a C pointer
    c_positions = positions_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # 3) Call the updated C function with the chosen line search type
    lib.bfgs_optimize(
        c_positions,
        n_beads,
        maxiter,
        tol,
        epsilon,
        sigma,
        b,
        k_b,
        chosen_ls
    )

    # 4) The positions array was modified in place, so reshape to (n_beads, 3)
    return positions_flat.reshape((n_beads, 3))

# ----------------------------------------------------------------------
# total_energy function remains the same
# ----------------------------------------------------------------------
lib.total_energy.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double
]
lib.total_energy.restype = ctypes.c_double

def compute_total_energy(positions, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    """
    Compute total energy by calling the C function total_energy.

    Parameters
    ----------
    positions : array-like
        Coordinates, can be shape (n_beads,3) or flattened.
    epsilon, sigma, b, k_b : floats, optional
        Energy parameters.

    Returns
    -------
    energy : float
        The total potential energy.
    """
    positions_flat = np.asarray(positions, dtype=np.float64).ravel()
    n_beads = positions_flat.size // 3

    c_positions = positions_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    return lib.total_energy(c_positions, n_beads, epsilon, sigma, b, k_b)
