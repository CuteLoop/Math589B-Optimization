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
                       linesearch_choice=1):
    """
    Call the (updated) C implementation of BFGS to optimize the protein positions.

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
    linesearch_choice : int, optional
        0 => Armijo line search
        1 => Wolfe line search
        2 => Strong Wolfe line search
        (default=0, i.e., Armijo)

    Returns
    -------
    optimized_positions : np.ndarray
        A 2D NumPy array of shape (n_beads, 3) with the optimized coordinates.
    """
    # 1) Flatten if needed, ensure double precision
    positions_flat = np.asarray(positions, dtype=np.float64).ravel()

    # 2) Create a C pointer
    c_positions = positions_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # 3) Call the updated C function with line search choice
    lib.bfgs_optimize(
        c_positions,
        n_beads,
        maxiter,
        tol,
        epsilon,
        sigma,
        b,
        k_b,
        linesearch_choice
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
