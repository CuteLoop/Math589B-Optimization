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
# STEP 2: Define the function prototypes for bfgs_optimize and total_energy
# ----------------------------------------------------------------------

# bfgs_optimize(double *x, int n_beads, int maxiter, double tol,
#               double epsilon, double sigma, double b, double k_b)
lib.bfgs_optimize.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # positions
    ctypes.c_int,                     # n_beads
    ctypes.c_int,                     # maxiter
    ctypes.c_double,                  # tol
    ctypes.c_double, ctypes.c_double, # epsilon, sigma
    ctypes.c_double, ctypes.c_double  # b, k_b
]
lib.bfgs_optimize.restype = None

def optimize_protein_c(positions, n_beads, maxiter=1000, tol=1e-6,
                       epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    """
    Call the C implementation of BFGS to optimize the protein positions.
    
    positions: 1D or 2D array of shape (n_beads * 3,)
    n_beads: number of beads
    returns: 2D array (n_beads, 3) of optimized positions
    """
    # 1) Flatten if needed, ensure double precision
    positions_flat = np.asarray(positions, dtype=np.float64).ravel()
    
    # 2) Create a C pointer to the data
    c_positions = positions_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # 3) Call the C function
    lib.bfgs_optimize(c_positions, n_beads, maxiter, tol, epsilon, sigma, b, k_b)

    # 4) Reshape the array back to (n_beads,3)
    #    Note: positions_flat is modified in place.
    return positions_flat.reshape((n_beads, 3))

# total_energy(double *positions, int n_beads,
#              double epsilon, double sigma, double b, double k_b)
lib.total_energy.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double
]
lib.total_energy.restype = ctypes.c_double

def compute_total_energy(positions, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    """
    Compute total energy by calling the C function total_energy.
    
    positions: 1D or 2D array. If 2D, shape (n_beads,3).
    returns: double - the total energy
    """
    # 1) Flatten and convert to double
    positions_flat = np.asarray(positions, dtype=np.float64).ravel()

    # 2) Determine n_beads from length
    n_beads = positions_flat.size // 3

    # 3) Create a pointer to pass to the C function
    c_positions = positions_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    # 4) Call total_energy
    return lib.total_energy(c_positions, n_beads, epsilon, sigma, b, k_b)
