import ctypes
import numpy as np

# Load the shared library
lib = ctypes.CDLL('./libbfgs.so')

# Define function prototype
lib.bfgs_optimize.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # Initial positions
    ctypes.c_int,                     # Number of beads
    ctypes.c_int,                     # maxiter
    ctypes.c_double,                   # tol
    ctypes.c_double, ctypes.c_double,  # epsilon, sigma
    ctypes.c_double, ctypes.c_double   # b, k_b
]
lib.bfgs_optimize.restype = None  # No return value

def optimize_protein_c(positions, n_beads, maxiter=1000, tol=1e-6,
                        epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    """
    Call the C implementation of BFGS to optimize the protein positions.
    """
    positions = positions.flatten().astype(np.float64)  # Ensure contiguous double array
    c_positions = positions.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    lib.bfgs_optimize(c_positions, n_beads, maxiter, tol, epsilon, sigma, b, k_b)

    return positions.reshape((n_beads, 3))  # Reshape back to original format
