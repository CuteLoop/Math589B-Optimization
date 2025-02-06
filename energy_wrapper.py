import ctypes
import numpy as np

# Load the shared library (adjust the filename for your platform)
lib = ctypes.CDLL('./libbfgs.dll')  # Use 'libbfgs.so' on Linux

# Define the prototype for bfgs_optimize from the library
lib.bfgs_optimize.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # positions (pointer to double)
    ctypes.c_int,                     # number of beads
    ctypes.c_int,                     # maxiter
    ctypes.c_double,                  # tol
    ctypes.c_double, ctypes.c_double, # epsilon, sigma
    ctypes.c_double, ctypes.c_double  # b, k_b
]
lib.bfgs_optimize.restype = None  # This function returns void

def optimize_protein_c(positions, n_beads, maxiter=1000, tol=1e-6,
                        epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    """
    Call the C implementation of BFGS to optimize the protein positions.
    """
    positions = positions.flatten().astype(np.float64)  # Ensure a contiguous double array
    c_positions = positions.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    lib.bfgs_optimize(c_positions, n_beads, maxiter, tol, epsilon, sigma, b, k_b)

    return positions.reshape((n_beads, 3))  # Reshape back to (n_beads, 3)

# Define the prototype for total_energy from the library
lib.total_energy.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # positions (pointer to double)
    ctypes.c_int,                     # n_beads
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double  # epsilon, sigma, b, k_b
]
lib.total_energy.restype = ctypes.c_double

def compute_total_energy(positions, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    """
    Compute total energy by calling the C function total_energy.
    """
    # Determine the number of beads assuming each bead has 3 coordinates.
    n_beads = len(positions) // 3
    positions_array = np.array(positions, dtype=np.float64)
    c_positions = positions_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    return lib.total_energy(c_positions, n_beads, epsilon, sigma, b, k_b)
