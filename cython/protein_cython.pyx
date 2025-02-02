# protein_cython.pyx

# Import Python and C-level NumPy API
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, sin, pow

# Define a C-level type for double precision
ctypedef np.float64_t DTYPE_t

#######################################
# Cython implementation of initialize_protein
#######################################
cpdef np.ndarray initialize_protein(int n_beads, int dimension=3, double fudge=1e-5):
    """
    Initialize a protein with `n_beads` arranged almost linearly in `dimension`-dimensional space.
    A small fudge factor adds a spiral structure.
    """
    cdef int i
    # Create a 2D numpy array of type double with shape (n_beads, dimension)
    cdef np.ndarray[DTYPE_t, ndim=2] positions = np.zeros((n_beads, dimension), dtype=np.float64)
    
    for i in range(1, n_beads):
        positions[i, 0] = positions[i-1, 0] + 1  # Fixed bond length along x
        positions[i, 1] = fudge * sin(i)           # Perturbation in y
        positions[i, 2] = fudge * sin(i * i)         # Perturbation in z
    return positions

#######################################
# Cython implementation of potential functions
#######################################
cpdef double lennard_jones_potential(double r, double epsilon=1.0, double sigma=1.0):
    """
    Compute Lennard-Jones potential between two beads.
    """
    if r < 1e-12:
        return 1e12  # Avoid division by zero
    cdef double sr = sigma / r
    cdef double sr6 = pow(sr, 6)
    return 4 * epsilon * (pow(sr6, 2) - sr6)

cpdef double bond_potential(double r, double b=1.0, double k_b=100.0):
    """
    Compute harmonic bond potential between two bonded beads.
    """
    return k_b * pow(r - b, 2)

#######################################
# Cython implementation of total_energy
#######################################
cpdef double total_energy(np.ndarray[DTYPE_t, ndim=2] positions, int n_beads,
                          double epsilon=1.0, double sigma=1.0,
                          double b=1.0, double k_b=100.0):
    """
    Compute the total energy of the protein conformation.
    Assumes positions is a 2D numpy array of shape (n_beads, dimension).
    """
    cdef int i, j, dimension = positions.shape[1]
    cdef double energy = 0.0, r, dx, dy, dz

    # Compute bond energy (for consecutive beads)
    for i in range(n_beads - 1):
        dx = positions[i+1, 0] - positions[i, 0]
        dy = positions[i+1, 1] - positions[i, 1]
        dz = positions[i+1, 2] - positions[i, 2]
        r = sqrt(dx*dx + dy*dy + dz*dz)
        energy += bond_potential(r, b, k_b)

    # Compute Lennard-Jones potential for non-bonded interactions
    for i in range(n_beads):
        for j in range(i+1, n_beads):
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]
            r = sqrt(dx*dx + dy*dy + dz*dz)
            if r > 1e-2:  # Avoid division by zero
                energy += lennard_jones_potential(r, epsilon, sigma)
    return energy
