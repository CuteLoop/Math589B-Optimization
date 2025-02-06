# full_optimization.pyx
# cython: language_level=3

"""
This module provides Cython-accelerated functions for a protein energy model.
It includes:
  - Particle initialization (initialize_protein)
  - Lennard-Jones potential (lennard_jones_potential)
  - Bond potential (bond_potential)
  - Total energy computation (total_energy)
  - Analytical gradient of the total energy (total_energy_grad)
  - A BFGS optimizer (bfgs)

These functions are intended to be imported and called from Python.
"""

###############################################
# Imports and Type Definitions
###############################################
import numpy as np
cimport numpy as np            # For C-level access to NumPy arrays
from libc.math cimport sqrt, sin, pow  # C math functions for speed

ctypedef np.float64_t DTYPE_t  # Alias for double-precision floats

###############################################
# Particle Initialization
###############################################
cpdef np.ndarray initialize_protein(int n_beads, int dimension=3, double fudge=1e-5):
    """
    Initialize a protein with `n_beads` arranged nearly linearly in `dimension`-dimensional space.
    A small fudge factor adds a slight spiral structure.
    
    Returns:
        A 2D NumPy array of shape (n_beads, dimension) with dtype=float64.
    """
    cdef int i
    cdef np.ndarray[DTYPE_t, ndim=2] positions = np.zeros((n_beads, dimension), dtype=np.float64)
    for i in range(1, n_beads):
        positions[i, 0] = positions[i-1, 0] + 1   # Beads are one unit apart in x.
        positions[i, 1] = fudge * sin(i)            # Sinusoidal perturbation in y.
        positions[i, 2] = fudge * sin(i * i)          # Perturbation in z.
    return positions

###############################################
# Potential Functions
###############################################
cpdef double lennard_jones_potential(double r, double epsilon=1.0, double sigma=1.0):
    """
    Compute the Lennard-Jones potential between two beads.
    
    Returns:
        The Lennard-Jones energy (double). For very small r, returns a large number.
    """
    if r < 1e-12:
        return 1e12  # Avoid division by zero.
    cdef double sr = sigma / r
    cdef double sr6 = pow(sr, 6)
    return 4 * epsilon * (pow(sr6, 2) - sr6)

cpdef double bond_potential(double r, double b=1.0, double k_b=100.0):
    """
    Compute the harmonic bond potential between two adjacent beads.
    
    Returns:
        The bond energy (double).
    """
    return k_b * pow(r - b, 2)

###############################################
# Total Energy Function
###############################################
cpdef double total_energy(np.ndarray[DTYPE_t, ndim=2] positions, int n_beads,
                          double epsilon=1.0, double sigma=1.0,
                          double b=1.0, double k_b=100.0):
    """
    Compute the total energy of a protein conformation.
    
    The energy includes:
      - Bond energy for consecutive beads.
      - Lennard-Jones energy for each unique pair of beads.
    
    Parameters:
        positions: A 2D NumPy array of shape (n_beads, dimension).
    
    Returns:
        Total energy as a double.
    """
    cdef int i, j, dim = positions.shape[1]
    cdef double energy = 0.0, r, dx, dy, dz
    # Bond energy: iterate over consecutive bead pairs.
    for i in range(n_beads - 1):
        dx = positions[i+1, 0] - positions[i, 0]
        dy = positions[i+1, 1] - positions[i, 1]
        dz = positions[i+1, 2] - positions[i, 2]
        r = sqrt(dx*dx + dy*dy + dz*dz)
        energy += bond_potential(r, b, k_b)
    # Lennard-Jones energy: iterate over all unique bead pairs.
    for i in range(n_beads):
        for j in range(i+1, n_beads):
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]
            r = sqrt(dx*dx + dy*dy + dz*dz)
            if r > 1e-2:  # Avoid singularity.
                energy += lennard_jones_potential(r, epsilon, sigma)
    return energy

###############################################
# Analytical Gradient of Total Energy
###############################################
cpdef np.ndarray total_energy_grad(np.ndarray[DTYPE_t, ndim=2] positions, int n_beads,
                                     double epsilon=1.0, double sigma=1.0,
                                     double b=1.0, double k_b=100.0):
    """
    Compute the analytical gradient of the total energy with respect to the bead coordinates.
    
    Returns:
        A 2D NumPy array of shape (n_beads, 3) containing the gradient.
    """
    cdef int i, j, dim = positions.shape[1]
    cdef double dx, dy, dz, r, factor
    cdef np.ndarray[DTYPE_t, ndim=2] grad = np.zeros((n_beads, dim), dtype=np.float64)
    
    # Bond energy gradient.
    for i in range(n_beads - 1):
        dx = positions[i+1, 0] - positions[i, 0]
        dy = positions[i+1, 1] - positions[i, 1]
        dz = positions[i+1, 2] - positions[i, 2]
        r = sqrt(dx*dx + dy*dy + dz*dz)
        if r < 1e-12:
            continue
        factor = -2.0 * k_b * (r - b) / r
        grad[i, 0] += factor * dx
        grad[i, 1] += factor * dy
        grad[i, 2] += factor * dz
        grad[i+1, 0] -= factor * dx
        grad[i+1, 1] -= factor * dy
        grad[i+1, 2] -= factor * dz
    
    # Lennard-Jones gradient.
    for i in range(n_beads):
        for j in range(i+1, n_beads):
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]
            r = sqrt(dx*dx + dy*dy + dz*dz)
            if r < 1e-2:
                continue
            factor = 4 * epsilon * (-12 * pow(sigma, 12) / pow(r, 13) + 6 * pow(sigma, 6) / pow(r, 7))
            grad[i, 0] += factor * (-dx / r)
            grad[i, 1] += factor * (-dy / r)
            grad[i, 2] += factor * (-dz / r)
            grad[j, 0] += factor * (dx / r)
            grad[j, 1] += factor * (dy / r)
            grad[j, 2] += factor * (dz / r)
    
    return grad

###############################################
# BFGS Optimization Function
###############################################
cpdef tuple bfgs(np.ndarray[DTYPE_t, ndim=1] x0, object func, object grad_func, 
                 double tol=1e-6, int max_iter=1000):
    """
    Minimize a function using the BFGS algorithm.
    
    Parameters:
        x0: Initial guess (1D NumPy array, float64).
        func: Objective function (callable).
        grad_func: Function to compute the gradient.
        tol: Tolerance for the gradient norm.
        max_iter: Maximum number of iterations.
    
    Returns:
        A tuple (x, f_val) where:
          x: The estimated minimum (1D array).
          f_val: The function value at x.
    """
    cdef int n = x0.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] x = x0.copy()
    cdef np.ndarray[DTYPE_t, ndim=2] I = np.eye(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] H = I.copy()
    cdef np.ndarray[DTYPE_t, ndim=1] g = grad_func(x)
    cdef int k
    cdef double norm_g, alpha, c_const = 1e-4, rho = 0.9, ys, temp
    cdef np.ndarray[DTYPE_t, ndim=1] p = np.empty(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] s = np.empty(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] y = np.empty(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] x_new, g_new

    for k in range(max_iter):
        norm_g = np.linalg.norm(g)
        if norm_g < tol:
            print(f"Converged in {k} iterations.")
            break

        p = -np.dot(H, g)
        alpha = 1.0
        while func(x + alpha * p) > func(x) + c_const * alpha * np.dot(g, p):
            alpha *= rho

        x_new = x + alpha * p
        g_new = grad_func(x_new)
        s = x_new - x
        y = g_new - g
        ys = np.dot(y, s)

        if ys > 1e-10:
            temp = 1.0 / ys
            H = (np.eye(n) - temp * np.outer(s, y)) @ H @ (np.eye(n) - temp * np.outer(y, s)) \
                + temp * np.outer(s, s)
        else:
            H = np.eye(n, dtype=np.float64)

        x = x_new
        g = g_new
    else:
        print(f"Maximum iterations ({max_iter}) reached.")
    return x, func(x)

###############################################
# (The Energy Optimization wrapper has been removed.)
###############################################
