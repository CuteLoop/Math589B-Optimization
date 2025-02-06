import numpy as np

def lennard_jones_potential(r, epsilon=1.0, sigma=1.0):
    """
    Compute the Lennard-Jones potential energy for a given interparticle distance r.
    
    Parameters:
        r (float): Distance between two particles.
        epsilon (float): Depth of the potential well.
        sigma (float): Finite distance at which the potential is zero.
        
    Returns:
        float: Lennard-Jones potential energy.
    """
    if r < 1e-12:
        return 1e12  
    sr6 = (sigma / r) ** 6
    return 4 * epsilon * (sr6**2 - sr6)

def bond_potential(r, b=1.0, k_b=100.0):
    """
    Compute the harmonic bond potential energy for a bond.
    
    Parameters:
        r (float): Current bond length.
        b (float): Equilibrium bond length.
        k_b (float): Bond stiffness (force constant).
    
    Returns:
        float: Bond potential energy.
    """
    return k_b * (r - b) ** 2

def total_energy(positions, n_beads, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    """
    Compute the total potential energy for a system of beads.
    
    Parameters:
        positions (np.ndarray): Flattened array of shape (n_beads*3,) representing bead positions.
        n_beads (int): Number of beads/particles.
        epsilon (float): Lennard-Jones parameter.
        sigma (float): Lennard-Jones parameter.
        b (float): Equilibrium bond length.
        k_b (float): Bond stiffness.
        
    Returns:
        float: Total potential energy.
    """
    energy = 0.0
    positions = positions.reshape(n_beads, 3)
    
    # Bond potential energy (adjacent beads).
    for i in range(n_beads - 1):
        r = np.linalg.norm(positions[i+1] - positions[i])
        energy += bond_potential(r, b, k_b)
    
    # Lennard-Jones potential energy (all unique pairs).
    for i in range(n_beads):
        for j in range(i + 1, n_beads):
            r = np.linalg.norm(positions[j] - positions[i])
            energy += lennard_jones_potential(r, epsilon, sigma)
    
    return energy

def compute_gradient(positions, n_beads, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    """
    Compute the analytical gradient (dE/dx) of the total potential energy.
    
    For the bond potential:
        V_bond = k_b * (r - b)^2 with r = ||x_(i+1) - x_i||
        dV/dr = 2*k_b*(r - b) and dr/dx_i = -(x_{i+1} - x_i)/r,
        so that dE/dx_i = -2*k_b*(r - b)*(x_{i+1}-x_i)/r,
        and dE/dx_(i+1) = 2*k_b*(r - b)*(x_{i+1}-x_i)/r.
    
    For the Lennard-Jones potential:
        V_LJ = 4*epsilon*((sigma/r)^12 - (sigma/r)^6),
        dV/dr = -48*epsilon*sigma^12/r^13 + 24*epsilon*sigma^6/r^7,
        with dr/dx_i = -(x_j - x_i)/r, yielding
        dE/dx_i = 48*epsilon*sigma^12*(x_j-x_i)/r^14 - 24*epsilon*sigma^6*(x_j-x_i)/r^8.
    
    Parameters:
        positions (np.ndarray): Flattened array of shape (n_beads*3,).
        n_beads (int): Number of beads.
        epsilon, sigma: Lennard-Jones parameters.
        b, k_b: Bond potential parameters.
        
    Returns:
        np.ndarray: Flattened gradient array (dE/dx) of shape (n_beads*3,).
    """
    positions = positions.reshape(n_beads, 3)
    gradient = np.zeros_like(positions)
    
    # Bond potential gradient.
    for i in range(n_beads - 1):
        r_vec = positions[i+1] - positions[i]  # vector from bead i to bead i+1.
        r = np.linalg.norm(r_vec)
        if r > 1e-12:
            deriv = 2 * k_b * (r - b)  # dV/dr.
            grad_contrib = deriv * (r_vec / r)  # chain rule.
            # dE/dx_i = -grad_contrib, dE/dx_(i+1) = +grad_contrib.
            gradient[i]   -= grad_contrib
            gradient[i+1] += grad_contrib
    
    # Lennard-Jones potential gradient.
    for i in range(n_beads):
        for j in range(i + 1, n_beads):
            r_vec = positions[j] - positions[i]
            r = np.linalg.norm(r_vec)
            if r > 1e-12:
                deriv = -48 * epsilon * (sigma**12) / (r**13) + 24 * epsilon * (sigma**6) / (r**7)
                grad_contrib = deriv * (r_vec / r)
                # Change here: for energy we assign:
                # dE/dx_i = -grad_contrib and dE/dx_j = +grad_contrib.
                gradient[i]   -= grad_contrib
                gradient[j]   += grad_contrib
    
    return gradient.flatten()

def numerical_gradient(f, x, n_beads, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0, h=1e-5):
    """
    Compute the numerical gradient using central finite differences.
    
    Approximates dE/dx as:
        (E(x + h) - E(x - h)) / (2h)
    
    Parameters:
        f (callable): Energy function.
        x (np.ndarray): Flattened positions.
        n_beads (int): Number of beads.
        epsilon, sigma, b, k_b: Energy parameters.
        h (float): Step size.
    
    Returns:
        np.ndarray: Numerical gradient as a flattened array.
    """
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_forward = x.copy()
        x_backward = x.copy()
        x_forward[i] += h
        x_backward[i] -= h
        grad[i] = (f(x_forward, n_beads, epsilon, sigma, b, k_b) -
                   f(x_backward, n_beads, epsilon, sigma, b, k_b)) / (2 * h)
    return grad
