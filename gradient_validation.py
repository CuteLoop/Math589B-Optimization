from functools import partial
import numpy as np
import jax
import jax.numpy as jnp



def jax_bond_energy(i, positions, n_beads, b, k_b):
    def compute_bond():
        r = jnp.linalg.norm(positions[i + 1] - positions[i])
        return k_b * (r - b) ** 2
    # Only compute bond energy if i is not the last bead.
    return jax.lax.cond(i < n_beads - 1, compute_bond, lambda: 0.0)

def jax_lennard_jones_energy(i, positions, n_beads, epsilon, sigma):
    """Compute the Lennard-Jones energy contribution for bead i.

    Instead of using a dynamic range starting at i+1, we generate a full range
    of indices and then mask out indices ≤ i.
    """
    def lj_potential(j):
        r = jnp.linalg.norm(positions[j] - positions[i])
        sr6 = (sigma / r) ** 6
        return 4 * epsilon * (sr6 ** 2 - sr6)
    
    # Use a full, static range and then mask indices.
    all_indices = jnp.arange(n_beads)  # n_beads is static.
    mask = all_indices > i
    energies = jax.vmap(lj_potential)(all_indices)
    energies = jnp.where(mask, energies, 0.0)
    return jnp.sum(energies)

def jax_total_energy(positions, n_beads, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    """Compute the total energy (bond + Lennard-Jones) of the system."""
    # Reshape positions to (n_beads, 3); n_beads is static.
    positions = jnp.reshape(positions, (n_beads, 3))
    bond_energy = jnp.sum(jax.vmap(lambda i: jax_bond_energy(i, positions, n_beads, b, k_b))
                          (jnp.arange(n_beads - 1)))
    lj_energy = jnp.sum(jax.vmap(lambda i: jax_lennard_jones_energy(i, positions, n_beads, epsilon, sigma))
                        (jnp.arange(n_beads)))
    return bond_energy + lj_energy

def jax_gradient(x, n_beads, epsilon, sigma, b, k_b):
    def energy_fn(pos):
        return jax_total_energy(pos, n_beads, epsilon, sigma, b, k_b)
    return jax.grad(energy_fn)(x)

# Mark n_beads as static so that it is known at compile time.
jax_gradient = jax.jit(jax_gradient, static_argnums=(1,))

if __name__ == "__main__":
    np.random.seed(42)
    n_beads = 10
    positions = np.random.rand(n_beads * 3)
    
    # Import your analytical energy and gradient functions for comparison.
    from energy import total_energy, compute_gradient, numerical_gradient

    grad_analytical = compute_gradient(positions, n_beads)
    grad_numerical = numerical_gradient(total_energy, positions, n_beads, 1.0, 1.0, 1.0, 100.0)
    grad_jax = np.array(jax_gradient(positions, n_beads, 1.0, 1.0, 1.0, 100.0))



    error_num = np.linalg.norm(grad_analytical - grad_numerical)
    error_jax = np.linalg.norm(grad_analytical - grad_jax)
    print(f"Gradient (Analytical): {grad_analytical}")
    print(f"Gradient (Numerical): {grad_numerical}")
    print(f"Gradient (JAX): {grad_jax}")
