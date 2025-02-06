#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Constants for numerical stability */
#define EPSILON_SING 1e-12

// Compute Lennard-Jones potential energy for distance r.
double lennard_jones_potential(double r, double epsilon, double sigma) {
    if (r < EPSILON_SING) {
        return 1e12;  // Avoid singularity
    }
    double sr6 = pow(sigma / r, 6);
    return 4.0 * epsilon * (sr6 * sr6 - sr6);
}

// Compute harmonic bond potential energy for a bond of length r.
double bond_potential(double r, double b, double k_b) {
    return k_b * (r - b) * (r - b);
}

// Compute total energy given positions (flattened array of length n_beads*3).
double total_energy(const double *positions, int n_beads, 
                    double epsilon, double sigma, double b, double k_b) {
    double energy = 0.0;
    int i, j;
    // Bond energy: over consecutive beads.
    for (i = 0; i < n_beads - 1; i++) {
        double dx = positions[(i+1)*3 + 0] - positions[i*3 + 0];
        double dy = positions[(i+1)*3 + 1] - positions[i*3 + 1];
        double dz = positions[(i+1)*3 + 2] - positions[i*3 + 2];
        double r = sqrt(dx*dx + dy*dy + dz*dz);
        energy += bond_potential(r, b, k_b);
    }
    // Lennard-Jones energy: over all unique pairs.
    for (i = 0; i < n_beads; i++) {
        for (j = i+1; j < n_beads; j++) {
            double dx = positions[j*3 + 0] - positions[i*3 + 0];
            double dy = positions[j*3 + 1] - positions[i*3 + 1];
            double dz = positions[j*3 + 2] - positions[i*3 + 2];
            double r = sqrt(dx*dx + dy*dy + dz*dz);
            energy += lennard_jones_potential(r, epsilon, sigma);
        }
    }
    return energy;
}

/*
 * Compute the analytical gradient dE/dx (for all coordinates) and write the result into gradient.
 *
 * For the bond potential:
 *   V_bond = k_b * (r - b)^2,   with r = ||x_(i+1) - x_i||
 *   dV/dr = 2 * k_b * (r - b)
 *   Chain rule gives:
 *     dE/dx_i   = -2 * k_b * (r - b) * ( (x_{i+1} - x_i) / r )
 *     dE/dx_{i+1} = +2 * k_b * (r - b) * ( (x_{i+1} - x_i) / r )
 *
 * For the Lennard-Jones potential:
 *   V_LJ = 4 * epsilon * [ (sigma/r)^12 - (sigma/r)^6 ]
 *   dV/dr = -48 * epsilon * sigma^12 / r^13 + 24 * epsilon * sigma^6 / r^7
 *   Using dr/dx_i = -(x_j - x_i)/r, we get:
 *     dE/dx_i = 48 * epsilon * sigma^12 * (x_j - x_i) / r^14 
 *                - 24 * epsilon * sigma^6 * (x_j - x_i) / r^8,
 *   and the opposite for bead j.
 *
 * The gradient vector is stored as a flattened array of length n_beads*3.
 */
void compute_gradient(const double *positions, int n_beads, 
                      double epsilon, double sigma, double b, double k_b, 
                      double *gradient) {
    int i, j;
    // Initialize gradient to zero.
    for (i = 0; i < n_beads * 3; i++) {
        gradient[i] = 0.0;
    }
    
    // Bond potential gradient.
    for (i = 0; i < n_beads - 1; i++) {
        double dx = positions[(i+1)*3 + 0] - positions[i*3 + 0];
        double dy = positions[(i+1)*3 + 1] - positions[i*3 + 1];
        double dz = positions[(i+1)*3 + 2] - positions[i*3 + 2];
        double r = sqrt(dx*dx + dy*dy + dz*dz);
        if (r > EPSILON_SING) {
            double deriv = 2.0 * k_b * (r - b);  // dV/dr for bond
            double gx = deriv * (dx / r);
            double gy = deriv * (dy / r);
            double gz = deriv * (dz / r);
            // For energy: bead i gets -grad_contrib, bead i+1 gets +grad_contrib.
            gradient[i*3 + 0]   -= gx;
            gradient[i*3 + 1]   -= gy;
            gradient[i*3 + 2]   -= gz;
            gradient[(i+1)*3 + 0] += gx;
            gradient[(i+1)*3 + 1] += gy;
            gradient[(i+1)*3 + 2] += gz;
        }
    }
    
    // Lennard-Jones potential gradient.
    for (i = 0; i < n_beads; i++) {
        for (j = i+1; j < n_beads; j++) {
            double dx = positions[j*3 + 0] - positions[i*3 + 0];
            double dy = positions[j*3 + 1] - positions[i*3 + 1];
            double dz = positions[j*3 + 2] - positions[i*3 + 2];
            double r = sqrt(dx*dx + dy*dy + dz*dz);
            if (r > EPSILON_SING) {
                double term1 = 48.0 * epsilon * pow(sigma, 12) / pow(r, 13);
                double term2 = 24.0 * epsilon * pow(sigma, 6) / pow(r, 7);
                double deriv = -term1 + term2;  // dV/dr for Lennard-Jones.
                double gx = deriv * (dx / r);
                double gy = deriv * (dy / r);
                double gz = deriv * (dz / r);
                // For energy: bead i gets -grad_contrib and bead j gets +grad_contrib.
                gradient[i*3 + 0]   -= gx;
                gradient[i*3 + 1]   -= gy;
                gradient[i*3 + 2]   -= gz;
                gradient[j*3 + 0]   += gx;
                gradient[j*3 + 1]   += gy;
                gradient[j*3 + 2]   += gz;
            }
        }
    }
}

/*
 * Initialize a protein with n_beads arranged almost linearly in 3D.
 * The "fudge" parameter adds a small spiral perturbation in the y- and z-directions.
 *
 * This function mimics the following Python code:
 *
 *   def initialize_protein(n_beads, dimension=3, fudge=1e-5):
 *       positions = np.zeros((n_beads, dimension))
 *       for i in range(1, n_beads):
 *           positions[i, 0] = positions[i-1, 0] + 1
 *           positions[i, 1] = fudge * np.sin(i)
 *           positions[i, 2] = fudge * np.sin(i*i)
 *       return positions
 *
 * The positions array is a flattened array of length n_beads*3.
 */
void initialize_protein(double *positions, int n_beads, int dimension, double fudge) {
    int i, d;
    // Set all positions to zero.
    for (i = 0; i < n_beads * dimension; i++) {
        positions[i] = 0.0;
    }
    // Initialize positions for i >= 1.
    for (i = 1; i < n_beads; i++) {
        // x-coordinate: previous x + 1.0
        positions[i * dimension + 0] = positions[(i - 1) * dimension + 0] + 1.0;
        // y-coordinate: fudge * sin(i) (if dimension > 1)
        if (dimension > 1) {
            positions[i * dimension + 1] = fudge * sin((double)i);
        }
        // z-coordinate: fudge * sin(i*i) (if dimension > 2)
        if (dimension > 2) {
            positions[i * dimension + 2] = fudge * sin((double)(i * i));
        }
    }
}

#ifdef TEST_ENERGY
// Main function for testing.
int main(void) {
    int n_beads = 10;
    int dimension = 3;
    int n_coords = n_beads * dimension;
    double epsilon = 1.0, sigma = 1.0, b = 1.0, k_b = 100.0;
    double fudge = 1e-5;
    
    // Allocate positions and gradient.
    double *positions = (double *)malloc(n_coords * sizeof(double));
    double *grad = (double *)malloc(n_coords * sizeof(double));
    if (!positions || !grad) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }
    
    // Initialize positions reproducibly.
    initialize_protein(positions, n_beads, dimension, fudge);
    
    double energy = total_energy(positions, n_beads, epsilon, sigma, b, k_b);
    compute_gradient(positions, n_beads, epsilon, sigma, b, k_b, grad);
    
    printf("Total Energy: %e\n", energy);
    printf("Gradient:\n");
    for (int i = 0; i < n_coords; i++) {
        printf("%e ", grad[i]);
        if ((i+1) % dimension == 0) {
            printf("\n");
        }
    }
    
    free(positions);
    free(grad);
    return 0;
}
#endif
