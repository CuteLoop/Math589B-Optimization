#include <cmath>
#include <vector>
#include "energy.hpp"

double lennard_jones_potential(double r, double epsilon, double sigma) {
  if (r < 1e-12) return 1e12; // Avoid division by zero or extremely small r
  double sr6 = pow(sigma / r, 6);
  return 4 * epsilon * (sr6 * sr6 - sr6);
}

double bond_potential(double r, double b, double k_b) {
  return k_b * pow(r - b, 2);
}

double total_energy(double* positions, int n_beads, double epsilon, double sigma, double b, double k_b) {
  double energy = 0.0;

  // Bond potential
  for (int i = 0; i < n_beads - 1; ++i) {
    int idx1 = i * 3;
    int idx2 = (i + 1) * 3;
    double dx = positions[idx2] - positions[idx1];
    double dy = positions[idx2 + 1] - positions[idx1 + 1];
    double dz = positions[idx2 + 2] - positions[idx1 + 2];
    double r = sqrt(dx * dx + dy * dy + dz * dz);
    energy += bond_potential(r, b, k_b);
  }

  // Lennard-Jones potential
  for (int i = 0; i < n_beads; ++i) {
    for (int j = i + 1; j < n_beads; ++j) {
      int idx1 = i * 3;
      int idx2 = j * 3;
      double dx = positions[idx2] - positions[idx1];
      double dy = positions[idx2 + 1] - positions[idx1 + 1];
      double dz = positions[idx2 + 2] - positions[idx1 + 2];
      double r = sqrt(dx * dx + dy * dy + dz * dz);
      energy += lennard_jones_potential(r, epsilon, sigma);
    }
  }

  return energy;
}

