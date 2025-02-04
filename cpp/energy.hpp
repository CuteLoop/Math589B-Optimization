#ifndef ENERGY_HPP
#define ENERGY_HPP


extern "C" {

  double lennard_jones_potential(double r, double epsilon, double sigma);
  double total_energy(double* positions, int n_beads, double epsilon, double sigma, double b, double k_b);

}

#endif
