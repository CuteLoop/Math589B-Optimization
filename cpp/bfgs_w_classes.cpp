/*----------------------------------------------------------------
* File:     bfgs_w_classes.cpp
*----------------------------------------------------------------
*
* Author:   Marek Rychlik (rychlik@arizona.edu)
* Date:     Sat Jan 25 13:22:46 2025
* Copying:  (C) Marek Rychlik, 2020. All rights reserved.
*
*----------------------------------------------------------------*/
// An AI-generated implementation of BFGS

#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <algorithm>
#include <iomanip>

class BFGS {
public:
  BFGS(std::function<double(const std::vector<double>&)> objective,
       std::function<std::vector<double>(const std::vector<double>&)> gradient,
       int max_iters, double tol, double c1)
    : objective_(objective), gradient_(gradient),
      max_iters_(max_iters), tol_(tol), c1_(c1) {}

  std::vector<double> optimize(std::vector<double> x) {
    int n = x.size();
    std::vector<std::vector<double>> H(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) H[i][i] = 1.0; // Initialize H as identity matrix

    auto grad = gradient_(x);
    for (int iter = 0; iter < max_iters_; ++iter) {
      double grad_norm = norm(grad);
      if (grad_norm < tol_) {
	std::cout << "Converged after " << iter << " iterations.\n";
	return x;
      }

      // Compute p = -H * grad
      auto p = mat_vec_mult(H, grad);
      for (auto& val : p) val = -val;

      // Line search with Armijo condition
      double alpha = 1.0;
      while (!armijo_condition(x, grad, p, alpha)) {
	alpha *= 0.5;
      }

      // Update x and compute s and y
      std::vector<double> s(n), y(n);
      for (int i = 0; i < n; ++i) {
	s[i] = alpha * p[i];
	x[i] += s[i];
      }
      auto grad_new = gradient_(x);
      for (int i = 0; i < n; ++i) {
	y[i] = grad_new[i] - grad[i];
      }

      // Update H using BFGS formula
      update_hessian(H, s, y);

      grad = grad_new;
    }

    std::cerr << "Failed to converge within " << max_iters_ << " iterations.\n";
    return x;
  }

private:
  std::function<double(const std::vector<double>&)> objective_;
  std::function<std::vector<double>(const std::vector<double>&)> gradient_;
  int max_iters_;
  double tol_;
  double c1_;

  static double norm(const std::vector<double>& v) {
    double sum = 0.0;
    for (double val : v) sum += val * val;
    return std::sqrt(sum);
  }

  static std::vector<double> mat_vec_mult(const std::vector<std::vector<double>>& mat,
					  const std::vector<double>& vec) {
    int n = vec.size();
    std::vector<double> result(n, 0.0);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
	result[i] += mat[i][j] * vec[j];
      }
    }
    return result;
  }

  bool armijo_condition(const std::vector<double>& x,
			const std::vector<double>& grad,
			const std::vector<double>& p, double alpha) {
    auto x_new = x;
    for (size_t i = 0; i < x.size(); ++i) {
      x_new[i] += alpha * p[i];
    }
    double f_x = objective_(x);
    double f_x_new = objective_(x_new);

    double dot_product = 0.0;
    for (size_t i = 0; i < grad.size(); ++i) {
      dot_product += grad[i] * p[i];
    }

    return f_x_new <= f_x + c1_ * alpha * dot_product;
  }

  void update_hessian(std::vector<std::vector<double>>& H,
		      const std::vector<double>& s,
		      const std::vector<double>& y) {
    int n = s.size();
    double sy = 0.0;
    for (int i = 0; i < n; ++i) sy += s[i] * y[i];

    if (sy <= 0.0) return; // Avoid negative curvature

    double rho = 1.0 / sy;
    std::vector<std::vector<double>> outer_s(n, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> Hy_outer(n, std::vector<double>(n, 0.0));
    std::vector<double> Hy = mat_vec_mult(H, y);

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
	outer_s[i][j] = s[i] * s[j];
	Hy_outer[i][j] = Hy[i] * Hy[j];
      }
    }

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
	H[i][j] += rho * (outer_s[i][j] - Hy_outer[i][j]);
      }
    }
  }
};

int main() {
  auto objective = [](const std::vector<double>& x) {
    // Example: f(x) = 0.5 * ||x||^2
    double sum = 0.0;
    for (double val : x) sum += 0.5 * val * val;
    return sum;
  };

  auto gradient = [](const std::vector<double>& x) {
    // Gradient of f(x) = x
    return x;
  };

  std::vector<double> x = {1.0, 1.0}; // Initial guess
  int max_iters = 100;
  double tol = 1e-6;
  double c1 = 1e-4;

  BFGS bfgs(objective, gradient, max_iters, tol, c1);
  auto result = bfgs.optimize(x);

  std::cout << "Optimal solution: ";
  for (double val : result) {
    std::cout << std::fixed << std::setprecision(6) << val << " ";
  }
  std::cout << "\n";

  return 0;
}
