#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- Function Prototypes ---
// Externally defined functions in energy.c:
double total_energy(const double *positions, int n_beads, 
                    double epsilon, double sigma, double b, double k_b);
void compute_gradient(const double *positions, int n_beads, 
                      double epsilon, double sigma, double b, double k_b, 
                      double *gradient);

// Exposed function for Python wrapper.
void bfgs_optimize(double *x, int n_beads, int maxiter, double tol, 
                   double epsilon, double sigma, double b, double k_b);

// --- Objective Function Pointer Type ---
typedef double (*ObjectiveFunc)(const double *x, double *grad, int n, va_list args);

// --- Armijo Line Search ---
// This function determines an acceptable step length alpha such that:
//    f(x + alpha*p) <= f(x) + c1 * alpha * (grad^T * p)
// It uses a backtracking approach with an initial alpha and multiplies alpha by beta (0.5) until the condition is met.
double armijo_line_search(const double *x, const double *p, int n, double f_x, 
                           const double *grad, double c1, double initial_alpha, 
                           ObjectiveFunc objective, va_list args) {
    double alpha = initial_alpha;
    double dot = 0.0;
    for (int i = 0; i < n; i++) {
        dot += grad[i] * p[i];
    }
    double *x_new = (double *)malloc(n * sizeof(double));
    if (x_new == NULL) {
        perror("malloc");
        exit(1);
    }
    double beta = 0.5;  // Reduction factor
    while (1) {
        // Compute x_new = x + alpha * p
        for (int i = 0; i < n; i++) {
            x_new[i] = x[i] + alpha * p[i];
        }
        // Since objective uses a va_list, make a copy to avoid consuming the original.
        va_list args_copy;
        va_copy(args_copy, args);
        double f_new = objective(x_new, NULL, n, args_copy);
        va_end(args_copy);
        // Check the Armijo condition.
        if (f_new <= f_x + c1 * alpha * dot) {
            break;
        }
        alpha *= beta;
        if (alpha < 1e-10) { // Minimum allowed step size.
            break;
        }
    }
    free(x_new);
    return alpha;
}

// --- BFGS Optimization Implementation ---
// This function now uses the Armijo line search to select a good step length
// and fixes the bugs by (1) computing the number of beads from n (n = n_beads * 3)
// and (2) calling compute_gradient when grad is non-NULL.
void bfgs(double *x, int n, int max_iters, double tol, ObjectiveFunc objective, ...) {
    va_list args;
    double c1 = 1e-4;  // Armijo condition parameter

    double *grad = (double *)malloc(n * sizeof(double));
    double *p = (double *)malloc(n * sizeof(double));
    double *H = (double *)malloc(n * n * sizeof(double));
    double *s = (double *)malloc(n * sizeof(double));
    double *y = (double *)malloc(n * sizeof(double));

    // Initialize H to identity.
    for (int i = 0; i < n * n; i++) {
        H[i] = (i % (n + 1) == 0) ? 1.0 : 0.0;
    }

    // Evaluate initial objective value and gradient.
    va_start(args, objective);
    double f_x = objective(x, grad, n, args);
    va_end(args);

    for (int iter = 0; iter < max_iters; iter++) {
        // Compute the norm of the gradient.
        double grad_norm = 0.0;
        for (int i = 0; i < n; i++) {
            grad_norm += grad[i] * grad[i];
        }
        grad_norm = sqrt(grad_norm);
        if (grad_norm < tol) {
            printf("Converged after %d iterations\n", iter);
            break;
        }

        // Compute search direction: p = -H * grad.
        for (int i = 0; i < n; i++) {
            p[i] = 0.0;
            for (int j = 0; j < n; j++) {
                p[i] -= H[i * n + j] * grad[j];
            }
        }

        // Determine an acceptable step length via Armijo line search.
        va_start(args, objective);
        double alpha = armijo_line_search(x, p, n, f_x, grad, c1, 1.0, objective, args);
        va_end(args);

        // Update x: x = x + alpha * p, and store s = alpha * p.
        for (int i = 0; i < n; i++) {
            s[i] = alpha * p[i];
            x[i] += s[i];
        }

        // Compute new gradient and new objective value.
        double *grad_new = (double *)malloc(n * sizeof(double));
        va_start(args, objective);
        double f_new = objective(x, grad_new, n, args);
        va_end(args);

        // Compute y = grad_new - grad.
        for (int i = 0; i < n; i++) {
            y[i] = grad_new[i] - grad[i];
        }

        // BFGS update.
        double ys = 0.0;
        for (int i = 0; i < n; i++) {
            ys += y[i] * s[i];
        }
        if (ys <= 1e-10) {
            // If the curvature condition fails, reset H to the identity.
            for (int i = 0; i < n * n; i++) {
                H[i] = (i % (n + 1) == 0) ? 1.0 : 0.0;
            }
        } else {
            double rho = 1.0 / ys;
            // Compute H * y and y^T * H * y.
            double *Hy = (double *)malloc(n * sizeof(double));
            double yHy = 0.0;
            for (int i = 0; i < n; i++) {
                Hy[i] = 0.0;
                for (int j = 0; j < n; j++) {
                    Hy[i] += H[i * n + j] * y[j];
                }
                yHy += y[i] * Hy[i];
            }
            // Standard BFGS update:
            // H = H - (H*y*s^T + s*y^T*H)/(y^T*s) + (1 + y^T*H*y/(y^T*s))*(s*s^T)/(y^T*s)
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    H[i * n + j] = H[i * n + j] - (s[i] * Hy[j] + Hy[i] * s[j]) * rho
                                   + (1 + yHy * rho) * s[i] * s[j] * rho;
                }
            }
            free(Hy);
        }

        // Prepare for the next iteration.
        for (int i = 0; i < n; i++) {
            grad[i] = grad_new[i];
        }
        f_x = f_new;
        free(grad_new);
    }

    free(grad);
    free(p);
    free(H);
    free(s);
    free(y);
}

// --- Objective Function ---
// This function wraps the energy evaluation and gradient computation.
// IMPORTANT: It computes the number of beads as n/3 (since x is a flattened array
// with 3 coordinates per bead). It calls total_energy and, when grad is non-NULL,
// compute_gradient.
double objective_function(const double *x, double *grad, int n, va_list args) {
    double epsilon = va_arg(args, double);
    double sigma   = va_arg(args, double);
    double b       = va_arg(args, double);
    double k_b     = va_arg(args, double);

    int n_beads = n / 3;
    double energy = total_energy(x, n_beads, epsilon, sigma, b, k_b);

    if (grad != NULL) {
        compute_gradient(x, n_beads, epsilon, sigma, b, k_b, grad);
    }
    return energy;
}

// --- Exposed Function for Python Wrapper ---
// This is the function that will be called (via ctypes) from Python.
// It converts the number of beads into the total number of coordinates (n_beads * 3)
// and calls the BFGS optimization routine.
void bfgs_optimize(double *x, int n_beads, int maxiter, double tol, 
                   double epsilon, double sigma, double b, double k_b) {
    bfgs(x, n_beads * 3, maxiter, tol, objective_function, epsilon, sigma, b, k_b);
}

#ifdef __cplusplus
}
#endif
