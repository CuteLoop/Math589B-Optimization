#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// Function prototype
void bfgs_optimize(double *x, int n_beads, int maxiter, double tol, 
                   double epsilon, double sigma, double b, double k_b);

// External energy function (defined in energy.c)
double total_energy(const double *positions, int n_beads, 
                    double epsilon, double sigma, double b, double k_b);

// Define the prototype for the objective function
typedef double (*ObjectiveFunc)(const double *x, double *grad, int n, va_list args);

// BFGS implementation
void bfgs(double *x, int n, int max_iters, double tol, ObjectiveFunc objective, ...) {
    va_list args;

    double *grad = malloc(n * sizeof(double));
    double *p = malloc(n * sizeof(double));
    double *H = malloc(n * n * sizeof(double));
    double *s = malloc(n * sizeof(double));
    double *y = malloc(n * sizeof(double));

    // Initialize H to identity
    for (int i = 0; i < n * n; i++) {
        H[i] = (i % (n + 1) == 0) ? 1.0 : 0.0;
    }

    va_start(args, objective);
    objective(x, grad, n, args);
    va_end(args);

    for (int iter = 0; iter < max_iters; iter++) {
        double grad_norm = 0.0;
        for (int i = 0; i < n; i++) {
            grad_norm += grad[i] * grad[i];
        }
        grad_norm = sqrt(grad_norm);
        if (grad_norm < tol) {
            printf("Converged after %d iterations\n", iter);
            break;
        }

        // Compute search direction p = -H * grad
        for (int i = 0; i < n; i++) {
            p[i] = 0.0;
            for (int j = 0; j < n; j++) {
                p[i] -= H[i * n + j] * grad[j];
            }
        }

        // Update x
        for (int i = 0; i < n; i++) {
            s[i] = p[i];  // Assume step size = 1 for simplicity
            x[i] += s[i];
        }

        double *grad_new = malloc(n * sizeof(double));
        va_start(args, objective);
        objective(x, grad_new, n, args);
        va_end(args);

        for (int i = 0; i < n; i++) {
            y[i] = grad_new[i] - grad[i];
        }

        // Update H
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                H[i * n + j] += s[i] * y[j] - H[i * n + j] * y[i] * y[j];
            }
        }

        for (int i = 0; i < n; i++) {
            grad[i] = grad_new[i];
        }
        free(grad_new);
    }

    free(grad);
    free(p);
    free(H);
    free(s);
    free(y);
}

// Objective function that calls total_energy
double objective_function(const double *x, double *grad, int n, va_list args) {
    double epsilon = va_arg(args, double);
    double sigma = va_arg(args, double);
    double b = va_arg(args, double);
    double k_b = va_arg(args, double);

    return total_energy(x, n, epsilon, sigma, b, k_b);
}

// Exposed function for Python wrapper
void bfgs_optimize(double *x, int n_beads, int maxiter, double tol, 
                   double epsilon, double sigma, double b, double k_b) {
    bfgs(x, n_beads * 3, maxiter, tol, objective_function, epsilon, sigma, b, k_b);
}

#ifdef __cplusplus
}
#endif
