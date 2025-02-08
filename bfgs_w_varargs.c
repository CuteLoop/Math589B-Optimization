#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------
// External (energy.c) - Provided by you
// ---------------------------------------------------------------------
double total_energy(const double *positions, int n_beads, 
                    double epsilon, double sigma, double b, double k_b);

void compute_gradient(const double *positions, int n_beads, 
                      double epsilon, double sigma, double b, double k_b, 
                      double *gradient);

// ---------------------------------------------------------------------
// We define an enum to pick the line-search approach at runtime.
// ---------------------------------------------------------------------
typedef enum {
    LINESEARCH_ARMIJO,
    LINESEARCH_WOLFE,
    LINESEARCH_STRONG_WOLFE
} LineSearchType;

// Exposed function for Python:
void bfgs_optimize(double *x, int n_beads, int maxiter, double tol, 
                   double epsilon, double sigma, double b, double k_b,
                   int linesearch_choice);

// ---------------------------------------------------------------------
// ObjectiveFunc pointer + objective function that calls total_energy + compute_gradient
// n = 3*n_beads
// ---------------------------------------------------------------------
typedef double (*ObjectiveFunc)(const double *x, double *grad, int n, va_list args);

static double objective_function(const double *x, double *grad, int n, va_list args)
{
    double epsilon = va_arg(args, double);
    double sigma   = va_arg(args, double);
    double bb      = va_arg(args, double);
    double kb      = va_arg(args, double);

    int n_beads = n / 3;
    double energy = total_energy(x, n_beads, epsilon, sigma, bb, kb);
    if(grad != NULL) {
        compute_gradient(x, n_beads, epsilon, sigma, bb, kb, grad);
    }
    return energy;
}

// ---------------------------------------------------------------------
// Helper: Dot product
// ---------------------------------------------------------------------
static double dot_product(const double *a, const double *b, int n)
{
    double s = 0.0;
    for(int i = 0; i < n; i++){
        s += a[i] * b[i];
    }
    return s;
}

// =====================================================================
// (1) Armijo (Backtracking) line search with dynamic alpha minimum
// =====================================================================
static double armijo_line_search(const double *x, const double *p, int n,
                                 double f_x, const double *grad,
                                 double c1, double alpha_init,
                                 ObjectiveFunc objective,
                                 va_list args_in,
                                 double min_alpha) // dynamic minimum alpha
{
    double alpha = alpha_init;
    double dotg  = dot_product(grad, p, n);
    double beta  = 0.5;  // shrink factor (unchanged for Armijo)

    double *x_new = (double*)malloc(n * sizeof(double));
    if(!x_new){
        fprintf(stderr,"Alloc failed.\n");
        exit(1);
    }

    // Up to 50 attempts to reduce alpha
    for(int tries = 0; tries < 50; tries++){
        // x_new = x + alpha * p
        for(int i = 0; i < n; i++){
            x_new[i] = x[i] + alpha * p[i];
        }
        // Evaluate f(x_new)
        va_list acopy;
        va_copy(acopy, args_in);
        double f_new = objective(x_new, NULL, n, acopy);
        va_end(acopy);

        // Armijo condition: f(x+alpha*p) <= f(x) + c1 * alpha * (grad^T * p)
        if(f_new <= f_x + c1 * alpha * dotg){
            break; // success
        }
        alpha *= beta;
        if(alpha < min_alpha){ // dynamic lower bound check
            break;
        }
    }
    free(x_new);
    return alpha;
}

// =====================================================================
// (2) Wolfe (Weak Wolfe) line search with dynamic alpha minimum,
// updated to mimic the Python tuning (c1=1e-4, c2=0.9, shrink factor=0.9)
// =====================================================================
static double wolfe_line_search(const double *x, const double *p, int n,
                                double f_x, const double *grad,
                                double c1, double c2, double alpha_init,
                                ObjectiveFunc objective,
                                va_list args_in,
                                double min_alpha) // dynamic minimum alpha
{
    double alpha = alpha_init;
    double dotg  = dot_product(grad, p, n);    // p^T * grad(x)
    double shrink = 0.9;  // Updated shrink factor from 0.5 to 0.9

    // Allocate workspace
    double *x_new    = (double*)malloc(n * sizeof(double));
    double *grad_new = (double*)malloc(n * sizeof(double));
    if(!x_new || !grad_new){
        fprintf(stderr,"Allocation error in wolfe_line_search.\n");
        exit(1);
    }

    for(int tries = 0; tries < 50; tries++){
        // x_new = x + alpha * p
        for(int i = 0; i < n; i++){
            x_new[i] = x[i] + alpha * p[i];
        }

        // Evaluate f(x_new) and grad(x_new)
        va_list acopy;
        va_copy(acopy, args_in);
        double f_new = objective(x_new, grad_new, n, acopy);
        va_end(acopy);

        // Condition (i): Armijo condition
        if(f_new > f_x + c1 * alpha * dotg) {
            alpha *= shrink;
        } else {
            // Condition (ii): Curvature condition: check absolute directional derivative.
            double dotg_new = dot_product(grad_new, p, n);
            if(fabs(dotg_new) > fabs(c2 * dotg)){
                alpha *= shrink;
            } else {
                break; // Both conditions satisfied.
            }
        }
        if(alpha < min_alpha){ // enforce dynamic lower bound
            break;
        }
    }
    free(x_new);
    free(grad_new);
    return alpha;
}

// =====================================================================
// (3) Strong Wolfe line search (Bracket + Zoom) with dynamic alpha minimum
// =====================================================================

// Evaluate phi(alpha) = f(x+alpha*p)
static double phi_val(const double *x, const double *p, int n, double alpha,
                      ObjectiveFunc objective, va_list args_in)
{
    double *temp = (double*)malloc(n * sizeof(double));
    for(int i = 0; i < n; i++){
        temp[i] = x[i] + alpha * p[i];
    }
    va_list acopy;
    va_copy(acopy, args_in);
    double val = objective(temp, NULL, n, acopy);
    va_end(acopy);
    free(temp);
    return val;
}

static double phi_val_and_grad(const double *x, const double *p, int n, double alpha,
                               ObjectiveFunc objective, va_list args_in,
                               double *grad_out)
{
    double *temp = (double*)malloc(n * sizeof(double));
    for(int i = 0; i < n; i++){
        temp[i] = x[i] + alpha * p[i];
    }
    va_list acopy;
    va_copy(acopy, args_in);
    double val = objective(temp, grad_out, n, acopy);
    va_end(acopy);
    free(temp);
    return val;
}

static double phi_prime(const double *grad_at_xnew, const double *p, int n)
{
    return dot_product(grad_at_xnew, p, n);
}

static double zoom_strong(double alpha_lo, double alpha_hi,
                          const double *x, const double *p, int n,
                          double f0, double dphi0, double c1, double c2,
                          ObjectiveFunc objective, va_list args_in,
                          double min_alpha) // dynamic minimum alpha
{
    const int MAX_ZOOM = 20;
    double *grad_lo = (double*)malloc(n * sizeof(double));
    double f_lo = phi_val_and_grad(x, p, n, alpha_lo, objective, args_in, grad_lo);
    double alpha_star = 0.5 * (alpha_lo + alpha_hi);
    for(int iter = 0; iter < MAX_ZOOM; iter++){
        double alpha_mid = 0.5 * (alpha_lo + alpha_hi);
        if(alpha_mid < min_alpha) { // enforce dynamic lower bound
            alpha_mid = min_alpha;
        }
        double *grad_mid = (double*)calloc(n, sizeof(double));
        double f_mid = phi_val_and_grad(x, p, n, alpha_mid, objective, args_in, grad_mid);
        if((f_mid > f0 + c1 * alpha_mid * dphi0) || (f_mid >= f_lo)){
            alpha_hi = alpha_mid;
        } else {
            double dphi_mid = phi_prime(grad_mid, p, n);
            if(fabs(dphi_mid) <= -c2 * dphi0){
                alpha_star = alpha_mid;
                free(grad_mid);
                break;
            }
            if(dphi_mid * (alpha_hi - alpha_lo) >= 0){
                alpha_hi = alpha_lo;
            }
            alpha_lo = alpha_mid;
            f_lo = f_mid;
            for(int i = 0; i < n; i++){
                grad_lo[i] = grad_mid[i];
            }
        }
        free(grad_mid);
        alpha_star = alpha_mid;
        if(fabs(alpha_hi - alpha_lo) < min_alpha)
            break;
    }
    free(grad_lo);
    return alpha_star;
}

static double strong_wolfe_line_search(const double *x, const double *p, int n,
                                       double f_x, const double *grad, 
                                       double c1, double c2,
                                       ObjectiveFunc objective, va_list args_in,
                                       double min_alpha) // dynamic minimum alpha
{
    double dphi0 = dot_product(grad, p, n);
    double alpha0 = 0.0;
    double alpha1 = 1.0;
    double f0 = f_x;
    double f1 = phi_val(x, p, n, alpha1, objective, args_in);
    const int MAX_EXPAND = 10;
    double alpha_prev = alpha0;
    double f_prev = f0;

    for(int i = 0; i < MAX_EXPAND; i++){
        if((f1 > f0 + c1 * alpha1 * dphi0) || ((i > 0) && (f1 >= f_prev))){
            return zoom_strong(alpha_prev, alpha1, x, p, n, f0, dphi0, c1, c2, objective, args_in, min_alpha);
        }
        double *gtemp = (double*)calloc(n, sizeof(double));
        double ftemp = phi_val_and_grad(x, p, n, alpha1, objective, args_in, gtemp);
        double dphi1 = dot_product(gtemp, p, n);
        free(gtemp);

        if(fabs(dphi1) <= -c2 * dphi0){
            return alpha1;
        }
        if(dphi1 >= 0){
            return zoom_strong(alpha1, alpha_prev, x, p, n, f0, dphi0, c1, c2, objective, args_in, min_alpha);
        }
        alpha_prev = alpha1;
        f_prev = f1;
        alpha1 *= 2.0;
        if(alpha1 > 1e5)
            break;
        f1 = phi_val(x, p, n, alpha1, objective, args_in);
    }
    return alpha1;
}

// ---------------------------------------------------------------------
// The actual BFGS routine with a line-search function pointer
// ---------------------------------------------------------------------
static double bfgs_line_search_call(LineSearchType ls_choice,
                                    const double *x, const double *p, int n,
                                    double f_x, const double *grad,
                                    ObjectiveFunc objective,
                                    va_list args_in,
                                    double min_alpha) // dynamic minimum alpha
{
    // For Wolfe, we now use c1 = 1e-4 and c2 = 0.9.
    double c1 = 1e-4; 
    double c2 = 0.9;  
    double alpha_init = 1.0;

    switch(ls_choice){
        case LINESEARCH_ARMIJO:
            return armijo_line_search(x, p, n, f_x, grad, c1, alpha_init, objective, args_in, min_alpha);
        case LINESEARCH_WOLFE:
            return wolfe_line_search(x, p, n, f_x, grad, c1, c2, alpha_init, objective, args_in, min_alpha);
        case LINESEARCH_STRONG_WOLFE:
        default:
            return strong_wolfe_line_search(x, p, n, f_x, grad, c1, c2, objective, args_in, min_alpha);
    }
}

// ---------------------------------------------------------------------
// BFGS Implementation
// ---------------------------------------------------------------------
static void bfgs_impl(double *x, int n, int max_iters, double tol,
                      LineSearchType ls_choice,
                      ObjectiveFunc objective, ... )
{
    va_list args;
    va_start(args, objective);

    double *grad = (double*)calloc(n, sizeof(double));
    double *p    = (double*)calloc(n, sizeof(double));
    double *H    = (double*)calloc(n * n, sizeof(double));
    double *s    = (double*)calloc(n, sizeof(double));
    double *y    = (double*)calloc(n, sizeof(double));

    if(!grad || !p || !H || !s || !y){
        fprintf(stderr,"Allocation error in bfgs_impl.\n");
        exit(1);
    }

    // Initialize H = I
    for(int i = 0; i < n; i++){
        H[i * n + i] = 1.0;
    }

    // Initial function value and gradient
    double f_x = objective(x, grad, n, args);
    va_end(args);

    for(int iter = 0; iter < max_iters; iter++){
        // Check gradient norm
        double grad_norm = 0.0;
        for(int i = 0; i < n; i++){
            grad_norm += grad[i] * grad[i];
        }
        grad_norm = sqrt(grad_norm);
        if(grad_norm < tol){
            printf("Converged after %d iterations\n", iter);
            break;
        }

        // p = -H * grad
        for(int i = 0; i < n; i++){
            double tmp = 0.0;
            for(int j = 0; j < n; j++){
                tmp += H[i * n + j] * grad[j];
            }
            p[i] = -tmp;
        }

        // Compute dynamic minimum alpha: ensure that alpha * ||p|| >= tol
        double norm_p = sqrt(dot_product(p, p, n));
        double dynamic_alpha_min = tol / (norm_p > 0 ? norm_p : 1.0);

        // Call the chosen line search
        va_list args2;
        va_start(args2, objective); // re-extract same varargs
        double alpha = bfgs_line_search_call(ls_choice, x, p, n, f_x, grad, objective, args2, dynamic_alpha_min);
        va_end(args2);

        // Update: s = alpha * p, and x = x + s
        for(int i = 0; i < n; i++){
            s[i] = alpha * p[i];
            x[i] += s[i];
        }

        // Compute new gradient
        double *grad_new = (double*)calloc(n, sizeof(double));
        if(!grad_new){
            fprintf(stderr, "Allocation error for grad_new.\n");
            exit(1);
        }
        va_list args3;
        va_start(args3, objective);
        double f_new = objective(x, grad_new, n, args3);
        va_end(args3);

        // y = grad_new - grad
        for(int i = 0; i < n; i++){
            y[i] = grad_new[i] - grad[i];
        }
        double ys = 0.0;
        for(int i = 0; i < n; i++){
            ys += y[i] * s[i];
        }

        // BFGS update
        if(ys > 1e-12){
            double rho = 1.0 / ys;
            // Compute Hy = H * y
            double *Hy = (double*)calloc(n, sizeof(double));
            double yHy = 0.0;
            for(int i = 0; i < n; i++){
                double tmp = 0.0;
                for(int j = 0; j < n; j++){
                    tmp += H[i * n + j] * y[j];
                }
                Hy[i] = tmp;
                yHy += y[i] * Hy[i];
            }

            // Update H: H = H - (s*Hy^T + Hy*s^T)*rho + (1 + y^T*H*y/(y^T*s))*(s*s^T)*rho
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    double term1 = -rho * (s[i] * Hy[j] + Hy[i] * s[j]);
                    double term2 = (1.0 + yHy * rho) * rho * s[i] * s[j];
                    H[i * n + j] += term1 + term2;
                }
            }
            free(Hy);
        } else {
            // Reset H = I if y^T*s is too small
            for(int i = 0; i < n * n; i++){
                H[i] = 0.0;
            }
            for(int i = 0; i < n; i++){
                H[i * n + i] = 1.0;
            }
        }

        // Update for next iteration
        for(int i = 0; i < n; i++){
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

// ---------------------------------------------------------------------
// bfgs_optimize: The function exposed to Python
//
// linesearch_choice: 0 => Armijo, 1 => Wolfe, 2 => Strong Wolfe
// ---------------------------------------------------------------------
void bfgs_optimize(double *x, int n_beads, int maxiter, double tol, 
                   double epsilon, double sigma, double b, double k_b,
                   int linesearch_choice)
{
    LineSearchType chosen_ls = LINESEARCH_ARMIJO;
    if(linesearch_choice == 1){
        chosen_ls = LINESEARCH_WOLFE;
    } else if(linesearch_choice == 2){
        chosen_ls = LINESEARCH_STRONG_WOLFE;
    }

    bfgs_impl(x, n_beads * 3, maxiter, tol, chosen_ls,
              objective_function,
              epsilon, sigma, b, k_b);
}

#ifdef __cplusplus
}
#endif
