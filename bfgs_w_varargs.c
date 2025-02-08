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
// Exposed function: bfgs_optimize(...)
//
// We'll define an enum to pick the line-search approach at runtime.
// ---------------------------------------------------------------------
typedef enum {
    LINESEARCH_ARMIJO,
    LINESEARCH_WOLFE,
    LINESEARCH_STRONG_WOLFE
} LineSearchType;

// The Python-exposed function
void bfgs_optimize(double *x, int n_beads, int maxiter, double tol, 
                   double epsilon, double sigma, double b, double k_b,
                   int linesearch_choice);

// ---------------------------------------------------------------------
// ObjectiveFunc pointer
// ---------------------------------------------------------------------
typedef double (*ObjectiveFunc)(const double *x, double *grad, int n, va_list args);

// ---------------------------------------------------------------------
// Objective function that calls total_energy + compute_gradient
// n = 3*n_beads
// ---------------------------------------------------------------------
static double objective_function(const double *x, double *grad, int n, va_list args)
{
    double epsilon = va_arg(args, double);
    double sigma   = va_arg(args, double);
    double bb      = va_arg(args, double);
    double kb      = va_arg(args, double);

    int n_beads = n/3;
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
    double s=0.0;
    for(int i=0; i<n; i++){
        s += a[i]*b[i];
    }
    return s;
}

// ---------------------------------------------------------------------
// 1) Armijo (Backtracking) line search
//    alpha starts at initial_alpha, shrinks by beta if the Armijo condition fails:
//      f(x+alpha*p) <= f(x) + c1 * alpha * grad^T p
// ---------------------------------------------------------------------
static double armijo_line_search(const double *x, const double *p, int n,
                                 double f_x, const double *grad,
                                 double c1, double initial_alpha,
                                 ObjectiveFunc objective,
                                 va_list args_in)
{
    double alpha = initial_alpha;
    double dotg = dot_product(grad, p, n);
    double beta = 0.5; // reduction factor

    double *x_new = (double*)malloc(n*sizeof(double));
    if(!x_new){
        fprintf(stderr,"Allocation failed.\n");
        exit(1);
    }

    // max attempts
    for(int tries=0; tries<50; tries++){
        // x_new = x + alpha*p
        for(int i=0; i<n; i++){
            x_new[i] = x[i] + alpha*p[i];
        }
        // Evaluate f(x_new)
        va_list acopy;
        va_copy(acopy, args_in);
        double f_new = objective(x_new, NULL, n, acopy);
        va_end(acopy);

        if(f_new <= f_x + c1*alpha*dotg){
            break; // success
        }
        alpha *= beta;
        if(alpha<1e-12){ // too small
            break;
        }
    }
    free(x_new);
    return alpha;
}

// ---------------------------------------------------------------------
// 2) Wolfe (Weak Wolfe) line search (simplified backtracking approach)
//
// Conditions:
//   1) f(x+alpha*p) <= f(x) + c1 alpha grad^T p
//   2) grad(x+alpha*p)^T p >= c2 grad^T p
// we reduce alpha if condition 1 fails, or expand alpha if condition 2 fails
// ---------------------------------------------------------------------
static double wolfe_line_search(const double *x, const double *p, int n,
                                double f_x, const double *grad,
                                double c1, double c2, double initial_alpha,
                                ObjectiveFunc objective,
                                va_list args_in)
{
    double alpha = initial_alpha;
    double dotg = dot_product(grad, p, n);
    double beta_shrink = 0.5;
    double beta_expand = 2.0;

    // allocate workspace
    double *x_new     = (double*)malloc(n*sizeof(double));
    double *grad_new  = (double*)malloc(n*sizeof(double));
    if(!x_new || !grad_new){
        fprintf(stderr,"Allocation error\n");
        exit(1);
    }

    for(int tries=0; tries<50; tries++){
        // build x_new
        for(int i=0; i<n; i++){
            x_new[i] = x[i] + alpha*p[i];
        }
        // Evaluate f(x_new) + grad(x_new)
        va_list acopy;
        va_copy(acopy, args_in);
        double f_new = objective(x_new, grad_new, n, acopy);
        va_end(acopy);

        // check Armijo
        if(f_new > f_x + c1*alpha*dotg){
            // reduce alpha
            alpha *= beta_shrink;
        } else {
            // check curvature
            double dotg_new = dot_product(grad_new, p, n);
            if(dotg_new < c2*dotg){
                // expand alpha
                alpha *= beta_expand;
            } else {
                // success
                break;
            }
        }
        if(alpha<1e-12 || alpha>1e12)
            break;
    }
    free(x_new);
    free(grad_new);
    return alpha;
}

// ---------------------------------------------------------------------
// 3) Strong Wolfe line search (Bracket + Zoom)
//   We'll implement the "complete" approach with bracket and zoom
// ---------------------------------------------------------------------

// Evaluate phi(alpha) = f(x+alpha*p), grad => phi'(alpha)= grad(x+alpha*p)^T p
static double phi_val(const double *x, const double *p, int n, double alpha,
                      ObjectiveFunc objective, va_list args_in)
{
    double *temp = (double*)malloc(n*sizeof(double));
    for(int i=0; i<n; i++){
        temp[i] = x[i] + alpha*p[i];
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
    double *temp = (double*)malloc(n*sizeof(double));
    for(int i=0; i<n; i++){
        temp[i] = x[i] + alpha*p[i];
    }
    va_list acopy;
    va_copy(acopy, args_in);
    double val = objective(temp, grad_out, n, acopy);
    va_end(acopy);
    free(temp);
    return val;
}

// phi'(alpha) = grad(x+alpha*p)^T p
static double phi_prime(const double *grad_at_xnew, const double *p, int n)
{
    return dot_product(grad_at_xnew, p, n);
}

static double zoom_strong(double alpha_lo, double alpha_hi,
                          const double *x, const double *p, int n,
                          double f0, double dphi0, double c1, double c2,
                          ObjectiveFunc objective, va_list args_in)
{
    const int MAX_ZOOM=20;

    double *grad_lo = (double*)malloc(n*sizeof(double));
    double f_lo=phi_val_and_grad(x, p, n, alpha_lo, objective, args_in, grad_lo);

    double alpha_star=0.5*(alpha_lo+alpha_hi);

    for(int iter=0; iter<MAX_ZOOM; iter++){
        double alpha_mid = 0.5*(alpha_lo + alpha_hi);

        double *grad_mid=(double*)calloc(n,sizeof(double));
        double f_mid=phi_val_and_grad(x, p, n, alpha_mid, objective, args_in, grad_mid);

        if( (f_mid > f0 + c1*alpha_mid*dphi0) || (f_mid>= f_lo) ){
            alpha_hi= alpha_mid;
        } else {
            double dphi_mid= phi_prime(grad_mid, p, n);
            if(fabs(dphi_mid)<= -c2*dphi0){
                alpha_star= alpha_mid;
                free(grad_mid);
                break;
            }
            if(dphi_mid*(alpha_hi - alpha_lo)>=0){
                alpha_hi=alpha_lo;
            }
            alpha_lo=alpha_mid;
            // update f_lo
            f_lo=f_mid;
            for(int i=0; i<n; i++){
                grad_lo[i]= grad_mid[i];
            }
        }
        free(grad_mid);
        alpha_star=alpha_mid;
        if(fabs(alpha_hi-alpha_lo)<1e-12)
            break;
    }
    free(grad_lo);
    return alpha_star;
}

static double strong_wolfe_line_search(const double *x, const double *p, int n,
                                       double f_x, const double *grad, 
                                       double c1, double c2,
                                       ObjectiveFunc objective, va_list args_in)
{
    double dphi0= dot_product(grad,p,n);

    double alpha0=0.0;
    double alpha1=1.0;
    double f0 = f_x;

    double f1= phi_val(x,p,n, alpha1, objective, args_in);
    const int MAX_EXPAND=10;
    double alpha_prev= alpha0;
    double f_prev= f0;

    for(int i=0; i<MAX_EXPAND; i++){
        if( (f1> f0 + c1*alpha1*dphi0) || ((i>0)&&(f1>= f_prev)) ){
            // bracket
            return zoom_strong(alpha_prev, alpha1, x,p,n, f0, dphi0, c1,c2, objective,args_in);
        }
        // check curvature
        double *gtemp=(double*)calloc(n,sizeof(double));
        double ftemp= phi_val_and_grad(x,p,n, alpha1, objective, args_in, gtemp);
        double dphi1= dot_product(gtemp,p,n);
        free(gtemp);

        if(fabs(dphi1)<= -c2*dphi0){
            return alpha1; // done
        }
        if(dphi1>=0){
            // bracket
            return zoom_strong(alpha1, alpha_prev, x,p,n,f0,dphi0,c1,c2,objective,args_in);
        }
        // expand alpha
        alpha_prev= alpha1;
        f_prev= f1;
        alpha1*=2.0;
        if(alpha1>1e5)
            break;
        f1= phi_val(x,p,n, alpha1, objective, args_in);
    }
    return alpha1;
}

// ---------------------------------------------------------------------
// The actual BFGS routine with a line-search function pointer
// ---------------------------------------------------------------------
typedef double (*LineSearchFunc)(const double *x, const double *p, int n,
                                 double f_x, const double *grad,
                                 double c1, double c2_or_unused, double init_alpha,
                                 ObjectiveFunc objective,
                                 va_list args_in);

static double bfgs_line_search_call(LineSearchType ls_choice,
                                    const double *x, const double *p, int n,
                                    double f_x, const double *grad,
                                    ObjectiveFunc objective,
                                    va_list args_in)
{
    // typical constants for line search
    double c1=1e-4; 
    double c2=0.9;  
    double alpha_init=1.0;

    switch(ls_choice){
    case LINESEARCH_ARMIJO:
        // c2 not used
        return armijo_line_search(x,p,n,f_x, grad,c1, alpha_init,objective,args_in);
    case LINESEARCH_WOLFE:
        return wolfe_line_search(x,p,n,f_x, grad,c1, c2, alpha_init, objective,args_in);
    case LINESEARCH_STRONG_WOLFE:
    default:
        return strong_wolfe_line_search(x,p,n,f_x, grad,c1, c2, objective,args_in);
    }
}

// ---------------------------------------------------------------------
// BFGS Implementation
// ---------------------------------------------------------------------
static void bfgs_impl(double *x, int n, int max_iters, double tol,
                      LineSearchType ls_choice,
                      ObjectiveFunc objective, ...)
{
    va_list args;
    va_start(args, objective);

    double *grad = (double*)calloc(n, sizeof(double));
    double *p    = (double*)calloc(n, sizeof(double));
    double *H    = (double*)calloc(n*n, sizeof(double));
    double *s    = (double*)calloc(n, sizeof(double));
    double *y    = (double*)calloc(n, sizeof(double));

    // init H= I
    for(int i=0; i<n; i++){
        H[i*n + i]=1.0;
    }

    // initial function + gradient
    double f_x= objective(x, grad, n, args);
    va_end(args);

    for(int iter=0; iter<max_iters; iter++){
        // check norm
        double grad_norm=0.0;
        for(int i=0; i<n; i++){
            grad_norm+= grad[i]*grad[i];
        }
        grad_norm= sqrt(grad_norm);
        if(grad_norm<tol){
            printf("Converged after %d iterations\n", iter);
            break;
        }

        // p= -H*grad
        for(int i=0; i<n; i++){
            double tmp=0.0;
            for(int j=0; j<n; j++){
                tmp += H[i*n+j]* grad[j];
            }
            p[i]= -tmp;
        }

        // call the chosen line search
        va_list args2;
        va_start(args2, objective);
        // to re-extract the same parameters we used above
        double alpha= bfgs_line_search_call(ls_choice, x, p, n, f_x, grad, objective, args2);
        va_end(args2);

        // s= alpha*p, x+= s
        for(int i=0; i<n; i++){
            s[i]= alpha*p[i];
            x[i]+= s[i];
        }

        // new gradient
        double *grad_new= (double*)calloc(n,sizeof(double));
        va_list args3;
        va_start(args3, objective);
        double f_new= objective(x, grad_new, n, args3);
        va_end(args3);

        // y= grad_new - grad
        for(int i=0; i<n; i++){
            y[i]= grad_new[i]- grad[i];
        }
        double ys=0.0;
        for(int i=0; i<n; i++){
            ys+= y[i]*s[i];
        }
        // BFGS update
        if(ys>1e-12){
            double rho=1.0/ ys;
            // Hy= H*y
            double *Hy= (double*)calloc(n,sizeof(double));
            double yHy= 0.0;
            for(int i=0; i<n; i++){
                double tmp=0.0;
                for(int j=0; j<n; j++){
                    tmp+= H[i*n+j]* y[j];
                }
                Hy[i]= tmp;
                yHy+= y[i]*Hy[i];
            }
            for(int i=0; i<n; i++){
                for(int j=0; j<n; j++){
                    double term1= -rho*( s[i]*Hy[j]+ Hy[i]*s[j]);
                    double term2= (1.0+ yHy*rho)* rho * s[i]*s[j];
                    H[i*n+j]+= term1+ term2;
                }
            }
            free(Hy);
        } else {
            // reset H=I
            for(int i=0; i<n*n; i++){
                H[i]=0.0;
            }
            for(int i=0; i<n; i++){
                H[i*n+i]=1.0;
            }
        }

        free(grad_new);
        for(int i=0; i<n; i++){
            grad[i]= grad_new[i];
        }
        f_x= f_new;
    }
    free(grad); free(p); free(H); free(s); free(y);
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
    LineSearchType chosen_ls= LINESEARCH_ARMIJO;
    if(linesearch_choice==1){
        chosen_ls= LINESEARCH_WOLFE;
    } else if(linesearch_choice==2){
        chosen_ls= LINESEARCH_STRONG_WOLFE;
    }
    bfgs_impl(x, n_beads*3, maxiter, tol, chosen_ls,
              objective_function,
              epsilon, sigma, b, k_b);
}

#ifdef __cplusplus
}
#endif
