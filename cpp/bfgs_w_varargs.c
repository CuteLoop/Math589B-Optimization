/*----------------------------------------------------------------
* File:     bfgs_w_varargs.c
*----------------------------------------------------------------
*
* Author:   Marek Rychlik (rychlik@arizona.edu)
* Date:     Sat Jan 25 12:49:10 2025
* Copying:  (C) Marek Rychlik, 2020. All rights reserved.
*
*----------------------------------------------------------------*/
/* NOTE: There is a bug in function matrix_update. Can you spot and fix it? */
/* This file demonstrates two things:
 * 1. An implementation of BFGS in C
 * 2. Using variable arguments in C (like **kwargs in Python)
 *
 * NOTES:
 * Most of the BFGS implementation was written by ChatGPT, produced
 * by the prompt: "Can you implement bfgs with Armijo condition in C?"
 *
 * However, when you want to pass extra arguments to the objective function,
 * as in our Assignment1 (parameters of the protein potential), while
 * maintaining generality of BFGS implementation, we need to pass
 * extra parameters to the objective function. Thus, I issued a prompt
 *
 * "In the C version, I would want to pass a function 'objective' that
 * accepts a variable number of extra arguments. How would the code
 * for that look like?"
 *
 * The code below is mostly in response to the second prompt. However,
 * there were several bugs, which I fixed. I also provided an
 * example that makes use of the extra argument.
 * 
 */


#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

// Define the prototype for the objective function
typedef double (*ObjectiveFunc)(const double *x, double *grad, int n, va_list args);

// Example objective function: f(x) = 0.5 * x^T * x + b^T * x
double objective(const double *x, double *grad, int n, va_list args)
{
  double *b = va_arg(args, double *);
  printf("%f %f\n", b[0], b[1]);

  double sum = 0.0;

  // Compute the value of the objective function
  for (int i = 0; i < n; i++) {
    sum += 0.5 * x[i] * x[i] + b[i] * x[i];
  }

  // If gradient requested, find it
  if(grad) {
    for (int i = 0; i < n; i++) {
      grad[i] = x[i] + b[i];
    }
  }

  return sum;
}

void matrix_vector_mult(const double *matrix, const double *vector, double *result, int n)
{
  for (int i = 0; i < n; i++) {
    result[i] = 0.0;
    for (int j = 0; j < n; j++) {
      result[i] += matrix[i * n + j] * vector[j];
    }
  }
}

void matrix_update(double *H, const double *s, const double *y, int n) {
  double sy = 0.0;
  for (int i = 0; i < n; i++) {
    sy += s[i] * y[i];
  }
  if (sy <= 0.0) {
    return; // Avoid division by zero or negative curvature
  }

  double rho = 1.0 / sy;
  double *Hy = malloc(n * sizeof(double));
  matrix_vector_mult(H, y, Hy, n);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double correction = rho * (s[i] * s[j] - Hy[i] * Hy[j]);
      H[i * n + j] += correction;
    }
  }

  free(Hy);
}

int armijo_condition(const double *x, const double *grad, const double *p, double alpha, int n,
                     double c1, ObjectiveFunc objective, va_list args)
{
  double *x_new = malloc(n * sizeof(double));
  va_list args_loc;

  for (int i = 0; i < n; i++) {
    x_new[i] = x[i] + alpha * p[i];
  }

  /* We need a copy, as args spoil when used */
  va_copy(args_loc, args);
  double f_x = objective(x, NULL, n, args);

  double f_x_new = objective(x_new, NULL, n, args_loc);

  double dot_product = 0.0;
  for (int i = 0; i < n; i++) {
    dot_product += grad[i] * p[i];
  }

  free(x_new);

  return f_x_new <= f_x + c1 * alpha * dot_product;
}

void bfgs(double *x, int n, int max_iters, double tol, double c1, ObjectiveFunc objective, ...)
{
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

    // Compute p = -H * grad
    matrix_vector_mult(H, grad, p, n);
    for (int i = 0; i < n; i++) {
      p[i] = -p[i];
    }

    // Armijo line search
    double alpha = 1.0;

    while (1) {
      va_start(args, objective);
      int status = armijo_condition(x, grad, p, alpha, n, c1, objective, args);
      if(status) {
	va_end(args);
	break;
      }
      alpha *= 0.5;
      va_end(args);
    }

    // Update x, s, and y
    for (int i = 0; i < n; i++) {
      s[i] = alpha * p[i];
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
    matrix_update(H, s, y, n);

    // Update grad
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

int main()
{
  int n = 2; // Number of variables
  double x[] = {1.0, 1.0}; // Initial guess
  int max_iters = 100;
  double tol = 1e-6;
  double c1 = 1e-4;
  double b[] = {40.0, 30.0};

  bfgs(x, n, max_iters, tol, c1, objective, b);

  printf("Optimal solution: ");
  for (int i = 0; i < n; i++) {
    printf("%f ", x[i]);
  }
  printf("\n");

  return 0;
}
