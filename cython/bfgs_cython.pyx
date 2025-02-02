# bfgs_cython.pyx
# cython: language_level=3
# This directive ensures that we are using Python 3 semantics.

import numpy as np
cimport numpy as np        # C-level access to NumPy arrays
from libc.math cimport sqrt  # Import C sqrt for faster math

# Define our double precision type for convenience
ctypedef np.float64_t DTYPE_t

###############################################
# Rosenbrock Function and Its Gradient
###############################################

cpdef double rosenbrock(np.ndarray[DTYPE_t, ndim=1] x):
    """
    Compute the Rosenbrock function:
       f(x) = sum_{i=0}^{n-2} [100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2]
    """
    cdef int i, n = x.shape[0]
    cdef double sum_val = 0.0, term1, term2
    for i in range(n - 1):
        term1 = 100.0 * (x[i+1] - x[i]*x[i])**2  # 100*(x[i+1]-x[i]^2)^2
        term2 = (1 - x[i])**2                     # (1 - x[i])^2
        sum_val += term1 + term2
    return sum_val

cpdef np.ndarray rosenbrock_grad(np.ndarray[DTYPE_t, ndim=1] x):
    """
    Compute the gradient of the Rosenbrock function.
    Returns a 1D NumPy array with the same shape as x.
    """
    cdef int n = x.shape[0]
    cdef int i
    cdef np.ndarray[DTYPE_t, ndim=1] grad = np.zeros(n, dtype=np.float64)
    # First element of gradient
    grad[0] = -400.0 * x[0] * (x[1] - x[0]*x[0]) - 2.0 * (1 - x[0])
    for i in range(1, n - 1):
        grad[i] = 200.0 * (x[i] - x[i-1]*x[i-1]) \
                  - 400.0 * x[i] * (x[i+1] - x[i]*x[i]) \
                  - 2.0 * (1 - x[i])
    grad[n-1] = 200.0 * (x[n-1] - x[n-2]*x[n-2])
    return grad

###############################################
# BFGS Optimization Function
###############################################

cpdef tuple bfgs(np.ndarray[DTYPE_t, ndim=1] x0, object func, object grad_func, double tol=1e-6, int max_iter=1000):
    """
    Minimize a function using the BFGS algorithm.
    
    Parameters:
      - x0: Initial guess (1D NumPy array, float64).
      - func: Objective function (callable from Python).
      - grad_func: Function to compute the gradient.
      - tol: Tolerance for the gradient norm.
      - max_iter: Maximum number of iterations.
    
    Returns:
      A tuple (x, f_val) where:
        x: The estimated minimum (1D array).
        f_val: The function value at x.
    """
    cdef int n = x0.shape[0]
    # Copy the initial guess to avoid modifying the input
    cdef np.ndarray[DTYPE_t, ndim=1] x = x0.copy()
    # Create an identity matrix of size n
    cdef np.ndarray[DTYPE_t, ndim=2] I = np.eye(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] H = I.copy()  # H is our inverse Hessian approximation
    # Compute the initial gradient using the provided grad_func (a Python callable)
    cdef np.ndarray[DTYPE_t, ndim=1] g = grad_func(x)
    cdef int k, i, j
    cdef double norm_g, alpha, c_const = 1e-4, rho = 0.9, ys, temp
    # Temporary arrays for search direction, and differences
    cdef np.ndarray[DTYPE_t, ndim=1] p = np.empty(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] s = np.empty(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] y = np.empty(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] x_new, g_new

    # Main iteration loop
    for k in range(max_iter):
        norm_g = np.linalg.norm(g)  # Compute the norm of the gradient
        if norm_g < tol:
            print(f"Converged in {k} iterations.")
            break

        # Compute search direction p = -H * g
        p = -np.dot(H, g)

        # Backtracking line search: start with alpha = 1 and reduce by factor rho until Armijo condition is met
        alpha = 1.0
        while func(x + alpha * p) > func(x) + c_const * alpha * np.dot(g, p):
            alpha *= rho

        # Update step: x_new = x + alpha * p
        x_new = x + alpha * p
        # Compute new gradient at x_new
        g_new = grad_func(x_new)
        # Differences: s = x_new - x, y = g_new - g
        s = x_new - x
        y = g_new - g
        ys = np.dot(y, s)  # y^T * s

        if ys > 1e-10:
            temp = 1.0 / ys
            # Update the inverse Hessian using the BFGS update formula:
            # H = (I - temp * outer(s, y)) * H * (I - temp * outer(y, s)) + temp * outer(s, s)
            H = (np.eye(n) - temp * np.outer(s, y)) @ H @ (np.eye(n) - temp * np.outer(y, s)) \
                + temp * np.outer(s, s)
        else:
            # If ys is too small, reset H to the identity matrix
            H = np.eye(n, dtype=np.float64)

        # Prepare for next iteration
        x = x_new
        g = g_new
    else:
        print(f"Maximum iterations ({max_iter}) reached.")

    # Return the final x and the function value at x
    return x, func(x)

###############################################
# Main function for testing
###############################################

def main():
    # Initial guess for the Rosenbrock function
    cdef np.ndarray[DTYPE_t, ndim=1] x0 = np.array([-1.2, 1.0], dtype=np.float64)
    # Run the BFGS optimization using our Cythonized Rosenbrock function and its gradient
    cdef tuple result = bfgs(x0, rosenbrock, rosenbrock_grad)
    cdef np.ndarray[DTYPE_t, ndim=1] x_min = result[0]
    cdef double f_min = result[1]
    
    print("Optimal solution:")
    for i in range(x_min.shape[0]):
        print(f"{x_min[i]:.6f}", end=" ")
    print(f"\nFunction value at minimum: {f_min:.6f}")

if __name__ == "__main__":
    main()
