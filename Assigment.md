
# Math589B Assignment 1: Protein Folding Optimization Project

This assignment is purely educational and focuses on optimizing the conformation of a simplified protein model. In our model, a protein is represented as a chain of beads connected by spring-like bonds. Non-bonded beads interact via the Lennard-Jones potential, which captures both the attractive and repulsive forces acting between them.

---

## Background

Protein folding is the process by which a protein attains its functional three-dimensional structure. In nature, the folded state corresponds to a local minimum of the protein’s potential energy, which balances various forces such as van der Waals interactions, electrostatics, and bond constraints. Solving the full protein folding problem is computationally intensive, so simplified models (like the one used in this assignment) allow us to study essential aspects of the folding process.

The total potential energy of the system is modeled as

\[
U_{\text{total}} = \sum_{i=1}^{n-1} k_b \left(\|x_{i+1} - x_i\| - b\right)^2 + \sum_{i=1}^{n}\sum_{j=i+1}^{n} 4\epsilon \left[\left(\frac{\sigma}{\|x_j-x_i\|}\right)^{12} - \left(\frac{\sigma}{\|x_j-x_i\|}\right)^6 \right],
\]

where:

- \(\|x_j-x_i\|\) is the Euclidean distance between bead \(i\) and bead \(j\),
- \(k_b\) is the bond stiffness constant,
- \(b\) is the equilibrium bond length,
- \(\epsilon\) is the depth of the Lennard-Jones potential well,
- \(\sigma\) is the distance at which the Lennard-Jones potential is zero.

---

## Project Objectives

The main goals of this assignment are to:

- **Implement a Custom BFGS Algorithm:**  
  Replace the provided BFGS optimizer in `protein_folding_3d.py` with your own implementation.

- **Scale-Up with C Integration:**  
  Optimize computationally intensive operations (such as pairwise distance calculations and energy/gradient evaluations) by writing these parts in C and integrating them with Python (using libraries such as `ctypes`, `cffi`, or `cython`).

- **Handle Different Protein Sizes:**  
  Compute optimized configurations for protein chains with \( n_{\text{beads}} = 10 \), \( 100 \), and \( 500 \). The solution for \( n_{\text{beads}} = 500 \) must complete within 600 seconds of CPU time on the Gradescope virtual machine.

---

## Requirements and Deliverables

Your submission must include the following:

1. **Code Repository on GitHub:**  
   - Fork the provided startup code.
   - Commit all changes and push to GitHub.
   - Submit your modified GitHub repository to Gradescope.

2. **Python Script Integrating C Code:**  
   - The main file, `protein_folding_3d.py`, must implement a function `optimize_protein` that initializes the beads (using `initialize_protein`) and performs the optimization using your custom BFGS.
   - Your code should produce three CSV files: `protein10.csv`, `protein100.csv`, and `proteing500.csv`, containing the final configurations.

3. **C Code and Makefile:**  
   - Write the performance-critical portions (energy calculations, gradients, and BFGS updates) in C.
   - Provide a Makefile to compile the C code (e.g., generating a shared library such as `libenergy.so`).

4. **Report:**  
   - A brief report (in GitHub Markdown format) explaining your implementation:
     - Describe your BFGS algorithm and any challenges you encountered.
     - Explain how you optimized the code for scalability using C.
     - Include runtime performance analysis for each configuration.
   - The report should be included as part of this README or as a separate Markdown file.

---

## Evaluation Criteria

Your project will be evaluated based on:

- **Correctness:**  
  Does your solution converge to a valid local minimum of the potential energy?

- **Efficiency:**  
  Can the configuration for \( n_{\text{beads}} = 500 \) be computed within 600 seconds on the Gradescope VM?

- **Code Quality:**  
  Is your code clean, modular, and well-documented?

- **Report Quality:**  
  Is your report clear and does it address all required points?

---

## Implementation Hints

- **BFGS Algorithm:**  
  Refer to standard pseudocode for BFGS. Make sure your implementation efficiently calculates the inverse Hessian approximation and performs a proper line search (e.g., using the Armijo condition).

- **Python-C Integration:**  
  Use tools like `ctypes`, `cffi`, or `cython` to integrate your C code. Ensure your arrays are C-contiguous for performance reasons.

- **Optimization:**  
  Consider optimizing pairwise distance calculations with spatial partitioning techniques (e.g., bounding boxes or cell lists) to speed up the Lennard-Jones potential computations.

- **Benchmarking:**  
  Use Python’s `time.perf_counter` and `cProfile` modules to benchmark and record the performance of your code.

---

## Autograding Details

- The autograder will call your function `optimize_protein` (located in `protein_folding_3d.py`) to verify that it finds a valid local minimum.
- **Important:** Do not modify the function `initialize_protein`. Testing will be performed with beads initially arranged in a straight line, one unit apart.
- Plotting routines should not execute by default (as the autograder runs headless). Control plotting via extra keyword arguments or a global flag.

---

## Current Repository Status

- **Code Files:**  
  - `bfgs.py`: Contains the BFGS algorithm implementation (based on textbook code).
  - `energy.cpp`, `energy.hpp`: C++ implementation for energy calculations.
  - `grad_w_armijo.cpp`: Contains a gradient descent algorithm with Armijo backtracking.
  - `protein_folding_3d.py`: Main Python script for protein optimization and CSV output.
  - Additional files for different BFGS variants (`bfgs_w_classes.cpp`, `bfgs_w_varargs.c`) are included for reference and experimentation.
  
- **Images:**  
  - A graph of the Lennard-Jones potential is provided in the `images` directory.

- **Makefile:**  
  - Automates the compilation of C/C++ code and building of executables and shared libraries.

---

## Report of Findings

*Write your detailed report here in GitHub Markdown format, addressing the following:*

- **Implementation of BFGS:**  
  Describe your approach, key decisions made, and any difficulties encountered during development.

- **C-Based Optimizations:**  
  Explain how you integrated C code with Python, which functions were offloaded for performance reasons, and how these changes affected scalability.

- **Runtime Performance Analysis:**  
  Present timing results for \( n_{\text{beads}} = 10 \), \( 100 \), and \( 500 \). Explain your benchmarking methodology (e.g., using `time.perf_counter` and `cProfile`), and include any observations on performance improvements.

