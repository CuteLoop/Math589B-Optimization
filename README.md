
# Protein Folding Optimization Project Report

## Introduction

Protein folding is the process by which a protein acquires its functional three-dimensional structure. In our simplified model, a protein is represented as a chain of beads in 3D space. The total potential energy of the system is given by


$$ U_{\text{total}} = \sum_{i=1}^{n-1} k_b \left(\|x_{i+1} - x_i\| - b\right)^2 + \sum_{i=1}^{n}\sum_{j=i+1}^{n} 4\epsilon \left[\left(\frac{\sigma}{\|x_j-x_i\|}\right)^{12} - \left(\frac{\sigma}{\|x_j-x_i\|}\right)^6\right],$$

where:
- $\|x_j-x_i\|$ is the Euclidean distance between bead $i$ and bead $j$,
- $k_b$ is the bond stiffness constant and $b$ is the equilibrium bond length,
- $\epsilon$ and $\sigma$ characterize the depth and zero-crossing distance of the Lennard-Jones potential.

Our goal is to optimize the protein configuration by finding a local minimum of this energy function. We use the BFGS optimization algorithm provided by SciPy.

## Methodology

See Usage.md for instructions on how to run code.

We developed 3 implementations:

1. **Vanilla Python + SciPy:**  
   10 particles average: 4.22 s
   

2. **Cython + SciPy:**  
   10 particles average: o.32 s
   100 particles average: 399.5216 

3. cython self implemented BFGS.

