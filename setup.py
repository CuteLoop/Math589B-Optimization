from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="protein_cython",          # This is the module name (protein_cython.so)
        sources=["protein_cython.pyx"],   # Your Cython source file
        include_dirs=[np.get_include()],  # Include NumPy headers if needed
        extra_compile_args=["-O3"],        # Optimize for speed
    )
]

setup(
    name="protein_cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"}  # Use Python 3 syntax
    ),
)
