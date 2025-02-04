from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="full_optimization",            # The name of the extension module.
        sources=["full_optimization.pyx"],   # The source Cython file.
        include_dirs=[numpy.get_include()],  # Include the NumPy header directory.
        extra_compile_args=["-O3"],           # Optimization flag (may be ignored by some compilers, e.g. MSVC).
    )
]

setup(
    name="FullOptimizationModule",          # Your package/module name.
    version="0.1",
    description="Cython module for protein energy optimization and BFGS algorithm",
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"}
    ),
    zip_safe=False,
)
