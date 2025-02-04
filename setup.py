from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="protein_cython",                # module name without a package prefix
        sources=["protein_cython.pyx"],         # path to the .pyx file
        include_dirs=[np.get_include()],        # include NumPy headers
    )
]

setup(
    name="protein_cython",
    version="0.1",
    description="Protein simulation module using Cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"},  # for Python 3 syntax
    ),
)
