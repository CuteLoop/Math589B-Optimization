from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="protein_cython",
        sources=["protein_cython.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    )
]

setup(
    name="protein_cython",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
