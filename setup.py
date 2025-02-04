from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="protein_cython",
        sources=["protein_cython.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],  # Use optimization flag
    )
]

setup(
    name="ProteinCythonModule",
    ext_modules=cythonize(extensions),
)
