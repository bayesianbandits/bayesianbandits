from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        [
            Extension(
                "bayesianbandits._takahashi",
                ["src/bayesianbandits/_takahashi.pyx"],
                include_dirs=[numpy.get_include()],
            )
        ],
    ),
)
