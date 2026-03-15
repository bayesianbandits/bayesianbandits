import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

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
