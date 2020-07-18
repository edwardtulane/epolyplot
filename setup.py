from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
  name = 'cythonised PAD averaging',
  ext_modules = cythonize("epolyplot/epC.pyx"),
  include_dirs=[numpy.get_include()]
)
