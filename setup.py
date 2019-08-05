from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext = [Extension("pseudospectral", sources=['pseudospectral.pyx'],
	 include_dirs=[numpy.get_include()])
	 ]

setup(
	name = 'pseudospectral',
	ext_modules = cythonize(ext, annotate=True)
)
