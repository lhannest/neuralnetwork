from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

filename = 'convolution.pyx'

extensions=[
    Extension('convolution',
             [filename],
             include_dirs=[numpy.get_include()],
             extra_compile_args=["-w"]
            )
]

setup(ext_modules=cythonize(extensions))