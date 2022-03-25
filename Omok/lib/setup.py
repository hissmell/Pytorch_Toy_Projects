from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
setup(name = 'c_utils',ext_modules=cythonize('c_utils.pyx',compiler_directives = {'language_level' : '3'})
      ,include_dirs=[np.get_include()])