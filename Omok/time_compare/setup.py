from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
setup(ext_modules=cythonize('c_check_action_type.pyx',compiler_directives = {'language_level' : '3'})
      ,include_dirs=[np.get_include()])