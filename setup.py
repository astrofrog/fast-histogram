import os

import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [Extension("fast_histogram.histogram_cython",
                        [os.path.join('fast_histogram', 'histogram_cython.pyx')],
                        include_dirs=[np.get_include()])]

setup(name='fast-histogram',
      version='0.1.dev0',
      install_requires=['numpy', 'Cython'],
      author='Thomas Robitaille',
      author_email='thomas.robitaille@gmail.com',
      license='BSD',
      url='https://github.com/astrofrog/fastogram',
      packages=['fast_histogram'],
      ext_modules=cythonize(extensions))
