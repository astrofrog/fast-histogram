import os
import io

import numpy as np
from setuptools import setup
from setuptools.extension import Extension

extensions = [Extension("fast_histogram._histogram_core",
                        [os.path.join('fast_histogram', '_histogram_core.c')],
                        include_dirs=[np.get_include()])]

with io.open('README.rst', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setup(name='fast-histogram',
      version='0.4.dev0',
      description='Fast simple 1D and 2D histograms',
      long_description=LONG_DESCRIPTION,
      install_requires=['numpy'],
      author='Thomas Robitaille',
      author_email='thomas.robitaille@gmail.com',
      license='BSD',
      url='https://github.com/astrofrog/fast-histogram',
      packages=['fast_histogram', 'fast_histogram.tests'],
      ext_modules=extensions)
