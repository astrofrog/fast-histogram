import os
import io

# Note that numpy is included as a build-time dependency in pyproject.toml as
# described in PEP 518. This works with pip 10.x and later. For older version
# of pip, one could in principle use setup_requires=['numpy'] in the setup call
# below but this can cause issues since setup_requires is honored by easy_install
# rather than pip, and this can mean picking up pre-releases. See
# https://mail.python.org/pipermail/numpy-discussion/2019-January/079097.html
# for more details.
try:
    import numpy
except ImportError:
    raise ImportError("Numpy is required to install this package - either "
                      "install it first or update to pip 10.0 or later for it "
                      "to be automatically installed")

from setuptools import setup
from setuptools.extension import Extension

extensions = [Extension("fast_histogram._histogram_core",
                        [os.path.join('fast_histogram', '_histogram_core.c')],
                        include_dirs=[numpy.get_include()])]

with io.open('README.rst', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setup(name='fast-histogram',
      version='0.6',
      description='Fast simple 1D and 2D histograms',
      long_description=LONG_DESCRIPTION,
      install_requires=['numpy'],
      author='Thomas Robitaille',
      author_email='thomas.robitaille@gmail.com',
      license='BSD',
      url='https://github.com/astrofrog/fast-histogram',
      packages=['fast_histogram', 'fast_histogram.tests'],
      ext_modules=extensions)
