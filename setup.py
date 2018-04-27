import os
import io

from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext


class build_ext_with_numpy(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        import numpy
        self.include_dirs.append(numpy.get_include())


extra_compile_args = []
# Uncomment these lines to enable loop vectorization reports from the compiler
# if os.name == 'nt':
#       extra_compile_args = ['/Qvec-report:2']
# else:
#       extra_compile_args = ['-fopt-info-vec', '-fopt-info-vec-missed', '-fdiagnostics-color=always']

extensions = [Extension("fast_histogram._histogram_core",
                       [os.path.join('fast_histogram', '_histogram_core.c')],
                       extra_compile_args=extra_compile_args
                       )]

with io.open('README.rst', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setup(name='fast-histogram',
      version='0.5.dev0',
      description='Fast simple 1D and 2D histograms',
      long_description=LONG_DESCRIPTION,
      setup_requires=['numpy'],
      install_requires=['numpy'],
      author='Thomas Robitaille',
      author_email='thomas.robitaille@gmail.com',
      license='BSD',
      url='https://github.com/astrofrog/fast-histogram',
      packages=['fast_histogram', 'fast_histogram.tests'],
      ext_modules=extensions,
      cmdclass={'build_ext': build_ext_with_numpy})
