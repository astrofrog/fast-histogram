import os

import numpy
from setuptools import Extension


def get_extensions():
    return [
        Extension(
            "fast_histogram._histogram_core",
            [os.path.join("fast_histogram", "_histogram_core.c")],
            include_dirs=[numpy.get_include()],
        )
    ]
