project('purelib-and-platlib', 'c')

py = import('python').find_installation(pure: false)

npy_include_path = run_command(py, ['-c', 'import os; import numpy; print(os.path.abspath(numpy.get_include()))'],
                         ).stdout().strip()

npy_include = include_directories(npy_include_path)

py.extension_module(
    'fast_histogram._histogram_core',
    'fast_histogram/_histogram_core.c',
    limited_api: '3.9',
    include_directories: [npy_include],
    install: true,
)
