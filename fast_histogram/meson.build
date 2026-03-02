py.install_sources(
  '__init__.py',
  'histogram.py',
  subdir: 'fast_histogram',
)

# Use run_command to get numpy includes - this is more compatible with limited API
# since dependency('numpy') may add flags that conflict with Py_LIMITED_API
incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

inc_np = include_directories(incdir_numpy)

py.extension_module(
  '_histogram_core',
  '_histogram_core.c',
  include_directories: inc_np,
  limited_api: '3.10',
  install: true,
  subdir: 'fast_histogram',
)
