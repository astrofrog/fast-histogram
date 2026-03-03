py.install_sources(
  '__init__.py',
  'histogram.py',
  subdir: 'fast_histogram',
)

py.install_sources(
  'tests/__init__.py',
  'tests/test_histogram.py',
  subdir: 'fast_histogram/tests',
)

py.extension_module(
  '_histogram_core',
  '_histogram_core.c',
  dependencies: dependency('numpy'),
  limited_api: '3.10',
  install: true,
  subdir: 'fast_histogram',
)
