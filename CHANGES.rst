0.6 (2019-01-07)
----------------

- Define numpy as a build-time dependency in pyproject.toml. [#33]

- Release the GIL during calculations in C code. [#31]

0.5 (2018-09-26)
----------------

- Fix bug that caused histograms of n-dimensional arrays to
  not be computed correctly. [#21]

- Avoid memory copies for non-native endian 64-bit float arrays. [#18]

- Avoid memory copies for any numerical Numpy type and
  non-contiguous arrays. [#23]

- Raise a better error if arrays are passed to the ``bins`` argument. [#24]

0.4 (2018-02-12)
----------------

- Make sure that Numpy is not required to run setup.py. [#15]

- Fix installation on platforms with an ASCII locale. [#15]

0.3 (2017-10-28)
----------------

- Use long instead of int for x/y sizes and indices

- Implement support for weights= option

0.2.1 (2017-07-18)
------------------

- Fixed rst syntax in README

0.2 (2017-07-18)
----------------

- Fixed segmentation fault under certain conditions.

- Ensure that arrays are C-contiguous before passing them to the C code.

0.1 (2017-07-18)
----------------

- Initial version
