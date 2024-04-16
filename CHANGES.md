## v0.13 - 2024-04-16

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

The main change in this release is added support for Numpy 2.0.

#### Other Changes

* Clean up test matrix by @pllim in https://github.com/astrofrog/fast-histogram/pull/82
* Use v1 workflow, not main by @pllim in https://github.com/astrofrog/fast-histogram/pull/79
* Fix link to benchmark plot in README by @pllim in https://github.com/astrofrog/fast-histogram/pull/78
* Add PyPI badge by @pllim in https://github.com/astrofrog/fast-histogram/pull/75
* Remove wheel from build requirements by @pllim in https://github.com/astrofrog/fast-histogram/pull/74
* Build against Numpy 2.0.0rc1 for Python>=3.9 by @astrofrog in https://github.com/astrofrog/fast-histogram/pull/86

### New Contributors

* @pre-commit-ci made their first contribution in https://github.com/astrofrog/fast-histogram/pull/70
* @pllim made their first contribution in https://github.com/astrofrog/fast-histogram/pull/82

**Full Changelog**: https://github.com/astrofrog/fast-histogram/compare/v0.12...v0.13

## 0.12 (2023-10-17)

- Fix segmentation faults under various conditions. [#62, #66]

## 0.11 (2022-05-23)

- Use Python limited API to produce forward-compatible wheels. [#56]

## 0.10 (2021-09-06)

- Add function for histograms in arbitrarily high dimensions. [#54, #55]

## 0.9 (2020-05-24)

- Fixed a bug that caused incorrect results in the weighted 1-d histogram and the weighted and unweighted 2-d histogram functions if using arrays with different layouts in memory. [#52]

## 0.8 (2020-01-07)

- Fixed compatibility of test suite with latest version of the hypothesis package. [#40]

## 0.7 (2019-01-09)

- Fix definition of numpy as a build-time dependency. [#36]

## 0.6 (2019-01-07)

- Define numpy as a build-time dependency in pyproject.toml. [#33]
- Release the GIL during calculations in C code. [#31]

## 0.5 (2018-09-26)

- Fix bug that caused histograms of n-dimensional arrays to not be computed correctly. [#21]
- Avoid memory copies for non-native endian 64-bit float arrays. [#18]
- Avoid memory copies for any numerical Numpy type and non-contiguous arrays. [#23]
- Raise a better error if arrays are passed to the `bins` argument. [#24]

## 0.4 (2018-02-12)

- Make sure that Numpy is not required to run setup.py. [#15]
- Fix installation on platforms with an ASCII locale. [#15]

## 0.3 (2017-10-28)

- Use long instead of int for x/y sizes and indices
- Implement support for weights= option

## 0.2.1 (2017-07-18)

- Fixed rst syntax in README

## 0.2 (2017-07-18)

- Fixed segmentation fault under certain conditions.
- Ensure that arrays are C-contiguous before passing them to the C code.

## 0.1 (2017-07-18)

- Initial version
