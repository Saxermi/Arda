# Arda 0.15.5 — Mathematical Library

*Development codename: Rúmil*

This release adds scalar special functions, bracketed roots, linear and natural cubic interpolants, adaptive Gauss–Kronrod integration, probability functions, exponential sampling and batch dispersion statistics. AMDE documents one supported native range loop as a sum, product or recurrence. [Numerical contracts](../research/implemented/mathematical-library.md) · [AMDE contracts](../research/implemented/mathdoc.md#single-range-loops) · [Runnable example](../examples/mathematical_library.arda)

## Local validation

The extracted release source was tested on Debian 13 x86-64 with GCC 14.2.0, Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0. Optional builds used single-threaded OpenBLAS 0.3.29. Reference skips in the portable builds cover LAPACK-dependent APIs.

| Build | Regression passed | Reference passed | Reference skipped |
| --- | ---: | ---: | ---: |
| Portable | 189 | 32 | 5 |
| BLAS/LAPACK | 189 | 37 | 0 |
| Portable with ASan, UBSan and leak detection | 189 | 32 | 5 |
| BLAS/LAPACK with ASan, UBSan and leak detection | 189 | 37 | 0 |

The clang-format 19.1.7 and generated-header checks passed. The mathematical library example ran with GC stress in both metadata modes. The four loop fragments compiled with `pdflatex`. Sanitizer builds exercised cleanup after a frontend error with loop metadata present and reported the expected error. GCC emitted the existing `-Wclobbered` warning for `self` in `src/vm.c`.

The numerical tests include Gamma near poles, extreme Beta ratios, Bessel orders near the turning point, directly evaluated probability tails, root brackets, nonuniform spline knots and integration of a sampled endpoint singularity. For $B(0.5,10^6)$, SciPy 1.17 differs from the 90-digit mpmath reference by about $9.6\times10^{-10}$ relative. The test stores the high-precision result as a fixture. The loop tests include descending and empty ranges, unsupported control flow, rebound `range` names, metadata limits and collection of bound queries. [Independent reference tests](../tests/reference_mathematical_library.py) · [Loop tests](../tests/test_mathdoc.py)

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
make test test-reference check-generated format-check
./arda --gc-stress examples/mathematical_library.arda
./arda --gc-stress --mathdoc=none examples/mathematical_library.arda
make BLAS=1 LAPACK=1 test test-reference
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1
make SANITIZE=1 test test-reference
make BLAS=1 LAPACK=1 SANITIZE=1 test test-reference
```

The next roadmap task is the Arda Wiki, followed by 0.16. [Roadmap](../ROADMAP.md)
