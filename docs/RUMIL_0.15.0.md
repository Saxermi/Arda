# Arda 0.15.0 — Dense Factorizations and Direct Solvers

*Development codename: Rúmil*

This release adds reusable pivoted LU, Cholesky and Householder QR factors. `linalg.solve` takes a factor and one or more right-hand sides. The existing `numc.solve(A,b)` vector interface shares the pivoted LU kernel. The `linalg` module also provides determinant, trace, diagonal extraction, inverse via triangular solves, and two direct solvers for structured matrices. [Factorization example](../examples/factorizations.arda) · [Direct solver example](../examples/direct_linear_algebra.arda) · [Numerical contracts](../research/implemented/linear-algebra.md)

## Source archive validation

The fresh source archive passed all 152 regression tests and 19 independent NumPy/SciPy reference tests in each configuration below. The portable build also passed `make check-generated` and both linked examples with GC stress enabled. Validation ran locally on Debian 13 x86-64 with GCC 14.2.0, Python 3.12.14, NumPy 2.3.5, and SciPy 1.17.0. The optional build used OpenBLAS 0.3.29. The `linalg` factorization and structured solve kernels are portable C in both builds.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
make test test-reference check-generated
make BLAS=1 LAPACK=1 test test-reference
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    make SANITIZE=1 test test-reference
make arda
./arda --version
./arda --gc-stress examples/factorizations.arda
./arda --gc-stress examples/direct_linear_algebra.arda
```

The reference suite uses the development dependencies in [requirements-dev.txt](../requirements-dev.txt). The optional configuration needs BLAS and LAPACK development libraries.

## Numerical acceptance

The LU reference test uses seeded 12×12 matrices and four right-hand sides. It compares Arda's results with SciPy and with the existing vector solve. Reconstruction error is $\|LU-A[p,:]\|_F/\|A\|_F$. The normalized solve residual is $\|AX-B\|_F/(\|A\|_F\|X\|_F+\|B\|_F)$. Calculations use the Float64 values printed by Arda for the reference test.

| LU case | $\kappa_2(A)$ | Relative reconstruction error | Normalized solve residual | Relative difference from SciPy solution |
| --- | ---: | ---: | ---: | ---: |
| General square | $76.4$ | $2.72\times10^{-15}$ | $3.39\times10^{-16}$ | $1.29\times10^{-15}$ |
| Ill-conditioned | $1.00\times10^{10}$ | $1.74\times10^{-15}$ | $1.51\times10^{-16}$ | $2.71\times10^{-8}$ |
| Symmetric indefinite | $3.0$ | $2.77\times10^{-15}$ | $3.15\times10^{-16}$ | $1.71\times10^{-15}$ |

The ill-conditioned example shows how a small residual can coexist with a larger difference between two computed solutions. Other reference cases check Cholesky reconstruction and SPD solves, Householder QR reconstruction and orthogonality for rectangular and scaled inputs, inverse residuals on both sides, row-scaled Thomas solves, and pivoted band solves. The regression suite verifies that LU, `numc.solve`, `inv`, and the banded solver reject exact singularity, while a singular determinant returns zero. It also checks Cholesky rejection of indefinite input and QR least-squares rejection of rank-deficient input. [Reference tests](../tests/reference_linalg.py) · [Regression tests](../tests/test_linalg.py)

## Performance

The checked-in benchmark compares the portable build with single-threaded SciPy and checks the result of every workload. The synthetic workloads cover dense determinant and inverse, tridiagonal solves versus general dense solves, and a pentadiagonal solve with a forced initial row swap. Reproduce the seven-round comparison from the portable build. [Timing and peak RSS results](../research/implemented/linear-algebra.md#local-direct-solver-benchmark)

```sh
make arda
python3 benchmarks/direct_linalg_bench.py --rounds 7
```
