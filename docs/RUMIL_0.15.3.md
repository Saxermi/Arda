# Arda 0.15.3 — Complex Scalars and Structured Operators

*Development codename: Rúmil*

This release adds complex scalars and principal elementary functions, compact matrix operators, randomized SVD with oversampling and power iterations, and objective-only optimization with coordinate or seeded random directions. `linalg.rsvd` needs `LAPACK=1`. [Runnable example](../examples/structured_computing.arda) · [API contracts and benchmarks](../research/implemented/structured-computing.md)

## Source archive validation

The release suite has 164 regression tests. The portable build runs 26 reference tests and skips five that require LAPACK. The BLAS/LAPACK build and the LAPACK sanitizer build run all 31 reference tests. Local validation uses Debian 13 x86-64, GCC 14.2, OpenBLAS 0.3.29, NumPy 2.3.5 and SciPy 1.17.0.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
make test test-reference check-generated
./arda --gc-stress examples/structured_computing.arda
make BLAS=1 LAPACK=1 test test-reference
./arda --gc-stress examples/structured_computing.arda
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    make BLAS=1 LAPACK=1 SANITIZE=1 test test-reference
```

## Numerical acceptance

- Complex scalar tests compare principal elementary functions against Python `cmath`, exercise the two sides of the negative-real branch cut, and check arithmetic, scaling and division by zero.
- The seeded rectangular operator cases had a maximum relative product discrepancy of $1.12\times10^{-15}$ versus NumPy and SciPy. Tests cover forward and transpose products, strided blocks, immutable factor snapshots, and sparse rollback on product failure.
- The rank-five randomized SVD of a decaying 32-value spectrum had relative Frobenius error 0.226 and a probe estimate of 0.241. The exact rank-one 5,000×6,000 test had a probe error of $4.01\times10^{-15}$. A separate 50,000×60,000 rank-two run used 22.4 MiB of peak process RSS and took 236 ms.
- Coordinate and seeded random-direction optimization reached objectives of $1.33\times10^{-15}$ and $1.63\times10^{-15}$ on the same anisotropic quadratic, using 222 and 1,645 objective evaluations. Both minimizers agreed with the SciPy comparison within $2\times10^{-7}$ per component.

[Regression tests](../tests/test_structured.py) · [Reference tests](../tests/reference_structured.py)
