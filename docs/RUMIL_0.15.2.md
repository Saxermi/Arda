# Arda 0.15.2 — Random Generators and Statistics

*Development codename: Rúmil*

The portable `random` module adds explicit seeded PCG32 generators and scalar or batched sampling. The portable `stats` module adds descriptive statistics and a streaming accumulator for population moments. [Runnable example](../examples/random_statistics.arda) · [API contracts](../research/implemented/random-and-statistics.md)

## Source archive validation

Each configuration has 158 regression tests. The LAPACK build and its sanitizer build run all 27 reference tests. The portable build runs 23 and skips four LAPACK-only cases. The validation environment used Debian 13 x86-64, GCC 14.2, OpenBLAS 0.3.29, Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
make test test-reference check-generated
./arda --gc-stress examples/random_statistics.arda
make BLAS=1 LAPACK=1 test test-reference
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    make BLAS=1 LAPACK=1 SANITIZE=1 test test-reference
```

## Numerical acceptance

The regression suite reproduces the first six PCG32 words for seed 42 and stream 54, verifies independent generator state, and checks that a failed batch leaves its generator at the same position. The reference suite compares 12,000 draws from each of the uniform and normal distributions against SciPy's CDFs. The Kolmogorov–Smirnov distances were 0.00845 and 0.00685. For 14,000 draws across seven equally likely integers, Pearson's chi-square statistic was 4.8. The maximum empirical CDF distance for Poisson was 0.0103 across seven tested means from 0.125 to $10^8$, with 12,000 draws per mean.

Median, quantile, covariance, and correlation agreed with NumPy or SciPy within a maximum absolute difference of $1.78\times10^{-15}$ on the seeded strided-vector case. The same suite checks regression, histogram, cumulative sums, clipping and online moments. Separate cases recover a unit-sized term after cancellation at magnitude $10^{100}$ and calculate population variance $0.125$ about a mean of $10^{15}$. [Regression tests](../tests/test_probability.py) · [Reference tests](../tests/reference_probability.py) · [Benchmark results](../research/implemented/random-and-statistics.md#local-performance)
