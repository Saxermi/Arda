# Arda 0.14.4 — Sparse Computing Release Validation

*Development codename: Moria*

This release closes S9. The sparse numerical contracts are documented in the [0.14.0 foundations report](MORIA_0.14.0.md), the [0.14.3 sparse computing guide](MORIA_0.14.3.md), and the [scientific notes](../research/implemented/sparse-matrices.md).

## Local release checks

The source distribution is checked from a fresh extraction. Run `make test`, `make test-reference`, and `make check-generated` in the default build, repeat the test suites with `BLAS=1 LAPACK=1`, then run `make sanitize` and the reference suite with `SANITIZE=1`. The suite exercises Matrix Market exchange, GC stress, duplicate cancellation, budget failure recovery, and SciPy comparisons across seeded shapes and SPD systems. The scaled SPD reference includes an independently evaluated residual and checks the solution in the original scaled coordinates.

## Benchmark gates

Run on Debian 13, x86-64 Intel Xeon Platinum 8573C, GCC 14.2.0, default C11 `-O2` build, Python 3.12, NumPy 2.3.5 and SciPy 1.17.0. Each measured process was pinned to logical CPU 1, with OpenBLAS and OpenMP restricted to one thread. Timings are medians of five processes. RSS reports the highest Linux process high-water mark across those runs.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/moria_bench.py \
    --size 600000 --solver-size 1000000 --rounds 5 --scipy --cpu 1
```

| Gate | Observed result | Criterion |
| --- | --- | --- |
| S1 — direct CSR and SpMV, 1,000,000 rows | 88.81 MiB maximum RSS, 31.76 ms CSR build and 9.72 ms SpMV. All 1,000,000 output entries compared with SciPy. | Below 256 MiB and independent SpMV agreement |
| S2 — Jacobi-PCG, 1,000,000 rows | 134.58 MiB maximum RSS, 247.98 ms, 12 iterations, relative residual 5.79e-10 and relative solution error 3.59e-10 | Below 512 MiB, relative residual at most 1e-8, and solution error documented |
| S3 — 3,000,000 unsorted COO triplets | Both assembly paths returned 1,200,000 canonical entries matching SciPy exactly in row order, columns and values. | Record assembly and conversion separately and check canonical structure |

The large SPD workload uses `tridiag(-1,4,-1)`, a known solution with values linearly spaced from 0.5 to 1.5, and a relative solver tolerance of 1e-9. SciPy took 149.08 ms for PCG and its final independent residual check with the same iteration count and measured error. Arda includes its structural checks. The CSR and PCG tests build matrices sparsely.

| S3 method | Assemble entries | Construct COO | Canonicalize to CSR | Sum of timed stages | Maximum process RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| List → COO → CSR | 1395.97 ms | 136.89 ms | 66.05 ms | 1646.79 ms | 281.52 MiB |
| Typed builder → CSR | 875.64 ms | Included in assembly | 84.62 ms | 957.04 ms | 231.13 MiB |

The list case explicitly collects the staging lists before canonicalization. The aggregate median is computed from the sum of the timed stages per run, so it can differ from the sum of stage medians. Process RSS also includes the subsequent SpMV. Full SciPy result comparisons run separately and do not enter the reported time or RSS. No hosted CI jobs are used.
