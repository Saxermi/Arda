# Arda decision log

Record the question, evidence, decision or recommendation, and when to revisit it. Use **Proposed** while a decision is open and **Accepted** once it is agreed. Link to the benchmark or roadmap that supports each entry.

## 2026-11-24 — Sparse PCG performance and numerical checks

**Status** — Proposed

**Question** — Should Moria change its sparse solver in response to the SciPy benchmark?

On the single-threaded, million-row tridiagonal test, Arda and SciPy both converged in 12 iterations with a relative residual of about 5.8 × 10⁻¹⁰. Arda's PCG call took 255 ms and SciPy's took 127 ms. Standalone SpMV in the solver runs took 8.89 ms and 3.36 ms per call, respectively. Arda includes structural checks and a true-residual computation in its solver timing. The benchmark verifies SciPy's true residual after stopping its solver timer. The timings show where to investigate, but they do not isolate the cost of validation or extended-precision accumulation. [Moria benchmark results](docs/MORIA_0.14.3.md) · [Benchmark harness](benchmarks/moria_bench.py)

**Recommendation** — Keep the Moria defaults, including the recomputed residual, breakdown detection, symmetry and diagonal checks, and extended-precision CSR accumulation. The result contract is useful to callers, and one well-conditioned matrix is too narrow a basis for changing the numerical default.

**Revisit** — During the 0.18.0 kernel benchmarking work, time setup, SpMV, vector updates, and final verification separately. Include final verification in both end-to-end timings and report it separately as well. For repeated solves with the same immutable CSR matrix, measure whether reusing the symmetry check and Jacobi diagonal helps. Compare a Float64 SpMV prototype with the current kernel on representative SPD matrices, including ill-conditioned and cancellation-heavy cases. Consider adopting the faster kernel if it delivers a reproducible improvement while preserving residual and solution-error tolerances and breakdown reporting. [CPU kernel roadmap](ROADMAP.md#arda-018--high-performance-cpu-execution-and-adaptive-planning-ake-v1-ace-v1)

## 2026-11-24 — Dense Float64 matrix multiplication

**Status** — Accepted

Dense matrix multiplication optimization belongs in 0.18. Establish a reproducible baseline in 0.18.0 for the default and OpenBLAS builds against Eigen and NumPy. In 0.18.1, require a reproducible speedup for the default build at 512 × 512 and 1024 × 1024 while preserving numerical tolerances and applying the existing 10% regression target to the 128 × 128 case. [Roadmap update](https://github.com/Saxermi/Arda/commit/d160a8c97a3a6518b8db267862c64229102b432d)
