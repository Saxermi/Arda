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

## 2026-11-24 — Portable reusable dense factorizations

**Status** — Accepted for the 0.15 development checkout

Use partial-pivoted LU for square general systems, lower Cholesky for exactly symmetric SPD matrices, and unpivoted Householder QR for rectangular factors and full-column-rank least squares. Share pivoted LU with the existing vector `numc.solve` API. Keep factors immutable and let callers reuse them across vector or matrix right-hand sides. CholeskyQR and randomized variants warrant investigation for suitable tall matrices after the portable baseline is measured. Their performance claims depend on conditioning, precision and hardware. [LAPACK guide](https://www.netlib.org/lapack/lug/node37.html) · [Recent randomized CholeskyQR analysis](https://arxiv.org/abs/2607.15561) · [Arda methods](research/implemented/linear-algebra.md#reusable-float64-factorizations)

## 2026-11-24 — Guarded tridiagonal and pivoted band solves

**Status** — Accepted for 0.15 development

Use row-normalized Thomas elimination when the tridiagonal coefficients pass an explicit strict row diagonal dominance check. Use compact band storage with partial row pivoting when the caller needs more general diagonals or wider bands. The serial kernels give a portable baseline with workspace proportional to the stored band and held right-hand sides. Parallel algorithms are candidates for later benchmarks. [Contracts and validation](research/implemented/linear-algebra.md#structured-direct-solvers) · [LAPACK band driver](https://www.netlib.org/lapack/explore-html/db/df8/group__gbsv_gaff55317eb3aed2278a85919a488fec07.html) · [Recent parallel band work](https://doi.org/10.1016/j.cam.2025.117254)

## 2026-11-24 — Optional LAPACK SVD and rank cutoff

**Status** — Accepted for 0.15.1 development

Use `DGESVD` as the first rectangular Float64 SVD backend. It provides reduced vectors and a values-only mode with a linear minimum workspace bound. `DGESDD` uses divide and conquer for the vector path and can trade additional workspace for speed. Rank and pseudoinverse use a common relative cutoff. The condition number uses the raw computed singular-value ratio so users can distinguish conditioning from numerical truncation. Mixed-precision preconditioned Jacobi SVD is a candidate when accuracy of small singular values is a workload requirement. Its reported benefits depend on the cost of the higher-precision preconditioner and need independent validation on Arda's target machines. [Arda validation](research/implemented/linear-algebra.md#lapack-reduced-svd) · [DGESVD](https://www.netlib.org/lapack/explore-html/d1/d7f/group__gesvd_gac6bd5d4e645049e49bb70691180abf07.html) · [DGESDD](https://www.netlib.org/lapack/explore-html/df/d22/group__gesdd_ga8941e5ff50de36580dae8940015e9cb0.html) · [Zhou, Tisseur, and Webb 2026](https://arxiv.org/abs/2602.18134v2)

## 2026-11-24 — LAPACK eigenpair representation

**Status** — Accepted for 0.15.1

Share the existing `DSYEV` contract between `calc.eigh` and `linalg.eigh`. Use `DGEEV` and separate real and imaginary tensors for general right eigenpairs. This allows complex results on the current Float64 runtime. Mixed-precision Jacobi methods with relative-gap error bounds merit comparison for symmetric positive definite problems. The general nonsymmetric solver uses the established Hessenberg/QR LAPACK path. [Eigenpair contracts](research/implemented/linear-algebra.md#lapack-eigensolvers) · [Zhou, Tisseur, and Webb 2026](https://arxiv.org/abs/2606.28069)
