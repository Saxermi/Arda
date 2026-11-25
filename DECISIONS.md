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

## 2026-11-24 — Seeded sampling and serial statistics

**Status** — Accepted for 0.15.2

Use explicit PCG32 generators with a versioned seed and stream contract, unbiased integer rejection, Box–Muller normal draws, and Poisson product sampling or transformed rejection by mean. Use sorting for exact linear quantiles, shifted Welford updates for moments and Neumaier compensation for sums. Counter-based streams and pairwise-mergeable moments are candidates for parallel execution. For batch throughput, compare selection-based quantiles and faster normal samplers at matched statistical accuracy. Changing the sampler can change seeded sequences, so tie such a change to an explicit version contract. [Arda methods and measurements](research/implemented/random-and-statistics.md) · [Research on compensated parallel sums](https://doi.org/10.1002/cpe.70886)

## 2026-11-24 — Structured operators and sampled approximation

**Status** — Accepted for 0.15.3

Represent complex scalars with two Float64 components and delegate principal elementary functions to C11 libm. Keep matrix operators compact and share the forward and transpose product interface with dense and sparse inputs. Use Gaussian range finding, reorthogonalized subspace iteration, and the existing optional LAPACK SVD for the projected matrix. A separate stream estimates residual error so callers can judge a chosen rank. Block Krylov is a candidate for tighter spectra when a larger basis fits the memory budget. For objective-only optimization, start with central coordinate differences and seeded random directions, and record objective evaluations. Revisit orthogonal or adaptive directions on high-dimensional objectives after measuring convergence and estimator variance. [Contracts and measurements](research/implemented/structured-computing.md)

## 2026-11-24 — AMDE v1 metadata, call provenance and simplification

**Status** — Accepted by the project owner and locally validated for 0.15.4. [Release validation and overhead measurements](docs/RUMIL_0.15.4.md)

Enable bounded compact user-function MathIR by default; retain `--mathdoc=none` to skip collection. Dynamic calls remain opaque source expressions, not inferred native identities. This preserves static-query non-execution and avoids resolving mutable runtime bindings.

Allow bounded, display-only formal scalar-real identities in level 0 for call-free, external-free expressions: integer additive/multiplicative identity literals, double negation and repeated parameter factors. Levels 1 and 3 preserve source operations; assumptions distinguish the mathematical display from runtime type, domain, overflow and rounding semantics. Do not remove arbitrary operands, cancel denominators, fold numerical operations or change the executable IR. Broader symbolic algebra remains in 0.24.3.

**Revisit** — Measure compilation time, RSS and execution overhead with compact metadata enabled and disabled; expand simplification only with documented assumptions and regression evidence. [Contracts](research/implemented/mathdoc.md#limited-mathematical-simplification-accepted-0154-policy)

## 2026-11-24 — Scalar numerical methods and range-loop documentation

Use libm for established real special functions. Combine the log-Gamma terms analytically for unbalanced Beta arguments to reduce cancellation. Use sign brackets for Brent and safeguarded Newton. The natural cubic spline has a linear-cost tridiagonal setup. Gauss–Kronrod quadrature reuses samples between its two rules and uses a panel budget with explicit failure. Keep the older Simpson routine for uniform fixed grids. [Library contracts and method comparisons](research/implemented/mathematical-library.md)

AMDE models one source loop with a local accumulator and a literal nonzero step. Check that the captured `range` binding resolves to the native callable at query time. Render pure additions and multiplications in closed sum and product notation, and preserve the ordered update in recurrence and source views. The model uses metadata independent of the number of iterations. [Range-loop contract](research/implemented/mathdoc.md#single-range-loops)

## 2026-11-24 — CPU tensor buffers and file access

Share one GC-traced buffer across views. Keep the existing Float64 kernels on the legacy array interface and expose explicit materialization from the typed tensor API. Arbitrary strided views are read-only so overlapping coordinates have unambiguous mutation behavior.

Keep the first disk format contiguous. Map stable files read-only or read explicit flat blocks with `pread` when truncation must be recoverable. Publish complete replacement files by same-directory rename after flushing the file. This allows existing mappings to retain their old inode. [Tensor and storage contracts](research/implemented/typed-tensors.md)

## 2026-11-25 — Budgeted tensor workspace and tile writes

Charge heap-backed tensor payloads and native workspace to one per-VM budget. Report mappings separately so logical file size can grow without changing the heap limit. Keep one idle scratch allocation for equal-size reuse under the budget. Streaming mean and population variance use a single reservation and a shifted Welford accumulator.

Publish tile changes as complete replacement files in 0.16.3. This preserves existing read-only maps and file iterators and keeps the original file available if publication fails. Each edit takes a full file copy and bounded scratch. [Memory and tile contracts](research/implemented/typed-tensors.md#memory-budgets-and-streaming)
