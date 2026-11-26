# Changelog

All notable changes to Arda are recorded here. Planned features are documented in [ROADMAP.md](ROADMAP.md), not listed as implemented.

## 0.17.0 — Vilya runtime and tensor autodiff

- Extended reverse-mode autodiff to dense Float64 tensors. Backward checks saved tensor versions and publishes gradients after a successful sweep.
- Added lazy Int64 ranges, explicit standard-library imports, structured caught errors, and named reports for iterative solvers.
- Retired redundant dense access names and the legacy axis sum. Moved SGD and Adam to `opt`. Migrated examples and callbacks to the new contracts. [Vilya runtime contracts](docs/VILYA_RUNTIME.md)

## 0.16.5 — Scientific Types

- Added SI quantities with dimension-checked arithmetic, explicit conversion, absolute and interval temperatures, and Float64 tensor products and sums.
- Added metadata-only shape and dtype assertions, 64-bit logical shape specifications, and MathIR notation for straight-line subscripts.
- Extended the bounded direct-evaluation check for near-cancelling inner products to CP, Tucker, and tensor trains. [Scientific type contracts](research/implemented/scientific-types.md) · [Low-rank tensor contracts](research/implemented/low-rank-tensors.md)

## 0.16.4 — Low-Rank Tensors

- Added immutable CP, Tucker, and tensor train representations with element access, contraction, norms, same-format dot products, and explicit materialization.
- Added optional LAPACK TT-SVD and controlled tensor train rounding with a discarded-tail estimate. [Low-rank tensor contracts](research/implemented/low-rank-tensors.md)

## 0.16.3 — Tensor Memory Budgets and Tiled I/O

- Added per-VM tensor buffer budgets, byte counters and a reusable workspace buffer.
- Added explicit block iteration and tile reads and atomic tile writes for tensor files.
- Added workspace planning and streaming mean and population variance. [Memory budgets and streaming](research/implemented/typed-tensors.md#memory-budgets-and-streaming)

## 0.16.2 — Tensor File Storage

- Added read-only memory mapping, bounded block reads and atomic tensor file creation with a per-file byte budget. [Tensor contracts and validation](research/implemented/typed-tensors.md)

## 0.16.1 — Typed Tensors

- Added CPU tensor dtypes, explicit casts, broadcast arithmetic, reductions, batched multiplication, contractions, indexing helpers and complex FFTs. [Tensor contracts](research/implemented/typed-tensors.md)

## 0.16.0 — Shared Tensor Buffers

- Moved legacy Float64 arrays and their views onto shared buffer handles. Added read-only aliases and checked tensor views with signed or zero strides. [View contracts](research/implemented/typed-tensors.md#buffers-and-views)

## 0.15.5 — Mathematical Library

- Added Gamma, Beta, integer-order Bessel and error functions with logarithmic and tail APIs.
- Added bracketed Brent and Newton methods, linear and natural cubic interpolants, and adaptive Gauss–Kronrod quadrature.
- Added Gaussian and exponential density, CDF and survival functions, seeded exponential draws, and batch variance and standard deviation.
- Extended AMDE to describe a single supported range loop as a sum, product or recurrence. [Release validation](docs/RUMIL_0.15.5.md)

## 0.15.4 — Mathematical Documentation (AMDE v1)

- Added static native mathematical documentation through callable queries and the `mathdoc` module. Shape-based cost estimates cover dense solve, matrix multiply, dot product, and CSR matrix–vector multiply. [Contracts](research/implemented/mathdoc.md)

- Added bounded compiler-side MathIR for straight-line user functions, source-level LaTeX views, explicit unresolved call provenance and `--mathdoc=none|compact`. Compact metadata is the default. Level 0 applies limited scalar-real display identities. [Release validation](docs/RUMIL_0.15.4.md)

## 0.15.3 — Complex Scalars and Structured Operators

- Added complex scalars with mixed real arithmetic, principal elementary functions and real and imaginary accessors.
- Added portable low-rank, diagonal, banded, Toeplitz and Kronecker operators with forward and transpose products. Optional LAPACK builds provide randomized SVD with an independent probe error estimate.
- Added `opt.zeroth_order` with central coordinate differences and seeded random directions. [Release validation](docs/RUMIL_0.15.3.md)

## 0.15.2 — Random Generators and Statistics

- Added seeded PCG32 generator objects, uniform, normal, integer and Poisson draws, choice with replacement, and shuffled copies.
- Added median, quantile, sample covariance, Pearson correlation, linear regression, histograms, compensated sums and cumulative sums, extrema indices, clipping, and streaming population moments. [Release validation](docs/RUMIL_0.15.2.md)

## 0.15.1 — SVD, Eigenvalues, and Conditioning

- Added optional LAPACK reduced SVD, pseudoinverse, numerical rank, and 2-norm condition numbers for real Float64 matrices. [Contracts and validation](research/implemented/linear-algebra.md#lapack-reduced-svd)
- Added `linalg.eigh` and `linalg.eig` with optional LAPACK. General eigenpairs expose real and imaginary components. `calc.eigh` shares the symmetric solver and keeps its existing return layout. [Release validation](docs/RUMIL_0.15.1.md)

## 0.15.0 — Dense Factorizations and Direct Solvers

- Added immutable, reusable dense LU, Cholesky, and Householder QR factors and solves for vector and matrix right-hand sides. The one-shot `numc.solve` shares the LU kernel.
- Added determinant, trace, diagonal extraction, inverse via one LU factorization, guarded Thomas tridiagonal solves, and pivoted compact band solves.
- Checked reconstruction, solve residuals, singularity handling, and one-shot `numc.solve` compatibility with the portable, BLAS/LAPACK, and sanitizer builds. [Release validation](docs/RUMIL_0.15.0.md)

## 0.14.4 — Sparse Computing Release Validation (Moria)

- Compared canonical CSR for three million unsorted COO triplets and the full million-row SpMV result against SciPy.
- Measured assembly, COO construction, canonicalization, SpMV, and PCG on the release workloads. Added a scaled SPD reference case and clarified the sparse complexity and convergence documentation.
- See [local validation and reproducible measurements](docs/MORIA_0.14.4.md).

## 0.14.3 — Moria (Sparse Solvers and Memory Budgets)

- CG and Jacobi-PCG for sparse SPD systems. Results include the independently recomputed residual, solution, iteration count, and convergence reason.
- Reusable solver workspace, explicit residual verification, and a configurable budget for managed sparse payload and solver storage.
- See the [Moria 0.14.3 guide](docs/MORIA_0.14.3.md) for API contracts, validation, and local measurements.

## 0.14.2 — Moria (Sparse Algebra and Operators)

- Sparse matrix–vector and sparse–dense matrix multiplication, transpose, addition, scaling, and diagonal extraction.
- Real coordinate Matrix Market general and symmetric I/O, static operation descriptors, and a sparse matvec workspace estimate.
- Shared internal linear operator interface for dense, CSR, and matrix-free Poisson kernels.

## 0.14.1 — Moria (Sparse Assembly)

- Native typed assembly with atomic batch insertion, checked capacity growth, and canonical CSR finalization.
- Direct identity, diagonal, and tridiagonal CSR constructors.
- In-place row sorting bounds canonicalization scratch while preserving the order of duplicate summation.

## 0.14.0 — Moria (Sparse Foundations)

- Native immutable COO and canonical CSR/CRS matrices, checked indices, finite Float64 values and GC-owned, accounted payloads.
- Direct bidirectional COO/CSR conversion without dense intermediates; deterministic duplicate summation, removal of exact zeros and empty-row support.
- Shape, format, nnz, payload bytes, checked lookup, copied components and explicit dense bridges honoring tensor strides and the existing dense limit.
- Versioned static mathematical descriptions of conversion algorithms, preparing the future Rúmil engine.
- Ten new regression methods and one independent SciPy reference method (24 cases), with 136 regular and nine reference methods passing locally in default, BLAS/LAPACK and sanitizer configurations.
- See [Moria specification and validation](docs/MORIA_0.14.0.md). Assembly, algebra, solvers and budgets arrived in subsequent Moria milestones.

## 0.13.1 — Scientific maintenance

- Reformatted the C sources without altering numeric algorithms; added clang-format configuration for future formatting.
- Recoverable lexer, parser and compiler errors with scoped allocation cleanup, including module imports and interactive input.
- Bytecode compilation roots participate in GC reachability through frames, closures and parameter environments. Added `--gc-stats`.
- Enabled ASan/UBSan/LeakSanitizer; six maintenance regression tests (126 total). Eight optional independent numerical reference test methods.
- Per-file object builds, header dependencies, separate backend/sanitizer directories, flag stamps, automated embedded-module generation and check target.
- COO/CSR conversion is specified for 0.14, not implemented in this release.

## 0.13.0 – High-dimensional scientific computing

- Native `high`: Jacobi-PCG, matrix-free d-dimensional Poisson CG, rectangular CGLS, stencil, fused scale/add, tensor inner products and axis reductions; finite checks and RMS RK error estimator.
- Embedded `ode.rk45`: adaptive Dormand–Prince 5(4) for scalar and tensor state.
- Embedded `opt.lbfgs`: limited-memory quasi-Newton solver with user gradients, Armijo search.
- Faster dot (independent extended accumulators), runtime-dispatched AVX2 blocked matrix multiplication with SSE2/scalar fallbacks, strided copy odometer.
- 13 new test methods; 120 total passed across default, sanitizer and BLAS/LAPACK builds. Eight-dimensional manufactured Poisson test with 65,536 unknowns. No GPU, general sparse matrices, or tensor automatic differentiation yet.

## 0.12.0 — Nenya

- Scalar reverse-mode autodiff: GC-traced DAG, iterative postorder and gradient accumulation.
- Native SGD and Adam updates on Float64 parameter vectors; linear regression training example.
- No tensor AD, higher derivatives or GPU yet.

## 0.11.0 — Narya (integrated)

- Composite trapezoidal/Simpson quadrature; callback-based Simpson and classical RK4 in embedded Arda library.
- Native linear-system RK4, radix-2 FFT/IFFT; optional LAPACK DGESV and DSYEV.
- Fixed-step, uniform grids; no adaptive stiff solver or arbitrary FFT lengths.

## 0.10.0 — Anor (integrated)

- Rank 1–8 dense Float64 tensors with shape and strides; GC-traced zero-copy views.
- Trailing-axis broadcasting with zero strides and an outer-axis odometer.
- SSE2 contiguous arithmetic with scalar fallback; retained blocked matmul/CBLAS.
- Explicit reshape, positive-step slicing, permutation, indexing and copy APIs.

## 0.9.0 — Ithil Scientific

- GC-owned dense float64 vectors and matrices, heap accounting and native numerical kernels.
- `numc` module: 26 operations including linear algebra, statistics and elementwise functions.
- Scalar broadcasting, indexing, iteration, optional CBLAS matrix multiplication.
- 11 new test methods, including 9 error subcases, retaining 69 previous tests.
- Regression and linear-system example; reproducible numerical benchmarks.
- Limitations: rank 1/2, no tensor views, no LAPACK, no GPU, no automatic differentiation.

## 0.8.1 — Shadowfax Performance Update
- Prehashed names and stamped table-index inline caches, without cached GC pointers.
- Six fused bytecode instructions for assignment statements, integer-literal arithmetic and comparison branches.
- Inline value helpers, one allocation per function environment with parameter storage, fast GC threshold check.
- Seven additional regression tests; four verified comparative benchmarks; quest example.

## 0.8.0 — Rivendell
- Single inheritance, inherited constructors, method overriding, isinstance.
- Call-time default arguments, lambdas, filtered list comprehensions, list/string slicing.
- Validated UTF-8 text with codepoint indexing, length and iteration.
- Native math, time, text, io, collections modules; numeric conversion functions.

## 0.7.0 — Shadowfax (integrated in 0.8.0)
- Checked signed 64-bit integers and distinct floating-point values.
- Tagged-union Value representation, direct parameter slots, weak string interning.
- Elide scopes without local bindings and redundant if jumps.
- Reproducible benchmarks; regression and sanitizer tests.

## 0.6.0 — The Eldar VM
- Indentation syntax, bytecode compiler, stack VM, incremental marking GC.
- Classes, modules, exceptions, dictionaries, iterators, file I/O.
