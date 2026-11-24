# Changelog

All notable changes to Arda are recorded here. Planned features are documented in [ROADMAP.md](ROADMAP.md), not listed as implemented.

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
