# Aulë numerical solvers

## 0.19.0 operator and result contracts

`linop.info(A)` reports shape, Float64 dtype, apply workspace, COO conversion workspace and
transpose availability. `linop.apply(A,x)` and `linop.apply_transpose(A,x)` use the same
operator abstraction as the solvers. Supported representations are rank-two Float64
`numc` and typed tensors, COO/CSR, and the native structured and Poisson operators.
The operator may read a signed-stride Float64 view directly. Solvers accept one Float64
vector as the right-hand side and return a `numc` vector. COO is canonicalized to CSR within
the existing sparse allocation budget. Structured operands and matrix-free stencils are
applied directly. [Operator source](../../src/linop.c) · [Structured kernels](structured-computing.md#portable-matrix-operators)

```arda
tol solver as s
tol linop as l
tol numc as n
esta op = l.poisson([3])
esta b = n.array([0,0,4])
esta result = s.solve("cg",op,b,{"tol":1e-10,"trace":10})
pedo(result["converged"],result["residual"],result["reason"])
pedo(s.verify(op,b,result["solution"]))
```

The common result fields include `solution`, `iterations`, `residual`, `normal_residual`,
`converged`, `breakdown`, `reason`, `method`, `preconditioner`, `tolerance` and
`workspace_bytes`. The final relative residual is `||b-Ax||₂ / ||b||₂`, with an absolute
residual when the denominator is zero. Least-squares methods also report
`||Aᵀ(b-Ax)||₂ / ||Aᵀb||₂` with the same zero-denominator convention. A least-squares result
may succeed at the normal-equation tolerance for an inconsistent system. Convergence is
checked against recomputed products. Reported stops include `converged`, `least_squares`,
`max_iterations`, `numerical_breakdown`, `nonpositive_curvature` and `stagnation`.
Invalid arguments, incompatible options, and preconditioner setup errors raise error maps.

Common options are `tol` (default 1e-8), `maxiter` (method-specific bounds), `trace` (maximum
entries, 0–10,000) and `workspace_bytes` (default 256 MiB). The workspace covers numeric
scratch, including packed right-hand sides, Krylov basis, factors and operator apply scratch.
The tensor budget also covers outputs. The COO setup estimate is exposed separately, and
the sparse budget charges that conversion. Trace maps are VM objects. Allocated solver
tensor scratch is cleared on return or exception. The existing `high`, `sparse` and `linalg`
entry points remain available. User-defined callbacks are an optional future extension.

## 0.19.1 Krylov methods, preconditioning and partial spectra

| Method | Operator requirements | Storage and numerical details |
| --- | --- | --- |
| CG, including PCG | Symmetric positive definite | Fixed vector count. Exact symmetry and positive diagonal checks on explicit coefficients, plus curvature checks at runtime. |
| MINRES | Symmetric, including indefinite | Short symmetric Lanczos recurrence with Givens rotations. The initial implementation uses the unpreconditioned Euclidean norm. |
| Restarted GMRES | Square with general coefficients | A restart length of 1–256, default 30. Two modified Gram–Schmidt passes, left preconditioning and a final true residual. Workspace grows as `O(nk + k²)` for restart `k`. |
| BiCGSTAB | Square with general coefficients | Fixed vector count, right preconditioning and breakdown checks. |
| LSQR and LSMR | Rectangular or square | Golub–Kahan bidiagonalization using apply and transpose. Fixed vector count, no explicit normal-equation matrix. The initial path verifies the data and normal residual every iteration. |

Use `preconditioner` with `"jacobi"`, `"block_jacobi"`, `"ilu0"` or `"ic0"` where
supported. Jacobi uses positive diagonal entries. Block Jacobi inverts consecutive SPD
blocks of size two, with a final scalar block when needed. ILU(0) uses the CSR sparsity
pattern without pivoting. IC(0) uses a symmetric incomplete `LDLᵀ` representation
with positive pivots. Incomplete factors can fail at unsuitable pivots. CG rejects
ILU(0). MINRES and the least-squares paths currently use `"none"`. Optional sparse
direct LU is deferred. [Krylov source](../../src/krylov.c) · [Preconditioner source](../../src/preconditioner.c)

`solver.lanczos(A, start, k[, options])` finds up to `k` largest algebraic Ritz values
from a `basis` of at most 128 vectors. It uses two passes of reorthogonalization and
portable Jacobi rotations for the small projected problem, then verifies each Ritz
pair against the original operator. It returns `values`, `vectors` (columns),
`residuals`, and a status. A single start vector may only expose a smaller invariant
subspace, including when eigenvalues are repeated. Basis storage is `O(nk_b + k_b²)`
for basis size `k_b`, and the two orthogonalization passes cost `O(nk_b²)`. [Lanczos source](../../src/lanczos.c)

`solver.arnoldi(A,start,k[,options])` handles general square operators in a
`LAPACK=1` build. The bounded Hessenberg problem uses the existing DGEEV adapter
with fixed workspace. The result returns the largest-magnitude Ritz values and
vectors as real and imaginary arrays, with per-pair residuals checked on the
original operator. Its basis uses the same two-pass Gram–Schmidt as GMRES.
[Arnoldi source](../../src/arnoldi.c) · [LAPACK DGEEV](https://www.netlib.org/lapack/explore-html/d4/d68/group__geev_ga7d8afe93d23c5862e238626905ee145e.html)

## 0.19.2 geometric Multigrid

`linop.poisson(shape[, weights])` builds the negative discrete Laplacian in one to three
dimensions. The diagonal is twice the sum of the positive axis weights. Each direction
uses nearest neighbors with homogeneous Dirichlet values outside the interior grid.
Inputs to the solver use flattened row-major ordering.

`solver.multigrid(op,b[,options])` uses a V-cycle with three weighted Jacobi sweeps on
each side, tensor-product linear interpolation, volume-scaled transpose restriction, a
rediscretized coarse operator and a direct one-point solve. The transfer coordinates
use the same physical domain at each level, including even fine grid sizes. The hierarchy
semicoarsens the strongest active directions. An axis is eligible when its current weight
is at least one quarter of the largest active weight. This heuristic leaves weakly
coupled directions resolved on fine levels when point relaxation would damp them slowly.
For a fixed dimension the coarsening halves at least one nontrivial extent on each level.
Work and workspace for a V-cycle grow linearly with the fine-grid point count, with
additional one-dimensional interpolation tables. General AMG, other boundary conditions
and variable coefficients remain extensions. [Stencil source](../../src/poisson.c) · [Multigrid source](../../src/multigrid.c)

## 0.19.3 explicit mixed precision and structured inputs

`solver.refine(A,b[,options])` explicitly opts a square dense `numc` matrix into
partial-pivoted Float32 LU and Float32 correction solves. The iterate and stored residual
are Float64. The residual computation accumulates in C `long double`, whose precision
depends on the platform. Each correction scales the residual before conversion, and the
method switches to Float64 LU when the Float32 factors or corrections fail, stagnation is
detected or the correction limit is reached. In the result, `fallback`, `fallback_reason`
and `factor_dtype` describe that choice. A singular fallback is reported as breakdown.
The Float64 fallback obeys the same workspace budget. [Refinement source](../../src/refinement.c)

Low-rank and Kronecker operators are accepted by the Krylov and verification APIs without
dense expansion. They require explicit symmetry or positive-definiteness assumptions for
methods that need these properties. Mixed precision is currently a separate dense direct
refinement path. Further structured approximations and a low-precision Krylov backend are
future work.

## 0.19.4 verification, trace and execution contracts

`solver.verify(A,b,x[,options])` recomputes the relative residual. It also reports the
normwise infinity backward error and operator infinity norm for coefficient matrices,
diagonal operators and Poisson stencils. For a square dense `numc` matrix of size at most
128, `{"condition":true}` estimates `cond∞(A)` from the computed inverse. This opt-in
operation needs `O(n³)` work and can be sensitive to roundoff.

`solver.verify_eigen(A, lambda, x[,options])` accepts real and complex eigenpairs. Supply
complex vector parts as `{"real": real_vector, "imag": imag_vector}`. The residual scales
`||Ax-λx||₂` by `||Ax||₂+|λ| ||x||₂`. `solver.verify_svd(A, sigma, u, v[,options])` checks
both singular-vector equations and individual vector normalization. It verifies a triplet.
Checking a full decomposition's completeness and orthogonality requires additional checks.

`solver.requirements(method)` describes symmetry, positive definiteness, shape, transpose
and storage requirements. `assume` on a structured operator records `symmetric` or
`positive_definite` as a caller assertion. These declarations cannot bypass coefficient
checks for explicit matrices. Opt-in traces label recurrence estimates by iteration and
report truncation. GMRES traces use the left-preconditioned residual, while the final
report uses the original system. The common operator applies and local norm/dot kernels
are the boundaries for future halo exchange and global reductions. Native calls are
synchronous. Pipelined recurrences and distributed execution remain future work.
[Verification source](../../src/solver_verify.c)

## Research and provenance

Reference recurrences define the implementations. Source code is original. LAPACK is used
by existing optional backends and is unnecessary for the Aulë Krylov and Multigrid paths.
[CG](https://doi.org/10.6028/jres.049.044) · [MINRES](https://doi.org/10.1137/0712047) · [GMRES](https://doi.org/10.1137/0907058) · [BiCGSTAB](https://doi.org/10.1137/0913035) · [LSQR](https://doi.org/10.1145/355984.355989) · [LSMR](https://doi.org/10.1137/10079687X) · [Lanczos](https://doi.org/10.6028/jres.045.026)

The V-cycle follows classical geometric multigrid. The direction-selection heuristic above
is specific to Arda's constant-coefficient grid and was tested against an independently
assembled anisotropic Kronecker stencil. Higher-precision residuals and lower-precision
correction solves follow iterative refinement. Recent research on augmented mixed-precision
GMRES and normal-equation preconditioning gives candidates for future comparison on harder
systems. [Brandt](https://doi.org/10.1090/S0025-5718-1977-0431719-X) · [Semi-coarsening reference](https://doi.org/10.13182/NSE96-A24230) · [Carson and Higham](https://doi.org/10.1137/17M1140819) · [Augmented GMRES](https://doi.org/10.1002/nla.70117) · [Normal-equation preconditioning](https://arxiv.org/abs/2502.17767)

## Poisson acceptance test

The benchmark constructs a solution as a product of three distinct cubics
`fᵢ(x)=x(1-x)(1+aᵢx)`, with `a = [0.2, -0.3, 0.7]`, on the Dirichlet domain.
The centered second difference is exact for each cubic. Its analytical discrete
right-hand side uses the independent one-dimensional terms `-h² fᵢ''(x)`.
The Python harness checks the manufactured right-hand side and evaluates every
returned solution with an independently written NumPy stencil. This right-hand side
contains multiple eigencomponents.

The host was an x86-64 Intel Xeon Platinum 8573C with a two-CPU quota. All runs used
one CPU (CPU 0). The portable C build used GCC 14.2.0, `-std=c11 -O2`. Python used
NumPy 2.3.5 and SciPy 1.17.0. Times are medians of three processes per method and grid
size, with rotated method order. The solver timing excludes fixture generation, file
loading and result writing. Peak RSS measures the entire process, including Python
imports for SciPy. The external comparison uses matrix-free SciPy CG.

| Grid | Multigrid cycles and time | Existing Arda CG iterations and time | SciPy CG iterations and time |
| --- | ---: | ---: | ---: |
| 16³ | 15, 2.20 ms | 53, 1.24 ms | 53, 2.84 ms |
| 32³ | 16, 18.88 ms | 106, 21.07 ms | 106, 28.11 ms |
| 64³ | 16, 162.22 ms | 216, 426.06 ms | 216, 608.16 ms |
| 128³ | 16, 1.336 s | 436, 7.109 s | 436, 9.292 s |

| 128³ method | Independent relative residual | Relative solution error | Median peak RSS |
| --- | ---: | ---: | ---: |
| Arda Multigrid | 4.365e-9 | 4.032e-9 | 72.1 MiB |
| Existing Arda Poisson CG | 9.506e-9 | 3.869e-10 | 97.3 MiB |
| SciPy CG | 9.506e-9 | 3.869e-10 | 143.8 MiB |

All three methods passed the residual requirement of 1e-8. All Multigrid runs passed
the 512 MiB RSS target. Multigrid was 5.3× as fast as legacy CG on the 128³ case.
The measured error and scaling apply to this manufactured problem. A separately
assembled 7×9×11 stencil with coupling weights [1,1,100] also passed the independent
residual and solution-error assertions. The tests cover nonconvergence at an iteration
limit. [Benchmark harness](../../benchmarks/poisson_aule.py) · [Reference tests](../../tests/reference_aule.py)

```sh
make
python3 benchmarks/poisson_aule.py --arda ./arda --cpu 0 --rounds 3
```

The benchmark needs NumPy, SciPy and GNU `time`. Choose a CPU available to the local
process or omit the pinning option.

## Local validation

The runtime suite passed 265 tests in each of five configurations, including the
portable and OpenMP builds and AddressSanitizer/UndefinedBehaviorSanitizer checks.
Both OpenBLAS/LAPACK configurations passed all 75 numerical reference tests. The
other configurations passed 65, with ten expected LAPACK-specific skips. The Aulë
tests cover mixed-precision fallback, signed-stride views, workspace cleanup on
exceptions, inconsistent least squares, complex eigenpairs, a nonsymmetric restarted
solve and the anisotropic grid case. Formatting and generated-source checks passed.
Clang x86-64 and GCC AArch64 under QEMU each passed 16 focused Aulë tests, with the
LAPACK-only Arnoldi reference skipped. A final isolated run using the complete portable
binary reproduced the 128³ residual and solution error with 73,808 KiB peak RSS.
