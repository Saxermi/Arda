# Arda decision log

Record the question, evidence, decision or recommendation, and when to revisit it. Use **Proposed** while a decision is open and **Accepted** once it is agreed. Link to the benchmark or roadmap that supports each entry.

## 2026-11-27 — Streaming imports and tiled output publication

**Status** — Implemented in the 0.21 development branch and open to owner revision

Convert supported external formats to the existing `.atn` layout with bounded buffers. This makes imported data available to every existing file reader with one implementation of bounds and tile access. It takes an extra file and one conversion pass. Direct reads of validated NPY files could save that copy for binary inputs. Add a shared read-only source interface if repeated binary reads justify the additional format-specific lifetime and layout checks.

Use one temporary sibling for a tiled result. The caller writes only the tiles it needs and publishes the result at commit. Assembling a result from scratch has tile-sized data writes, and replacing an existing result gives current readers a stable snapshot. A persistent chunk store could support incremental edits and restartable writes, with the added cost of metadata and recovery. Revisit that alternative when implementing the cache and recovery work. [File contracts](research/implemented/typed-tensors.md#streaming-imports) · [0.21 roadmap](ROADMAP.md#arda-021--beyond-ram-data-processing-and-visualization-ame-v2)

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

## 2026-11-25 — Defer large-file performance benchmarks

**Status** — Deferred until a test machine with suitable SSD space is available. Small automated fixtures cover correctness and budget enforcement. Throughput and peak RSS remain unmeasured for the 20 GB streaming-statistics and 80 GB file-backed reduction targets. [Roadmap validation targets](ROADMAP.md#arda-016--virtual-tensors-and-memory-management-ame-v1)

Plan for about 30 GB of free SSD space for the 20 GB benchmark alone, 100 GB for both read-only benchmarks run one dataset at a time, or 120 GB to retain both datasets. An optional atomic tile-write benchmark on the 80 GB file needs about 180 GB free before creating the dataset, since the original and full temporary replacement coexist. Allow about 200 GB if the 20 GB dataset is also retained. These estimates are in decimal GB and include working headroom.

**Revisit** — Use fully allocated files and record throughput and peak RSS for cold and warm runs. Check output against an independent reference and measure tile writes separately from read-only reductions.

## 2026-11-25 — Low-rank contractions and scientific types

Use structured CP, Tucker and tensor train inner products and bounded direct evaluation when cancellation threatens the result on small shapes. Use deterministic LAPACK sweeps as the reference for TT approximation. Keep randomized rounding and TT-cross for later comparison. [Low-rank tensor contracts](research/implemented/low-rank-tensors.md)

Represent scientific quantities with SI values and separate display-unit metadata. Check dimensions once per operation and reuse the Float64 kernels. Separate absolute temperatures from temperature differences. Keep 64-bit logical dimensions in specifications and assertions while retaining compact indices in live layouts. MathIR displays source subscripts with unresolved shapes and bounds. [Scientific type contracts](research/implemented/scientific-types.md)

## 2026-11-26 — Vilya runtime and language contracts

**Status** — Accepted

- Use the existing typed tensor layout and shared buffers for tensor AD. Keep the optimized numerical kernels and retire redundant public names where operations have the same contract. Put mutation versions on shared buffers so writes through an owner or an alias invalidate a saved value.
- Extend `ad` to dense Float64 tensors first. Accumulate contributions within a reverse sweep. Each successful backward call replaces the gradients it visits. Publish gradients only after a successful sweep. Charge gradient payloads against the tensor memory budget and keep MathIR separate from runtime tracing.
- Use IEEE results for ordinary real floating-point operations and reject nonfinite values in differentiation. Keep checked integer arithmetic. Require an explicit scalar condition for tensors and tracked values.
- Use maps with named fields for caught errors and iterative solver reports. Migrate in-tree examples and callers directly when an interface changes. Present both English and Elvish-inspired spellings in the main tutorial.
- Make `range` lazy and make materialization explicit with `list`. Give the standard library an explicit import spelling and resolve bare standard names consistently. Keep quoted paths for local modules.

## 2026-11-26 — Derivative queries and fixed CSR patterns

**Status** — Accepted for the 0.17.1 development branch

Use the executed tape for seeded VJPs and JVPs. Treat the selected node as independent for partial derivatives. Queries leave published gradients intact. Compute a full Jacobian along the smaller basis and a scalar-output Hessian by differentiating reverse accumulation. Reuse scratch buffers between basis sweeps. Dense results follow the existing rank and heap limits.

Let `zero_grad` release the stored gradient and `detach` expose a shared primal without graph edges. Scope `no_grad` to a callable and restore the caller's recording state after errors. Keep CSR offsets and columns immutable. Differentiate an explicit vector of stored coefficients in CSR order, including a coefficient whose current value is zero. [API and evidence](docs/VILYA_RUNTIME.md#gradient-queries-and-sparse-matvec)

## 2026-11-26 — Retained gradients and elementwise checkpoint replay

**Status** — Adopted for the 0.17.2 development implementation and open to owner revision

Use selected-gradient backward as an opt-in mode. Reuse one private gradient buffer when the byte count matches. Allow explicit consumption of the output's history after publishing successfully. This preserves existing callers and makes release visible. It requires a retain list and can leave shared history owned by other nodes. An automatic policy could require fewer flags and release more memory. It would change immediate access to intermediate gradients or repeated backward behavior and needs owner approval.

Interpret `samples` as the weight of a batch-mean gradient. The published result is the full weighted mean, including unequal batch sizes. Callers must know the sample count and clear an unweighted gradient before mixing conventions. Accumulating raw sums would support custom normalization and weighting, with additional bookkeeping and a separate division step for callers.

Pack pure dense elementwise operations between one or two explicit boundary nodes. Replay their numerical recipe while retaining the already sampled random values. This makes side effects and generator state independent of recomputation. It limits checkpoint coverage to supported operations and adds compute time. Callback replay would cover more workloads and could save and restore RNG state, with stronger requirements for effect tracking, generator ownership and error recovery. Broader replay needs owner approval. [Contracts and measurements](docs/VILYA_RUNTIME.md#memory-efficient-backward-and-checkpoints)

## 2026-11-26 — Explicit activation offloading with bounded reads

**Status** — Adopted for the 0.17.3 development implementation and open to owner revision

Use an explicit disk quota and let callers name the dense Float64 intermediates to spill. This makes placement auditable and lets applications choose activations or checkpoint boundaries according to their workloads. Callers have to select and rebind those intermediates. A graph planner could choose placements automatically and reduce that work, at the cost of liveness and I/O estimates and new scheduling behavior.

Use immutable snapshots in immediately unlinked files. Mapping the snapshot preserves the ordinary tensor interfaces for forward reads, while chunked reads bound derivative workspace. Forward page-cache residency depends on the OS, and strided access can reread chunks. Reloading each whole activation would simplify read management and may help random access. It would make derivative workspace proportional to the largest reloaded tensor. [Runtime contract](docs/VILYA_RUNTIME.md#activation-offloading)

## 2026-11-26 — Local sensitivity products and factored covariance

**Status** — Adopted for the 0.17.4 development implementation and open to owner revision

Evaluate numerical sensitivity on a concrete tape and keep formal derivative TeX in MathIR. Execution captures the current arguments and control flow. Symbolic output works independently for a bounded subset under real differentiability assumptions. A shared symbolic graph could expose more numerical internals for explanation, at the cost of preserving effects, closure bindings, tensor shapes and dynamic dispatch in the symbolic representation.

Use seeded products and allow independent standard deviations or a supplied covariance factor. This bounds memory and lets a rectangular factor express rank-deficient covariance directly. The caller prepares a factor for general correlated inputs. A dense covariance overload could accept the original matrix and factor it internally for convenience. It would require quadratic storage, factorization work, and an explicit policy for symmetry, small negative eigenvalues and singular inputs. [Runtime contracts](docs/VILYA_RUNTIME.md#sensitivity-and-uncertainty)

## 2026-11-26 — Draft ownership for distributed training

**Status** — Proposal for owner review before any distributed implementation

Use stable logical parameter IDs and aggregate local gradient sums with exact sample counts. Start the design with replicated parameters and optimizer state because this gives each rank local reads and a straightforward comparison with serial training. Each rank pays the full model and optimizer memory cost. Sharded ownership would lower those memory costs and require an explicit map from parameters to ranks, remote reads and coordinated restart. The proposal leaves ownership granularity and the eventual distributed API open. [Draft contract and acceptance cases](research/planned/computing-systems.md#draft-distributed-parameter-and-gradient-ownership-contract)

## 2026-11-26 — Register blocking with separate multiply and add by default

**Status** — Implemented for 0.18.1 and open to owner revision

Use register tiles over the existing row-major Float64 inputs. This reuses right-hand values across rows without an additional packed matrix or tensor workspace. Keeping each dot product's order and leaving contraction disabled preserves the default rounding behavior. Packing panels would improve locality for more shapes and give larger microkernels predictable access. Packing also needs extra storage, data movement and budget checks.

Expose AVX2 FMA and GNU libmvec as explicit choices. They can improve throughput, and vector providers and fused arithmetic can differ in the last bits. Selecting them automatically would broaden their speed benefits and change the default numerical behavior. [Kernel contracts](research/implemented/transforms-and-kernels.md#0181-register-blocked-matrix-multiplication)

## 2026-11-26 — Optional OpenMP for coarse native CPU work

**Status** — Implemented for 0.18.2 and open to owner revision

Use compiler-managed OpenMP teams for large packed GEMM and AXPY. A per-VM setting defaults to one thread. Static output partitions preserve per-element arithmetic order and keep VM state and error handling on the caller thread. The runtime can reuse workers, and the implementation stays small. OpenMP adds a build dependency and leaves worker placement and some scheduling behavior to its provider.

A native pool could give Arda direct control of worker lifetime, CPU placement and shutdown. It would require new synchronization, lifecycle management and error tests. Use the linked provider's threading for CBLAS builds to avoid nested Arda and BLAS teams. [Thread controls and measurements](research/implemented/transforms-and-kernels.md#0182-controlled-cpu-parallelism)

## 2026-11-26 — Explicit symmetric Q4 and Q8, with widened arithmetic

**Status** — Implemented for 0.18.3 and open to owner revision

Use immutable weight snapshots with symmetric 32-value row blocks and a Float32 scale. This keeps storage predictable and makes zero easy to represent. Uniform per-block rounding can lose small coefficients near outliers. Asymmetric blocks could use the code range more effectively for skewed inputs, while adding zero-point metadata and arithmetic. Calibrated scales could improve model quality and require representative data and a calibration pass.

Decode one block at a time and multiply Float64 activations. This preserves the caller's activation precision and bounds scratch. Integer products with quantized activations could improve throughput for larger batches. They would introduce activation quantization error and require scale management and overflow analysis. The separate Float16/BFloat16 path widens into Float32 and gives callers explicit control over that trade-off. [Contracts and evidence](research/implemented/quantization.md)

## 2026-11-26 — Explicit fusion, calibrated plans and CPU placement

**Status** — Implemented for 0.18.4–0.18.6 and open to owner revision

Keep fusion explicit and round each Float64 operation in expression order. This reduces full
tensor materialization with predictable numerics. Callers must opt in and select one of the
supported regions. A compiler pass could recognize more expressions and require fewer source
changes. It would need dependency, aliasing and exception-order checks before applying a rewrite.

Use a bounded, per-VM search for ACE and report measured medians separately from the lower-bound
cost model. Cached measurements keep normal executions cheap but can age as machine load changes.
Always-on adaptive tuning could respond to load and locality changes. It would also add timing
variance and work to ordinary executions.

Keep prefetch and CPU pinning opt-in. One POSIX worker makes prefetch workspace and lifetime
predictable. An asynchronous I/O backend could overlap more requests but brings extra platform
and queue-management costs. Keep caller affinity and OpenMP worker affinity explicit. An
Arda-managed NUMA scheduler could put work and pages on chosen nodes directly, with added
worker lifecycle and memory-placement complexity. Cold SSD and multi-node NUMA data are
needed before revisiting these defaults. [Contracts and measurements](research/implemented/cpu-compute.md)

## 2026-11-27 — Aulë solver contracts, transfers and verification

**Status** — Implemented for 0.19.0–0.19.4 and open to owner revision

Use one additive `solver` interface over the existing native operator representations
and recompute the final residual. This keeps old calling conventions and makes stop
reasons comparable. It adds a second entry point and an extra operator pass. A direct
migration could reduce API surface, but would require changing existing programs.
Synchronous user-defined operators would extend coverage, with extra rooting, error
unwinding and VM-call costs. They remain an optional extension.

Start with bounded Krylov bases and caller-selected methods and preconditioners. This
keeps storage and assumptions visible. Automatic method selection can save choices when
operators have verified properties, while an incorrect classification can invalidate
a recurrence or waste work. Scope adaptive precision to explicit `solver.refine`.
Float32 factors can make corrections cheap, but the method may stagnate or require a
full Float64 factorization. An always-Float64 path avoids fallback handling and gives
up the lower-precision factorization.

Use rediscretized geometric V-cycles with linear transfer and semicoarsen strong
directions. The matrix-free stencil keeps storage bounded on regular grids. General
Galerkin or algebraic multigrid could serve more operators and variable coefficients,
with higher setup, storage and testing costs. Keep the known CG and SciPy paths as baseline
comparators. [Algorithms and local validation](research/implemented/aule-solvers.md)

## 2026-11-27 — Retire the parallel iterative-solver API

**Status** — API migration approved by the owner

Use `solver` for iterative solves and solution verification, and `linop` for operator
construction and application. Remove the duplicate entries and their private sparse workspace
type. The old reusable workspace avoided allocations across solves. The common solver instead
budgets transient scratch per call and uses tensor memory accounting. A compatibility layer
would preserve the old signatures and accounting contract alongside the new one.

Route least-squares examples through LSQR. CGLS was a separate recurrence for the same
least-squares problem. The current result preserves both data and normal residuals. Use
the existing specialized stencil for up to three axes and a general matrix-free stencil
for higher ranks. This keeps the four- and eight-dimensional examples runnable. Generalizing
the Multigrid hierarchy would add variable-rank transfer kernels and needs separate validation.
[API migration](research/implemented/aule-solvers.md#solver-api-migration)

## 2026-11-27 — Initial Ulmo output, implicit stepping and grid contracts

**Status** — Implemented for 0.20.0–0.20.2 and open to owner revision

Treat RK45 sampling and trajectory retention as separate options. Fixed output times
preserve the integration steps, while retained dense output trades memory for later
queries. A synchronous stream avoids keeping all snapshots. A forced-step approach
would simplify interpolation at fixed outputs but could add many steps and make the
integration depend on the sampling schedule. Checkpoints are in-memory snapshots at
segment boundaries. A persistent format needs versioning and I/O failure handling.

Start implicit integration with fixed-step Euler, BDF2 and Radau IIA order three.
Newton uses a dense residual Jacobian and existing LU or GMRES. This makes the first
implementations easier to verify and supports small stiff systems. Adaptive
higher-order methods or matrix-free Newton–Krylov would reduce work or storage for
some problems, with changes to step selection, preconditioning and failure recovery.

Use uniform Cartesian stencils with explicit homogeneous boundary conventions and
stability checks. Upwind fluxes conserve mass on closed or periodic grids but add
numerical diffusion. Higher-order limiters could sharpen transport and would require
additional stability and monotonicity tests. The Poisson wrapper applies an explicit
zero-mean gauge on the compatible constant-nullspace cases.
[Contracts and measurements](research/implemented/ulmo-simulation.md)

## 2026-11-27 — P1 transfer and fixed-step simulation derivatives

**Status** — Implemented for 0.20.3–0.20.4 and open to owner revision

Use conforming midpoint subdivision for segments and triangles. The coarse P1
space is contained in the fine space, so midpoint interpolation preserves the
field and its integral to roundoff. Local closure also splits neighbors of marked
cells. Propagating repeated partial splits can reduce element quality. Newest-vertex
bisection would give a stronger basis for shape-regularity and adaptive complexity
analysis at the cost of compatible edge labels and a refinement hierarchy. A
general remesher could optimize shape or remove nodes. Transfer in that case would
need a conservative projection and an overlap integration or mass solve.

Use continuous P1 pressure and cellwise velocity recovery for experimental steady
Darcy flow. It reuses the scalar diffusion solver and is exact for a constant
mobility and affine pressure. Cellwise recovered velocities can have unequal normal
fluxes across a face. Mixed Raviart–Thomas FEM would represent a locally conservative
flux and would need extra face degrees of freedom and a mixed-system solver.

Differentiate the selected discrete RK4 and Cartesian heat steps on a fixed time
schedule. Reverse replay stores block boundaries and rebuilds one block at a time.
This reproduces the numerical trajectory and gives an explicit bound on retained
checkpoint/replay states. Replay costs extra primal steps and requires a deterministic
RHS. Multistage checkpoint scheduling can spend a given memory budget more efficiently,
at the cost of stage-specific state management. Differentiating adaptive stepping,
implicit solves or events would require contracts for step selection, linear solves
and event derivatives. A continuous adjoint is also viable for smooth models and can
use a separate integrator. Its gradient approximates the continuous problem and can
differ from the derivative of the numerical trajectory.

[Methods, API and validation](research/implemented/ulmo-simulation.md)

## 2026-11-27 — Alternatives for Ulmo refinement, flow and reverse replay

**Status** — Owner requested the three alternatives and a comparison. Implemented
as opt-in choices, with the original defaults available for the same problems.

Newest-vertex bisection stores immutable parent and generation records, keeps
the newest vertex on each leaf and closes the selected edges across adjacent
triangles. It avoids the small angles observed in repeated partial midpoint
subdivision on the benchmark grid. Closure can add triangles and retaining the
tree uses memory. The existing midpoint path remains a compact option, and
uniform midpoint subdivision does not exhibit the local benchmark's small-angle
failure.

The mixed triangular Darcy solver uses RT0 integrated edge flux and P0 cell
pressure. It enforces the element balance and represents each internal edge
flux once. Diagonally scaled MINRES reuses the sparse Krylov solver. The system
has more unknowns and costs more than the P1 pressure solve followed by a
gradient. Hybridization or an equivalent nonconforming pressure solve followed
by flux reconstruction could reduce that cost and would require a separate
assembly and reconstruction path.

The stage-aware RK4 schedule holds a budgeted tail of differentiated steps
during replay. It reuses each retained four-stage tape in the reverse sweep
and cuts RHS invocations. The retained graphs increase tensor memory and GC
work. The block schedule remains the default. A stage-aware dynamic program
with model-specific checkpoint byte costs could pick better retention points
and would add scheduler and graph-memory accounting complexity.

[Contracts and direct measurements](research/implemented/ulmo-simulation.md#alternative-ulmo-methods)

## 2026-11-27 — Resource-scoped tuning and Krylov audit fixes

Filter ACE candidates by the live output and scratch budget before tuning.
Reuse measurements only with the same eligible choices, actual stable teams
and known CPU placement. This preserves useful entries when returning to an
earlier mask or budget. Timing reuse on multi-node systems would require a
placement policy or measurements associated with particular allocations.

Restart GMRES after a productive near-invariant cycle using the freshly
computed residual. This recovers correctable roundoff without adding an
unbounded retry policy. Keep the public tolerance tied to verified residuals.
Borrow packed right-hand sides and reuse transient CG vectors inside the
synchronous solver. The copy path stays necessary for signed-stride inputs.
For conditioning, keep the bounded dense path and support exact diagonal
ratios. Large sparse inverse-norm estimates would need a separate contract
for reliable transpose solves.
[Audit](research/implemented/ace-aule-audit.md)

## 2026-11-27 — Ulmo experiment scheduling, restart and domain contracts

Run sweep callbacks on the current VM in input order and stream each row
synchronously. This gives a bounded active case count while existing kernels
keep their own thread controls. An explicit LU, Cholesky or QR factor is passed
through unchanged. Parallel or isolated workers would need a callback and
object transfer contract, deterministic output ordering and shared resource
accounting.

Publish each checkpoint as one binary record with a solver identifier and
validated state and metadata. Reusing the tensor writer's atomic publication
keeps state and metadata together and preserves Float64 values. This format
covers the RK45 and heat contracts. General serialization of VM objects or
arbitrary callbacks would require an explicit integrity and compatibility
model. Heat copies the state only on request so a later mutation of the
public result cannot change that snapshot.

Use one axis of contiguous slabs, with packed external faces and the existing
stencil. This adds a local contract for testing future exchanges while keeping
the boundary discretization identical. Multi-axis blocks and MPI transport can
build on the same face ordering once a workload warrants their complexity.

Run the implementation gate on the cloud machine for 0.20.5.
[Experiment contracts](research/implemented/ulmo-simulation.md#0205-experiments-and-domains)

## 2026-11-27 — Ulmo hardware validation before Arda 1.0

Complete 0.20.6 with a clean source archive, cloud validation and measured
comparisons. Record Ulmo's physical 8 GB laptop check in the 0.28.2 hardware
matrix and run it on the final 1.0 release candidate alongside the full test
matrix. This ties the measurements to the actual 1.0 source. [Roadmap](ROADMAP.md#physischer-ulmo-laptop-test)
