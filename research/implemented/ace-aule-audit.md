# ACE and Aulë audit

Baseline commit `b9e81f5` in Arda 0.20.4. This audit covers compute planning,
Krylov termination, verification, tensor storage error paths and workspace use.

| Confirmed issue | Reproduction | Correction |
| --- | --- | --- |
| ACE rejected a legal candidate | Transposed 64×64 Float64 RHS, budget for the output and a 32×32 panel. The larger candidate made planning fail. | Filter before timing and scope each cached result to the eligible candidate set. A comparison accounts for both outputs. |
| ACE cached a timing without a placement or effective-team key | The original cache stored shape, strides, storage and requested team size. The public affinity setter cleared everything, including otherwise reusable timings. | Key by exact mask on single-node Linux, predicted eligible teams and resolved native dispatch. Cache only stable observed teams. Multi-node or unknown placement and dynamic parallel teams still run tuning. |
| GMRES stopped after a productive near-invariant cycle | For diagonal `[1,2]` and RHS `[1,1e-17]`, tolerance `1e-20`, the old solver stopped at iteration 1 with true relative residual `1e-17`. | Restart from the true updated residual when it measurably improves. The fixture converges in two iterations. Give the iteration limit priority when further work would be required. |
| CG allocated multiple avoidable vectors | The 128³ Poisson common solver allocated 96 MiB numeric workspace and had 129.3 MiB median peak RSS at the baseline. | Borrow a packed Float64 RHS during the synchronous solve, reuse the last matrix–vector product for verification and alias the unpreconditioned residual and search preparation vector. A strided RHS still needs packing. |
| Relative verification overflowed from a representable input pair | With `A=I`, `b=[1e308,0]` and `x=[-1e308,0]`, the old relative residual overflowed. With a subnormal well-conditioned diagonal dense matrix, explicit inverse formation could also fail. | Compute residuals in scaled coordinates, add the absolute residual and evaluate the backward error with exponent scaling. Use scaled LU with one-column solves for the bounded dense condition estimate, and a linear-time exact ratio for diagonal operators. |
| Failed tensor reads retained already charged buffers | Truncate a file after opening a block iterator, then catch the short-read error under a tight budget. A second read previously encountered the retained allocation. | Discard unpublished block and tile buffers on error. |
| ILU(0) on an empty sparsity pattern reached undefined behavior | UBSan reported a null source argument to a zero-byte copy during setup. | Skip the copy and report the missing pivot through the existing solver error path. |

## Decisions and limits

The ACE roofline report is an optimistic model based on logical byte counts and
provided bandwidth rates. Candidate selection uses resource checks, layout and
dispatch rules, and optional measured medians. Cache hierarchy, transient loads,
page placement on multi-node hosts, hardware throttling and arbitrary provider
thread pools are not measured by this model. Historical plan samples remain
available through `measured_seconds` and `measured_threads`. Every execution
checks its workspace again and reports the actual team.

GMRES convergence still requires a true residual check. Near-invariance is a
signal to stop extending the current basis, not a certificate of convergence.
An unusable projected column is numerical breakdown, and an invariant cycle
without meaningful residual improvement is stagnation. Orthogonality diagnostics
beyond the existing two Gram–Schmidt passes are outside the solver result contract.

A small residual measures the fit to the equations. Forward error additionally
depends on conditioning and the norm used. The LU estimate is not a certified
bound. A general sparse or structured estimate using Hager–Higham would need
reliable direct and transpose inverse actions with stated accuracy and a suitable
norm bound. The current public API does not promise these for arbitrary operators.
The diagonal path computes its ratio directly, with singular or unrepresentably
large ratios reported as infinity.

The 16-million-element ceiling remains a heap materialization policy. Typed views
and file metadata use checked shape and addressability rules separately. Tests
cover backwards and zero strides, extreme stride values and logical shapes above
the heap limit. Large-file decomposition and wider backend dimensions are
separate roadmap work. A mapped inode still needs to remain stable while in use.

## Validation and provenance

Portable GCC passed 297 runtime tests and 81 numerical references, with ten
LAPACK-specific references skipped. The baseline passed 288 runtime tests.
Focused regressions run under GC stress where applicable. SciPy GMRES with a
Jacobi preconditioner independently checked a scaled nonnormal system at two
restart sizes. Dense direct solves checked near-invariant cases at three
power-of-two scales.

[ACE source](../../src/compute_plan.c) · [GMRES source](../../src/krylov.c) ·
[Verification source](../../src/solver_verify.c) ·
[Saad and Schultz](https://doi.org/10.1137/0907058) ·
[PETSc happy-breakdown option](https://petsc.org/release/manualpages/KSP/KSPGMRESSetHapTol/) ·
[LAPACK Hager–Higham norm estimator](https://www.netlib.org/lapack/explore-html/d7/db3/group__lacon_ga1a635889a2510c62b1d93480232bf8d5.html)

## Independent review of `ulmo-0.20`

The comparison used `9773de0` and its published parent `660a0d7`. The parent
passed 288 runtime tests. The focused checks exercised the following behaviors
on the relevant revisions.

| Issue | Severity | Status | Location, cause and validation |
| --- | --- | --- | --- |
| ACE-01 | Medium | Fixed in `9773de0` | `compute_plan.c:239`, `tune`, and `compute_topology.c:175`, `compute_tuning_placement`. Timing reuse now follows the exact affinity mask and effective candidate teams. Returning to a previous mask or team can reuse its timing. Tests covered stable one- and two-thread teams, dynamic teams, and independent global matmul dispatch. |
| ACE-02 | Medium | Fixed in `9773de0` | `compute_plan.c:141`, `fits`, and `:239`, `tune`. With space for the output and an 8 KiB panel, a transposed 64×64 RHS now selects block 32. A one-byte smaller budget is rejected. Comparison accounts for both outputs. |
| AULE-01 | High | Fixed in `9773de0` | `krylov.c:188`, `gmres`. The near-invariant cycle on diagonal coefficients 1 and 2 with RHS `[1,1e-17]` and tolerance `1e-20` now restarts from the true residual and converges in two iterations. The iteration cap and singular case give distinct reasons. Independent references cover scaling, nonnormality, two restart sizes and Jacobi preconditioning. |
| AULE-02 | Medium | Fixed in `9773de0` | `solver_verify.c:141`, `linear_check`. At `A=I`, `b=[1e308,0]` and `x=[-1e308,0]`, the scaled difference gives relative residual 2, absolute residual infinity and backward error 1. Nonfinite RHS values raise a domain error and a nonfinite operator product fails verification. |
| AULE-03 | Medium | Fixed in this review | `linalg.c:896`, `dense_condition_inf`. Dense `diag(1, 6e-309)` with `b=[1,6e-309]` and `x=[1,1]` returned infinity for a Float64-representable condition near `1.667e308`. The previous scaling could reduce the norm below 1 and overflow an inverse column. The new scaling keeps the norm at least 1 at the same workspace and asymptotic cost. The GC-stress regression fails on `9773de0` and passes after the fix. |
| AULE-04 | Medium | Fixed in `9773de0` | `solver.c:86`, `packed_rhs`, and `krylov.c:71`, `cg`. Borrowing a packed RHS and sharing work vectors halves CG workspace on the 128³ Poisson fixture. A signed-stride RHS still has a correctly charged copy. |
| AME-01 | Medium | Fixed in `9773de0` | `tensor_storage.c:215` and `:251`, `tensor_block_next` and `tensor_read_tile`. Short reads now release unpublished buffers. The block regression restores the tensor byte counter after each of two failed attempts. |
| AULE-05 | Medium | Fixed in `9773de0` | `preconditioner.c:79`, `setup_incomplete`. UBSan diagnoses the zero-byte copy in the parent. The empty CSR case now reaches the missing-pivot diagnostic with sanitizers and GC stress. |
| AME-02 | Low | Existing bounds validated | `typed_tensor.c:25` and `:186`, `typed_layout_checked` and `strided_view`. Tests reject extreme strides and overflowing shape products, and accept valid reverse and broadcast views. `sparse_mul_size` checks allocation arithmetic and `decode_header` checks layout and payload lengths. The 16-million-element heap limit is a materialization policy. |
| ACE-03 | Low | Open model limitation | `compute_plan.c:408`, `estimate`. Calibration affects reporting. Layout, backend rules, memory eligibility and measured medians drive selection. The estimate uses logical traffic and leaves cache warmth, contention and actual memory transfers unmeasured. |

### Local validation

Linux x86-64 ran GCC 14.2, Clang 19.1 and OpenBLAS 0.3.29. GCC and Clang
OpenMP, GCC OpenBLAS/LAPACK and GCC ASan/UBSan with leak detection each passed
298 runtime tests. Portable GCC and Clang each passed 297 with one affinity
skip from CPU pinning. A two-CPU portable run passed all 19 focused ACE, kernel
and tensor memory tests, including the affinity case. Without LAPACK 74 of 91
references passed, with ten LAPACK and seven optional JAX skips. OpenBLAS/LAPACK
passed 84 of 91 with seven JAX skips. The references passed with the same
expected skips under ASan/UBSan. Four kernel tests passed with `VECMATH=1`,
and both prefetch tests passed under ThreadSanitizer. Generated sources and
formatting checks passed.

Clang's static analyzer ran on all 55 C translation units. Reviewing the reported
paths accounted for nonempty tensor shape invariants, CSR conversion before ILU
setup, sparse builder reserve-before-write, and advancement of the frontend
allocation list before free. Native AArch64, multi-node NUMA, a binary64
`long double` ABI and OpenMP race instrumentation remain unvalidated.

### Performance measurements

The AMD EPYC 9V74 VM exposed three CPUs with a two-CPU quota. Portable builds
used GCC 14.2, C11 and `-O2`. Each Poisson run pinned one CPU and checked a
Float64 result against the independent NumPy stencil at relative tolerance
`1e-8`. Table values give medians with ranges over three separate processes.
Peak RSS includes the entire process.

| 128³ method | Seconds, median [range] | Peak RSS MiB, median [range] | Iterations or cycles | Independent residual |
| --- | ---: | ---: | ---: | ---: |
| Common CG, parent | 8.12 [8.05, 10.57] | 129.28 [129.27, 129.30] | 436 | 9.5054e-9 |
| Common CG, `9773de0` | 7.52 [7.16, 11.15] | 81.28 [81.26, 81.38] | 436 | 9.5054e-9 |
| Common CG, final | 7.18 [7.01, 7.19] | 81.29 [81.27, 81.30] | 436 | 9.5054e-9 |
| Legacy CG, `edc7ba8` | 9.07 [7.57, 11.04] | 97.24 [97.24, 97.28] | 436 | 9.5065e-9 |
| Arda geometric multigrid, final | 1.43 [1.32, 1.44] | 72.17 [72.15, 72.17] | 16 | 4.3650e-9 |
| SciPy CG | 8.39 [8.30, 8.47] | 143.70 [143.15, 143.71] | 436 | 9.5056e-9 |

Arda's CG workspace counter fell from 96 to 48 MiB and its peak budgeted heap
from 112 to 64 MiB on the same input. The independent relative solution error
was `3.8690e-10` on both versions. The RSS reduction repeats across runs.
Wall-time samples have wider dispersion. [Poisson harness](../../benchmarks/poisson_aule.py)

| Other workload | Checked result |
| --- | --- |
| ACE tuning with an 8 KiB panel budget | Seven runs after one warm-up used a transposed 128×128 RHS. Median planning time was 3.03 ms on a miss and 12.4 µs on a hit, with ranges of 3.02–3.15 ms and 9.7–26.5 µs. Both plans selected block 32 and returned the checked value. |
| Native Float64 GEMM, square sizes 128, 512 and 1024 | Seven batches after warm-up. Parent medians 0.105, 15.49 and 111.81 ms. Branch medians 0.106, 15.16 and 124.89 ms. OpenBLAS CBLAS medians 0.095, 5.80 and 57.55 ms. Exact sampled values match at all sizes. At 1024 the parent and branch ranges overlap. |
| OpenMP GEMM and GEMV | Five batches after warm-up. Actual teams respected the two-thread ceiling. Final 768² GEMM medians were 35.37 ms with one thread and 26.45 ms with two, with the latter ranging from 22.18 to 57.54 ms. |
| Sparse SpMV | All 262,144 output values matched SciPy exactly. The final one-thread Arda and SciPy medians were 2.37 and 0.84 ms, with ranges of 1.67–2.50 and 0.75–1.96 ms. Arda used a serial kernel for all requested teams. |
| PyAMG 5.4.1 smoothed aggregation, 32³ | The CSR comparison passed the same residual gate at `6.05e-9`. Three runs after warm-up took median 126 ms setup and 291 ms solve with 15 V-cycles. Arda's matrix-free geometric multigrid at the same grid size had a median solve time of 17.7 ms on the parent and 17.2 ms on the branch. The runs used different cores on the same VM. |
| Fusion and buffered streaming | Three rounds per revision passed the output checks. Fused RSS was about 459 MiB and unfused RSS about 688 MiB on both. Final file-reduction medians were 123 ms serial and 136 ms with prefetch. |
| Q4/Q8 GEMV and GEMM | Five batches after warm-up with 1024×1024 weights. Q8/Q4 GEMV medians were 0.406/0.388 ms with 0.51%/9.56% relative L2 error. GEMM medians were 2.660/2.674 ms with 0.53%/9.65% error. The parent gave the same error values. |
| AMDE | Five interleaved measured rounds after warm-up validated startup, compilation, VM calls and dense GEMM outputs. Compact VM-call time was 72.42 ms in the final build and 72.55 ms in the parent. |

General condition estimation still needs a contract for inverse and
transpose-inverse actions and their tolerances. A verified residual measures
equation fit. Forward-error bounds also depend on the relevant norm,
conditioning and rounding. Wider backend dimensions and large-file execution
remain architecture topics with separate resource constraints.
