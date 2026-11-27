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
