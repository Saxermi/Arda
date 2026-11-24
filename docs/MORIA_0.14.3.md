# Arda 0.14.3 — Sparse Solvers and Memory Budgets

*Development codename: Moria*

This guide covers 0.14.1 through 0.14.3. See [the 0.14.0 sparse matrix foundations report](MORIA_0.14.0.md) for the original COO/CSR representation, inspection, and conversion contracts. Import the module with `tol sparse as sp`.

## Assembly

| API | Behavior |
| --- | --- |
| `sp.builder(rows, cols[, capacity[, max_bytes]])` | Creates a mutable, empty builder. Capacity is an optional initial entry reservation. `max_bytes` is an optional cap on builder buffer storage, including overlapping old and new capacity during growth. |
| `sp.append(builder, row, col, value)` | Appends one zero-based integer coordinate with a finite value. |
| `sp.extend(builder, rows, cols, values)` | Appends equal-length lists or rank-one numeric tensors. Tensor indices must be exact integers within bounds. The full batch is validated before the builder changes. |
| `sp.finish(builder)` | Returns immutable canonical CSR. On success it releases the builder buffers and closes the builder. A failed finish leaves the builder open with its inputs intact. Further calls on a finalized builder raise an error. |
| `sp.eye(n)` | Builds an `n × n` identity in CSR. |
| `sp.diags(rows, cols, offsets, diagonals)` | Builds rectangular CSR from unique diagonal offsets `col - row`. Each diagonal is a finite scalar to broadcast or a list or rank-one tensor of the exact diagonal length. |
| `sp.tridiag(n, lower, main, upper)` | Builds square CSR for offsets −1, 0, and +1. Each argument follows the scalar or vector diagonal contract. |

Builder storage uses one native Float64 value and two uint32 coordinates per capacity slot. Finalization uses the same canonicalization kernel as COO conversion. Per-row sorting is in-place and includes insertion order in its key. Duplicate values sum in insertion order, nonfinite intermediate sums raise an error, and exact zeros are omitted. Direct diagonal constructors count and write stored entries in two passes without forming triplets.

## Algebra and exchange

| API | Behavior |
| --- | --- |
| `sp.matvec(A, x)` | Sparse matrix times a rank-one Float64 tensor. |
| `sp.spmm(A, X)` | Sparse matrix times a rank-two Float64 tensor, including strided views. |
| `sp.transpose(A)`, `sp.add(A, B)`, `sp.scale(A, factor)` | Return canonical CSR. Addition requires matching shapes. Scaling by zero returns an empty CSR of the same shape. |
| `sp.diagonal(A)` | Returns the main diagonal as a rank-one Float64 tensor. |
| `sp.estimate(A)` | Returns `rows`, `cols`, `nnz`, and `matvec_workspace_bytes`. CSR matvec needs no scratch beyond its result. The COO estimate is an upper bound on managed temporary CSR conversion storage and excludes the result vector. |
| `sp.read_mm(path)` | Reads Matrix Market coordinate real general or symmetric into canonical CSR. Input coordinates are one-based. Symmetric files use the lower triangle, which is expanded on import. Data records support decimal and scientific notation. |
| `sp.write_mm(path, A[, symmetric])` | Writes general Matrix Market by default. Passing true requires square exact symmetry and writes only the lower triangle. The destination is replaced after a complete write. |
| `sp.describe(name)` | Returns static versioned descriptions for conversion, matvec, SpMM, transpose, addition, scaling, the diagonal, CG, PCG, and verification. |

The serial matvec kernel accumulates each CSR row in extended precision and stores Float64 output. SpMM applies the same kernel to each dense input column. CSR matvec takes O(rows + nnz), transpose takes O(rows + cols + nnz), and addition merges rows in O(rows + nnz(A) + nnz(B)). COO operands are canonicalized when needed. Sparse inputs never require a dense matrix intermediate. Dense results keep Arda’s existing 16,000,000-element limit.

## Solvers and budget

`sp.cg(A, b, tolerance, maxiter[, workspace])` starts at zero and uses unpreconditioned CG. `sp.pcg` takes the same arguments and uses Jacobi preconditioning. The matrix must be square, symmetric, and positive definite. Symmetry and a positive diagonal are checked explicitly. Positive definiteness is a caller precondition, with nonpositive curvature detected during iteration. Tolerance is in (0, 1) and `maxiter` is in 1–1,000,000.

The result map has `solution`, `iterations`, `residual`, `converged`, and `reason`. Reasons include `converged`, `max_iterations`, `nonpositive_curvature`, and `numerical_breakdown`. Convergence uses the recomputed Euclidean norm of `b - A*x` divided by the norm of `b`. `sp.verify(A, b, x)` computes the same quantity, or the absolute residual norm when `b` is zero. For a valid zero right-hand side, the solver returns zero in zero iterations.

Use `sp.workspace(n)` to reuse five Float64 work vectors across sequential solves. With no supplied workspace, the solver allocates them once per nonzero solve and frees them before returning. Each solve returns a separate dense solution.

`sp.set_budget(bytes)` sets a per-interpreter limit. Use `sp.set_budget(mun)` to clear it. `sp.memory()` reports `used` and `limit`. The limit covers managed sparse payloads, overlapping builder buffers during growth, conversion scratch, and solver workspace. Solution storage is checked against the budget during the solve. Once returned, the solution is an ordinary dense tensor. The reported bytes exclude object headers, VM lists, ordinary dense tensors, allocator overhead, and I/O buffers. The budget is distinct from process peak RSS. Reducing the limit below currently owned bytes raises an error. Unreachable objects are reclaimed by the ordinary GC.

Run `./arda examples/moria.arda` for a builder and PCG example.

## Local validation and benchmarks

Measurements use `python3 benchmarks/moria_bench.py --rounds 5 --scipy --cpu 1` on Linux. The harness runs each case in a fresh process. Timings are medians, and peak RSS is the median per-process Linux high-water mark. The list and builder workloads consume identical triplets in identical order. SciPy uses the same tridiagonal system, zero initial guess, Jacobi preconditioner, and relative tolerance of 1e-9.

The local run used Debian 13 on an Intel Xeon Platinum 8573C, one pinned logical CPU, GCC 14.2.0, and the default Arda build (`-O2`, without BLAS). The reference used NumPy 2.3.5 and SciPy 1.17.0 with OpenBLAS limited to one thread.

| 100,000-row workload | List → COO → CSR | Typed builder → CSR | List path peak RSS | Builder path peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Tridiagonal (299,998 triplets) | 145.24 ms | 93.59 ms | 34.39 MiB | 24.30 MiB |
| Unsorted duplicates (500,000 triplets) | 215.33 ms | 133.62 ms | 56.26 MiB | 36.80 MiB |

Assembly time includes entry generation and CSR finalization. Peak RSS also includes the subsequent SpMV checks.

| 1,000,000-row tridiagonal SPD case | Direct CSR build | SpMV | PCG | Peak RSS with PCG | Iterations | Relative residual | Relative solution error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arda | 25.73 ms | 8.89 ms | 255.32 ms | 134.58 MiB | 12 | 5.79e-10 | 3.59e-10 |
| SciPy | 21.19 ms | 3.36 ms | 126.87 ms | 146.32 MiB | 12 | 5.79e-10 | 3.59e-10 |

The separate Arda direct-build and SpMV case peaked at 88.81 MiB. SpMV timings are averages of eight calls after a warm-up within each process, followed by the median across five processes. The PCG figures include solver setup, iteration, and result creation. SciPy's process RSS includes the Python runtime and imports.

Local validation passed 145 regression tests and 12 independent reference test methods in each of the default, BLAS/LAPACK, and ASan/UBSan/LeakSanitizer configurations. The suite covers invalid and empty inputs, builder atomicity, budget failure recovery, Matrix Market round trips, and comparison with SciPy for sparse algebra and SPD solves. Generated-file checks passed. The sanitizer build emitted the existing `-Wclobbered` warning in the VM. The 0.14.4 release validation and archive remain the next milestone.
