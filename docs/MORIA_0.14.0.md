# Arda 0.14.0 — Moria: Sparse Foundations

This release implements work packages S0–S5 of the master roadmap. It is the first Moria milestone, not the entire 0.14 series. No builder, sparse algebra, sparse solver, hard memory budget, MPI or general MathIR is implemented yet.

## API and representation

Import `sparse` as `sp`. COO and CSR are immutable native objects, not dense arrays; `type(A)` is `sparse` and `sp.format(A)` returns `coo` or `csr`. Dimensions and coordinates are zero-based signed Arda integer inputs, with dimensions in 0..UINT32_MAX and indices less than their corresponding dimension. Native local coordinates are uint32_t, row offsets size_t, values Float64. Future HPC global indices and partitions must be distinct from these local buffers. No dependency on MPI.

| API | Contract |
| --- | --- |
| `sp.coo(rows,cols,row,col,values)` | Component lists of equal length; finite numeric values, duplicates and explicit zeros allowed; preserves input order |
| `sp.csr(rows,cols,row_ptr,col,values)` / `sp.crs` | Strict canonical CSR, validates offsets and indices before buffer allocation; finite nonzero values |
| `sp.to_csr(A)` / `sp.to_crs(A)` | Canonicalize COO; return the same immutable object if already CSR |
| `sp.to_coo(A)` | Expand CSR into sorted canonical COO; return the same object if already COO |
| `sp.shape(A)`, `sp.nnz(A)`, `sp.format(A)` | Shape, stored entry count (not distinct mathematical nonzeros for COO), format |
| `sp.bytes(A)` | Owned payload bytes, excluding object header, allocator overhead, conversion workspace, and VM lists |
| `sp.get(A,r,c)` | Checked nonnegative integer indices; missing coordinate gives zero; COO sums duplicates in input order, CSR binary-searches the row |
| `sp.components(A)` | Independent copies as three lists: `[row or row_ptr, col, values]`; may be expensive for large matrices |
| `sp.from_dense(A)` | Explicit conversion from rank-two Float64 array to canonical CSR; honors strides, rejects nonfinite values |
| `sp.todense(A)` | Explicit bridge to rank-two dense array; checks the existing dense cap before canonicalizing; no implicit densification |
| `sp.describe("to_csr")`, `sp.describe("to_coo")` | Static versioned mathematical algorithm descriptions, no numeric execution; precursor to Rúmil, not `getTex()` |

CSR requires `rows+1` row offsets, first zero, last nnz, nondecreasing; columns strictly increase within each row. Empty rows repeat offsets. Rectangular and zero-sized matrices are supported. COO → CSR counts rows, computes prefix sums, scatters in input order, sorts each unsorted row by column and original position, then merges duplicates using Float64 addition in input order and removes exact zeros. Nonfinite intermediate sums are errors, even if later cancellation would restore a finite sum. Round trips preserve the canonical mathematical matrix, not original ordering, duplicate identity or stored zero signs. Floating-point reassociation is not performed. For already sorted rows the sort is skipped. Time is O(rows + nnz + sum(k_r log k_r)) with a conventional O(k log k) library sort; C does not specify qsort's complexity. CSR → COO is O(rows + nnz). COO lookup is O(nnz); CSR lookup is O(log(k_r+1)).

Payload storage: COO = 16*nnz bytes; CSR = 12*nnz + sizeof(size_t)*(rows+1) bytes (on ordinary platforms with 8-byte doubles and 4-byte uint32_t). COO conversion scratch uses one Coordinate per original entry (24 bytes on the validated x86-64 build), plus the output offsets and exact canonical payload; libc qsort may use additional temporary memory. Scratch is accounted while live and freed on success, or GC-owned until reclamation on error. No dense allocation in format conversion. Native buffer size arithmetic and VM accounting are checked; successful allocations immediately belong to the GC object. OS overcommit, general runtime OOM, and overall process memory are not guaranteed by these checks. Large row pointer arrays or exported VM lists may be expensive; a hard sparse budget and typed builder follow in later Moria milestones.

## Example

Run `./arda examples/moria.arda`. The unordered 3x4 COO example produces CSR row pointers `[0,2,2,4]`, columns `[1,3,0,3]`, values `[5,-1,4,7]`, and dense matrix `[[0,5,0,-1],[0,0,0,0],[4,0,0,7]]`. The original COO and canonical CSR both use 80 payload bytes on x86-64 (five original entries vs four canonical entries and four offsets).

## Local validation

Environment: Debian 13, x86-64, GCC 14.2.0, Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0. Baseline: 126 regular and eight reference methods passed before changes. After changes: 136 regular and nine independent reference methods passed in each of default, BLAS/LAPACK and ASan/UBSan/LeakSanitizer builds. New reference test covers 24 randomized/empty rectangular COO cases and canonical round trips against SciPy. Tests include stable cancellation, invalid CSR/COO indices and offsets, duplicate overflow, empty matrices, strided dense bridges, immutable component copies and repeated failed conversions with GC stress. `make check-generated` passed. Existing sanitizer compiler warning about `self` potentially clobbered by longjmp remains; no sanitizer failures. clang-format unavailable, so format-check was not executed. ARM/Ubuntu and million-row builder/SpMV/CG gates are not validated here. No hosted CI or GitHub runners were used.

## Conversion microbenchmark

`benchmarks/moria_conversion.arda` assembles a 100,000 x 100,000 tridiagonal matrix with 299,998 entries using VM lists, constructs COO, releases the lists, converts to CSR and back to COO. Seven local runs on one allowed CPU (affinity CPU 0), default GCC -O2 build: median list assembly 91.616 ms, COO construction 3.300 ms, COO → CSR 6.866 ms, CSR → COO 0.413 ms; median end-to-end peak RSS 34,176 KiB (33.375 MiB), measured by GNU time. Payloads: COO 4,799,968 B, CSR 4,399,984 B; hypothetical dense 80 GB. These results are specific to the test environment, include list assembly in peak RSS, and do not measure the future native builder, SpMV or CG. The timings use time.monotonic around each operation and include native call overhead. No speed comparison to SciPy is claimed.

Next: 0.14.1 typed assembly and direct sparse constructors, 0.14.2 algebra and operators, 0.14.3 iterative sparse solvers and budgets, 0.14.4 final Moria validation.
