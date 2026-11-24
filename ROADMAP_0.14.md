# Arda 0.14 — Resource-Efficient Sparse Computing

*Development codename: Moria*

**Status** — Assembly, algebra, and solvers are implemented through 0.14.3. See the [foundations report](docs/MORIA_0.14.0.md) and the [current API and validation guide](docs/MORIA_0.14.3.md). See the [0.14.4 release validation](docs/MORIA_0.14.4.md) for the completed gates and source-archive checks.

CSR (Compressed Sparse Row) and CRS (Compressed Row Storage) are synonyms. Use `csr` as the primary API name and consider `crs` as an alias. COO (Coordinate format) is also a first-class format, not a temporary dense matrix.

## Foundation APIs

- `sp.coo(rows, cols, row_indices, col_indices, values)`
- `sp.csr(rows, cols, row_ptr, col_indices, values)`
- `sp.to_csr(coo)` and `sp.to_coo(csr)`; optionally `sp.to_crs` alias
- `sp.shape`, `sp.nnz`, `sp.bytes`, `sp.todense`, `sp.from_dense`, `sp.matvec`

For the implemented builder, diagonal constructors, sparse algebra, and CG/PCG, see [the 0.14.3 sparse computing guide](docs/MORIA_0.14.3.md).

## Storage and conversion contract

Use zero-based integer row/column indices; use checked size_t row offsets, Float64 values, and GC-managed buffers. Validate dimensions, array lengths, bounds, overflow and row pointer monotonicity. `row_ptr` has `rows + 1` entries, begins at 0 and ends at nnz. Empty rows have repeated offsets. Rectangular and empty matrices are supported; no dense allocation during conversion. Canonical CSR sorts column indices per row and combines duplicate COO coordinates by summation. Canonical CSR drops exact zeros after summation. Preserve the mathematical matrix on round trips, not original ordering or duplicate identity. Conversion uses counting and prefix sums followed by in-place sorting of unsorted rows. CSR -> COO takes O(rows + nnz).

Example (3 x 4):

```
COO row = [2, 0, 0, 2, 0]
COO col = [3, 1, 1, 0, 3]
COO val = [7, 2, 3, 4, -1]
CSR row_ptr = [0, 2, 2, 4]
CSR col = [1, 3, 0, 3]
CSR val = [5, -1, 4, 7]
Canonical COO row = [0, 0, 2, 2]
Canonical COO col = [1, 3, 0, 3]
Canonical COO val = [5, -1, 4, 7]
```

The duplicate (0,1) is combined to 5; row 1 is empty. The dense matrix is [[0,5,0,-1],[0,0,0,0],[4,0,0,7]]. Multiplication by [1,2,3,4] gives [6,0,32].

For a 10,000 x 10,000 tridiagonal matrix: nnz = 29,998. Dense Float64: 800,000,000 bytes. COO (32-bit row and column indices): 479,968 bytes. CSR (32-bit columns, 64-bit row pointers): 439,984 bytes. Excludes metadata and allocation overhead.

Test against SciPy COO/CSR, including unordered duplicates, cancelling entries, zero dimensions, empty rows, non-square matrices, invalid indices, overflow, round trips, and GC stress. Keep all 0.13.1 tests passing. Do not increase the existing dense-array limit merely to implement sparse matrices.
