# Beyond-RAM processing and visualization

## 0.21.2 — File computation

These operations take `ARDATEN1` files. Input readers retain open descriptors, so replacing a path preserves each active reader's source. Use stable source files throughout a run. Operations that write a file publish an atomic replacement after successful completion. The byte quota includes the file header. [Tensor file contracts](typed-tensors.md#tensor-files)

| Operation | Contract |
| --- | --- |
| `compute.stream_stats(path[,block_elements[,options]])` | Existing one-pass real-valued count, mean and population variance. Supports optional prefetch and a memory-budget plan. |
| `compute.reduce_file(source,initial,step[,block_elements])` | Visit flat typed blocks in row-major order and return the result of repeated `step(accumulator,block)` calls. Returns the initial value for an empty file. |
| `compute.map_file(source,destination,callback[,dtype[,block_elements[,disk_budget]]])` | The callback returns a packed vector of the same length and selected dtype for each input block. The file retains the original shape and defaults to the original dtype. Returns the number of written elements. |
| `compute.filter_file(source,destination,predicate[,block_elements[,disk_budget]])` | Call a Boolean predicate once per scalar and write a packed vector of selected values in input order and dtype. The staged file initially has the input element count, which must fit a single dimension and the disk budget. The final file is shortened to the selected count. Returns that count. |
| `compute.file_matmul(left,right,destination[,tile[,disk_budget]])` | Compatible Float64 matrices in separate files. Produces a Float64 matrix file. Square tile width defaults to 64 and accepts 1–256. |
| `compute.file_spmv(row_ptr,col_idx,values,x,destination[,block_elements[,disk_budget]])` | Rank-one Int64 CSR row pointers and columns, Float64 values, and a Float64 input vector in four files. Writes a Float64 output vector. Row pointers start at zero and end at the stored entry count. Each row has strictly increasing zero-based columns. |
| `compute.file_qr_r(source[,block_rows])` | Reduce a nonempty tall Float64 matrix to its upper-triangular factor `R`. Returns the square factor as a Float64 numeric array. Block height defaults to 256 and must be at least the column count. |
| `compute.file_svd_right(source[,block_rows])` | With LAPACK, return `[singular_values,Vt]` from the SVD of the file QR's `R`. This gives the full right singular spectrum and vectors of the tall input up to numerical error. |
| `compute.file_rsvd_right(source,rank[,oversample[,seed[,block_rows]]])` | With LAPACK, return approximations to the leading singular values and right singular vectors. Oversampling defaults to 8, the seed to 0, and the block height to 256. The block height must cover the sketch width. |

Callback block sizes default to 65,536 elements. Callbacks run synchronously. Arda releases each consumed block and runs collection before the next block; callbacks control the memory of retained data and the accumulator. The filter builds bounded VM lists for each block and a typed result block. The tensor budget tracks tensor buffers and workspace; VM lists and callback allocations also contribute to process RSS. [Callback implementation](../../lib/compute.arda) · [Streaming statistics](typed-tensors.md#memory-budgets-and-streaming)

The tiled product holds two input tiles and one output tile, with at most `24 * tile ** 2` workspace bytes for large axes. It accumulates in ascending inner-index order and writes each output tile once. With matrix dimensions $m \times k$ and $k \times n$, input payload reads are $8mk\lceil n/b\rceil + 8kn\lceil m/b\rceil$ bytes for nonempty products with tile width $b$. The result payload is $8mn$ bytes. This counts logical requests. Filesystem cache and readahead affect device traffic. Empty inner axes produce a logically zero-filled result. [Tiled product and file workspace](../../src/file_compute.c)

The CSR product reads indices and values in bounded blocks with one cached segment of the input vector. It processes rows and their entries in stored order, accumulates in `long double`, and checks that each result is representable as Float64. The workspace is at most 24 bytes per requested block element. Irregular access can reread vector segments. CSR structural or numeric errors roll back publication.

The tall QR uses a flat sequential reduction. It factors each new row panel stacked below the previous triangular factor using the shared scaled Householder implementation. It reads each input element once and holds at most $8(b+n)n$ bytes of workspace plus $8n^2$ bytes for an $n$-column result. Its work is $O(mn^2)$ for $b \ge n$. The SVD stage operates on the square factor using the existing LAPACK workspace checks. [Communication-optimal QR and LU](https://doi.org/10.1137/080731992) · [Householder implementation](../../src/linalg.c)

The randomized SVD is a two-pass row-space range finder. The first pass forms $A^T\Omega$ with streamed Gaussian weights and constructs a Householder basis $Z$. The second pass reduces $AZ$ to a small triangular factor, from which LAPACK computes the singular spectrum and right vectors. Those vectors are lifted back through $Z$. A sketch of width $p$ uses $O(np+(b+p)p)$ storage before the final LAPACK stage, independent of the number of input rows. The two data passes read $16mn$ payload bytes for an $m$-by-$n$ Float64 matrix. Approximation quality depends on the spectrum, oversampling, and seed. This implementation uses no power iterations and returns right factors. Left or orthogonal output factors and external COO sorting remain open. [Randomized range finding](https://doi.org/10.1137/090771806) · [Arda implementation](../../src/file_compute.c)

Recent randomized CholeskyQR variants use a preconditioner to improve accuracy on full-rank tall matrices. The current Householder path also handles dependent columns, and the reference test exercises that case. [RCLUPPr analysis](https://arxiv.org/abs/2607.15561)

## 0.21.3 — SVG and VTK exports

`plot.svg(kind,source,destination[,options])` reads a real-valued tensor file and writes a standalone SVG. It uses a bounded scan buffer, two such buffers for contour cells, and bounded bin counters for histograms. Sampled values must be finite. `disk_budget` optionally caps the exact output bytes. The renderer uses the same samples for the range and drawing passes.

| Kind | Source and options |
| --- | --- |
| `"line"` | Rank-one ordinates with zero-based x coordinates, or `[points,2]` x/y pairs in stored order. |
| `"scatter"` | `[points,2]` x/y coordinates. |
| `"histogram"` | Any nonempty real tensor. `bins` defaults to 32 and accepts 1–4,096. Equal-width bins include the maximum in the final bin. A constant series uses the first bin. The histogram scans every value. |
| `"heatmap"` | A scalar grid `[rows,columns]`, with the first row at the top. Linear color scale over sampled extrema. |
| `"contour"` | A scalar grid and a required finite `level`. Marching squares uses linear edge interpolation and the cell-center value to resolve alternating corners. At least two sampled nodes per axis are needed. |
| `"vector"` | `[rows,columns,2]` with horizontal and upward components in that order. Arrow lengths share a scale based on the largest sampled absolute component, and zero vectors are omitted. |

`stride` defaults to 1 for series and grids. A larger stride selects indices `0,stride,2*stride,...` along each displayed axis. This is explicit point sampling for visualization. The result can miss features between selected indices. The SVG emitter scans in file order where applicable and writes primitives incrementally. [SVG paths](https://www.w3.org/TR/SVG2/paths.html) · [Renderer and publication](../../src/file_plot.c)

`plot.vtk(source,destination[,disk_budget])` exports a nonempty Float64 scalar grid in legacy binary VTK 3.0 `STRUCTURED_POINTS`. The shape is `[rows,columns]` or `[depth,rows,columns]` with up to 2,147,483,647 points. VTK receives x-fastest ordering, unit spacing and zero origin. The values are copied in bounded chunks with big-endian byte order. PNG and additional VTK data layouts remain open. [VTK legacy file format](https://docs.vtk.org/en/latest/vtk_file_formats/vtk_legacy_file_format.html)

```arda
tol compute as c
tol tensor as t
tol plot as p
t.save("grid.atn",t.array([0,1,2,3],"float64",[2,2]))
c.file_matmul("grid.atn","grid.atn","product.atn",1,160)
p.svg("heatmap","product.atn","product.svg",{"disk_budget":2048})
p.vtk("product.atn","product.vtk",512)
pedo(t.tolist(t.read_tile("product.atn",[0,0],[2,2])))
```

The program prints `[[2, 3], [6, 11]]`.

## Validation

The regression cases check budgeted block lifetime, rollback on callback and CSR validation failures, empty results, SVG geometry and exact VTK binary layout. Independent NumPy and SciPy fixtures check matrix products, $R^TR$, singular values, right-vector orthogonality, spectral reconstruction, and a seeded low-rank approximation. [Regression cases](../../tests/test_beyond_ram.py) · [Reference comparisons](../../tests/reference_beyond_ram.py)

The portable GCC 14.2 `-O2` build ran on a Linux cloud AMD EPYC 9V74 with one Arda thread and an overlay filesystem. The dense comparison used seeded normal inputs of size 512 × 384 and 384 × 512 (NumPy generator seed 21023) with tiles of width 64. Both paths included publishing the 2 MiB output and produced byte-identical results. The maximum absolute error against NumPy was 4.97 × 10⁻¹⁴. Median wall time and peak RSS over seven invocations after two warmups were 47.85 ms and 7,008 KiB for mapped inputs and in-heap tiled output, and 56.29 ms and 1,908 KiB for the file product. The file path requested 24 MiB of input payload and used 96 KiB of workspace. The measurement used warm files.

A separate run multiplied a 4,081 × 1 matrix filled with 2 by a 1 × 4,000 matrix filled with 3. The 16,324,000-element output used 130,592,768 allocated filesystem bytes and exceeded both the heap tensor element limit and the process address-space limit of 64 MiB. With 128-wide tiles and a 256 KiB tensor memory budget, publication took 0.368 s and reached 1,788 KiB peak RSS. A separate full-file scan under the same memory limits returned the expected count, mean 6 and variance 0. The real-file 20 GB and 80 GB acceptance gates are pending.
