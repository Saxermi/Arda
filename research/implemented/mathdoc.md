# Static mathematical documentation

**Status** — Native descriptor and cost models in the 0.15.4 development checkout.

## Queries

For documented native functions, call `f.getTex([level])`, `f.getAlgorithm()`, `f.getAssumptions()`, `f.getComplexity()`, or `f.getCost(spec)`. The `mathdoc` module exposes the same queries with `f` supplied as the first argument. Both forms look up documentation independently of numerical execution. The registry resolves the native function identity, so importing an alias preserves its documentation. A callable-derived query bound to a function can outlive the original variable.

`getTex` returns a LaTeX fragment without math delimiters. Its four levels are independent views.

| Level | Content |
| --- | --- |
| 0 (default) | Mathematical definition |
| 1 | Intermediate mathematical relations |
| 2 | Shapes and assumptions |
| 3 | Available backend algorithm formulas |

An unavailable field returns `mun`. This includes user-defined functions and methods in this development slice. Invalid level values raise an error. The libm `cos` descriptors expose cosine semantics and the provider boundary. The implementation formula and numerical cost of the provider are unknown. `numc.solve.getTex(3)` gives pivot selection, elimination and triangular substitutions. `numc.matmul` describes the selected build's portable or CBLAS path, and `linalg.rsvd` reports whether its optional LAPACK implementation is available. Static metadata lives outside numerical kernels and contains only pre-authored TeX fragments. [Source](../../src/mathdoc.c) · [Runnable example](../../examples/mathdoc_queries.arda)

## Shape-based estimates

`getCost` models supported work from a map with a `shape` list of integer dimensions. `dtype` is optional and defaults to `"float64"`. CSR accepts `nnz` and an optional `format` that defaults to `"csr"`. A request for `"coo"` returns `mun` because the CSR estimate excludes canonicalization. The estimates assume inputs already have the expected dtype and layout.

| Callable | Specification | Floating arithmetic model |
| --- | --- | --- |
| `numc.solve` | `{"shape":[n,n]}` | Successful LU factorization and one vector solve. Counts elimination, forward and backward substitutions, and divisions. |
| `numc.matmul` | `{"shape":[m,k,n]}` | Nominal $2mkn$ operations, counting a multiply-add as two |
| `numc.dot` | `{"shape":[n]}` | Nominal $2n$ operations, counting a multiply-add as two |
| `sparse.matvec` | `{"shape":[m,n],"nnz":z}` | Nominal $2z$ operations for canonical CSR |

Results contain `estimated`, `flops`, `output_bytes`, `workspace_bytes`, `backend`, and `model`. Counts exclude comparisons, row swaps, dispatch and initialization. Byte counts are for known Arda numeric buffers, with scalar output counted as one Float64. They exclude input storage, object headers, implicit strided-input packing and private provider workspace. `workspace_bytes` for dense solve includes the copied matrix and row permutation. The fields describe a size model and can be computed for a large symbolic shape without allocating its inputs. [Linear algebra implementations](linear-algebra.md) · [CSR implementation](sparse-matrices.md)
