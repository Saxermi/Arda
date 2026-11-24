# Static mathematical documentation

**Status** — Native descriptor and cost models plus user-function MathIR in the 0.15.4 development branch; release validation pending.

## Queries

For documented native functions, call `f.getTex([level])`, `f.getAlgorithm()`, `f.getAssumptions()`, `f.getComplexity()`, or `f.getCost(spec)`. The `mathdoc` module exposes the same queries with `f` supplied as the first argument. Both forms look up documentation independently of numerical execution. The registry resolves the native function identity, so importing an alias preserves its documentation. A callable-derived query bound to a function can outlive the original variable.

`getTex` returns a LaTeX fragment without math delimiters. Its four levels are independent views.

| Level | Content |
| --- | --- |
| 0 (default) | Mathematical definition |
| 1 | Intermediate mathematical relations |
| 2 | Shapes and assumptions |
| 3 | Available backend algorithm formulas |

An unavailable field returns `mun`. This includes unsupported user-defined functions and bound instance methods. Invalid level values raise an error. The libm `cos` descriptors expose cosine semantics and the provider boundary. The implementation formula and numerical cost of the provider are unknown. `numc.solve.getTex(3)` gives pivot selection, elimination and triangular substitutions. `numc.matmul` describes the selected build's portable or CBLAS path, and `linalg.rsvd` reports whether its optional LAPACK implementation is available. Native metadata lives outside numerical kernels and contains pre-authored TeX fragments. [Source](../../src/mathdoc.c) · [Runnable example](../../examples/mathdoc_queries.arda)

## Shape-based estimates

`getCost` models supported work from a map with a `shape` list of integer dimensions. `dtype` is optional and defaults to `"float64"`. CSR accepts `nnz` and an optional `format` that defaults to `"csr"`. A request for `"coo"` returns `mun` because the CSR estimate excludes canonicalization. The estimates assume inputs already have the expected dtype and layout.

| Callable | Specification | Floating arithmetic model |
| --- | --- | --- |
| `numc.solve` | `{"shape":[n,n]}` | Successful LU factorization and one vector solve. Counts elimination, forward and backward substitutions, and divisions. |
| `numc.matmul` | `{"shape":[m,k,n]}` | Nominal $2mkn$ operations, counting a multiply-add as two |
| `numc.dot` | `{"shape":[n]}` | Nominal $2n$ operations, counting a multiply-add as two |
| `sparse.matvec` | `{"shape":[m,n],"nnz":z}` | Nominal $2z$ operations for canonical CSR |

Results contain `estimated`, `flops`, `output_bytes`, `workspace_bytes`, `backend`, and `model`. Counts exclude comparisons, row swaps, dispatch and initialization. Byte counts are for known Arda numeric buffers, with scalar output counted as one Float64. They exclude input storage, object headers, implicit strided-input packing and private provider workspace. `workspace_bytes` for dense solve includes the copied matrix and row permutation. The fields describe a size model and can be computed for a large symbolic shape without allocating its inputs. [Linear algebra implementations](linear-algebra.md) · [CSR implementation](sparse-matrices.md)

## User-defined functions

The compiler extracts immutable MathIR before releasing the AST. It supports integer/finite floating literals, parameters, external name references, unary negation, addition, subtraction, multiplication, division, straight-line local declarations, a final return expression and expression lambdas. Calls through a source name or single qualified name are represented as opaque calls: even a spelling such as `cos` may be rebound at runtime. Native callable queries still resolve actual native identity; user-function queries do not infer that identity from source spelling or invoke a callee. External values are not captured or evaluated by the query. Numeric semantics are conditional on suitable operands; types, domains, overflow, shape, aliasing and exact floating-point equivalence are not inferred.

Level 0 expands local names when safe and within limits, falling back to named relations otherwise; level 1 preserves local declarations; level 2 labels unknown assumptions and dependencies; level 3 shows source-level assignments and return, not a backend algorithm. User-function `getAlgorithm` and `getAssumptions` explain this limited coverage; `getComplexity` and `getCost` return `mun`. Bound instance methods, default arguments, reassignments, branching, loops, recursive analysis, and unsupported expressions return `mun` or, for syntactically supported opaque calls, explicitly unresolved documentation. No algebraic rewrites or execution tracing occur.

The IR is one frontend allocation per supported prototype, including copied identifier strings. It has at most 128 nodes, 32 parameters, 128 bytes per identifier and 32 extraction levels. Rendering has a depth limit of 64 and an 8192-byte buffer including the terminator. Overflow returns `mun` or falls back to named steps at level 0. Underscores are escaped; non-ASCII identifier code points are rendered as explicit U+ tokens. The prototype owns the allocation and frees it with its compilation root, including frontend error cleanup. No VM object or AST pointer is retained. Ordinary numerical execution has no new instructions or per-element instrumentation.

`--mathdoc=compact` (proposed default) collects this bounded metadata during compilation, including imported modules and REPL submissions. `--mathdoc=none` skips user-function extraction; existing native descriptors remain queryable. No AST reconstruction is attempted when metadata is unavailable. These are CLI modes, not runtime profiling switches. See [implementation](../../src/mathir.c), [tests](../../tests/test_mathdoc.py) and [validation checklist](../../docs/RUMIL_0.15.4.md).
