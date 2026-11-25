# Static mathematical documentation

**Status** — Arda 0.15.4 introduced static native and straight-line queries. Arda 0.15.5 adds a limited range-loop model. Arda 0.16.5 adds straight-line subscript notation. [Release validation](../../docs/RUMIL_0.15.5.md)

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

For straight-line functions, level 0 expands local names within limits and falls back to named relations. Level 1 preserves local declarations, level 2 labels unknown assumptions and dependencies, and level 3 shows source-level assignments and return. Floating-point literals use enough significant digits to recover the stored value. User-function `getAlgorithm` and `getAssumptions` describe the supported model. User-function `getComplexity` and `getCost` return `mun`. Bound methods, default arguments, general assignments, branching, recursion and unsupported expressions return `mun` or explicitly unresolved documentation for syntactically supported opaque calls.

The IR is one frontend allocation per supported prototype, including copied identifier strings. It has at most 128 nodes, 32 parameters, 128 bytes per identifier and 32 extraction levels. Rendering has a depth limit of 64 and an 8192-byte buffer including the terminator. Overflow returns `mun` or falls back to named steps at level 0. Underscores are escaped; non-ASCII identifier code points are rendered as explicit U+ tokens. The prototype owns the allocation and frees it with its compilation root, including frontend error cleanup. No VM object or AST pointer is retained. Ordinary numerical execution has no new instructions or per-element instrumentation.

`--mathdoc=compact` (default) collects bounded metadata during compilation, including imported modules and REPL submissions. `--mathdoc=none` skips user-function extraction. Native descriptors remain queryable. [Implementation](../../src/mathir.c) · [Tests](../../tests/test_mathdoc.py)

### Subscripts

Straight-line bracket expressions such as `x[i + 1]` and `A[i][j]` display their source indices as LaTeX subscripts. The result preserves the original index expression. Indexed functions retain their named relations and source operations at all formula levels. The query explains that operand types, shapes and bounds are unresolved. Actual shape and dtype assertions are available at runtime through `tensor.contract`. [Scientific type contracts](scientific-types.md)

### Single range loops

Arda recognizes one `for`/`an` loop after scalar local declarations. Initialize a local accumulator, update it once per iteration with an arithmetic expression, then return it. Bounds may use literals, parameters and prior locals. They must evaluate to integral values within the native range limit. Supply a nonzero literal step or omit it for the default step of 1. Dynamic calls, external values, nested loops, control flow and mutation outside the accumulator cause the query to return `mun`. The query checks the closure's `range` binding against the native builtin without invoking it. Changing that binding can change the result of a later query.

Let `start`, `stop` and `step` be the resolved bounds. The index at iteration $k$ is $i_k=\mathrm{start}+k\,\mathrm{step}$ for $0\le k<N$, where $N=\max(0,\lceil(\mathrm{stop}-\mathrm{start})/\mathrm{step}\rceil)$. The stop is exclusive in either direction. Empty ranges return the initial accumulator. Level 0 displays a sum for `state = state + term` or a product for `state = state * term` when the term does not depend on the state. Other supported updates display a recurrence. Level 1 keeps the order of source operations in the recurrence, level 2 gives its assumptions, and level 3 shows the loop and update. Algebraic sums and products can round differently from sequential Float64 updates. Queries use bounded metadata independent of the iteration count. [Runnable loop example](../../examples/mathematical_library.arda)

### Limited mathematical simplification (accepted 0.15.4 policy)

The default compact metadata mode and opaque call treatment are accepted. Only user-function `getTex(0)` in the call-free, external-free compact view applies bounded formal scalar-real display identities: `x+0`, `0+x`, `x-0`, `x*1`, `1*x`, `x/1` (integer identity literals), double negation, and adjacent repeated parameter factors (such as `v*v` or `0.5*m*v*v` rendered with a square). No constant evaluation, cancellation, multiplication by zero elimination, reassociation of executable arithmetic, or arbitrary symbolic reasoning is performed. These identities are conditional on ordinary finite scalar real mathematics; they do not guarantee the same dynamic type, IEEE signed zero, overflow behavior, or runtime result for arbitrary inputs. Levels 1 and 3 retain source operations; level 2 and `getAssumptions` state the boundary. Calls and external dependencies disable simplification altogether. If expansion exceeds existing limits, the renderer falls back to unsimplified named steps. The immutable source IR, bytecode and VM are unchanged by rendering.
