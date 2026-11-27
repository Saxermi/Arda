# Sparse matrices and iterative solvers

**Status** — Implemented through Moria 0.14.3. Let $m,n$ be row and column counts, $z$ the input stored entries, and $z'$ the canonical stored entries.

## Formats, assembly and algebra

COO holds triplets $(i,j,v)$. CSR holds a row-offset array plus aligned column/value arrays, with entries for row $i$ in $[p_i,p_{i+1})$. Arda's CSR uses strictly increasing columns and no explicit zeros. Canonicalization counts rows, prefix-sums, scatters in input order, sorts unsorted row segments by column and original position using an in-place heap sort, then folds duplicates in input order and drops exact zero. Given $k_i$ triplets in row $i$, work is $O(m+z+\sum_i k_i\log(\max(1,k_i)))$, with $O(m+z+z')$ total conversion scratch and output. Already sorted rows skip sorting. CSR → COO takes $O(m+z')$ time and $O(z')$ output. Duplicate reduction uses Float64 and rejects an intermediate nonfinite result. It is sensitive to input order and cancellation. [Source, `canonical` and `to_coo`](../../src/sparse.c) · [Davis 2006](https://doi.org/10.1137/1.9780898718881)

The typed builder buffers triplets with geometric growth when the budget permits and validates batches before mutation. An append is normally amortized $O(1)$ under geometric growth, while a growth step copies existing entries and can require old and new storage simultaneously. The direct diagonal constructors visit the supplied diagonal entries and count/write CSR without generating triplets, including sorting requested offsets when needed. Matrix Market coordinate real general and symmetric files use one-based indices. Symmetric input is expanded from its lower triangle before canonicalization. [Builder and diagonal routines](../../src/sparse.c) · [Matrix Market implementation](../../src/sparse_market.c) · [NIST format](https://math.nist.gov/MatrixMarket/formats.html)

CSR SpMV computes $y_i=\sum_{j\in\mathcal S_i}a_{ij} x_j$ with extended accumulation and a Float64 result. The kernel uses $O(m+z')$ time with zero additional managed scratch. The public CSR matvec scans the entire input and output for finiteness, giving $O(m+n+z')$ time and $O(m)$ output storage. SpMM applies the kernel to each column and validates the full dense input and result. CSR transpose counts, scans and scatters in $O(m+n+z')$ time, with space proportional to output. Addition merges sorted rows in two passes with $O(m+z'_A+z'_B)$ time and output-sized storage. Scaling drops exact zeros, and diagonal extraction uses per-row binary search. COO may require conversion first. Sparse algebra operates without a dense matrix intermediate. [Source, `csr_apply`, `sparse_transpose`, `merge_rows`, `sparse_scale`, `sparse_diagonal`](../../src/sparse_algebra.c)

## CG, Jacobi-PCG, and matrix-free Poisson

For SPD $A$, CG minimizes the error in the $A$-norm over a Krylov subspace. For an SPD preconditioner $M$, set $z_k=M^{-1}r_k$ and $\rho_k=r_k^Tz_k$. Then $\alpha_k=\rho_k/(p_k^TAp_k)$, $x_{k+1}=x_k+\alpha_kp_k$, $r_{k+1}=r_k-\alpha_kAp_k$ and $p_{k+1}=z_{k+1}+(\rho_{k+1}/\rho_k)p_k$. Plain CG uses $M=I$. Jacobi uses $M=\operatorname{diag}(A)$. For $\kappa=\kappa_2(M^{-1/2}AM^{-1/2})$, the standard exact-arithmetic bound is $\|e_k\|_A\le2((\sqrt\kappa-1)/(\sqrt\kappa+1))^k\|e_0\|_A$. Finite precision can lose conjugacy and delay convergence. Significant cancellation or small residual alone need not imply small forward error. [Hestenes and Stiefel 1952](https://doi.org/10.6028/jres.049.044) · [Barrett et al. 1994](https://doi.org/10.1137/1.9781611971538) · [Wathen 2015](https://doi.org/10.1017/S0962492915000021)

`solver.solve("cg",A,b[,options])` starts at zero for dense and sparse matrices.
Select `"preconditioner":"jacobi"` to obtain PCG. The solver checks symmetry and diagonal
positivity on stored coefficients and detects nonpositive curvature during iteration.
It recomputes the final relative residual, which `solver.verify(A,b,x)["residual"]` can check
independently. A CSR iteration takes O(n + nnz) work and O(n) numeric workspace. For a
dense matrix the product takes O(n²) work. [Krylov implementation](../../src/krylov.c) · [Current contracts](aule-solvers.md)

`linop.poisson(shape)` represents the negative unit-spacing discrete Laplacian with
homogeneous Dirichlet boundaries on a rectangular grid of one through eight dimensions.
Apply costs O(dN) time for d axes and N grid sites. CG uses O(N) workspace. The grid
operator also supports directional weights. [Stencil](../../src/poisson.c) · [Brandt 1977](https://doi.org/10.1090/S0025-5718-1977-0431719-X)

## Rectangular least squares

Use `solver.solve("lsqr",A,b[,options])` or `"lsmr"`. Both use bidiagonalization via
apply and transpose with linear vector workspace. A least-squares search can satisfy
the normal-residual tolerance for an inconsistent system while retaining a nonzero data
residual. [Solver contracts](aule-solvers.md#0191-krylov-methods-preconditioning-and-partial-spectra) · [Paige and Saunders 1982](https://doi.org/10.1145/355984.355989)
