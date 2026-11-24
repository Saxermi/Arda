# Complex scalars, structured operators and randomized SVD

**Status** — Implemented in Arda 0.15.3. Complex scalars, operator products, and zeroth-order optimization run in the portable build. Randomized SVD needs `LAPACK=1`. [Release validation](../../docs/RUMIL_0.15.3.md)

## Complex scalars

`complex.new(re,im)` makes an immutable scalar with two Float64 components. `real`, `imag`, `conj`, `abs` and `arg` expose its components, conjugate, magnitude and principal phase. Scalars support mixed real and complex `+`, `-`, `*`, `/`, numeric equality, and unary negation. A zero complex divisor raises an error. Zero complex scalars are false in conditions. Integer operands are converted to floating point for arithmetic, with comparisons against integers retaining the runtime's exact integer-to-Float64 comparison. `exp`, `log`, `sqrt`, `sin`, `cos`, `tan` and `pow(z,w)` use C's complex math functions, including signed-zero branch conventions. The global `abs` also accepts a complex scalar. Fixed-size scalar work and storage are independent of input magnitude. Arrays and linear operators still store real Float64 entries. [Source](../../src/complex.c) · [C11 draft, §§6.2.5 and 7.3](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf) · [DLMF elementary functions](https://dlmf.nist.gov/4)

## Portable matrix operators

`linop` constructors snapshot factor tensors. `shape(A)` returns `[rows,cols]`. `apply(A,x)` and `apply_transpose(A,x)` accept real rank-one vectors or rank-two blocks with right-hand sides in columns, including strided inputs. Products also accept dense and COO/CSR matrices through the existing `LinearOperator` interface. COO conversion uses temporary sparse storage and honors the sparse memory budget. Nonfinite coefficients, right-hand sides and results raise errors. Two-stage contractions store the intermediate vector in Float64. [Source](../../src/linop.c)

| Constructor | Map and cost per vector |
| --- | --- |
| `lowrank(U,V)` | $UV^T$ for equal factor column counts $r$. Apply $V^Tx$ then $U$ in $O((m+n)r)$ work, with an $r$-element scratch vector. |
| `diagonal(d)` | Entrywise product, $O(n)$ work and factor storage. |
| `banded(l,u,bands)` | Square matrix with `bands[u+i-j,j]` encoding $A_{ij}$ inside the band. $O(n(l+u+1))$ work and factor storage. |
| `toeplitz(c,r)` | Rectangular Toeplitz map using first column `c` and first row `r` with the same first value. Direct multiplication takes $O(mn)$ work and $O(m+n)$ factor storage. |
| `kron(A,B)` | Kronecker product of two rectangular factors. With the input reshaped in row-major order, evaluate $AXB^T$ and flatten. Choose a contraction order that uses the smaller intermediate. |

Transpose products exchange factor roles or index orientations. The Kronecker map chooses between an $n_A\times m_B$ and an $m_A\times n_B$ intermediate. Consequently its scratch is bounded by the smaller cross product, while work is the sum of the two factor contractions. [Van Loan 2000](https://doi.org/10.1016/S0377-0427(00)00393-9)

## Randomized SVD

`linalg.rsvd(A,k[,oversampling[,power_iterations[,seed]]])` returns `[U,s,Vt,estimated_relative_error]`. Defaults are eight oversampling columns, one power iteration and seed zero. Power iterations are in `[0,32]`. For dense inputs, the rank and sketch width are capped at the smaller axis. For sparse and structured operators they are strictly below it. All output and workspace tensors obey the existing element limit. The output has shapes `[m,k]`, `[k]`, and `[k,n]` for its factors.

Let $\ell$ be the capped sketch width. Apply $A$ to a seeded Gaussian sketch and obtain $Q$ by Householder QR. Each power step applies $A^T$ and $A$, with reorthogonalization after each product. Form $A^TQ$ as a skinny matrix, apply LAPACK `DGESVD`, and map back and truncate to $k$ components. Storage, including factors and reused sketch buffers, is $O((m+n)(\ell+k)+\ell^2+w)$ plus operator scratch and provider workspace $w$. The two-sided iteration uses $2q+2$ forward or transpose block products for $q$ power steps. QR and projection work outside the applies scales as $O((q+1)(m+n)\ell^2)$, plus the rectangular SVD. [Halko, Martinsson and Tropp 2011](https://doi.org/10.1137/090771806) · [Source](../../src/linalg.c)

The error is estimated as $\|(A-U\operatorname{diag}(s)V^T)G\|_F/\|AG\|_F$ using eight Gaussian vectors from a separate seeded stream. Error estimation applies both maps without reconstructing a dense approximation. The ratio is a sampled diagnostic with no guaranteed bound. The reference suite compares it with an explicitly measured Frobenius error on a small matrix and checks a 5,000×6,000 exact rank-one operator. Block Krylov methods can retain a larger subspace and recover clustered spectra in fewer products. Arda uses subspace iteration as the smaller-workspace baseline. [Chen et al. 2026](https://doi.org/10.1137/1.9781611978971.42) · [Reference tests](../../tests/reference_structured.py)

## Objective-only optimization

`opt.zeroth_order(f,x0[,method[,radius[,tolerance[,maxiter[,seed[,directions]]]]]])` minimizes a real scalar callback over a nonempty real vector. Defaults are `"coordinate"`, radius $10^{-4}$, tolerance $10^{-6}$, 200 iterations, seed zero and eight random directions. It returns `[x,iterations,gradient_estimate_norm,converged,value,evaluations]`, with `converged` set when the estimated norm reaches tolerance. Coordinate differences use two samples per coordinate. The random method uses normalized Gaussian directions and scales averaged directional differences by the dimension. Both count objective calls and require a sufficient objective decrease during backtracking. A fixed probe radius introduces finite-difference error. Random directions add variance, so the stopping flag reports an estimator test. With dimension $n$ and $d$ random directions, the two modes use $2n$ or $2d$ objective calls per gradient estimate and $O(n)$ live computational scratch. Each candidate is given to the callback as a copy. [Source](../../lib/opt.arda) · [Nesterov and Spokoiny 2017](https://doi.org/10.1007/s10208-015-9296-2) · [Ma and Huang 2025](https://arxiv.org/abs/2510.19975)

## Local performance

The low-rank product used $m=768$, $n=640$, $r=4$, $U_{ik}=\sin(0.013ik)$, $V_{jk}=\cos(0.017jk)$ and $x_j=\sin(0.019j)$ with one-based indices. Builds used GCC 14.2 at `-O2` on one pinned logical CPU of an Intel Xeon Platinum 8573C, with NumPy 2.3.5 and SciPy 1.17.0. Product times are medians of seven calls after warm-up, excluding setup and validation. Whole-process peak RSS includes the runtime and imports.

| Product | Arda portable (ms) | NumPy (ms) | Arda / NumPy peak RSS (MiB) |
| --- | ---: | ---: | ---: |
| Apply preformed dense matrix | 0.594 | 0.145 | 11.7 / 51.2 |
| Apply low-rank factors | 0.00786 | 0.00428 | 11.9 / 47.7 |

Both products agreed with the factor expression within $8\times10^{-16}$ relative error. One randomized-SVD run using the same factor formulas with a 50,000×60,000 rank-two operator took 236 ms and 22.4 MiB peak RSS with single-threaded OpenBLAS 0.3.29. Its probe residual was $5.02\times10^{-15}$, and the singular values agreed with an independent SciPy QR and rank-two SVD computation within $1.5\times10^{-15}$ relative error. A dense Float64 copy of this operator would require 24 GB. [Release checks](../../docs/RUMIL_0.15.3.md)

**Bibliography keys** — `ChenEtAlBlockKrylov2026`, `HalkoEtAl2011`, `MaHuang2025`, `NesterovSpokoiny2017`, `NISTDLMF`, `ReferenceLAPACK`, `VanLoan2000`.
