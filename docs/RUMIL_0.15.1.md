# Arda 0.15.1 — SVD, Eigenvalues, and Conditioning

*Development codename: Rúmil*

This release adds reduced SVD, pseudoinverse, numerical rank, condition estimates, and symmetric and general eigenvalue solvers to `linalg`. These APIs use the optional LAPACK build. `eigh` returns real eigenvalues and column eigenvectors. `eig` returns real and imaginary components for its eigenvalues and right eigenvectors. [Eigenvalue example](../examples/eigenpairs.arda) · [Numerical contracts](../research/implemented/linear-algebra.md#lapack-eigensolvers)

## Source archive validation

Run the portable, LAPACK and LAPACK-enabled sanitizer configurations locally. Each configuration has 153 regression tests. The LAPACK configurations run 23 reference tests. The portable configuration runs 19 and skips four that require LAPACK. The validation environment used Debian 13 x86-64, GCC 14.2, OpenBLAS 0.3.29, NumPy 2.3.5 and SciPy 1.17.0.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
make test test-reference check-generated
make BLAS=1 LAPACK=1 test test-reference
./arda --gc-stress examples/eigenpairs.arda
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    make BLAS=1 LAPACK=1 SANITIZE=1 test test-reference
```

## Numerical acceptance

The symmetric reference case checks the eigenvalues and a repeated-eigenvalue subspace against SciPy. It has relative residual $\|AQ-Q\Lambda\|_F/\|A\|_F=8.39\times10^{-16}$ and orthogonality error $\|Q^TQ-I\|_F=2.33\times10^{-15}$. The existing `calc.eigh` entry point gives the same results.

For the general cases, the maximum column residual $\max_j\|Av_j-\lambda_jv_j\|_2/\|A\|_F$ was $4.92\times10^{-16}$ for the mixed real and complex spectrum, $3.05\times10^{-16}$ for a real spectrum, and $1.19\times10^{-16}$ for a defective Jordan block. The reference tests verify eigenvalues, conjugate pairing, eigenvector normalization and input preservation. SVD reference tests cover reconstruction, orthogonality, rank cutoff, condition numbers and the Moore–Penrose identities. [Reference tests](../tests/reference_linalg.py)
