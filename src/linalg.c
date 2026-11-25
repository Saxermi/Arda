#include "arda.h"
#include "linear_operator.h"
#include <float.h>

#ifdef ARDA_LAPACK
extern void dgesvd_(const char *, const char *, const int *, const int *, double *, const int *,
                    double *, double *, const int *, double *, const int *, double *, const int *,
                    int *);
extern void dsyev_(const char *, const char *, const int *, double *, const int *, double *,
                   double *, const int *, int *);
extern void dgeev_(const char *, const char *, const int *, double *, const int *, double *,
                   double *, double *, const int *, double *, const int *, double *, const int *,
                   int *);
#endif

#define A(o) ((o)->as.array)
#define F(o) ((o)->as.factor)

typedef enum { LU, CHOLESKY, QR } FactorKind;
enum {
    PARTS = QR + 1,
    SOLVE,
    DET,
    TRACE,
    DIAG,
    INV,
    SOLVE_TRIDIAG,
    SOLVE_BANDED,
    SVD,
    PINV,
    RANK,
    COND,
    EIGH,
    EIG,
    RSVD
};

static const char *names[] = {"lu",    "cholesky", "qr",   "parts",         "solve",        "det",
                              "trace", "diag",     "inv",  "solve_tridiag", "solve_banded", "svd",
                              "pinv",  "rank",     "cond", "eigh",          "eig",          "rsvd"};
static const int arities[] = {1, 1, 1, 1, 2, 1, 1, 1, 1, 4, 4, 1, 1, 1, 1, 1, 1, 2};
_Static_assert(sizeof names / sizeof *names == sizeof arities / sizeof *arities,
               "linalg arity table");

const char *linalg_name(int id) {
    int op = id - LINALG_BASE;
    return op >= 0 && (size_t)op < sizeof names / sizeof *names ? names[op] : NULL;
}

static size_t index_at(const Obj *o, int row, int col) {
    return (size_t)row * A(o).cols + col;
}

/* Both work arrays and published factors are GC-owned. No public operation
 * writes a packed factor or returns an alias to it. */
static Obj *new_factor(VM *v, Obj *input, FactorKind kind) {
    int m = A(input).rows, n = A(input).cols;
    int k = m < n ? m : n;
    size_t extra = kind == LU   ? (size_t)n * sizeof(int)
                   : kind == QR ? (size_t)k * sizeof(double)
                                : 0;
    Obj *f = newobj_extra(v, O_FACTOR, extra);
    F(f).kind = kind;
    F(f).packed = tensor_copy(v, input);
    return f;
}

static double checked(VM *v, long double value) {
    if (!isfinite(value) || fabsl(value) > DBL_MAX)
        vmerror(v, "nonfinite factorization or solution");
    return (double)value;
}

static void finite_packed(VM *v, const Obj *x, const char *message) {
    for (size_t i = 0; i < A(x).n; i++)
        if (!isfinite(A(x).data[i]))
            vmerror(v, "%s", message);
}

static int lu_factor(VM *v, Obj *f) {
    Obj *a = F(f).packed;
    double *w = A(a).data;
    int n = A(a).rows;
    int *perm = (int *)(f + 1);
    int sign = 1;
    for (int i = 0; i < n; i++)
        perm[i] = i;
    for (int k = 0; k < n; k++) {
        int pivot = k;
        for (int i = k + 1; i < n; i++)
            if (fabs(w[index_at(a, i, k)]) > fabs(w[index_at(a, pivot, k)]))
                pivot = i;
        double diag = w[index_at(a, pivot, k)];
        if (!isfinite(diag))
            vmerror(v, "nonfinite LU factorization");
        if (diag == 0.0) {
            finite_packed(v, a, "nonfinite LU factorization");
            return 0;
        }
        if (pivot != k) {
            sign = -sign;
            for (int j = 0; j < n; j++) {
                size_t i = index_at(a, k, j), p = index_at(a, pivot, j);
                double tmp = w[i];
                w[i] = w[p];
                w[p] = tmp;
            }
            int tmp = perm[k];
            perm[k] = perm[pivot];
            perm[pivot] = tmp;
        }
        for (int i = k + 1; i < n; i++) {
            size_t p = index_at(a, i, k);
            double multiplier = w[p] / diag;
            w[p] = multiplier;
            for (int j = k + 1; j < n; j++)
                w[index_at(a, i, j)] -= multiplier * w[index_at(a, k, j)];
        }
    }
    finite_packed(v, a, "nonfinite LU factorization");
    return sign;
}

static void cholesky_factor(VM *v, Obj *a) {
    double *w = A(a).data;
    int n = A(a).rows;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++)
            if (w[index_at(a, i, j)] != w[index_at(a, j, i)])
                vmerror(v, "Cholesky requires a symmetric matrix");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            long double value = w[index_at(a, i, j)];
            for (int k = 0; k < j; k++)
                value -= (long double)w[index_at(a, i, k)] * w[index_at(a, j, k)];
            if (i == j) {
                if (!(value > 0.0L))
                    vmerror(v, "Cholesky requires a positive definite matrix");
                double pivot = checked(v, sqrtl(value));
                if (pivot == 0.0)
                    vmerror(v, "Cholesky pivot underflow");
                w[index_at(a, i, i)] = pivot;
            } else
                w[index_at(a, i, j)] = checked(v, value / w[index_at(a, j, j)]);
        }
    }
}

static void apply_reflector(VM *v, const Obj *packed, int k, double tau, Obj *target,
                            int first_col) {
    if (tau == 0.0)
        return;
    const double *w = A(packed).data;
    double *data = A(target).data;
    for (int j = first_col; j < A(target).cols; j++) {
        long double dot = data[index_at(target, k, j)];
        for (int i = k + 1; i < A(packed).rows; i++)
            dot += (long double)w[index_at(packed, i, k)] * data[index_at(target, i, j)];
        long double scale = tau * dot;
        size_t first = index_at(target, k, j);
        data[first] = checked(v, (long double)data[first] - scale);
        for (int i = k + 1; i < A(packed).rows; i++) {
            size_t next = index_at(target, i, j);
            data[next] = checked(v, (long double)data[next] - scale * w[index_at(packed, i, k)]);
        }
    }
}

/* Generate each reflector in scaled coordinates. This avoids squaring large
 * entries and forming alpha - beta in the original scale. */
static void qr_factor(VM *v, Obj *f) {
    Obj *a = F(f).packed;
    double *w = A(a).data, *tau = (double *)(f + 1);
    int m = A(a).rows, n = A(a).cols;
    for (int k = 0; k < m && k < n; k++) {
        double tail_max = 0.0;
        for (int i = k + 1; i < m; i++)
            tail_max = fmax(tail_max, fabs(w[index_at(a, i, k)]));
        if (tail_max == 0.0)
            continue;
        size_t diag = index_at(a, k, k);
        double scale = fmax(fabs(w[diag]), tail_max);
        double alpha = w[diag] / scale;
        double norm = fabs(alpha);
        for (int i = k + 1; i < m; i++)
            norm = hypot(norm, w[index_at(a, i, k)] / scale);
        double beta = -copysign(norm, w[diag]);
        tau[k] = (beta - alpha) / beta;
        for (int i = k + 1; i < m; i++) {
            size_t p = index_at(a, i, k);
            w[p] = (w[p] / scale) / (alpha - beta);
        }
        w[diag] = checked(v, (long double)scale * beta);
        apply_reflector(v, a, k, tau[k], a, k + 1);
    }
}

static Obj *factorize(VM *v, Obj *input, FactorKind kind, const char *nonfinite, int *det_sign) {
    if (A(input).rank != 2)
        vmerror(v, "factorization expects a matrix");
    if (kind != QR && A(input).rows != A(input).cols)
        vmerror(v, "factorization expects a square matrix");
    Obj *f = new_factor(v, input, kind);
    Obj *w = F(f).packed;
    finite_packed(v, w, nonfinite);
    switch (kind) {
    case LU: {
        int sign = lu_factor(v, f);
        if (det_sign)
            *det_sign = sign;
        else if (!sign)
            vmerror(v, "singular matrix");
        break;
    }
    case CHOLESKY:
        cholesky_factor(v, w);
        break;
    case QR:
        qr_factor(v, f);
        break;
    }
    return f;
}

static Obj *rhs_copy(VM *v, const Obj *b, const int *perm) {
    Obj *x = array_new(v, A(b).rank, A(b).rows, A(b).cols);
    size_t cs = A(b).rank == 2 ? A(b).stride[1] : 0;
    for (int i = 0; i < A(b).rows; i++) {
        size_t row = (size_t)(perm ? perm[i] : i) * A(b).stride[0];
        for (int j = 0; j < A(b).cols; j++)
            A(x).data[index_at(x, i, j)] = A(b).data[row + (size_t)j * cs];
    }
    return x;
}

static void back_substitute(VM *v, const Obj *a, Obj *x, int n, int transposed) {
    for (int i = n - 1; i >= 0; i--) {
        for (int c = 0; c < A(x).cols; c++) {
            size_t p = index_at(x, i, c);
            long double total = A(x).data[p];
            for (int j = i + 1; j < n; j++) {
                double w = A(a).data[transposed ? index_at(a, j, i) : index_at(a, i, j)];
                total -= (long double)w * A(x).data[index_at(x, j, c)];
            }
            long double value = total / A(a).data[index_at(a, i, i)];
            if (!isfinite(value) || fabsl(value) > DBL_MAX)
                vmerror(v, "nonfinite solution");
            A(x).data[p] = (double)value;
        }
    }
}

static void lu_forward(const Obj *a, Obj *x) {
    /* Apply the unit-lower triangular factor in the same forward order as
     * Gaussian elimination, with Float64 updates to the right-hand side. */
    for (int k = 0; k < A(a).rows; k++)
        for (int i = k + 1; i < A(a).rows; i++) {
            double multiplier = A(a).data[index_at(a, i, k)];
            for (int c = 0; c < A(x).cols; c++)
                A(x).data[index_at(x, i, c)] -= multiplier * A(x).data[index_at(x, k, c)];
        }
}

static void cholesky_forward(VM *v, const Obj *a, Obj *x) {
    for (int i = 0; i < A(a).rows; i++)
        for (int c = 0; c < A(x).cols; c++) {
            size_t p = index_at(x, i, c);
            long double total = A(x).data[p];
            for (int k = 0; k < i; k++)
                total -= (long double)A(a).data[index_at(a, i, k)] * A(x).data[index_at(x, k, c)];
            A(x).data[p] = checked(v, total / A(a).data[index_at(a, i, i)]);
        }
}

static void require_qr_full_rank(VM *v, const Obj *a) {
    int m = A(a).rows, n = A(a).cols;
    if (m < n)
        vmerror(v, "QR solve requires at least as many rows as columns");
    double max_diag = 0;
    for (int i = 0; i < n; i++)
        max_diag = fmax(max_diag, fabs(A(a).data[index_at(a, i, i)]));
    double threshold = DBL_EPSILON * m;
    for (int i = 0; i < n; i++)
        if (max_diag == 0.0 || fabs(A(a).data[index_at(a, i, i)]) / max_diag <= threshold)
            vmerror(v, "QR solve requires a numerically full column rank matrix");
}

static Obj *solve_factor(VM *v, const Obj *f, const Obj *b) {
    const Obj *a = F(f).packed;
    if ((A(b).rank != 1 && A(b).rank != 2) || A(b).rows != A(a).rows)
        vmerror(v, "factor solve requires a matching vector or matrix right-hand side");
    if (F(f).kind == QR)
        require_qr_full_rank(v, a);
    const int *perm = F(f).kind == LU ? (const int *)(f + 1) : NULL;
    Obj *x = rhs_copy(v, b, perm);
    finite_packed(v, x, "solve requires finite inputs");
    if (F(f).kind == LU) {
        lu_forward(a, x);
        back_substitute(v, a, x, A(a).rows, 0);
    } else if (F(f).kind == CHOLESKY) {
        cholesky_forward(v, a, x);
        back_substitute(v, a, x, A(a).rows, 1);
    } else {
        const double *tau = (const double *)(f + 1);
        int n = A(a).cols;
        for (int k = 0; k < n; k++)
            apply_reflector(v, a, k, tau[k], x, 0);
        back_substitute(v, a, x, n, 0);
        if (A(x).rows != n) {
            Obj *solution = array_new(v, A(x).rank, n, A(x).cols);
            if (A(solution).n)
                memcpy(A(solution).data, A(x).data, A(solution).n * sizeof(double));
            x = solution;
        }
    }
    return x;
}

Obj *dense_lu_solve(VM *v, Obj *a, Obj *b) {
    if (A(a).rows != A(a).cols || A(a).rows != A(b).rows)
        vmerror(v, "solve requires square matrix and matching vector");
    finite_packed(v, b, "solve requires finite inputs");
    return solve_factor(v, factorize(v, a, LU, "solve requires finite inputs", NULL), b);
}

static void qr_basis(VM *v, const Obj *f, Obj *q) {
    const Obj *a = F(f).packed;
    int k = A(q).cols;
    memset(A(q).data, 0, A(q).n * sizeof(double));
    for (int i = 0; i < k; i++)
        A(q).data[index_at(q, i, i)] = 1;
    const double *tau = (const double *)(f + 1);
    for (int i = k - 1; i >= 0; i--)
        apply_reflector(v, a, i, tau[i], q, 0);
}

static Val factor_parts(VM *v, const Obj *f) {
    const Obj *a = F(f).packed;
    int m = A(a).rows, n = A(a).cols;
    Obj *out = list(v);
    if (F(f).kind == QR) {
        int k = m < n ? m : n;
        Obj *q = array_new(v, 2, m, k), *r = array_new(v, 2, k, n);
        for (int i = 0; i < k; i++)
            for (int j = i; j < n; j++)
                A(r).data[index_at(r, i, j)] = A(a).data[index_at(a, i, j)];
        qr_basis(v, f, q);
        append(v, out, objval(q));
        append(v, out, objval(r));
    } else {
        Obj *l = array_new(v, 2, n, n);
        int lu = F(f).kind == LU;
        for (int i = 0; i < n; i++) {
            A(l).data[index_at(l, i, i)] = lu ? 1.0 : A(a).data[index_at(a, i, i)];
            for (int j = 0; j < i; j++)
                A(l).data[index_at(l, i, j)] = A(a).data[index_at(a, i, j)];
        }
        if (lu) {
            Obj *p = list(v), *u = array_new(v, 2, n, n);
            const int *perm = (const int *)(f + 1);
            for (int i = 0; i < n; i++) {
                append(v, p, integer(perm[i]));
                for (int j = i; j < n; j++)
                    A(u).data[index_at(u, i, j)] = A(a).data[index_at(a, i, j)];
            }
            append(v, out, objval(p));
            append(v, out, objval(l));
            append(v, out, objval(u));
        } else
            append(v, out, objval(l));
    }
    return objval(out);
}

static Obj *numeric_array(VM *v, Val value) {
    if (!isobj(value, O_ARRAY))
        vmerror(v, "expected numeric array");
    return value.o;
}

static double at(const Obj *a, int row, int col) {
    return A(a).data[(size_t)row * A(a).stride[0] + (size_t)col * A(a).stride[1]];
}

static double determinant(VM *v, Obj *input) {
    int sign;
    Obj *f = factorize(v, input, LU, "determinant requires finite inputs", &sign);
    if (!sign)
        return 0;
    const Obj *a = F(f).packed;
    /* Renormalize each term to keep the intermediate product in range.
     * A square Float64 tensor has at most 4000 rows, so int is enough
     * for the sum of the Float64 exponents. */
    long double mantissa = sign;
    int exponent = 0;
    for (int i = 0; i < A(a).rows; i++) {
        int diagonal_exp, product_exp;
        double diagonal = frexp(A(a).data[index_at(a, i, i)], &diagonal_exp);
        mantissa = frexpl(mantissa * diagonal, &product_exp);
        exponent += diagonal_exp + product_exp;
    }
    long double result = ldexpl(mantissa, exponent);
    if (!isfinite(result) || fabsl(result) > DBL_MAX)
        vmerror(v, "determinant overflow");
    return (double)result;
}

static Val diagonal(VM *v, const Obj *a, int trace) {
    if (A(a).rank != 2)
        vmerror(v, "trace and diag expect a matrix");
    int n = A(a).rows < A(a).cols ? A(a).rows : A(a).cols;
    Obj *result = trace ? NULL : array_new(v, 1, n, 1);
    long double sum = 0;
    for (int i = 0; i < n; i++) {
        double value = at(a, i, i);
        if (!isfinite(value))
            vmerror(v, "trace and diag require finite diagonal entries");
        if (trace)
            sum += value;
        else
            A(result).data[i] = value;
    }
    return trace ? number(checked(v, sum)) : objval(result);
}

static Obj *inverse(VM *v, Obj *input) {
    Obj *f = factorize(v, input, LU, "inverse requires finite inputs", NULL);
    const Obj *a = F(f).packed;
    int n = A(a).rows;
    Obj *out = array_new(v, 2, n, n);
    const int *perm = (const int *)(f + 1);
    /* Form P*I directly in the output before solving with L and U. */
    for (int i = 0; i < n; i++)
        A(out).data[index_at(out, i, perm[i])] = 1;
    lu_forward(a, out);
    back_substitute(v, a, out, n, 0);
    return out;
}

static Obj *structured_rhs(VM *v, Val rhs, int n) {
    Obj *b = numeric_array(v, rhs);
    if ((A(b).rank != 1 && A(b).rank != 2) || A(b).rows != n)
        vmerror(v, "structured solve requires a matching vector or matrix right-hand side");
    Obj *out = rhs_copy(v, b, NULL);
    finite_packed(v, out, "solve requires finite right-hand sides");
    return out;
}

static Obj *solve_tridiag(VM *v, Val *args) {
    const Obj *lower = numeric_array(v, args[0]);
    const Obj *diag = numeric_array(v, args[1]);
    const Obj *upper = numeric_array(v, args[2]);
    int n = A(diag).rows, edges = n ? n - 1 : 0;
    if (A(lower).rank != 1 || A(diag).rank != 1 || A(upper).rank != 1 || A(lower).rows != edges ||
        A(upper).rows != edges)
        vmerror(v, "tridiagonal solve expects vectors of lengths n-1, n, n-1");
    Obj *out = structured_rhs(v, args[3], n);
    Obj *workspace = array_new(v, 1, edges, 1);
    double *super = A(workspace).data, *x = A(out).data;
    int nrhs = A(out).cols;
    for (int i = 0; i < n; i++) {
        double d = at(diag, i, 0);
        double l = i ? at(lower, i - 1, 0) : 0;
        double u = i < edges ? at(upper, i, 0) : 0;
        if (!isfinite(d) || !isfinite(l) || !isfinite(u))
            vmerror(v, "tridiagonal solve requires finite coefficients");
        double largest = fmax(fabs(l), fabs(u));
        if (!(fabs(d) - largest > fmin(fabs(l), fabs(u))))
            vmerror(v, "Thomas solve requires strict row diagonal dominance");

        /* Normalize each row by its diagonal before elimination. Its two
         * off-diagonal coefficients then have absolute sum below one. */
        long double ell = (long double)l / d;
        long double pivot = i ? 1.0L - ell * super[i - 1] : 1.0L;
        if (!(pivot > 0.0L))
            vmerror(v, "Thomas pivot breakdown");
        if (i < edges)
            super[i] = checked(v, ((long double)u / d) / pivot);
        for (int c = 0; c < nrhs; c++) {
            size_t p = index_at(out, i, c);
            long double value = (long double)x[p] / d;
            if (i)
                value -= ell * x[index_at(out, i - 1, c)];
            x[p] = checked(v, value / pivot);
        }
    }
    for (int i = edges - 1; i >= 0; i--)
        for (int c = 0; c < nrhs; c++) {
            size_t p = index_at(out, i, c);
            x[p] =
                checked(v, (long double)x[p] - (long double)super[i] * x[index_at(out, i + 1, c)]);
        }
    return out;
}

static int bandwidth(VM *v, Val value, int limit) {
    if (value.t != V_INT || value.i < 0 || value.i > limit)
        vmerror(v, "bandwidth must be a nonnegative integer within the matrix size");
    return (int)value.i;
}

static size_t band_at(const Obj *band, int lower, int row, int col) {
    return index_at(band, row, lower + col - row);
}

static void swap(double *a, double *b) {
    double tmp = *a;
    *a = *b;
    *b = tmp;
}

static Obj *solve_banded(VM *v, Val *args) {
    const Obj *input = numeric_array(v, args[2]);
    if (A(input).rank != 2)
        vmerror(v, "band storage must be a rank-two tensor");
    int n = A(input).cols, last = n ? n - 1 : 0;
    int lower = bandwidth(v, args[0], last), upper = bandwidth(v, args[1], last);
    if (A(input).rows != lower + upper + 1)
        vmerror(v, "band storage must have lower + upper + 1 rows");
    Obj *out = structured_rhs(v, args[3], n);
    int fill_upper = lower + upper < last ? lower + upper : last;
    Obj *band = array_new(v, 2, n, lower + fill_upper + 1);
    double *a = A(band).data, *x = A(out).data;
    int nrhs = A(out).cols;
    for (int j = 0; j < n; j++) {
        int first = j > upper ? j - upper : 0;
        for (int i = first; i < n && i <= j + lower; i++) {
            double value = at(input, upper + i - j, j);
            if (!isfinite(value))
                vmerror(v, "banded solve requires finite coefficients");
            a[band_at(band, lower, i, j)] = value;
        }
    }
    for (int k = 0; k < n; k++) {
        int pivot_row = k;
        for (int i = k + 1; i < n && i <= k + lower; i++)
            if (fabs(a[band_at(band, lower, i, k)]) > fabs(a[band_at(band, lower, pivot_row, k)]))
                pivot_row = i;
        double pivot = a[band_at(band, lower, pivot_row, k)];
        if (pivot == 0)
            vmerror(v, "singular banded matrix");
        int end = k + fill_upper < last ? k + fill_upper : last;
        /* Only the active columns are needed since row operations on the
         * right-hand sides are performed alongside elimination. */
        if (pivot_row != k) {
            for (int j = k; j <= end; j++)
                swap(&a[band_at(band, lower, k, j)], &a[band_at(band, lower, pivot_row, j)]);
            for (int c = 0; c < nrhs; c++)
                swap(&x[index_at(out, k, c)], &x[index_at(out, pivot_row, c)]);
        }
        for (int i = k + 1; i < n && i <= k + lower; i++) {
            double multiplier = a[band_at(band, lower, i, k)] / pivot;
            if (multiplier == 0)
                continue;
            for (int j = k + 1; j <= end; j++) {
                size_t p = band_at(band, lower, i, j);
                a[p] = checked(v, (long double)a[p] -
                                      (long double)multiplier * a[band_at(band, lower, k, j)]);
            }
            for (int c = 0; c < nrhs; c++) {
                size_t p = index_at(out, i, c);
                x[p] = checked(v, (long double)x[p] -
                                      (long double)multiplier * x[index_at(out, k, c)]);
            }
        }
    }
    for (int i = n - 1; i >= 0; i--)
        for (int c = 0; c < nrhs; c++) {
            size_t p = index_at(out, i, c);
            long double value = x[p];
            for (int j = i + 1; j < n && j <= i + fill_upper; j++)
                value -= (long double)a[band_at(band, lower, i, j)] * x[index_at(out, j, c)];
            x[p] = checked(v, value / a[band_at(band, lower, i, i)]);
        }
    return out;
}

#ifdef ARDA_LAPACK
typedef struct {
    Obj *u, *s, *vt;
} Svd;

static Svd singular_value_decomposition(VM *v, Obj *input, int vectors) {
    int m = A(input).rows, n = A(input).cols, k = m < n ? m : n;
    Svd result = {.s = array_new(v, 1, k, 1)};
    if (vectors) {
        result.u = array_new(v, 2, m, k);
        result.vt = array_new(v, 2, k, n);
    }
    if (!k)
        return result;

    /* LAPACK sees the row-major copy as A^T. Its column-major left vectors
     * land directly in the row-major V^T, and its right transpose in U. */
    Obj *packed = tensor_copy(v, input);
    finite_packed(v, packed, "SVD requires finite inputs");
    int ldu = vectors ? n : 1, ldvt = vectors ? k : 1;
    double dummy_u, dummy_vt;
    double *left = vectors ? A(result.vt).data : &dummy_u;
    double *right_t = vectors ? A(result.u).data : &dummy_vt;
    const char *job = vectors ? "S" : "N";
    int lwork = -1, info = 0;
    double optimum = 0;
    dgesvd_(job, job, &n, &m, A(packed).data, &n, A(result.s).data, left, &ldu, right_t, &ldvt,
            &optimum, &lwork, &info);
    if (info || !isfinite(optimum) || optimum < 1)
        vmerror(v, "LAPACK dgesvd workspace query failed (%d)", info);
    if (optimum > ARDA_MAX_ELEMENTS) {
        /* This bound covers all DGESVD paths when optimum exceeds the cap. */
        int longest = m > n ? m : n;
        if (3 * k + longest > (int)ARDA_MAX_ELEMENTS || 5 * k > (int)ARDA_MAX_ELEMENTS)
            vmerror(v, "LAPACK dgesvd workspace exceeds the array size limit");
        lwork = ARDA_MAX_ELEMENTS;
    } else {
        lwork = (int)ceil(optimum);
    }
    Obj *work = array_new(v, 1, lwork, 1);
    dgesvd_(job, job, &n, &m, A(packed).data, &n, A(result.s).data, left, &ldu, right_t, &ldvt,
            A(work).data, &lwork, &info);
    if (info > 0)
        vmerror(v, "LAPACK dgesvd failed to converge (%d)", info);
    if (info < 0)
        vmerror(v, "LAPACK dgesvd failed (%d)", info);
    finite_packed(v, result.s, "nonfinite singular values");
    if (vectors) {
        finite_packed(v, result.u, "nonfinite singular vectors");
        finite_packed(v, result.vt, "nonfinite singular vectors");
    }
    return result;
}

static int singular_value_rank(const Obj *values, double rcond) {
    const double *s = A(values).data;
    int k = A(values).rows, rank = 0;
    /* Comparing ratios avoids an underflowed cutoff for tiny matrices. */
    while (rank < k && s[rank] > 0 && (rcond == 0 || (long double)s[rank] / s[0] > rcond))
        rank++;
    return rank;
}

void linalg_reduced_svd(VM *v, Obj *input, Obj **u, Obj **s, Obj **vt) {
    Svd result = singular_value_decomposition(v, input, 1);
    *u = result.u;
    *s = result.s;
    *vt = result.vt;
}

static Obj *pseudoinverse(VM *v, Svd svd, int rank) {
    int m = A(svd.u).rows, n = A(svd.vt).cols;
    Obj *out = array_new(v, 2, n, m);
    if (!rank)
        return out;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            long double value = 0;
            for (int p = 0; p < rank; p++)
                value += (long double)A(svd.vt).data[index_at(svd.vt, p, i)] *
                         A(svd.u).data[index_at(svd.u, j, p)] / A(svd.s).data[p];
            A(out).data[index_at(out, i, j)] = checked(v, value);
        }
    }
    return out;
}

static void transpose_square(Obj *matrix) {
    int n = A(matrix).rows;
    double *data = A(matrix).data;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++)
            swap(&data[index_at(matrix, i, j)], &data[index_at(matrix, j, i)]);
}

static int eigen_workspace(VM *v, double optimum, int info) {
    if (info || !isfinite(optimum) || optimum < 1 || optimum > ARDA_MAX_ELEMENTS)
        vmerror(v, "LAPACK workspace query failed");
    return (int)ceil(optimum);
}

static Val symmetric_eigen(VM *v, Obj *vectors) {
    int n = A(vectors).rows;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++)
            if (at(vectors, i, j) != at(vectors, j, i))
                vmerror(v, "eigh requires symmetric matrix");

    Obj *values = array_new(v, 1, n, 1);
    int lw = -1, info = 0;
    double optimum = 0;
    double *data = A(vectors).data, *w = A(values).data;
    dsyev_("V", "U", &n, data, &n, w, &optimum, &lw, &info);
    lw = eigen_workspace(v, optimum, info);
    Obj *work = array_new(v, 1, lw, 1);
    dsyev_("V", "U", &n, data, &n, w, A(work).data, &lw, &info);
    if (info)
        vmerror(v, "LAPACK dsyev failed (%d)", info);
    finite_packed(v, values, "nonfinite eigenvalues");
    finite_packed(v, vectors, "nonfinite eigenvectors");
    transpose_square(vectors);
    Obj *out = list(v);
    append(v, out, objval(values));
    append(v, out, objval(vectors));
    return objval(out);
}

static Val general_eigen(VM *v, Obj *matrix) {
    int n = A(matrix).rows;
    transpose_square(matrix);
    Obj *wr = array_new(v, 1, n, 1);
    Obj *wi = array_new(v, 1, n, 1);
    Obj *vr = array_new(v, 2, n, n);
    int lw = -1, info = 0, one = 1;
    double optimum = 0, unused = 0;
    double *a = A(matrix).data, *re = A(wr).data, *im = A(wi).data, *right = A(vr).data;
    dgeev_("N", "V", &n, a, &n, re, im, &unused, &one, right, &n, &optimum, &lw, &info);
    lw = eigen_workspace(v, optimum, info);
    Obj *work = array_new(v, 1, lw, 1);
    dgeev_("N", "V", &n, a, &n, re, im, &unused, &one, right, &n, A(work).data, &lw, &info);
    if (info)
        vmerror(v, "LAPACK dgeev failed (%d)", info);
    finite_packed(v, wr, "nonfinite eigenvalues");
    finite_packed(v, wi, "nonfinite eigenvalues");
    finite_packed(v, vr, "nonfinite eigenvectors");

    /* Reuse the overwritten input buffer for the imaginary parts. LAPACK
     * packs each conjugate pair as two real columns, one per component. */
    transpose_square(vr);
    memset(a, 0, A(matrix).n * sizeof(double));
    for (int j = 0; j < n; j++) {
        if (im[j] == 0)
            continue;
        if (im[j] < 0 || j + 1 == n || im[j + 1] != -im[j])
            vmerror(v, "invalid LAPACK conjugate eigenpair");
        for (int i = 0; i < n; i++) {
            size_t first = index_at(matrix, i, j), second = first + 1;
            a[first] = right[second];
            a[second] = -right[second];
            right[second] = right[first];
        }
        j++;
    }
    Obj *out = list(v);
    append(v, out, objval(wr));
    append(v, out, objval(wi));
    append(v, out, objval(vr));
    append(v, out, objval(matrix));
    return objval(out);
}

static void gaussian(Obj *out, RandomState *rng) {
    for (size_t i = 0; i < A(out).n; i++)
        A(out).data[i] = (double)random_normal_unit(rng);
}

static void orthonormalize(VM *v, Obj *factor, Obj *input, Obj *basis) {
    F(factor).packed = input;
    barrier(v, factor, objval(input));
    memset(factor + 1, 0, (size_t)A(input).cols * sizeof(double));
    qr_factor(v, factor);
    qr_basis(v, factor, basis);
}

static double sketch_residual(VM *v, const LinearOperator *op, Svd factors, int64_t seed) {
    int m = A(factors.u).rows, n = A(factors.vt).cols, k = A(factors.s).rows;
    Obj *probe = array_new(v, 1, n, 1), *target = array_new(v, 1, m, 1);
    Obj *coordinate = array_new(v, 1, k, 1);
    RandomState rng;
    random_seed(&rng, seed, 1);
    long double error = 0, norm = 0;
    for (int trial = 0; trial < 8; trial++) {
        gaussian(probe, &rng);
        linop_apply_into(v, op, probe, target, 0);
        for (int j = 0; j < k; j++)
            A(coordinate).data[j] =
                (double)linear_dot(n, A(factors.vt).data + (size_t)j * n, A(probe).data);
        for (int i = 0; i < m; i++) {
            long double value = 0;
            for (int j = 0; j < k; j++)
                value += (long double)A(factors.u).data[index_at(factors.u, i, j)] *
                         A(factors.s).data[j] * A(coordinate).data[j];
            norm = hypotl(norm, A(target).data[i]);
            error = hypotl(error, A(target).data[i] - value);
        }
    }
    if (!isfinite(norm) || !isfinite(error) || (norm == 0 && error != 0))
        vmerror(v, "randomized SVD residual could not be estimated");
    return norm == 0 ? 0 : checked(v, error / norm);
}

static int rsvd_integer(VM *v, Val a, int low, int high, const char *name) {
    if (a.t != V_INT || a.i < low || a.i > high)
        vmerror(v, "randomized SVD %s must be an integer in [%d, %d]", name, low, high);
    return (int)a.i;
}
#endif

static Val randomized_svd(VM *v, int argc, Val *a) {
#ifndef ARDA_LAPACK
    (void)argc;
    (void)a;
    vmerror(v, "LAPACK unavailable: rebuild with LAPACK=1");
#else
    if (a[0].t != V_OBJ)
        vmerror(v, "expected a matrix or linear operator");
    LinopContext context;
    LinearOperator op = linop_open(v, a[0].o, &context);
    if (!op.rows || !op.cols || op.rows > ARDA_MAX_ELEMENTS || op.cols > ARDA_MAX_ELEMENTS)
        vmerror(v, "randomized SVD needs nonempty axes within the array limit");
    int m = (int)op.rows, n = (int)op.cols, limit = m < n ? m : n;
    if (!isobj(a[0], O_ARRAY))
        limit--; /* Keep operator sketches and factors strictly below full width. */
    if (limit < 1)
        vmerror(v, "randomized SVD needs a positive rank below the smallest operator axis");
    int k = rsvd_integer(v, a[1], 1, limit, "rank");
    int extra = argc > 2 ? rsvd_integer(v, a[2], 0, ARDA_MAX_ELEMENTS, "oversampling") : 8;
    int iterations = argc > 3 ? rsvd_integer(v, a[3], 0, 32, "power iterations") : 1;
    if (argc > 4 && a[4].t != V_INT)
        vmerror(v, "randomized SVD seed must be an integer");
    int64_t seed = argc > 4 ? a[4].i : 0;
    int width = extra < limit - k ? k + extra : limit;
    RandomState rng;
    random_seed(&rng, seed, 0);
    Obj *omega = array_new(v, 2, n, width);
    Obj *range = array_new(v, 2, m, width), *basis = array_new(v, 2, m, width);
    Obj *corange = array_new(v, 2, n, width);
    Obj *qr = newobj_extra(v, O_FACTOR, (size_t)width * sizeof(double));
    F(qr).kind = QR;
    gaussian(omega, &rng);
    linop_apply_into(v, &op, omega, range, 0);
    orthonormalize(v, qr, range, basis);
    for (int i = 0; i < iterations; i++) {
        linop_apply_into(v, &op, basis, corange, 1);
        orthonormalize(v, qr, corange, omega);
        linop_apply_into(v, &op, omega, range, 0);
        orthonormalize(v, qr, range, basis);
    }

    /* SVD of (Q^T A)^T lets the skinny product stay in its native layout. */
    linop_apply_into(v, &op, basis, corange, 1);
    Svd small = singular_value_decomposition(v, corange, 1);
    Svd result = {
        .u = array_new(v, 2, m, k), .s = array_new(v, 1, k, 1), .vt = array_new(v, 2, k, n)};
    for (int j = 0; j < k; j++) {
        A(result.s).data[j] = A(small.s).data[j];
        for (int i = 0; i < m; i++)
            A(result.u).data[index_at(result.u, i, j)] =
                checked(v, linear_dot(width, A(basis).data + (size_t)i * width,
                                      A(small.vt).data + (size_t)j * width));
        for (int i = 0; i < n; i++)
            A(result.vt).data[index_at(result.vt, j, i)] = A(small.u).data[index_at(small.u, i, j)];
    }
    double error = sketch_residual(v, &op, result, seed);
    linop_close(v, &context);
    Obj *out = list(v);
    append(v, out, objval(result.u));
    append(v, out, objval(result.s));
    append(v, out, objval(result.vt));
    append(v, out, number(error));
    return objval(out);
#endif
}

static Val linalg_eigen(VM *v, Val a, int symmetric) {
#ifndef ARDA_LAPACK
    (void)a;
    (void)symmetric;
    vmerror(v, "LAPACK unavailable: rebuild with LAPACK=1");
#else
    if (!isobj(a, O_ARRAY) || A(a.o).rank != 2)
        vmerror(v, "expected rank-2 array");
    if (!A(a.o).rows || A(a.o).rows != A(a.o).cols)
        vmerror(v, "expected nonempty square matrix");
    Obj *packed = tensor_copy(v, a.o);
    finite_packed(v, packed, "expected finite array");
    return symmetric ? symmetric_eigen(v, packed) : general_eigen(v, packed);
#endif
}

Val linalg_eigh(VM *v, Val a) {
    return linalg_eigen(v, a, 1);
}

static Val spectral_call(VM *v, int op, int argc, Val *args) {
#ifndef ARDA_LAPACK
    (void)op;
    (void)argc;
    (void)args;
    vmerror(v, "LAPACK unavailable: rebuild with LAPACK=1");
#else
    Obj *input = numeric_array(v, args[0]);
    if (A(input).rank != 2)
        vmerror(v, "SVD requires a matrix");
    int m = A(input).rows, n = A(input).cols;
    double rcond = (m > n ? m : n) * DBL_EPSILON;
    if (argc == 2) {
        rcond = asnum(v, args[1]);
        if (!isfinite(rcond) || rcond < 0)
            vmerror(v, "rcond must be finite and nonnegative");
    }
    if (op == COND && (!m || !n))
        vmerror(v, "condition number requires a nonempty matrix");
    Svd svd = singular_value_decomposition(v, input, op == SVD || op == PINV);
    if (op == SVD) {
        Obj *out = list(v);
        append(v, out, objval(svd.u));
        append(v, out, objval(svd.s));
        append(v, out, objval(svd.vt));
        return objval(out);
    }
    if (op == COND) {
        double *s = A(svd.s).data;
        double smallest = s[A(svd.s).rows - 1];
        return number(smallest > 0 ? s[0] / smallest : INFINITY);
    }
    int rank = singular_value_rank(svd.s, rcond);
    return op == RANK ? integer(rank) : objval(pseudoinverse(v, svd, rank));
#endif
}

Val linalg_call(VM *v, int id, int argc, Val *a) {
    int op = id - LINALG_BASE;
    if (op < 0 || op >= (int)(sizeof names / sizeof *names))
        vmerror(v, "unknown linalg operation");
    if (argc != arities[op] && !(argc == 2 && (op == RANK || op == PINV)) &&
        !(op == RSVD && argc >= 3 && argc <= 5))
        vmerror(v, "invalid linalg.%s arity", names[op]);
    if (op <= QR) {
        if (!isobj(a[0], O_ARRAY))
            vmerror(v, "factorization expects a numeric array");
        return objval(
            factorize(v, a[0].o, (FactorKind)op, "factorization requires finite inputs", NULL));
    }
    switch (op) {
    case DET:
        return number(determinant(v, numeric_array(v, a[0])));
    case TRACE:
    case DIAG:
        return diagonal(v, numeric_array(v, a[0]), op == TRACE);
    case INV:
        return objval(inverse(v, numeric_array(v, a[0])));
    case SOLVE_TRIDIAG:
        return objval(solve_tridiag(v, a));
    case SOLVE_BANDED:
        return objval(solve_banded(v, a));
    case SVD:
    case PINV:
    case RANK:
    case COND:
        return spectral_call(v, op, argc, a);
    case EIGH:
        return linalg_eigh(v, a[0]);
    case EIG:
        return linalg_eigen(v, a[0], 0);
    case RSVD:
        return randomized_svd(v, argc, a);
    default:
        break;
    }
    if (!isobj(a[0], O_FACTOR))
        vmerror(v, "expected a linear algebra factor");
    if (op == PARTS)
        return factor_parts(v, a[0].o);
    if (!isobj(a[1], O_ARRAY))
        vmerror(v, "expected a numeric right-hand side");
    return objval(solve_factor(v, a[0].o, a[1].o));
}

Obj *linalg_module(VM *v) {
    const char *key = "<native:linalg>";
    Entry *e = tablefind(&v->modules, key);
    if (e)
        return e->val.o;
    Obj *o = newobj(v, O_MODULE);
    o->as.mod.path = copystr(key);
    account(v, o, 0, strlen(key) + 1);
    o->as.mod.state = 1;
    o->as.mod.env = environment(v, NULL);
    for (size_t i = 0; i < sizeof names / sizeof *names; i++) {
        Obj *f = newobj(v, O_NATIVE);
        f->as.native.id = LINALG_BASE + (int)i;
        tableset(v, o->as.mod.env, &o->as.mod.env->as.env.tab, names[i], objval(f));
    }
    tableset(v, NULL, &v->modules, key, objval(o));
    return o;
}
