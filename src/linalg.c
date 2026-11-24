#include "arda.h"
#include <float.h>

#define A(o) ((o)->as.array)
#define F(o) ((o)->as.factor)

typedef enum { LU, CHOLESKY, QR } FactorKind;
enum { PARTS = QR + 1, SOLVE };

static const char *names[] = {"lu", "cholesky", "qr", "parts", "solve"};

static size_t index_at(const Obj *o, int row, int col) {
    return (size_t) row * A(o).cols + col;
}

/* Both work arrays and published factors are GC-owned. No public operation
 * writes a packed factor or returns an alias to it. */
static Obj *new_factor(VM *v, Obj *input, FactorKind kind) {
    int m = A(input).rows, n = A(input).cols;
    int k = m < n ? m : n;
    size_t extra = kind == LU ? (size_t) n * sizeof(int) :
                   kind == QR ? (size_t) k * sizeof(double) : 0;
    Obj *f = newobj_extra(v, O_FACTOR, extra);
    F(f).kind = kind;
    F(f).packed = tensor_copy(v, input);
    return f;
}

static double checked(VM *v, long double value) {
    if (!isfinite(value) || fabsl(value) > DBL_MAX)
        vmerror(v, "nonfinite factorization or solution");
    return (double) value;
}

static void finite_packed(VM *v, const Obj *x, const char *message) {
    for (size_t i = 0; i < A(x).n; i++)
        if (!isfinite(A(x).data[i]))
            vmerror(v, "%s", message);
}

static void lu_factor(VM *v, Obj *f) {
    Obj *a = F(f).packed;
    double *w = A(a).data;
    int n = A(a).rows;
    int *perm = (int *) (f + 1);
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
        if (diag == 0.0)
            vmerror(v, "singular matrix");
        if (pivot != k) {
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
                value -= (long double) w[index_at(a, i, k)] * w[index_at(a, j, k)];
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

static void apply_reflector(VM *v, const Obj *packed, int k, double tau, Obj *target, int first_col) {
    if (tau == 0.0)
        return;
    const double *w = A(packed).data;
    double *data = A(target).data;
    for (int j = first_col; j < A(target).cols; j++) {
        long double dot = data[index_at(target, k, j)];
        for (int i = k + 1; i < A(packed).rows; i++)
            dot += (long double) w[index_at(packed, i, k)] * data[index_at(target, i, j)];
        long double scale = tau * dot;
        size_t first = index_at(target, k, j);
        data[first] = checked(v, (long double) data[first] - scale);
        for (int i = k + 1; i < A(packed).rows; i++) {
            size_t next = index_at(target, i, j);
            data[next] = checked(v, (long double) data[next] - scale * w[index_at(packed, i, k)]);
        }
    }
}

/* Generate each reflector in scaled coordinates. This avoids squaring large
 * entries and forming alpha - beta in the original scale. */
static void qr_factor(VM *v, Obj *f) {
    Obj *a = F(f).packed;
    double *w = A(a).data, *tau = (double *) (f + 1);
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
        w[diag] = checked(v, (long double) scale * beta);
        apply_reflector(v, a, k, tau[k], a, k + 1);
    }
}

static Obj *factorize(VM *v, Obj *input, FactorKind kind, const char *nonfinite) {
    if (A(input).rank != 2)
        vmerror(v, "factorization expects a matrix");
    if (kind != QR && A(input).rows != A(input).cols)
        vmerror(v, "factorization expects a square matrix");
    Obj *f = new_factor(v, input, kind);
    Obj *w = F(f).packed;
    finite_packed(v, w, nonfinite);
    switch (kind) {
    case LU:
        lu_factor(v, f);
        break;
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
        size_t row = (size_t) (perm ? perm[i] : i) * A(b).stride[0];
        for (int j = 0; j < A(b).cols; j++)
            A(x).data[index_at(x, i, j)] = A(b).data[row + (size_t) j * cs];
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
                total -= (long double) w * A(x).data[index_at(x, j, c)];
            }
            long double value = total / A(a).data[index_at(a, i, i)];
            if (!isfinite(value) || fabsl(value) > DBL_MAX)
                vmerror(v, "nonfinite solution");
            A(x).data[p] = (double) value;
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
                total -= (long double) A(a).data[index_at(a, i, k)] * A(x).data[index_at(x, k, c)];
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
    const int *perm = F(f).kind == LU ? (const int *) (f + 1) : NULL;
    Obj *x = rhs_copy(v, b, perm);
    finite_packed(v, x, "solve requires finite inputs");
    if (F(f).kind == LU) {
        lu_forward(a, x);
        back_substitute(v, a, x, A(a).rows, 0);
    } else if (F(f).kind == CHOLESKY) {
        cholesky_forward(v, a, x);
        back_substitute(v, a, x, A(a).rows, 1);
    } else {
        const double *tau = (const double *) (f + 1);
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
    return solve_factor(v, factorize(v, a, LU, "solve requires finite inputs"), b);
}

static Val factor_parts(VM *v, const Obj *f) {
    const Obj *a = F(f).packed;
    int m = A(a).rows, n = A(a).cols;
    Obj *out = list(v);
    if (F(f).kind == QR) {
        int k = m < n ? m : n;
        Obj *q = array_new(v, 2, m, k), *r = array_new(v, 2, k, n);
        for (int i = 0; i < k; i++) {
            A(q).data[index_at(q, i, i)] = 1.0;
            for (int j = i; j < n; j++)
                A(r).data[index_at(r, i, j)] = A(a).data[index_at(a, i, j)];
        }
        const double *tau = (const double *) (f + 1);
        for (int i = k - 1; i >= 0; i--)
            apply_reflector(v, a, i, tau[i], q, 0);
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
            const int *perm = (const int *) (f + 1);
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

Val linalg_call(VM *v, int id, int argc, Val *a) {
    int op = id - LINALG_BASE;
    if (op < 0 || op >= (int) (sizeof names / sizeof *names))
        vmerror(v, "unknown linalg operation");
    if (argc != (op == SOLVE ? 2 : 1))
        vmerror(v, "invalid linalg.%s arity", names[op]);
    if (op <= QR) {
        if (!isobj(a[0], O_ARRAY))
            vmerror(v, "factorization expects a numeric array");
        return objval(factorize(v, a[0].o, (FactorKind) op, "factorization requires finite inputs"));
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
        f->as.native.id = LINALG_BASE + (int) i;
        tableset(v, o->as.mod.env, &o->as.mod.env->as.env.tab, names[i], objval(f));
    }
    tableset(v, NULL, &v->modules, key, objval(o));
    return o;
}
