#include "arda.h"
#include <float.h>
#define A(o) ((o)->as.array)
/* Native calls are atomic with respect to GC. All scratch arrays are owned by
 * the collector even if vmerror unwinds the call. */
static Obj *arr(VM *v, Val x) {
    if (!isobj(x, O_ARRAY))
        vmerror(v, "expected numeric array");
    return tensor_contiguous(x.o) ? x.o : tensor_copy(v, x.o);
}

static int same(Obj *a, Obj *b) {
    return A(a).rank == A(b).rank && !memcmp(A(a).shape, A(b).shape, (size_t) A(a).rank * sizeof(int));
}

static Obj *like(VM *v, Obj *a) {
    return tensor_new(v, A(a).rank, A(a).shape);
}

static double finite_num(VM *v, Val x) {
    double d = asnum(v, x);
    if (!isfinite(d))
        vmerror(v, "expected finite number");
    return d;
}

static void finite_arr(VM *v, Obj *a) {
    for (size_t i = 0; i < A(a).n; i++)
        if (!isfinite(A(a).data[i]))
            vmerror(v, "expected finite array");
}

static int steps(VM *v, Val x) {
    if (x.t != V_INT || x.i < 1 || x.i > 1000000)
        vmerror(v, "maxiter must be 1..1000000");
    return (int) x.i;
}
/* Extended precision accumulation, four independent chains to reduce latency. */
static long double dotl(size_t n, const double *a, const double *b) {
    long double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        s0 += (long double) a[i] * b[i];
        s1 += (long double) a[i + 1] * b[i + 1];
        s2 += (long double) a[i + 2] * b[i + 2];
        s3 += (long double) a[i + 3] * b[i + 3];
    }
    for (; i < n; i++)
        s0 += (long double) a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}

static double norm(size_t n, const double *x) {
    return (double) sqrtl(dotl(n, x, x));
}

static void dense_mv(size_t n, const double *a, const double *x, double *y) {
    for (size_t i = 0; i < n; i++)
        y[i] = (double) dotl(n, a + i * n, x);
}
/* Negative discrete Laplacian with homogeneous Dirichlet boundaries on a
 * Cartesian grid, unit spacing. No dense matrix or sparse index arrays. */
static void laplace(Obj *grid, const double *x, double *y) {
    size_t n = A(grid).n;
    int r = A(grid).rank;
    for (size_t i = 0; i < n; i++)
        y[i] = 2.0 * r * x[i];
    for (int axis = 0; axis < r; axis++) {
        size_t stride = A(grid).stride[axis], span = stride * (size_t) A(grid).shape[axis];
        if (span == stride)
            continue;
        for (size_t base = 0; base < n; base += span) {
            for (size_t i = base; i < base + span - stride; i++)
                y[i] -= x[i + stride];
            for (size_t i = base + stride; i < base + span; i++)
                y[i] -= x[i - stride];
        }
    }
}

static void apply(Obj *matrix, Obj *grid, const double *x, double *y) {
    if (matrix)
        dense_mv(A(grid).n, A(matrix).data, x, y);
    else
        laplace(grid, x, y);
}

static Val report(VM *v, Obj *x, int it, double residual, int ok) {
    Obj *l = list(v);
    append(v, l, objval(x));
    append(v, l, integer(it));
    append(v, l, number(residual));
    append(v, l, boolean(ok));
    return objval(l);
}
/* Jacobi-preconditioned conjugate gradients for SPD dense systems; constant
 * diagonal for the stencil. True residual recomputed before reporting success. */
static Val cg(VM *v, Obj *matrix, Obj *b, double tol, int maxit) {
    size_t n = A(b).n;
    if (!n || tol <= 0 || tol >= 1)
        vmerror(v, "CG needs nonempty rhs and tolerance in (0,1)");
    finite_arr(v, b);
    if (matrix) {
        if (A(b).rank != 1 || A(matrix).rank != 2 || (size_t) A(matrix).rows != n || (size_t) A(matrix).cols != n)
            vmerror(v, "CG shape mismatch");
        finite_arr(v, matrix);
        for (size_t i = 0; i < n; i++) {
            if (A(matrix).data[i * n + i] <= 0)
                vmerror(v, "CG requires positive diagonal");
            for (size_t j = 0; j < i; j++)
                if (A(matrix).data[i * n + j] != A(matrix).data[j * n + i])
                    vmerror(v, "CG requires symmetric matrix");
        }
    }
    Obj *out = like(v, b), *rr = like(v, b), *zz = like(v, b), *pp = like(v, b), *aa = like(v, b);
    double *x = A(out).data, *r = A(rr).data, *z = A(zz).data, *p = A(pp).data, *ap = A(aa).data;
    const double *rhs = A(b).data;
    double bn = norm(n, rhs);
    if (!isfinite(bn))
        vmerror(v, "CG rhs norm overflow");
    if (bn == 0)
        return report(v, out, 0, 0, 1);
    memcpy(r, rhs, n * sizeof(double));
    double rel = 1;
    int it = 0, restart = 1;
    long double prev = 0;
    for (; it < maxit; ) {
        for (size_t i = 0; i < n; i++)
            z[i] = matrix ? r[i] / A(matrix).data[i * n + i] : r[i];
        long double rho = dotl(n, r, z);
        if (!(rho > 0) || !isfinite(rho))
            vmerror(v, "CG numerical breakdown");
        if (restart)
            memcpy(p, z, n * sizeof(double));
        else {
            double beta = (double)(rho / prev);
            for (size_t i = 0; i < n; i++)
                p[i] = z[i] + beta * p[i];
        }
        apply(matrix, b, p, ap);
        long double den = dotl(n, p, ap);
        if (!(den > 0) || !isfinite(den))
            vmerror(v, "CG nonpositive curvature (matrix may not be SPD)");
        double alpha = (double)(rho / den);
        if (!isfinite(alpha))
            vmerror(v, "CG nonfinite step");
        for (size_t i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        it++;
        rel = norm(n, r) / bn;
        if (!isfinite(rel))
            vmerror(v, "CG nonfinite residual");
        restart = 0;
        prev = rho;
        if (rel <= tol) {
            apply(matrix, b, x, ap);
            for (size_t i = 0; i < n; i++)
                r[i] = rhs[i] - ap[i];
            rel = norm(n, r) / bn;
            if (rel <= tol)
                break;
            restart = 1;
        }
    }
    apply(matrix, b, x, ap);
    for (size_t i = 0; i < n; i++)
        r[i] = rhs[i] - ap[i];
    rel = norm(n, r) / bn;
    if (!isfinite(rel))
        vmerror(v, "CG nonfinite final residual");
    return report(v, out, it, rel, rel <= tol);
}
/* CGLS: rectangular least squares without forming A^T A. */
static void mv_rect(size_t m, size_t n, const double *a, const double *x, double *y) {
    for (size_t i = 0; i < m; i++)
        y[i] = (double) dotl(n, a + i * n, x);
}

static void mtv(size_t m, size_t n, const double *a, const double *x, double *y) {
    memset(y, 0, n * sizeof(double));
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++)
            y[j] += a[i * n + j] * x[i];
}

static Val cgls(VM *v, Obj *mat, Obj *b, double tol, int maxit) {
    if (A(mat).rank != 2 || A(b).rank != 1 || A(mat).rows != A(b).rows || !A(mat).rows || !A(mat).cols || tol <= 0 || tol >= 1)
        vmerror(v, "CGLS shape or tolerance mismatch");
    finite_arr(v, mat);
    finite_arr(v, b);
    int m = A(mat).rows, n = A(mat).cols;
    Obj *out = array_new(v, 1, n, 1), *rr = like(v, b), *ss = like(v, out), *pp = like(v, out), *qq = like(v,
            b);
    double *x = A(out).data, *r = A(rr).data, *s = A(ss).data, *p = A(pp).data, *q = A(qq).data,
           *a = A(mat).data;
    memcpy(r, A(b).data, (size_t) m * sizeof(double));
    mtv(m, n, a, r, s);
    double gn = norm(n, s);
    if (!isfinite(gn))
        vmerror(v, "CGLS gradient overflow");
    if (gn == 0)
        return report(v, out, 0, 0, 1);
    memcpy(p, s, (size_t) n * sizeof(double));
    long double gamma = dotl(n, s, s);
    double rel = 1;
    int it = 0;
    for (; it < maxit; ) {
        mv_rect(m, n, a, p, q);
        long double d = dotl(m, q, q);
        if (!(d > 0) || !isfinite(d))
            vmerror(v, "CGLS numerical breakdown");
        double alpha = (double)(gamma / d);
        for (int j = 0; j < n; j++)
            x[j] += alpha * p[j];
        for (int i = 0; i < m; i++)
            r[i] -= alpha * q[i];
        mtv(m, n, a, r, s);
        long double g = dotl(n, s, s);
        it++;
        rel = (double) sqrtl(g) / gn;
        if (!isfinite(rel))
            vmerror(v, "CGLS nonfinite residual");
        if (rel <= tol) {
            mv_rect(m, n, a, x, q);
            for (int i = 0; i < m; i++)
                r[i] = A(b).data[i] - q[i];
            mtv(m, n, a, r, s);
            g = dotl(n, s, s);
            rel = (double) sqrtl(g) / gn;
            if (rel <= tol)
                break;
            memcpy(p, s, (size_t) n * sizeof(double));
            gamma = g;
            continue;
        }
        double beta = (double)(g / gamma);
        for (int j = 0; j < n; j++)
            p[j] = s[j] + beta * p[j];
        gamma = g;
    }
    mv_rect(m, n, a, x, q);
    for (int i = 0; i < m; i++)
        r[i] = A(b).data[i] - q[i];
    mtv(m, n, a, r, s);
    rel = norm(n, s) / gn;
    if (!isfinite(rel))
        vmerror(v, "CGLS nonfinite final residual");
    return report(v, out, it, rel, rel <= tol);
}

static const char *names[] = {
    "cg", "poisson", "laplace", "lstsq", "axpy", "inner", "sum_axis", "rkerror", "finite"
};
Val high_call(VM *v, int id, int argc, Val *a) {
    int k = id - HIGH_BASE;
    const int ar[] = {
        4, 3, 1, 4, 3, 2, 2, 5, 1
    };
    if (k < 0 || k >= 9 || argc != ar[k])
        vmerror(v, "invalid highdim function arity");
    if (k == 8) {
        if (!isobj(a[0], O_ARRAY))
            return boolean(isfinite(asnum(v, a[0])));
        Obj *x = arr(v, a[0]);
        for (size_t i = 0; i < A(x).n; i++)
            if (!isfinite(A(x).data[i]))
                return boolean(0);
        return boolean(1);
    }
    if (k == 7) {
        double rt = finite_num(v, a[3]), at = finite_num(v, a[4]);
        if (rt <= 0 || at <= 0)
            vmerror(v, "invalid RK45 tolerance");
        if (!isobj(a[0], O_ARRAY)) {
            double y = finite_num(v, a[0]), z = finite_num(v, a[1]), l = finite_num(v, a[2]);
            return number((double)(fabsl((long double) z - l) / (at + (long double) rt * fmax(fabs(y), fabs(z)))));
        }
        Obj *x = arr(v, a[0]), *z = arr(v, a[1]), *l = arr(v, a[2]);
        size_t n = A(x).n;
        if (!n || !same(x, z) || !same(x, l))
            vmerror(v, "RK45 state shape mismatch");
        long double sum = 0;
        for (size_t i = 0; i < n; i++) {
            double u = A(x).data[i], w = A(z).data[i], q = A(l).data[i];
            if (!isfinite(u) || !isfinite(w) || !isfinite(q))
                vmerror(v, "nonfinite RK45 state");
            long double e = ((long double) w - q) / (at + (long double) rt * fmax(fabs(u), fabs(w)));
            sum += e * e;
        }
        return number((double) sqrtl(sum / n));
    }
    if (k == 0 || k == 3) {
        Obj *m = arr(v, a[0]), *b = arr(v, a[1]);
        double tol = finite_num(v, a[2]);
        int max = steps(v, a[3]);
        return k == 0 ? cg(v, m, b, tol, max) : cgls(v, m, b, tol, max);
    }
    Obj *x = arr(v, a[0]);
    if (k == 1)
        return cg(v, NULL, x, finite_num(v, a[1]), steps(v, a[2]));
    if (k == 2) {
        finite_arr(v, x);
        Obj *o = like(v, x);
        if (A(x).n)
            laplace(x, A(x).data, A(o).data);
        return objval(o);
    }
    if (k == 6) {
        int axis = indexnum(v, a[1], A(x).rank);
        size_t inner = A(x).stride[axis], len = A(x).shape[axis];
        int shape[8], r = 0;
        for (int j = 0; j < A(x).rank; j++)
            if (j != axis)
                shape[r++] = A(x).shape[j];
        if (!r) {
            long double s = 0;
            for (size_t i = 0; i < A(x).n; i++)
                s += A(x).data[i];
            return number((double) s);
        }
        Obj *o = tensor_new(v, r, shape);
        if (len && inner)
            for (size_t i = 0; i < A(o).n; i++) {
                size_t base = (i / inner) * inner * len + i % inner;
                long double s = 0;
                for (size_t j = 0; j < len; j++)
                    s += A(x).data[base + j * inner];
                A(o).data[i] = (double) s;
            }
        return objval(o);
    }
    Obj *y = arr(v, a[1]);
    if (!same(x, y))
        vmerror(v, "highdim shape mismatch");
    if (k == 5)
        return number((double) dotl(A(x).n, A(x).data, A(y).data));
    double alpha = finite_num(v, a[2]);
    Obj *o = like(v, x);
    for (size_t i = 0; i < A(x).n; i++)
        A(o).data[i] = alpha * A(x).data[i] + A(y).data[i];
    return objval(o);
}

Obj *high_module(VM *v) {
    const char * key = "<native:high>";
    Entry *e = tablefind(&v->modules, key);
    if (e)
        return e->val.o;
    Obj *o = newobj(v, O_MODULE);
    o->as.mod.path = copystr(key);
    account(v, o, 0, strlen(key) + 1);
    o->as.mod.state = 1;
    o->as.mod.env = environment(v, NULL);
    for (int i = 0; i < 9; i++) {
        Obj *f = newobj(v, O_NATIVE);
        f->as.native.id = HIGH_BASE + i;
        tableset(v, o->as.mod.env, &o->as.mod.env->as.env.tab, names[i], objval(f));
    }
    tableset(v, NULL, &v->modules, key, objval(o));
    return o;
}
