#include "arda.h"
#include "cpu_kernels.h"
#include "linear_operator.h"
#define A(o) ((o)->as.array)
/* Native calls are atomic with respect to GC. All scratch arrays are owned by
 * the collector even if vmerror unwinds the call. */
static Obj *arr(VM *v, Val x) {
    if (!isobj(x, O_ARRAY))
        vmerror(v, "expected numeric array");
    return tensor_contiguous(x.o) ? x.o : tensor_copy(v, x.o);
}

static int same(Obj *a, Obj *b) {
    return A(a).rank == A(b).rank &&
           !memcmp(A(a).shape, A(b).shape, (size_t)A(a).rank * sizeof(int));
}

static Obj *like(VM *v, Obj *a) {
    return tensor_new(v, A(a).rank, A(a).shape);
}

static void axpy_span(size_t n, double alpha, const double *x, const double *y, double *out) {
    for (size_t i = 0; i < n; i++)
        out[i] = alpha * x[i] + y[i];
}

static void axpy(VM *v, size_t n, double alpha, const double *x, const double *y, double *out) {
#if defined(_OPENMP) && !defined(ARDA_BLAS)
    enum { CHUNK = 131072 };
    int partitions = (int)(n / CHUNK);
    if (partitions >= 2) {
        int team = cpu_parallel_team(v, partitions);
        if (team > 1) {
#pragma omp parallel for num_threads(team) schedule(static) default(none)                          \
    firstprivate(partitions, n, alpha, x, y, out)
            for (int part = 0; part < partitions; part++) {
                size_t begin = n * (uint64_t)part / partitions;
                size_t end = n * (uint64_t)(part + 1) / partitions;
                axpy_span(end - begin, alpha, x + begin, y + begin, out + begin);
            }
            return;
        }
    }
#else
    (void)v;
#endif
    axpy_span(n, alpha, x, y, out);
}

static double finite_num(VM *v, Val x) {
    double d = asnum(v, x);
    if (!isfinite(d))
        vmerror(v, "expected finite number");
    return d;
}

/* Extended precision accumulation, four independent chains to reduce latency. */
long double linear_dot(size_t n, const double *a, const double *b) {
    long double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        s0 += (long double)a[i] * b[i];
        s1 += (long double)a[i + 1] * b[i + 1];
        s2 += (long double)a[i + 2] * b[i + 2];
        s3 += (long double)a[i + 3] * b[i + 3];
    }
    for (; i < n; i++)
        s0 += (long double)a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}

/* Scale the sum of squares so that very large or small finite rhs and residual
 * vectors do not overflow or underflow merely due to squaring. */
long double linear_norm(size_t n, const double *x, size_t stride) {
    long double scale = 0, squares = 1;
    for (size_t i = 0; i < n; i++) {
        long double value = fabsl(x[i * stride]);
        if (!isfinite(value))
            return INFINITY;
        if (value == 0)
            continue;
        if (value > scale) {
            long double ratio = scale / value;
            squares = 1 + squares * ratio * ratio;
            scale = value;
        } else {
            long double ratio = value / scale;
            squares += ratio * ratio;
        }
    }
    return scale * sqrtl(squares);
}

static void dense_apply(const void *context, const double *x, size_t xs, double *y, size_t ys) {
    const Obj *matrix = context;
    size_t rows = A(matrix).rows, cols = A(matrix).cols;
    for (size_t i = 0; i < rows; i++) {
        const double *row = A(matrix).data + i * A(matrix).stride[0];
        size_t column_stride = A(matrix).stride[1];
        if (xs == 1 && column_stride == 1)
            y[i * ys] = (double)linear_dot(cols, row, x);
        else {
            long double sum = 0;
            for (size_t j = 0; j < cols; j++)
                sum += (long double)row[j * column_stride] * x[j * xs];
            y[i * ys] = (double)sum;
        }
    }
}

static void dense_apply_transpose(const void *context, const double *x, size_t xs, double *y,
                                  size_t ys) {
    const Obj *matrix = context;
    size_t rows = A(matrix).rows, cols = A(matrix).cols;
    for (size_t j = 0; j < cols; j++)
        y[j * ys] = 0;
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            y[j * ys] +=
                A(matrix).data[i * A(matrix).stride[0] + j * A(matrix).stride[1]] * x[i * xs];
}

LinearOperator dense_operator(Obj *matrix) {
    return (LinearOperator){.rows = A(matrix).rows,
                            .cols = A(matrix).cols,
                            .context = matrix,
                            .apply = dense_apply,
                            .apply_transpose = dense_apply_transpose};
}

enum { AXPY, INNER, RKERROR, FINITE };
static const char *const names[] = {"axpy", "inner", "rkerror", "finite"};
Val high_call(VM *v, int id, int argc, Val *a) {
    int k = id - HIGH_BASE;
    const int ar[] = {3, 2, 5, 1};
    if (k < 0 || k >= (int)(sizeof ar / sizeof *ar) || argc != ar[k])
        vmerror(v, "invalid highdim function arity");
    if (k == FINITE) {
        if (!isobj(a[0], O_ARRAY))
            return boolean(isfinite(asnum(v, a[0])));
        Obj *x = arr(v, a[0]);
        for (size_t i = 0; i < A(x).n; i++)
            if (!isfinite(A(x).data[i]))
                return boolean(0);
        return boolean(1);
    }
    if (k == RKERROR) {
        double rt = finite_num(v, a[3]), at = finite_num(v, a[4]);
        if (rt <= 0 || at <= 0)
            vmerror(v, "invalid RK45 tolerance");
        if (!isobj(a[0], O_ARRAY)) {
            double y = finite_num(v, a[0]), z = finite_num(v, a[1]), l = finite_num(v, a[2]);
            return number((double)(fabsl((long double)z - l) /
                                   (at + (long double)rt * fmax(fabs(y), fabs(z)))));
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
            long double e = ((long double)w - q) / (at + (long double)rt * fmax(fabs(u), fabs(w)));
            sum += e * e;
        }
        return number((double)sqrtl(sum / n));
    }
    Obj *x = arr(v, a[0]);
    Obj *y = arr(v, a[1]);
    if (!same(x, y))
        vmerror(v, "highdim shape mismatch");
    if (k == INNER)
        return number((double)linear_dot(A(x).n, A(x).data, A(y).data));
    double alpha = finite_num(v, a[2]);
    Obj *o = like(v, x);
    axpy(v, A(x).n, alpha, A(x).data, A(y).data, A(o).data);
    return objval(o);
}

Obj *high_module(VM *v) {
    return named_native_module(v, "high", names, sizeof names / sizeof *names, HIGH_BASE);
}
