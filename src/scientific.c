#include "arda.h"
#include <float.h>
#if defined(__SSE2__)
#include <emmintrin.h>
#endif
#ifdef ARDA_BLAS
#include <cblas.h>
#endif
/* Dense float64 arrays, row-major. All allocations belong to the GC; native
 * calls do not run the collector until the instruction has returned. */
#define MAX_ELEMENTS 16000000u
#define A(o) ((o)->as.array)
static Obj *asarray(VM *v, Val x) {
    if (!isobj(x, O_ARRAY))
        vmerror(v, "expected numeric array");
    return x.o;
}

static int dim(VM *v, Val x) {
    if (x.t != V_INT || x.i < 0 || x.i > MAX_ELEMENTS)
        vmerror(v, "invalid array dimension");
    return (int) x.i;
}

Obj *array_new(VM *v, int rank, int rows, int cols) {
    int s[] = {
        rows, cols
    };
    return tensor_new(v, rank, s);
}

static Obj *like(VM *v, Obj *a) {
    return tensor_new(v, A(a).rank, A(a).shape);
}

static int same(Obj *a, Obj *b) {
    return A(a).rank == A(b).rank && !memcmp(A(a).shape, A(b).shape, (size_t) A(a).rank * sizeof(int));
}

static void vector(VM *v, Obj *a) {
    if (A(a).rank != 1)
        vmerror(v, "expected vector");
}

static void matrix(VM *v, Obj *a) {
    if (A(a).rank != 2)
        vmerror(v, "expected matrix");
}

Val array_index(VM *v, Val a, Val b) {
    Obj *o = asarray(v, a);
    return tensor_index(v, o, indexnum(v, b, A(o).rows));
}

void array_set_index(VM *v, Val a, Val b, Val x) {
    Obj *o = asarray(v, a);
    vector(v, o);
    A(o).data[(size_t) indexnum(v, b, A(o).rows) * A(o).stride[0]] = asnum(v, x);
}

void array_print(Obj *o) {
    tensor_print(o);
}

Val array_binary(VM *v, Op op, Val a, Val b) {
    return tensor_binary(v, op, a, b);
}

static long double total(Obj *a) {
    long double s = 0;
    for (size_t i = 0; i < A(a).n; i++)
        s += A(a).data[i];
    return s;
}

static double dot(Obj *a, Obj *b) {
    long double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    size_t i = 0, n = A(a).n;
    const double *p = A(a).data, *q = A(b).data;
    for (; i + 3 < n; i += 4) {
        s0 += (long double) p[i] * q[i];
        s1 += (long double) p[i + 1] * q[i + 1];
        s2 += (long double) p[i + 2] * q[i + 2];
        s3 += (long double) p[i + 3] * q[i + 3];
    }
    for (; i < n; i++)
        s0 += (long double) p[i] * q[i];
    return (double)((s0 + s1) + (s2 + s3));
}
#if !defined(ARDA_BLAS) && defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
__attribute__((target("avx2"))) static void matmul_avx2(int m, int n, int k, const double * x,
        const double *y, double *z) {
    const int block = 64;
    for (int ii = 0; ii < m; ii += block)
        for (int kk = 0; kk < k; kk += block)
            for (int jj = 0; jj < n; jj += block)
                for (int i = ii; i < m && i < ii + block; i++)
                    for (int t = kk; t < k && t < kk + block; t++) {
                        double at = x[(size_t) i * k + t];
                        __m256d aa = _mm256_set1_pd(at);
                        double *dst = z + (size_t) i * n;
                        const double *src = y + (size_t) t * n;
                        int j = jj, end = n < jj + block ? n : jj + block;
                        for (; j + 15 < end; j += 16)
                            for (int u = 0; u < 16; u += 4)
                                _mm256_storeu_pd(dst + j + u, _mm256_add_pd(_mm256_loadu_pd(dst + j + u), _mm256_mul_pd(aa, _mm256_loadu_pd(src + j + u))));
                        for (; j + 3 < end; j += 4)
                            _mm256_storeu_pd(dst + j, _mm256_add_pd(_mm256_loadu_pd(dst + j), _mm256_mul_pd(aa, _mm256_loadu_pd(src + j))));
                        for (; j < end; j++)
                            dst[j] += at * src[j];
                    }
}
#endif
static Obj *multiply(VM *v, Obj *a, Obj *b) {
    matrix(v, a);
    matrix(v, b);
    int m = A(a).rows, k = A(a).cols, n = A(b).cols;
    if (k != A(b).rows)
        vmerror(v, "matmul shape mismatch");
    Obj *o = array_new(v, 2, m, n);
    if (!m || !k || !n)
        return o;
    double *x = A(a).data, *y = A(b).data, *z = A(o).data;
#ifdef ARDA_BLAS
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, x, k, y, n, 0, z, n);
#else
    /* Blocked row-major multiplication; innermost loop is contiguous. */
    const int block = 32;
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    if (__builtin_cpu_supports("avx2")) {
        matmul_avx2(m, n, k, x, y, z);
        return o;
    }
#endif
    for (int ii = 0; ii < m; ii += block)
        for (int kk = 0; kk < k; kk += block)
            for (int jj = 0; jj < n; jj += block)
                for (int i = ii; i < m && i < ii + block; i++)
                    for (int t = kk; t < k && t < kk + block; t++) {
                        double at = x[(size_t) i * k + t];
                        double *dst = z + (size_t) i * n;
                        const double *src = y + (size_t) t * n;
                        int j = jj, end = n < jj + block ? n : jj + block;
#if defined(__SSE2__)
                        __m128d aa = _mm_set1_pd(at);
                        for (; j + 7 < end; j += 8) {
                            for (int u = 0; u < 8; u += 2)
                                _mm_storeu_pd(dst + j + u, _mm_add_pd(_mm_loadu_pd(dst + j + u), _mm_mul_pd(aa, _mm_loadu_pd(src + j + u))));
                        }
                        for (; j + 1 < end; j += 2)
                            _mm_storeu_pd(dst + j, _mm_add_pd(_mm_loadu_pd(dst + j), _mm_mul_pd(aa, _mm_loadu_pd(src + j))));
#endif
                        for (; j < end; j++)
                            dst[j] += at * src[j];
                    }
#endif
    return o;
}
/* Gaussian elimination with partial pivoting. Input is never modified.
 * Temporary arrays are GC-owned, including on a caught exception. */
static Obj *solve(VM *v, Obj *a, Obj *b) {
    matrix(v, a);
    vector(v, b);
    int n = A(a).rows;
    if (n != A(a).cols || n != A(b).rows)
        vmerror(v, "solve requires square matrix and matching vector");
    Obj *w = like(v, a), *x = like(v, b);
    double *m = A(w).data, *r = A(x).data;
    memcpy(m, A(a).data, A(a).n * sizeof(double));
    memcpy(r, A(b).data, A(b).n * sizeof(double));
    for (size_t i = 0; i < A(a).n; i++)
        if (!isfinite(m[i]))
            vmerror(v, "solve requires finite inputs");
    for (int i = 0; i < n; i++)
        if (!isfinite(r[i]))
            vmerror(v, "solve requires finite inputs");
    for (int k = 0; k < n; k++) {
        int p = k;
        for (int i = k + 1; i < n; i++)
            if (fabs(m[(size_t) i * n + k]) > fabs(m[(size_t) p * n + k]))
                p = i;
        if (m[(size_t) p * n + k] == 0)
            vmerror(v, "singular matrix");
        if (p != k) {
            for (int j = k; j < n; j++) {
                double tmp = m[(size_t) k * n + j];
                m[(size_t) k * n + j] = m[(size_t) p * n + j];
                m[(size_t) p * n + j] = tmp;
            }
            double tmp = r[k];
            r[k] = r[p];
            r[p] = tmp;
        }
        for (int i = k + 1; i < n; i++) {
            double f = m[(size_t) i * n + k] / m[(size_t) k * n + k];
            for (int j = k + 1; j < n; j++)
                m[(size_t) i * n + j] -= f * m[(size_t) k * n + j];
            r[i] -= f * r[k];
        }
    }
    for (int i = n - 1; i >= 0; i--) {
        long double s = r[i];
        for (int j = i + 1; j < n; j++)
            s -= (long double) m[(size_t) i * n + j] * r[j];
        r[i] = (double)(s / m[(size_t) i * n + i]);
        if (!isfinite(r[i]))
            vmerror(v, "nonfinite solution");
    }
    return x;
}

static const char *names[] = {
    "array", "zeros", "ones", "linspace", "shape", "tolist", "sum", "mean", "var", "std", "dot", "matmul",
    "transpose", "solve", "sin", "cos", "exp", "log", "sqrt", "norm", "min", "max", "get", "set",
    "backend", "bytes"
};
enum {
    S_ARRAY, S_ZEROS, S_ONES, S_LINSPACE, S_SHAPE, S_TOLIST, S_SUM, S_MEAN, S_VAR, S_STD, S_DOT, S_MATMUL,
    S_TRANSPOSE, S_SOLVE, S_SIN, S_COS, S_EXP, S_LOG, S_SQRT, S_NORM, S_MIN, S_MAX, S_GET, S_SET,
    S_BACKEND, S_BYTES
};
Val scientific_call(VM *v, int id, int argc, Val *a) {
    int k = id - SCI_BASE;
    if (k < 0 || k > S_BYTES)
        vmerror(v, "invalid numeric function");
    int expected = (k == S_BACKEND) ? 0 : (k == S_LINSPACE || k == S_GET) ? 3 : k == S_SET ? 4 : (k == S_DOT || k == S_MATMUL || k == S_SOLVE) ? 2 : 1;
    if (k == S_ZEROS || k == S_ONES) {
        if (argc < 1 || argc > 8)
            vmerror(v, "%s expects one to eight dimensions", names[k]);
    } else if (argc != expected)
        vmerror(v, "%s expects %d arguments", names[k], expected);
    if (k == S_BACKEND) {
#ifdef ARDA_BLAS
        return objval(string(v, "cblas"));
#else
        return objval(string(v, "portable-c"));
#endif
    }
    if (k == S_ARRAY)
        return objval(tensor_from_list(v, a[0]));
    if (k == S_ZEROS || k == S_ONES) {
        int s[8];
        for (int i = 0; i < argc; i++)
            s[i] = dim(v, a[i]);
        Obj *o = tensor_new(v, argc, s);
        if (k == S_ONES)
            for (size_t i = 0; i < A(o).n; i++)
                A(o).data[i] = 1;
        return objval(o);
    }
    if (k == S_LINSPACE) {
        double start = asnum(v, a[0]), end = asnum(v, a[1]);
        int n = dim(v, a[2]);
        if (!isfinite(start) || !isfinite(end))
            vmerror(v, "linspace requires finite endpoints");
        Obj *o = array_new(v, 1, n, 1);
        for (int i = 0; i < n; i++) {
            double t = n > 1 ? (double) i / (n - 1) : 0;
            A(o).data[i] = (1 - t) * start + t * end;
        }
        if (n > 1)
            A(o).data[n - 1] = end;
        return objval(o);
    }
    Obj *x = asarray(v, a[0]);
    double *p = A(x).data;
    size_t n = A(x).n;
    if (k == S_SHAPE) {
        Obj *l = list(v);
        for (int i = 0; i < A(x).rank; i++)
            append(v, l, integer(A(x).shape[i]));
        return objval(l);
    }
    if (k == S_BYTES)
        return integer((int64_t)(n * sizeof(double)));
    if (k == S_TOLIST)
        return tensor_tolist(v, x);
    if (k == S_GET || k == S_SET) {
        matrix(v, x);
        int i = indexnum(v, a[1], A(x).rows), j = indexnum(v, a[2], A(x).cols);
        size_t q = (size_t) i * A(x).stride[0] + (size_t) j * A(x).stride[1];
        if (k == S_GET)
            return number(p[q]);
        p[q] = asnum(v, a[3]);
        return nil();
    }
    if (!tensor_contiguous(x)) {
        x = tensor_copy(v, x);
        p = A(x).data;
    }
    if (k == S_DOT) {
        Obj *y = asarray(v, a[1]);
        if (!tensor_contiguous(y))
            y = tensor_copy(v, y);
        vector(v, x);
        vector(v, y);
        if (!same(x, y))
            vmerror(v, "dot shape mismatch");
        return number(dot(x, y));
    }
    if (k == S_MATMUL) {
        Obj *y = asarray(v, a[1]);
        return objval(multiply(v, x, tensor_contiguous(y) ? y : tensor_copy(v, y)));
    }
    if (k == S_SOLVE) {
        Obj *y = asarray(v, a[1]);
        return objval(solve(v, x, tensor_contiguous(y) ? y : tensor_copy(v, y)));
    }
    if (k == S_TRANSPOSE) {
        matrix(v, x);
        Obj *o = array_new(v, 2, A(x).cols, A(x).rows);
        for (int i = 0; i < A(x).rows; i++)
            for (int j = 0; j < A(x).cols; j++)
                A(o).data[(size_t) j * A(x).rows + i] = p[(size_t) i * A(x).cols + j];
        return objval(o);
    }
    if (k == S_SUM)
        return number((double) total(x));
    if (k == S_NORM) {
        double scale = 0, ssq = 1;
        int inf = 0;
        for (size_t i = 0; i < n; i++) {
            double z = fabs(p[i]);
            if (isnan(z))
                return number(NAN);
            if (isinf(z)) {
                inf = 1;
                continue;
            }
            if (z) {
                if (z > scale) {
                    double r = scale / z;
                    ssq = 1 + ssq * r * r;
                    scale = z;
                } else {
                    double r = z / scale;
                    ssq += r * r;
                }
            }
        }
        return number(inf ? INFINITY : scale * sqrt(ssq));
    }
    if (k == S_MEAN || k == S_VAR || k == S_STD || k == S_MIN || k == S_MAX) {
        if (!n)
            vmerror(v, "empty array reduction");
        if (k == S_MEAN)
            return number((double)(total(x) / n));
        if (k == S_MIN || k == S_MAX) {
            double r = p[0];
            for (size_t i = 0; i < n; i++) {
                if (isnan(p[i]))
                    return number(NAN);
                if (k == S_MIN ? p[i] < r : p[i] > r)
                    r = p[i];
            }
            return number(r);
        }
        long double mean = 0, m2 = 0;
        for (size_t i = 0; i < n; i++) {
            long double d = p[i] - mean;
            mean += d / (i + 1);
            m2 += d * (p[i] - mean);
        }
        double r = (double)(m2 / n);
        return number(k == S_STD ? sqrt(r) : r);
    }
    Obj *o = like(v, x);
    for (size_t i = 0; i < n; i++) {
        double z = p[i];
        switch (k) {
        case S_SIN:
            z = sin(z);
            break;
        case S_COS:
            z = cos(z);
            break;
        case S_EXP:
            z = exp(z);
            break;
        case S_LOG:
            z = log(z);
            break;
        case S_SQRT:
            z = sqrt(z);
            break;
        default:
            vmerror(v, "unknown numeric operation");
        }
        A(o).data[i] = z;
    }
    return objval(o);
}

Obj *scientific_module(VM *v) {
    const char * key = "<native:numc>";
    Entry *cached = tablefind(&v->modules, key);
    if (cached)
        return cached->val.o;
    Obj *o = newobj(v, O_MODULE);
    o->as.mod.path = copystr(key);
    account(v, o, 0, strlen(key) + 1);
    o->as.mod.state = 1;
    o->as.mod.env = environment(v, NULL);
    for (int i = 0; i < (int)(sizeof names / sizeof * names); i++) {
        Obj *f = newobj(v, O_NATIVE);
        f->as.native.id = SCI_BASE + i;
        tableset(v, o->as.mod.env, &o->as.mod.env->as.env.tab, names[i], objval(f));
    }
    install_tensor(v, o);
    tableset(v, NULL, &v->modules, key, objval(o));
    return o;
}
