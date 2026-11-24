#include "arda.h"
#define A(x) ((x)->as.array)
#ifdef ARDA_LAPACK
extern void dgesv_(const int * , const int * , double * , const int * , int * , double * , const int * ,
        int * );
#endif
static Obj *array(VM *v, Val a, int rank) {
    if (!isobj(a, O_ARRAY) || a.o->as.array.rank != rank)
        vmerror(v, "expected rank-%d array", rank);
    return tensor_contiguous(a.o) ? a.o : tensor_copy(v, a.o);
}

static double finite_num(VM *v, Val a) {
    double x = asnum(v, a);
    if (!isfinite(x))
        vmerror(v, "expected finite number");
    return x;
}

static void finite_array(VM *v, Obj *a) {
    for (size_t i = 0; i < A(a).n; i++)
        if (!isfinite(A(a).data[i]))
            vmerror(v, "expected finite array");
}

static int count(VM *v, Val a, int max) {
    if (a.t != V_INT || a.i < 1 || a.i > max)
        vmerror(v, "invalid step count");
    return (int) a.i;
}

static const char *names[] = {
    "trapz", "simpson", "fft", "ifft", "rk4_linear", "lapack_solve", "eigh", "backend"
};
/* In-place radix-2 DIT FFT, bit reversal, recurrence for twiddles. */
static Obj *fft(VM *v, Obj *a, int inv) {
    int n = A(a).rows;
    if (n < 1 || (n & (n - 1)) || n > 1048576)
        vmerror(v, "FFT length must be a power of two <= 1048576");
    if (inv && A(a).cols != 2)
        vmerror(v, "ifft expects Nx2 real/imag array");
    if (!inv && A(a).rank == 2 && A(a).cols != 2)
        vmerror(v, "fft expects vector or Nx2 real/imag array");
    finite_array(v, a);
    Obj *o = array_new(v, 2, n, 2);
    double *p = A(o).data;
    for (int i = 0; i < n; i++) {
        p[2 * i] = A(a).data[A(a).rank == 1 ? i : 2 * i];
        if (A(a).rank == 2)
            p[2 * i + 1] = A(a).data[2 * i + 1];
    }
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            double t = p[2 * i];
            p[2 * i] = p[2 * j];
            p[2 * j] = t;
            t = p[2 * i + 1];
            p[2 * i + 1] = p[2 * j + 1];
            p[2 * j + 1] = t;
        }
    }
    for (int m = 2; m <= n; m *= 2) {
        double ang = (inv ? 2 : -2) * acos(-1.0) / m, cr = cos(ang), ci = sin(ang);
        for (int i = 0; i < n; i += m) {
            double wr = 1, wi = 0;
            for (int j = 0; j < m / 2; j++) {
                int u = 2 * (i + j), w = 2 * (i + j + m / 2);
                double tr = wr * p[w] - wi * p[w + 1], ti = wr * p[w + 1] + wi * p[w], ur = p[u], ui = p[u + 1];
                p[u] = ur + tr;
                p[u + 1] = ui + ti;
                p[w] = ur - tr;
                p[w + 1] = ui - ti;
                double nr = wr * cr - wi * ci;
                wi = wr * ci + wi * cr;
                wr = nr;
            }
        }
    }
    if (inv)
        for (int i = 0; i < 2 * n; i++)
            p[i] /= n;
    return o;
}

static void mv(int n, const double *a, const double *x, double *y) {
    for (int i = 0; i < n; i++) {
        long double s = 0;
        for (int j = 0; j < n; j++)
            s += (long double) a[(size_t) i * n + j] * x[j];
        y[i] = (double) s;
    }
}
/* Fixed-step classical RK4 for y'=Ay. General callbacks supplied by ode.arda. */
static Obj *rk4(VM *v, Obj *a, Obj *y, double h, int steps) {
    int n = A(a).rows;
    if (!n || A(a).cols != n || A(y).rows != n)
        vmerror(v, "rk4 shape mismatch");
    if ((size_t) n * n > 100000000u / (unsigned) steps)
        vmerror(v, "rk4 work limit");
    finite_array(v, a);
    finite_array(v, y);
    Obj *out = array_new(v, 2, steps + 1, n), *work = array_new(v, 2, 5, n);
    double *p = A(out).data, *k1 = A(work).data, *k2 = k1 + n, *k3 = k2 + n, *k4 = k3 + n, *tmp = k4 + n;
    memcpy(p, A(y).data, (size_t) n * sizeof(double));
    for (int i = 0; i < steps; i++) {
        double *x = p + (size_t) i * n, *z = x + n;
        mv(n, A(a).data, x, k1);
        for (int j = 0; j < n; j++)
            tmp[j] = x[j] + h * k1[j] / 2;
        mv(n, A(a).data, tmp, k2);
        for (int j = 0; j < n; j++)
            tmp[j] = x[j] + h * k2[j] / 2;
        mv(n, A(a).data, tmp, k3);
        for (int j = 0; j < n; j++)
            tmp[j] = x[j] + h * k3[j];
        mv(n, A(a).data, tmp, k4);
        for (int j = 0; j < n; j++) {
            z[j] = x[j] + h * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6;
            if (!isfinite(z[j]))
                vmerror(v, "nonfinite rk4 result");
        }
    }
    return out;
}

Val calculus_call(VM *v, int id, int argc, Val *a) {
    int k = id - CALC_BASE;
    int ar[] = {
        2, 2, 1, 1, 4, 2, 1, 0
    };
    if (k < 0 || k >= 8 || argc != ar[k])
        vmerror(v, "invalid calculus function arity");
    if (k == 7) {
#ifdef ARDA_LAPACK
        return objval(string(v, "lapack"));
#else
        return objval(string(v, "portable-c"));
#endif
    }
    if (k == 0 || k == 1) {
        Obj *x = array(v, a[0], 1);
        double h = finite_num(v, a[1]);
        int n = A(x).rows;
        if (n < 2 || (k == 1 && (n < 3 || n % 2 == 0)))
            vmerror(v, "invalid sample count for quadrature");
        finite_array(v, x);
        double *p = A(x).data;
        long double s = (long double) p[0] + p[n - 1];
        for (int i = 1; i < n - 1; i++)
            s += (k == 0 ? 2 : i % 2 ? 4 : 2) * (long double) p[i];
        return number((double)(s * h / (k == 0 ? 2 : 3)));
    }
    if (k == 2 || k == 3) {
        if (!isobj(a[0], O_ARRAY))
            vmerror(v, "expected numeric array");
        int r = A(a[0].o).rank;
        if ((r != 1 && r != 2) || (k == 3 && r != 2))
            vmerror(v, "FFT expects vector or Nx2 array");
        return objval(fft(v, array(v, a[0], r), k == 3));
    }
    if (k == 4)
        return objval(rk4(v, array(v, a[0], 2), array(v, a[1], 1), finite_num(v, a[2]), count(v, a[3],
                        1000000)));
    if (k == 6)
        return linalg_eigh(v, a[0]);
#ifndef ARDA_LAPACK
    vmerror(v, "LAPACK unavailable: rebuild with LAPACK=1");
#else
    Obj *x = array(v, a[0], 2);
    int n = A(x).rows;
    if (!n || A(x).cols != n)
        vmerror(v, "expected nonempty square matrix");
    finite_array(v, x);
    Obj *w = array_new(v, 2, n, n);
    double *m = A(w).data;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            m[(size_t) j * n + i] = A(x).data[(size_t) i * n + j];
    int info = 0;
    Obj *b = array(v, a[1], 1);
    if (A(b).rows != n)
        vmerror(v, "solve shape mismatch");
    finite_array(v, b);
    Obj *r = tensor_copy(v, b);
    int *ip = mem((size_t) n * sizeof(int));
    int one = 1;
    dgesv_(&n, &one, m, &n, ip, A(r).data, &n, &info);
    free(ip);
    if (info)
        vmerror(v, "LAPACK dgesv failed (%d)", info);
    return objval(r);
#endif
}

const char **calculus_names(int *n) {
    *n = 8;
    return names;
}
