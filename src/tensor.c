#include "arda.h"
#if defined(__SSE2__)
#include <emmintrin.h>
#endif
#define A(x) ((x)->as.array)
static Obj *arr(VM *v, Val a) {
    if (!isobj(a, O_ARRAY))
        vmerror(v, "expected numeric array");
    return a.o;
}

static int dim(VM *v, Val a) {
    if (a.t != V_INT || a.i < 0 || a.i > ARDA_MAX_ELEMENTS)
        vmerror(v, "invalid array dimension");
    return (int) a.i;
}

static int shape_list(VM *v, Val a, int *s) {
    if (!isobj(a, O_LIST) || a.o->as.list.n < 1 || a.o->as.list.n > 8)
        vmerror(v, "shape must be a list of 1..8 dimensions");
    int r = a.o->as.list.n;
    for (int i = 0; i < r; i++)
        s[i] = dim(v, a.o->as.list.a[i]);
    return r;
}

static void layout(VM *v, Obj *x, int r, const int *s) {
    if (r < 1 || r > 8)
        vmerror(v, "tensor rank must be 1..8");
    size_t n = 1, span = 1;
    A(x).rank = r;
    for (int i = r - 1; i >= 0; i--) {
        if (s[i] < 0 || (unsigned) s[i] > ARDA_MAX_ELEMENTS || span > ARDA_MAX_ELEMENTS / (size_t)(s[i] ? s[i] : 1))
            vmerror(v, "array size limit: 16000000 elements");
        A(x).shape[i] = s[i];
        A(x).stride[i] = n;
        n *= s[i];
        span *= s[i] ? s[i] : 1;
    }
    A(x).n = n;
    A(x).rows = s[0];
    A(x).cols = r == 2 ? s[1] : 1;
}

Obj *tensor_new(VM *v, int r, const int *s) {
    Obj *x = newobj(v, O_ARRAY);
    layout(v, x, r, s);
    A(x).data = mem(A(x).n * sizeof(double));
    account(v, x, 0, A(x).n * sizeof(double));
    return x;
}

int tensor_contiguous(Obj *x) {
    size_t n = 1;
    for (int i = A(x).rank - 1; i >= 0; i--) {
        if (A(x).shape[i] > 1 && A(x).stride[i] != n)
            return 0;
        n *= A(x).shape[i];
    }
    return 1;
}

size_t tensor_offset(Obj *x, size_t i) {
    size_t off = 0;
    for (int j = A(x).rank - 1; j >= 0; j--) {
        size_t d = A(x).shape[j];
        off += (i % d) * A(x).stride[j];
        i /= d;
    }
    return off;
}

Obj *tensor_copy(VM *v, Obj *x) {
    Obj *y = tensor_new(v, A(x).rank, A(x).shape);
    if (tensor_contiguous(x))
        memcpy(A(y).data, A(x).data, A(x).n * sizeof(double));
    else {
        size_t off = 0;
        int coord[8] = {
            0
        };
        for (size_t i = 0; i < A(x).n; i++) {
            A(y).data[i] = A(x).data[off];
            if (i + 1 < A(x).n)
                for (int j = A(x).rank - 1; j >= 0; j--) {
                    if (++coord[j] < A(x).shape[j]) {
                        off += A(x).stride[j];
                        break;
                    }
                    coord[j] = 0;
                    off -= (size_t)(A(x).shape[j] - 1) * A(x).stride[j];
                }
        }
    }
    return y;
}

static Obj *view(VM *v, Obj *x, int r, const int *s, const size_t *st, size_t off) {
    Obj *y = newobj(v, O_ARRAY);
    layout(v, y, r, s);
    if (st)
        memcpy(A(y).stride, st, (size_t) r * sizeof * st);
    A(y).owner = A(x).owner ? A(x).owner : x;
    A(y).data = A(x).data + (A(y).n ? off : 0);
    barrier(v, y, objval(A(y).owner));
    return y;
}

static void infer(VM *v, Val a, int depth, int *r, int *s) {
    if (!isobj(a, O_LIST)) {
        if (depth == 0)
            vmerror(v, "array expects a list or nested lists");
        *r = depth;
        return;
    }
    if (depth == 8)
        vmerror(v, "tensor rank must be 1..8");
    s[depth] = a.o->as.list.n;
    if (!s[depth]) {
        *r = depth + 1;
        return;
    }
    infer(v, a.o->as.list.a[0], depth + 1, r, s);
}

static void flatten(VM *v, Val a, Obj *x, int d, size_t *i) {
    if (d == A(x).rank) {
        if (isobj(a, O_LIST))
            vmerror(v, "ragged tensor");
        A(x).data[(*i)++] = asnum(v, a);
        return;
    }
    if (!isobj(a, O_LIST) || a.o->as.list.n != A(x).shape[d])
        vmerror(v, "ragged tensor or matrix");
    for (int j = 0; j < a.o->as.list.n; j++)
        flatten(v, a.o->as.list.a[j], x, d + 1, i);
}

Obj *tensor_from_list(VM *v, Val a) {
    int r = 0, s[8] = {
        0
    };
    infer(v, a, 0, &r, s);
    Obj *x = tensor_new(v, r, s);
    size_t i = 0;
    flatten(v, a, x, 0, &i);
    return x;
}

Val tensor_index(VM *v, Obj *x, int i) {
    if (A(x).rank == 1)
        return number(A(x).data[(size_t) i * A(x).stride[0]]);
    Obj *y = view(v, x, A(x).rank - 1, A(x).shape + 1, A(x).stride + 1, (size_t) i * A(x).stride[0]);
    return objval(tensor_copy(v, y));
}

static void print_rec(Obj *x, int d, size_t off, size_t *shown) {
    putchar('[');
    for (int i = 0; i < A(x).shape[d]; i++) {
        if (i)
            fputs(", ", stdout);
        if (*shown >= 32 || i >= 32) {
            fputs("...", stdout);
            break;
        }
        size_t j = off + (size_t) i * A(x).stride[d];
        if (d + 1 == A(x).rank) {
            printf("%.15g", A(x).data[j]);
            (*shown)++;
        } else
            print_rec(x, d + 1, j, shown);
    }
    putchar(']');
}

void tensor_print(Obj *x) {
    size_t shown = 0;
    print_rec(x, 0, 0, &shown);
}

static Val tolist(VM *v, Obj *x, int d, size_t off) {
    Obj *l = list(v);
    for (int i = 0; i < A(x).shape[d]; i++) {
        size_t j = off + (size_t) i * A(x).stride[d];
        append(v, l, d + 1 == A(x).rank ? number(A(x).data[j]) : tolist(v, x, d + 1, j));
    }
    return objval(l);
}

Val tensor_tolist(VM *v, Obj *x) {
    return tolist(v, x, 0, 0);
}

static double opval(Op op, double a, double b) {
    switch (op) {
    case OP_ADD:
        return a + b;
    case OP_SUB:
        return a - b;
    case OP_MUL:
        return a * b;
    default:
        return a / b;
    }
}
/* Contiguous equal-shape/scalar path: two doubles per SSE2 instruction, scalar tail. */
static void dense(Op op, double *z, const double *p, const double *q, double a, double b, size_t n) {
    size_t i = 0;
#if defined(__SSE2__)
#define VEC(OP) do{for(;i+1<n;i+=2){__m128d x=p?_mm_loadu_pd(p+i):_mm_set1_pd(a),y=q?_mm_loadu_pd(q+i):_mm_set1_pd(b);_mm_storeu_pd(z+i,OP(x,y));}}while(0)
    switch (op) {
    case OP_ADD:
        VEC(_mm_add_pd);
        break;
    case OP_SUB:
        VEC(_mm_sub_pd);
        break;
    case OP_MUL:
        VEC(_mm_mul_pd);
        break;
    default:
        VEC(_mm_div_pd);
        break;
    }
#undef VEC
#endif
    for (; i < n; i++)
        z[i] = opval(op, p ? p[i] : a, q ? q[i] : b);
}

Val tensor_binary(VM *v, Op op, Val a, Val b) {
    Obj *x = isobj(a, O_ARRAY) ? a.o : NULL, *y = isobj(b, O_ARRAY) ? b.o : NULL;
    if (op != OP_ADD && op != OP_SUB && op != OP_MUL && op != OP_DIV)
        vmerror(v, "unsupported array operator");
    int rx = x ? A(x).rank : 0, ry = y ? A(y).rank : 0, r = rx > ry ? rx : ry, s[8] = {
        0
    };
    size_t sx[8] = {
        0
    }, sy[8] = {
        0
    };
    int fast = 1;
    for (int i = 0; i < r; i++) {
        int ix = i - r + rx, iy = i - r + ry, dx = ix < 0 ? 1 : A(x).shape[ix], dy = iy < 0 ? 1 : A(y).shape[iy];
        if (dx != dy && dx != 1 && dy != 1)
            vmerror(v, "array shape mismatch (broadcast)");
        s[i] = dx == 1 ? dy : dx;
        if (ix >= 0 && dx != 1)
            sx[i] = A(x).stride[ix];
        if (iy >= 0 && dy != 1)
            sy[i] = A(y).stride[iy];
        if ((x && dx != s[i]) || (y && dy != s[i]))
            fast = 0;
    }
    double ax = x ? 0 : asnum(v, a), by = y ? 0 : asnum(v, b);
    Obj *z = tensor_new(v, r, s);
    if (fast && (!x || tensor_contiguous(x)) && (!y || tensor_contiguous(y)))
        dense(op, A(z).data, x ? A(x).data : NULL, y ? A(y).data : NULL, ax, by, A(z).n);
    else if (A(z).n) {
        /* Odometer over outer axes; contiguous or broadcast inner axes use the same SIMD kernel. */
        size_t inner = (size_t) s[r - 1], ox = 0, oy = 0;
        int coord[8] = {
            0
        };
        for (size_t i = 0; i < A(z).n; i += inner) {
            const double *p = x ? A(x).data + ox : NULL, *q = y ? A(y).data + oy : NULL;
            if (sx[r - 1] <= 1 && sy[r - 1] <= 1)
                dense(op, A(z).data + i, p && sx[r - 1] ? p : NULL, q && sy[r - 1] ? q : NULL, p ? *p : ax, q ? *q : by,
                        inner);
            else
                for (size_t j = 0; j < inner; j++)
                    A(z).data[i + j] = opval(op, p ? p[j * sx[r - 1]] : ax, q ? q[j * sy[r - 1]] : by);
            if (i + inner < A(z).n)
                for (int j = r - 2; j >= 0; j--) {
                    if (++coord[j] < s[j]) {
                        ox += sx[j];
                        oy += sy[j];
                        break;
                    }
                    coord[j] = 0;
                    ox -= (size_t)(s[j] - 1) * sx[j];
                    oy -= (size_t)(s[j] - 1) * sy[j];
                }
        }
    }
    return objval(z);
}

static const char *names[] = {
    "reshape", "view", "permute", "copy", "at", "put", "contiguous", "simd"
};
void install_tensor(VM *v, Obj *m) {
    for (int i = 0; i < 8; i++) {
        Obj *f = newobj(v, O_NATIVE);
        f->as.native.id = TENSOR_BASE + i;
        tableset(v, m->as.mod.env, &m->as.mod.env->as.env.tab, names[i], objval(f));
    }
}

Val tensor_call(VM *v, int id, int argc, Val *a) {
    int k = id - TENSOR_BASE;
    int ar[] = {
        2, 5, 2, 1, 2, 3, 1, 0
    };
    if (k < 0 || k >= 8 || argc != ar[k])
        vmerror(v, "invalid tensor function arity");
    if (k == 7) {
#if defined(__SSE2__)
        return objval(string(v, "sse2"));
#else
        return objval(string(v, "scalar"));
#endif
    }
    Obj *x = arr(v, a[0]);
    if (k == 3)
        return objval(tensor_copy(v, x));
    if (k == 6)
        return boolean(tensor_contiguous(x));
    int s[8], r = A(x).rank;
    size_t st[8];
    if (k == 0) {
        int nr = shape_list(v, a[1], s);
        if (!tensor_contiguous(x))
            vmerror(v, "reshape needs contiguous array; use copy");
        Obj *y = view(v, x, nr, s, NULL, 0);
        if (A(y).n != A(x).n)
            vmerror(v, "reshape size mismatch");
        return objval(y);
    }
    if (k == 1) {
        int axis = indexnum(v, a[1], r), start = dim(v, a[2]), stop = dim(v, a[3]), step = dim(v, a[4]);
        if (!step || start > stop || stop > A(x).shape[axis])
            vmerror(v, "invalid view bounds");
        memcpy(s, A(x).shape, sizeof s);
        memcpy(st, A(x).stride, sizeof st);
        s[axis] = (stop - start + step - 1) / step;
        if (s[axis] > 1 && st[axis] > SIZE_MAX / (size_t) step)
            vmerror(v, "view stride overflow");
        if (s[axis] > 1)
            st[axis] *= step;
        return objval(view(v, x, r, s, st, (size_t) start * A(x).stride[axis]));
    }
    if (!isobj(a[1], O_LIST) || a[1].o->as.list.n != r)
        vmerror(v, "expected one index per dimension");
    if (k == 2) {
        unsigned used = 0;
        for (int i = 0; i < r; i++) {
            int j = indexnum(v, a[1].o->as.list.a[i], r);
            if (used & (1u << j))
                vmerror(v, "duplicate permutation axis");
            used |= 1u << j;
            s[i] = A(x).shape[j];
            st[i] = A(x).stride[j];
        }
        return objval(view(v, x, r, s, st, 0));
    }
    size_t off = 0;
    for (int i = 0; i < r; i++)
        off += (size_t) indexnum(v, a[1].o->as.list.a[i], A(x).shape[i]) * A(x).stride[i];
    if (k == 4)
        return number(A(x).data[off]);
    A(x).data[off] = asnum(v, a[2]);
    return nil();
}
