#include "arda.h"

#define S(o) ((o)->as.sparse)

typedef struct {
    uint32_t col;
    size_t order;
    double value;
} Coordinate;

typedef enum {
    SP_COO, SP_CSR, SP_CRS, SP_TO_CSR, SP_TO_CRS, SP_TO_COO, SP_SHAPE, SP_NNZ,
    SP_BYTES, SP_FORMAT, SP_GET, SP_COMPONENTS, SP_FROM_DENSE, SP_TO_DENSE, SP_DESCRIBE
} SparseOp;

static const char *names[] = {
    "coo", "csr", "crs", "to_csr", "to_crs", "to_coo", "shape", "nnz",
    "bytes", "format", "get", "components", "from_dense", "todense", "describe"
};
static const int nargs[] = {5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1};

/* Every successful allocation is immediately owned and accounted by the object.
 * Native calls are atomic with respect to GC; partially constructed objects
 * (including scratch storage) are reclaimed after a catchable error. */
static size_t mul_size(VM *v, size_t n, size_t width) {
    if (width && n > SIZE_MAX / width)
        vmerror(v, "sparse buffer size overflow");
    return n * width;
}

static size_t rows_plus_one(VM *v, uint32_t rows) {
    if ((uintmax_t) rows + 1 > SIZE_MAX)
        vmerror(v, "sparse row offset count overflow");
    return (size_t) rows + 1;
}

static void *buffer(VM *v, Obj *o, size_t n, size_t width, int zero) {
    size_t bytes = mul_size(v, n, width);
    if (bytes > SIZE_MAX - v->bytes || bytes > SIZE_MAX - o->bytes ||
        bytes > (uintmax_t) INT64_MAX - (o->bytes - sizeof *o))
        vmerror(v, "sparse allocation accounting overflow");
    if (!bytes)
        return NULL;
    void *p = zero ? calloc(n, width) : malloc(bytes);
    if (!p)
        vmerror(v, "sparse allocation failed");
    account(v, o, 0, bytes);
    return p;
}

static Obj *matrix(VM *v, uint32_t rows, uint32_t cols, int csr) {
    if (v->bytes > SIZE_MAX - sizeof(Obj))
        vmerror(v, "sparse object accounting overflow");
    Obj *o = newobj(v, O_SPARSE);
    S(o).rows = rows;
    S(o).cols = cols;
    S(o).csr = csr;
    if (csr)
        S(o).offset = buffer(v, o, rows_plus_one(v, rows), sizeof(size_t), 1);
    return o;
}

static void entries(VM *v, Obj *o, size_t n) {
    S(o).nnz = n;
    if (!S(o).csr)
        S(o).row = buffer(v, o, n, sizeof(uint32_t), 0);
    S(o).col = buffer(v, o, n, sizeof(uint32_t), 0);
    S(o).values = buffer(v, o, n, sizeof(double), 0);
}

static uint32_t dimension(VM *v, Val x) {
    if (x.t != V_INT || x.i < 0 || (uint64_t) x.i > UINT32_MAX)
        vmerror(v, "sparse dimensions must be integers in 0..UINT32_MAX");
    return (uint32_t) x.i;
}

static uint32_t coord(VM *v, Val x, uint32_t limit) {
    if (x.t != V_INT || x.i < 0 || (uint64_t) x.i >= limit)
        vmerror(v, "sparse index out of bounds or not an integer");
    return (uint32_t) x.i;
}

static double finite(VM *v, Val x) {
    double d = asnum(v, x);
    if (!isfinite(d))
        vmerror(v, "sparse values must be finite");
    return d;
}

static Obj *aslist(VM *v, Val x) {
    if (!isobj(x, O_LIST))
        vmerror(v, "sparse components must be lists");
    return x.o;
}

static Obj *as_sparse(VM *v, Val x) {
    if (!isobj(x, O_SPARSE))
        vmerror(v, "expected sparse matrix");
    return x.o;
}

static size_t row_offset(VM *v, Val x, size_t n) {
    if (x.t != V_INT || x.i < 0 || (uintmax_t) x.i > n)
        vmerror(v, "invalid CSR row offset");
    return (size_t) x.i;
}

static Obj *construct(VM *v, Val *a, int csr) {
    uint32_t rows = dimension(v, a[0]), cols = dimension(v, a[1]);
    Obj *r = aslist(v, a[2]), *c = aslist(v, a[3]), *x = aslist(v, a[4]);
    size_t n = (size_t) x->as.list.n;
    if ((size_t) c->as.list.n != n || (size_t) r->as.list.n !=
        (csr ? rows_plus_one(v, rows) : n))
        vmerror(v, "sparse component length mismatch");
    Val *ri = r->as.list.a, *ci = c->as.list.a, *vi = x->as.list.a;
    /* Validate before allocating substantial native buffers. */
    for (size_t i = 0; i < n; i++) {
        if (!csr)
            coord(v, ri[i], rows);
        coord(v, ci[i], cols);
        double d = finite(v, vi[i]);
        if (csr && d == 0.0)
            vmerror(v, "canonical CSR cannot store explicit zeros");
    }
    if (csr) {
        size_t prev = row_offset(v, ri[0], n);
        if (prev != 0)
            vmerror(v, "CSR row offsets must begin at zero");
        for (size_t i = 0; i < rows; i++) {
            size_t end = row_offset(v, ri[i + 1], n);
            if (end < prev)
                vmerror(v, "CSR row offsets must be nondecreasing");
            for (size_t j = prev; j < end; j++)
                if (j > prev && ci[j - 1].i >= ci[j].i)
                    vmerror(v, "CSR columns must strictly increase within each row");
            prev = end;
        }
        if (prev != n)
            vmerror(v, "CSR row offsets must end at nnz");
    }
    Obj *o = matrix(v, rows, cols, csr);
    entries(v, o, n);
    if (csr)
        for (size_t i = 0; i < rows_plus_one(v, rows); i++)
            S(o).offset[i] = (size_t) ri[i].i;
    for (size_t i = 0; i < n; i++) {
        if (!csr)
            S(o).row[i] = (uint32_t) ri[i].i;
        S(o).col[i] = (uint32_t) ci[i].i;
        S(o).values[i] = asnum(v, vi[i]);
    }
    return o;
}

static int by_column(const void *aa, const void *bb) {
    const Coordinate *a = aa, *b = bb;
    if (a->col != b->col)
        return a->col < b->col ? -1 : 1;
    return (a->order > b->order) - (a->order < b->order);
}

static double add_finite(VM *v, double a, double b) {
    double sum = a + b;
    if (!isfinite(sum))
        vmerror(v, "nonfinite sparse duplicate sum");
    return sum;
}

static Obj *to_csr(VM *v, Obj *x) {
    if (S(x).csr)
        return x;
    size_t n = S(x).nnz, rows = S(x).rows;
    Obj *o = matrix(v, S(x).rows, S(x).cols, 1);
    size_t *ptr = S(o).offset;
    Coordinate *work = buffer(v, o, n, sizeof *work, 0);
    S(o).work = work;
    for (size_t i = 0; i < n; i++)
        ptr[(size_t) S(x).row[i] + 1]++;
    for (size_t i = 0; i < rows; i++)
        ptr[i + 1] += ptr[i];
    /* Scatter in input order, temporarily using row starts as cursors. */
    for (size_t i = 0; i < n; i++)
        work[ptr[S(x).row[i]]++] = (Coordinate){S(x).col[i], i, S(x).values[i]};
    for (size_t i = rows; i > 0; i--)
        ptr[i] = ptr[i - 1];
    ptr[0] = 0;
    size_t used = 0;
    for (size_t i = 0; i < rows; i++) {
        size_t begin = ptr[i], end = ptr[i + 1];
        int sorted = 1;
        for (size_t j = begin + (begin < end); j < end; j++)
            if (work[j - 1].col > work[j].col) {
                sorted = 0;
                break;
            }
        if (!sorted)
            qsort(work + begin, end - begin, sizeof *work, by_column);
        ptr[i] = used;
        for (size_t j = begin; j < end;) {
            uint32_t col = work[j].col;
            double sum = work[j++].value;
            while (j < end && work[j].col == col)
                sum = add_finite(v, sum, work[j++].value);
            if (sum != 0.0)
                work[used++] = (Coordinate){col, 0, sum};
        }
    }
    ptr[rows] = used;
    entries(v, o, used);
    for (size_t i = 0; i < used; i++) {
        S(o).col[i] = work[i].col;
        S(o).values[i] = work[i].value;
    }
    free(work);
    S(o).work = NULL;
    account(v, o, n * sizeof *work, 0);
    return o;
}

static Obj *to_coo(VM *v, Obj *x) {
    if (!S(x).csr)
        return x;
    Obj *o = matrix(v, S(x).rows, S(x).cols, 0);
    size_t n = S(x).nnz;
    entries(v, o, n);
    for (size_t i = 0; i < S(x).rows; i++)
        for (size_t j = S(x).offset[i]; j < S(x).offset[i + 1]; j++)
            S(o).row[j] = (uint32_t) i;
    if (n) {
        memcpy(S(o).col, S(x).col, n * sizeof(uint32_t));
        memcpy(S(o).values, S(x).values, n * sizeof(double));
    }
    return o;
}

static double get(VM *v, Obj *o, uint32_t r, uint32_t c) {
    Sparse *s = &S(o);
    if (s->csr) {
        size_t lo = s->offset[r], end = s->offset[(size_t) r + 1], hi = end;
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (s->col[mid] < c)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo < end && s->col[lo] == c ? s->values[lo] : 0.0;
    }
    double sum = 0.0;
    for (size_t i = 0; i < s->nnz; i++)
        if (s->row[i] == r && s->col[i] == c)
            sum = add_finite(v, sum, s->values[i]);
    return sum;
}

static Val components(VM *v, Obj *o) {
    Sparse *s = &S(o);
    size_t nr = s->csr ? rows_plus_one(v, s->rows) : s->nnz;
    /* Existing VM lists use int capacities with geometric growth. */
    if (nr > INT_MAX / 2 || s->nnz > INT_MAX / 2)
        vmerror(v, "sparse components exceed VM list capacity");
    Obj *r = list(v), *c = list(v), *x = list(v), *out = list(v);
    for (size_t i = 0; i < nr; i++)
        append(v, r, integer((int64_t) (s->csr ? s->offset[i] : s->row[i])));
    for (size_t i = 0; i < s->nnz; i++) {
        append(v, c, integer(s->col[i]));
        append(v, x, number(s->values[i]));
    }
    append(v, out, objval(r));
    append(v, out, objval(c));
    append(v, out, objval(x));
    return objval(out);
}

static Obj *from_dense(VM *v, Val x) {
    if (!isobj(x, O_ARRAY) || x.o->as.array.rank != 2)
        vmerror(v, "from_dense expects a rank-two numeric array");
    Obj *a = x.o;
    size_t rows = a->as.array.shape[0], cols = a->as.array.shape[1], n = 0;
    size_t rs = a->as.array.stride[0], cs = a->as.array.stride[1];
    for (size_t r = 0; r < rows; r++)
        for (size_t c = 0; c < cols; c++) {
            double d = a->as.array.data[r * rs + c * cs];
            if (!isfinite(d))
                vmerror(v, "sparse values must be finite");
            n += d != 0.0;
        }
    Obj *o = matrix(v, (uint32_t) rows, (uint32_t) cols, 1);
    entries(v, o, n);
    size_t j = 0;
    for (size_t r = 0; r < rows; r++) {
        S(o).offset[r] = j;
        for (size_t c = 0; c < cols; c++) {
            double d = a->as.array.data[r * rs + c * cs];
            if (d != 0.0) {
                S(o).col[j] = (uint32_t) c;
                S(o).values[j++] = d;
            }
        }
    }
    S(o).offset[rows] = j;
    return o;
}

static Obj *todense(VM *v, Obj *x) {
    uint32_t r = S(x).rows, c = S(x).cols;
    if (r > ARDA_MAX_ELEMENTS || c > ARDA_MAX_ELEMENTS ||
        (r ? r : 1) > ARDA_MAX_ELEMENTS / (c ? c : 1))
        vmerror(v, "array size limit: 16000000 elements");
    Obj *s = to_csr(v, x);
    int shape[2] = {(int) r, (int) c};
    Obj *out = tensor_new(v, 2, shape);
    for (size_t i = 0; i < r; i++)
        for (size_t j = S(s).offset[i]; j < S(s).offset[i + 1]; j++)
            out->as.array.data[i * c + S(s).col[j]] = S(s).values[j];
    return out;
}

/* Versioned, static descriptors are independent of numerical execution. */
static Val describe(VM *v, Val name) {
    const char *s = asstr(v, name);
    if (!strcmp(s, "to_csr") || !strcmp(s, "to_crs"))
        return objval(string(v, "v1: A[r,c] = sum of COO values at (r,c) in input order; count rows, prefix sum, stable column sort, merge duplicates, omit exact zeros. O(rows + nnz + sum(k_r log k_r)) time; O(rows + nnz) storage."));
    if (!strcmp(s, "to_coo"))
        return objval(string(v, "v1: for each row r, emit (r,col[j],value[j]) for offset[r] <= j < offset[r+1]. O(rows + nnz) time; O(nnz) output."));
    vmerror(v, "unknown sparse mathematical descriptor");
}

Val sparse_call(VM *v, int id, int argc, Val *a) {
    int op = id - SPARSE_BASE;
    if (op < 0 || (size_t) op >= sizeof names / sizeof *names)
        vmerror(v, "unknown sparse operation");
    if (argc != nargs[op])
        vmerror(v, "sparse.%s expects %d arguments, got %d", names[op], nargs[op], argc);
    if (op == SP_COO || op == SP_CSR || op == SP_CRS)
        return objval(construct(v, a, op != SP_COO));
    if (op == SP_FROM_DENSE)
        return objval(from_dense(v, a[0]));
    if (op == SP_DESCRIBE)
        return describe(v, a[0]);
    Obj *o = as_sparse(v, a[0]);
    Sparse *s = &S(o);
    switch (op) {
    case SP_TO_CSR: case SP_TO_CRS: return objval(to_csr(v, o));
    case SP_TO_COO: return objval(to_coo(v, o));
    case SP_SHAPE: {
        Obj *l = list(v);
        append(v, l, integer(s->rows));
        append(v, l, integer(s->cols));
        return objval(l);
    }
    case SP_NNZ: return integer((int64_t) s->nnz);
    case SP_BYTES: return integer((int64_t) (o->bytes - sizeof *o));
    case SP_FORMAT: return objval(string(v, s->csr ? "csr" : "coo"));
    case SP_GET: return number(get(v, o, coord(v, a[1], s->rows), coord(v, a[2], s->cols)));
    case SP_COMPONENTS: return components(v, o);
    case SP_TO_DENSE: return objval(todense(v, o));
    default: vmerror(v, "unknown sparse operation");
    }
}

Obj *sparse_module(VM *v) {
    const char *key = "<native:sparse>";
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
        f->as.native.id = SPARSE_BASE + (int) i;
        tableset(v, o->as.mod.env, &o->as.mod.env->as.env.tab, names[i], objval(f));
    }
    tableset(v, NULL, &v->modules, key, objval(o));
    return o;
}
