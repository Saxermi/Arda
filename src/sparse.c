#include "sparse_internal.h"

#define S(o) ((o)->as.sparse)

typedef struct {
    uint32_t col;
    size_t order;
    double value;
} Coordinate;

typedef enum {
    SP_COO, SP_CSR, SP_CRS, SP_TO_CSR, SP_TO_CRS, SP_TO_COO, SP_SHAPE, SP_NNZ,
    SP_BYTES, SP_FORMAT, SP_GET, SP_COMPONENTS, SP_FROM_DENSE, SP_TO_DENSE, SP_DESCRIBE,
    SP_BUILDER, SP_APPEND, SP_EXTEND, SP_FINISH, SP_EYE, SP_DIAGS, SP_TRIDIAG,
    SP_MATVEC, SP_SPMM, SP_TRANSPOSE, SP_ADD, SP_SCALE, SP_DIAGONAL, SP_ESTIMATE, SP_READ_MM, SP_WRITE_MM,
    SP_CG, SP_PCG, SP_VERIFY, SP_WORKSPACE, SP_SET_BUDGET, SP_MEMORY
} SparseOp;

static const char *names[] = {
    "coo", "csr", "crs", "to_csr", "to_crs", "to_coo", "shape", "nnz",
    "bytes", "format", "get", "components", "from_dense", "todense", "describe",
    "builder", "append", "extend", "finish", "eye", "diags", "tridiag",
    "matvec", "spmm", "transpose", "add", "scale", "diagonal", "estimate", "read_mm", "write_mm",
    "cg", "pcg", "verify", "workspace", "set_budget", "memory"
};
static const int nargs[] = {
    5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1,
    2, 4, 4, 1, 1, 4, 4,
    2, 2, 1, 2, 2, 1, 1, 1, 2,
    4, 4, 3, 1, 1, 0
};
_Static_assert(sizeof names / sizeof *names == sizeof nargs / sizeof *nargs, "sparse dispatch arity table");

/* New sparse buffers are owned by their objects immediately. Native calls
 * cannot interleave with GC, so an error can release unpublished payloads. */
void sparse_release_payload(Obj *o) {
    if (o->type == O_SPARSE_BUILDER) {
        free(o->as.sparse_builder.data);
        o->as.sparse_builder.data = NULL;
        return;
    }
    if (o->type == O_SPARSE_WORKSPACE) {
        free(o->as.sparse_workspace.data);
        o->as.sparse_workspace.data = NULL;
        return;
    }
    free(S(o).row);
    free(S(o).col);
    free(S(o).offset);
    free(S(o).values);
    free(S(o).work);
    S(o) = (Sparse){0};
}

void sparse_drop_temporary(VM *v, Obj *temp, Obj *original) {
    if (temp != original) {
        sparse_release_payload(temp);
        account(v, temp, temp->bytes - sizeof *temp, 0);
    }
}

void sparse_abort(VM *v) {
    for (Obj *o = v->objects; o != v->sparse_checkpoint; o = o->next) {
        if (sparse_owns_payload(o->type))
            sparse_drop_temporary(v, o, NULL);
    }
    if (v->sparse_file) {
        fclose(v->sparse_file);
        v->sparse_file = NULL;
    }
    v->sparse_checkpoint = NULL;
    v->sparse_active = 0;
}
size_t sparse_mul_size(VM *v, size_t n, size_t width) {
    if (width && n > SIZE_MAX / width)
        vmerror(v, "sparse buffer size overflow");
    return n * width;
}

static size_t rows_plus_one(VM *v, uint32_t rows) {
    if ((uintmax_t) rows + 1 > SIZE_MAX)
        vmerror(v, "sparse row offset count overflow");
    return (size_t) rows + 1;
}

void sparse_budget_check(VM *v, size_t bytes) {
    if (bytes > v->sparse_limit - v->sparse_bytes || bytes > (uintmax_t) INT64_MAX - v->sparse_bytes)
        vmerror(v, "sparse memory budget exceeded");
}

static void allocation_check(VM *v, Obj *o, size_t bytes) {
    sparse_budget_check(v, bytes);
    if (bytes > SIZE_MAX - v->bytes || bytes > SIZE_MAX - o->bytes ||
        bytes > (uintmax_t) INT64_MAX - (o->bytes - sizeof *o))
        vmerror(v, "sparse allocation accounting overflow");
}

void *sparse_buffer(VM *v, Obj *o, size_t n, size_t width, int zero) {
    size_t bytes = sparse_mul_size(v, n, width);
    allocation_check(v, o, bytes);
    if (!bytes)
        return NULL;
    void *p = zero ? calloc(n, width) : malloc(bytes);
    if (!p)
        vmerror(v, "sparse allocation failed");
    account(v, o, 0, bytes);
    return p;
}

Obj *sparse_matrix(VM *v, uint32_t rows, uint32_t cols, int csr) {
    if (v->bytes > SIZE_MAX - sizeof(Obj))
        vmerror(v, "sparse object accounting overflow");
    Obj *o = newobj(v, O_SPARSE);
    S(o).rows = rows;
    S(o).cols = cols;
    S(o).csr = csr;
    if (csr)
        S(o).offset = sparse_buffer(v, o, rows_plus_one(v, rows), sizeof(size_t), 1);
    return o;
}

void sparse_entries(VM *v, Obj *o, size_t n) {
    S(o).nnz = n;
    if (!S(o).csr)
        S(o).row = sparse_buffer(v, o, n, sizeof(uint32_t), 0);
    S(o).col = sparse_buffer(v, o, n, sizeof(uint32_t), 0);
    S(o).values = sparse_buffer(v, o, n, sizeof(double), 0);
}

uint32_t sparse_dimension(VM *v, Val x) {
    if (x.t != V_INT || x.i < 0 || (uint64_t) x.i > UINT32_MAX)
        vmerror(v, "sparse dimensions must be integers in 0..UINT32_MAX");
    return (uint32_t) x.i;
}

static uint32_t coord(VM *v, Val x, uint32_t limit) {
    if (x.t != V_INT || x.i < 0 || (uint64_t) x.i >= limit)
        vmerror(v, "sparse index out of bounds or not an integer");
    return (uint32_t) x.i;
}

double sparse_finite(VM *v, Val x) {
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
    uint32_t rows = sparse_dimension(v, a[0]), cols = sparse_dimension(v, a[1]);
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
        double d = sparse_finite(v, vi[i]);
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
    Obj *o = sparse_matrix(v, rows, cols, csr);
    sparse_entries(v, o, n);
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

/* libc qsort may allocate hidden scratch storage. An in-place heap sort
 * keeps canonicalization and diagonal sorting within the managed budget. */
static inline void sift_records(unsigned char *base, size_t count, size_t root, size_t width,
                                int (*compare)(const void *, const void *), unsigned char *temp) {
    memcpy(temp, base + root * width, width);
    while (root < count / 2) {
        size_t child = 2 * root + 1;
        if (child + 1 < count && compare(base + child * width, base + (child + 1) * width) < 0)
            child++;
        if (compare(temp, base + child * width) >= 0)
            break;
        memcpy(base + root * width, base + child * width, width);
        root = child;
    }
    memcpy(base + root * width, temp, width);
}

static inline void sort_records(void *records, size_t count, size_t width,
                                int (*compare)(const void *, const void *), void *temp) {
    unsigned char *base = records;
    for (size_t i = count / 2; i > 0; i--)
        sift_records(base, count, i - 1, width, compare, temp);
    for (size_t i = count; i > 1; i--) {
        unsigned char *last = base + (i - 1) * width;
        memcpy(temp, last, width);
        memcpy(last, base, width);
        memcpy(base, temp, width);
        sift_records(base, i - 1, 0, width, compare, temp);
    }
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

size_t sparse_csr_estimate(VM *v, const Sparse *s) {
    size_t offsets = sparse_mul_size(v, rows_plus_one(v, s->rows), sizeof(size_t));
    size_t entries = sparse_mul_size(v, s->nnz, sizeof(Coordinate) + sizeof(uint32_t) + sizeof(double));
    if (entries > SIZE_MAX - offsets)
        vmerror(v, "sparse workspace estimate out of range");
    return offsets + entries;
}

static Obj *canonical(VM *v, const Sparse *input) {
    size_t n = input->nnz, rows = input->rows;
    Obj *o = sparse_matrix(v, input->rows, input->cols, 1);
    size_t *ptr = S(o).offset;
    Coordinate *work = sparse_buffer(v, o, n, sizeof *work, 0);
    S(o).work = work;
    for (size_t i = 0; i < n; i++)
        ptr[(size_t) input->row[i] + 1]++;
    for (size_t i = 0; i < rows; i++)
        ptr[i + 1] += ptr[i];
    /* Scatter in input order, temporarily using row starts as cursors. */
    for (size_t i = 0; i < n; i++)
        work[ptr[input->row[i]]++] = (Coordinate){input->col[i], i, input->values[i]};
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
        if (!sorted) {
            Coordinate temp;
            sort_records(work + begin, end - begin, sizeof *work, by_column, &temp);
        }
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
    sparse_entries(v, o, used);
    for (size_t i = 0; i < used; i++) {
        S(o).col[i] = work[i].col;
        S(o).values[i] = work[i].value;
    }
    free(work);
    S(o).work = NULL;
    account(v, o, n * sizeof *work, 0);
    return o;
}

Obj *sparse_as_csr(VM *v, Obj *x) {
    return S(x).csr ? x : canonical(v, &S(x));
}

static Obj *to_coo(VM *v, Obj *x) {
    if (!S(x).csr)
        return x;
    Obj *o = sparse_matrix(v, S(x).rows, S(x).cols, 0);
    size_t n = S(x).nnz;
    sparse_entries(v, o, n);
    for (size_t i = 0; i < S(x).rows; i++)
        for (size_t j = S(x).offset[i]; j < S(x).offset[i + 1]; j++)
            S(o).row[j] = (uint32_t) i;
    if (n) {
        memcpy(S(o).col, S(x).col, n * sizeof(uint32_t));
        memcpy(S(o).values, S(x).values, n * sizeof(double));
    }
    return o;
}

#define B(o) ((o)->as.sparse_builder)
#define A(o) ((o)->as.array)

typedef struct {
    const Val *list;
    const double *data;
    size_t length, stride;
} SparseInput;

typedef struct {
    int64_t offset;
    SparseInput input;
    double scalar;
    int constant;
} Diagonal;

static size_t nonnegative_size(VM *v, Val x) {
    if (x.t != V_INT || x.i < 0 || (uintmax_t) x.i > SIZE_MAX)
        vmerror(v, "expected a nonnegative sparse size");
    return (size_t) x.i;
}

static SparseInput sequence(VM *v, Val value) {
    if (isobj(value, O_LIST))
        return (SparseInput){.list = value.o->as.list.a, .length = (size_t) value.o->as.list.n};
    if (isobj(value, O_ARRAY) && A(value.o).rank == 1)
        return (SparseInput){.data = A(value.o).data, .length = A(value.o).n, .stride = A(value.o).stride[0]};
    vmerror(v, "expected list or rank-one numeric array");
}

static uint32_t seq_index(VM *v, SparseInput x, size_t i, uint32_t limit) {
    if (x.list)
        return coord(v, x.list[i], limit);
    double d = x.data[i * x.stride];
    if (!(d >= 0.0 && d < limit) || floor(d) != d)
        vmerror(v, "sparse index out of bounds or not an integer");
    return (uint32_t) d;
}

static double seq_value(VM *v, SparseInput x, size_t i) {
    if (x.list)
        return sparse_finite(v, x.list[i]);
    double d = x.data[i * x.stride];
    if (!isfinite(d))
        vmerror(v, "sparse values must be finite");
    return d;
}

static Sparse builder_view(const SparseBuilder *b) {
    Sparse view = {.rows = b->rows, .cols = b->cols, .nnz = b->count};
    if (b->capacity) {
        view.values = b->data;
        view.row = (uint32_t *) (view.values + b->capacity);
        view.col = view.row + b->capacity;
    }
    return view;
}

void sparse_builder_reserve(VM *v, Obj *o, size_t extra) {
    SparseBuilder *b = &B(o);
    const size_t width = sizeof(double) + 2 * sizeof(uint32_t);
    size_t old = b->capacity * width;
    if (extra > SIZE_MAX / width - b->count)
        vmerror(v, "sparse builder capacity overflow");
    size_t needed = b->count + extra;
    if (needed <= b->capacity)
        return;
    size_t max = (b->limit - old) / width;
    size_t budget = (v->sparse_limit - v->sparse_bytes) / width;
    if (max > budget)
        max = budget;
    if (needed > max)
        vmerror(v, "sparse builder memory budget exceeded");
    size_t capacity = b->capacity && b->capacity <= max / 2 ? 2 * b->capacity : needed;
    if (capacity < needed)
        capacity = needed;
    Sparse prior = builder_view(b);
    void *data = sparse_buffer(v, o, capacity, width, 0);
    void *old_data = b->data;
    b->data = data;
    b->capacity = capacity;
    if (b->count) {
        Sparse view = builder_view(b);
        memcpy(view.row, prior.row, b->count * sizeof *view.row);
        memcpy(view.col, prior.col, b->count * sizeof *view.col);
        memcpy(view.values, prior.values, b->count * sizeof *view.values);
    }
    free(old_data);
    account(v, o, old, 0);
}

Obj *sparse_builder_new(VM *v, uint32_t rows, uint32_t cols, size_t capacity, size_t limit) {
    if (v->bytes > SIZE_MAX - sizeof(Obj))
        vmerror(v, "sparse object accounting overflow");
    Obj *o = newobj(v, O_SPARSE_BUILDER);
    B(o) = (SparseBuilder){.rows = rows, .cols = cols, .limit = limit};
    sparse_builder_reserve(v, o, capacity);
    return o;
}

static Obj *open_builder(VM *v, Val arg) {
    if (!isobj(arg, O_SPARSE_BUILDER))
        vmerror(v, "expected sparse builder");
    if (B(arg.o).finished)
        vmerror(v, "sparse builder is already finalized");
    return arg.o;
}

void sparse_builder_write(Obj *o, uint32_t row, uint32_t col, double value) {
    Sparse view = builder_view(&B(o));
    size_t i = B(o).count++;
    view.row[i] = row;
    view.col[i] = col;
    view.values[i] = value;
}

static void builder_append(VM *v, Obj *o, Val *a) {
    uint32_t row = coord(v, a[0], B(o).rows), col = coord(v, a[1], B(o).cols);
    double value = sparse_finite(v, a[2]);
    sparse_builder_reserve(v, o, 1);
    sparse_builder_write(o, row, col, value);
}

static void builder_extend(VM *v, Obj *o, Val *a) {
    SparseInput row = sequence(v, a[0]), col = sequence(v, a[1]), values = sequence(v, a[2]);
    if (row.length != col.length || row.length != values.length)
        vmerror(v, "sparse batch component length mismatch");
    for (size_t i = 0; i < values.length; i++) {
        seq_index(v, row, i, B(o).rows);
        seq_index(v, col, i, B(o).cols);
        seq_value(v, values, i);
    }
    sparse_builder_reserve(v, o, values.length);
    for (size_t i = 0; i < values.length; i++)
        sparse_builder_write(o, seq_index(v, row, i, B(o).rows), seq_index(v, col, i, B(o).cols),
                      seq_value(v, values, i));
}

Obj *sparse_builder_finish(VM *v, Obj *o) {
    Sparse view = builder_view(&B(o));
    Obj *out = canonical(v, &view);
    sparse_drop_temporary(v, o, NULL);
    B(o).count = B(o).capacity = 0;
    B(o).finished = 1;
    return out;
}

static size_t diagonal_length(uint32_t rows, uint32_t cols, int64_t offset) {
    if (offset < -(int64_t) rows || offset > (int64_t) cols)
        return 0;
    size_t r = offset < 0 ? rows - (size_t) -offset : rows;
    size_t c = offset > 0 ? cols - (size_t) offset : cols;
    return r < c ? r : c;
}

static int by_offset(const void *left, const void *right) {
    const Diagonal *a = left, *b = right;
    return (a->offset > b->offset) - (a->offset < b->offset);
}

static double diagonal_value(VM *v, const Diagonal *diagonal, size_t i) {
    return diagonal->constant ? diagonal->scalar : seq_value(v, diagonal->input, i);
}

/* Offsets are sorted and unique, so each row is written in canonical order. */
static void fill_diagonals(VM *v, Obj *out, const Diagonal *diagonals, size_t count) {
    Sparse *s = &S(out);
    size_t nnz = 0;
    for (size_t d = 0; d < count; d++) {
        const Diagonal *diag = &diagonals[d];
        size_t length = diagonal_length(s->rows, s->cols, diag->offset);
        if (!diag->constant && diag->input.length != length)
            vmerror(v, "sparse diagonal length mismatch");
        if (diag->constant && diag->scalar == 0.0)
            continue;
        size_t row = diag->offset < 0 && length ? (size_t) -diag->offset : 0;
        for (size_t i = 0; i < length; i++) {
            if (diagonal_value(v, diag, i) != 0.0) {
                if (nnz == SIZE_MAX)
                    vmerror(v, "sparse diagonal count overflow");
                nnz++;
                s->offset[row + i + 1]++;
            }
        }
    }
    for (size_t row = 0; row < s->rows; row++)
        s->offset[row + 1] += s->offset[row];
    sparse_entries(v, out, nnz);
    for (size_t d = 0; d < count; d++) {
        const Diagonal *diag = &diagonals[d];
        if (diag->constant && diag->scalar == 0.0)
            continue;
        size_t length = diagonal_length(s->rows, s->cols, diag->offset);
        if (!length)
            continue;
        size_t row = diag->offset < 0 ? (size_t) -diag->offset : 0;
        size_t col = diag->offset > 0 ? (size_t) diag->offset : 0;
        for (size_t i = 0; i < length; i++) {
            double value = diagonal_value(v, diag, i);
            if (value != 0.0) {
                size_t target = s->offset[row + i]++;
                s->col[target] = (uint32_t) (col + i);
                s->values[target] = value;
            }
        }
    }
    for (size_t row = s->rows; row > 0; row--)
        s->offset[row] = s->offset[row - 1];
    s->offset[0] = 0;
}

static Diagonal diagonal_input(VM *v, Val arg, int64_t offset) {
    Diagonal diag = {.offset = offset};
    if (isobj(arg, O_LIST) || isobj(arg, O_ARRAY))
        diag.input = sequence(v, arg);
    else {
        diag.constant = 1;
        diag.scalar = sparse_finite(v, arg);
    }
    return diag;
}

static Obj *construct_diags(VM *v, Val *a) {
    uint32_t rows = sparse_dimension(v, a[0]), cols = sparse_dimension(v, a[1]);
    Obj *offsets = aslist(v, a[2]), *values = aslist(v, a[3]);
    int count = offsets->as.list.n;
    if (count != values->as.list.n)
        vmerror(v, "sparse diagonal count mismatch");
    Obj *out = sparse_matrix(v, rows, cols, 1);
    Diagonal *diags = sparse_buffer(v, out, (size_t) count, sizeof *diags, 0);
    S(out).work = diags;
    for (int i = 0; i < count; i++) {
        Val offset = offsets->as.list.a[i];
        if (offset.t != V_INT || offset.i < -(int64_t) rows || offset.i > (int64_t) cols)
            vmerror(v, "invalid sparse diagonal offset");
        diags[i] = diagonal_input(v, values->as.list.a[i], offset.i);
    }
    if (count > 1) {
        Diagonal temp;
        sort_records(diags, (size_t) count, sizeof *diags, by_offset, &temp);
    }
    for (int i = 1; i < count; i++)
        if (diags[i - 1].offset == diags[i].offset)
            vmerror(v, "duplicate sparse diagonal offset");
    fill_diagonals(v, out, diags, (size_t) count);
    free(diags);
    S(out).work = NULL;
    account(v, out, (size_t) count * sizeof *diags, 0);
    return out;
}

static double get(VM *v, Obj *o, uint32_t r, uint32_t c) {
    Sparse *s = &S(o);
    if (s->csr)
        return sparse_csr_at(s, r, c);
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
    Obj *o = sparse_matrix(v, (uint32_t) rows, (uint32_t) cols, 1);
    sparse_entries(v, o, n);
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
    Obj *s = sparse_as_csr(v, x);
    int shape[2] = {(int) r, (int) c};
    Obj *out = tensor_new(v, 2, shape);
    for (size_t i = 0; i < r; i++)
        for (size_t j = S(s).offset[i]; j < S(s).offset[i + 1]; j++)
            out->as.array.data[i * c + S(s).col[j]] = S(s).values[j];
    sparse_drop_temporary(v, s, x);
    return out;
}

/* Versioned, static descriptors are independent of numerical execution. */
static Val describe(VM *v, Val name) {
    const char *s = asstr(v, name);
    if (!strcmp(s, "to_csr") || !strcmp(s, "to_crs"))
        return objval(string(v, "v1: A[r,c] = sum of COO values at (r,c) in input order; count rows, prefix sum, stable column sort, merge duplicates, omit exact zeros. O(rows + nnz + sum(k_r log k_r)) time; O(rows + nnz) storage."));
    if (!strcmp(s, "matvec") || !strcmp(s, "spmm"))
        return objval(string(v, "v1: Y[r,j] = sum over the stored entries A[r,c] * X[c,j]. "
                             "Canonicalize COO first, then apply CSR row reductions. "
                             "O(rows + nnz) per output column."));
    if (!strcmp(s, "transpose"))
        return objval(string(v, "v1: T[c,r] = A[r,c]. Count columns, prefix sum, scatter in row order. "
                             "CSR time O(rows + cols + nnz)."));
    if (!strcmp(s, "add"))
        return objval(string(v, "v1: C[r,c] = A[r,c] + B[r,c]. Merge the sorted CSR rows and omit exact zeros. "
                             "O(rows + input nnz) in CSR."));
    if (!strcmp(s, "scale"))
        return objval(string(v, "v1: C[r,c] = k * A[r,c]. Multiply the stored CSR entries and omit exact zeros. "
                             "O(rows + nnz) in CSR."));
    if (!strcmp(s, "diagonal"))
        return objval(string(v, "v1: d[i] = A[i,i]. Binary search each applicable CSR row."));
    if (!strcmp(s, "cg") || !strcmp(s, "pcg") || !strcmp(s, "verify"))
        return objval(string(v, "v1: CG for symmetric positive definite A, with optional Jacobi "
                             "preconditioner M = diag(A). Start at x = 0. Verify with norm(b - A*x)/norm(b), "
                             "or the absolute norm if b is zero."));
    if (!strcmp(s, "to_coo"))
        return objval(string(v, "v1: for each row r, emit (r,col[j],value[j]) for offset[r] <= j < offset[r+1]. O(rows + nnz) time; O(nnz) output."));
    vmerror(v, "unknown sparse mathematical descriptor");
}

static Val sparse_dispatch(VM *v, int id, int argc, Val *a) {
    int op = id - SPARSE_BASE;
    if (op < 0 || (size_t) op >= sizeof names / sizeof *names)
        vmerror(v, "unknown sparse operation");
    if (op == SP_BUILDER) {
        if (argc < 2 || argc > 4)
            vmerror(v, "sparse.builder expects 2 to 4 arguments");
        uint32_t rows = sparse_dimension(v, a[0]), cols = sparse_dimension(v, a[1]);
        size_t cap = argc >= 3 ? nonnegative_size(v, a[2]) : 0;
        size_t limit = argc == 4 ? nonnegative_size(v, a[3]) : SIZE_MAX;
        return objval(sparse_builder_new(v, rows, cols, cap, limit));
    }
    if (op == SP_WRITE_MM) {
        if (argc != 2 && argc != 3)
            vmerror(v, "sparse.write_mm expects 2 or 3 arguments");
        if (argc == 3 && a[2].t != V_BOOL)
            vmerror(v, "Matrix Market symmetric flag must be a boolean");
        sparse_write_mm(v, asstr(v, a[0]), as_sparse(v, a[1]), argc == 3 && a[2].n);
        return nil();
    }
    if (op == SP_CG || op == SP_PCG) {
        if (argc != 4 && argc != 5)
            vmerror(v, "sparse.%s expects 4 or 5 arguments", names[op]);
        return sparse_solve(v, as_sparse(v, a[0]), a[1], a[2], a[3], argc == 5 ? a[4] : nil(), op == SP_PCG);
    }
    if (argc != nargs[op])
        vmerror(v, "sparse.%s expects %d arguments, got %d", names[op], nargs[op], argc);
    if (op == SP_COO || op == SP_CSR || op == SP_CRS)
        return objval(construct(v, a, op != SP_COO));
    if (op == SP_SET_BUDGET) {
        size_t limit = a[0].t == V_NIL ? SIZE_MAX : nonnegative_size(v, a[0]);
        if (limit < v->sparse_bytes)
            vmerror(v, "sparse budget is below currently owned bytes");
        v->sparse_limit = limit;
        return nil();
    }
    if (op == SP_MEMORY) {
        Obj *result = newobj(v, O_MAP);
        sparse_result_field(v, result, "used", integer((int64_t) v->sparse_bytes));
        sparse_result_field(v, result, "limit", v->sparse_limit == SIZE_MAX ? nil() : integer((int64_t) v->sparse_limit));
        return objval(result);
    }
    if (op == SP_WORKSPACE)
        return objval(sparse_workspace_new(v, nonnegative_size(v, a[0])));
    if (op == SP_READ_MM)
        return objval(sparse_read_mm(v, asstr(v, a[0])));
    if (op == SP_FROM_DENSE)
        return objval(from_dense(v, a[0]));
    if (op == SP_DESCRIBE)
        return describe(v, a[0]);
    if (op == SP_APPEND || op == SP_EXTEND || op == SP_FINISH) {
        Obj *builder = open_builder(v, a[0]);
        if (op == SP_FINISH)
            return objval(sparse_builder_finish(v, builder));
        if (op == SP_APPEND)
            builder_append(v, builder, a + 1);
        else
            builder_extend(v, builder, a + 1);
        return nil();
    }
    if (op == SP_DIAGS)
        return objval(construct_diags(v, a));
    if (op == SP_EYE || op == SP_TRIDIAG) {
        uint32_t n = sparse_dimension(v, a[0]);
        Diagonal diagonals[3] = {{.constant = 1, .scalar = 1.0}};
        if (op == SP_TRIDIAG)
            for (int i = 0; i < 3; i++)
                diagonals[i] = diagonal_input(v, a[i + 1], i - 1);
        Obj *out = sparse_matrix(v, n, n, 1);
        fill_diagonals(v, out, diagonals, op == SP_EYE ? 1u : 3u);
        return objval(out);
    }
    Obj *o = as_sparse(v, a[0]);
    Sparse *s = &S(o);
    switch (op) {
    case SP_TO_CSR: case SP_TO_CRS: return objval(sparse_as_csr(v, o));
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
    case SP_MATVEC: case SP_SPMM: return objval(sparse_product(v, o, a[1], op == SP_MATVEC ? 1 : 2));
    case SP_TRANSPOSE: return objval(sparse_transpose(v, o));
    case SP_ADD: return objval(sparse_add(v, o, as_sparse(v, a[1])));
    case SP_SCALE: return objval(sparse_scale(v, o, sparse_finite(v, a[1])));
    case SP_DIAGONAL: return objval(sparse_diagonal(v, o));
    case SP_ESTIMATE: return sparse_estimate(v, o);
    case SP_VERIFY: return sparse_verify(v, o, a[1], a[2]);
    case SP_TO_DENSE: return objval(todense(v, o));
    default: vmerror(v, "unknown sparse operation");
    }
}

Val sparse_call(VM *v, int id, int argc, Val *a) {
    v->sparse_checkpoint = v->objects;
    v->sparse_active = 1;
    Val result = sparse_dispatch(v, id, argc, a);
    v->sparse_active = 0;
    v->sparse_checkpoint = NULL;
    return result;
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
