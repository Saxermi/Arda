#include "sparse_internal.h"
#include "poisson.h"
#include "pde.h"
#include "typed_tensor.h"
#include <float.h>

#define A(o) ((o)->as.array)
#define L(o) ((o)->as.linop)

enum { SHAPE = LINOP_KRON + 1, APPLY, ADJOINT, POISSON, INFO };
static const char *const names[] = {"lowrank", "diagonal", "banded",          "toeplitz", "kron",
                                    "shape",   "apply",    "apply_transpose", "poisson",  "info"};
static const int arities[] = {2, 1, 3, 2, 2, 1, 2, 2, 1, 1};
_Static_assert(sizeof names / sizeof *names == sizeof arities / sizeof *arities,
               "linop arity table");

static void finite_array(VM *v, Obj *o) {
    int packed = tensor_contiguous(o);
    for (size_t i = 0; i < A(o).n; i++)
        if (!isfinite(A(o).data[packed ? i : tensor_offset(o, i)]))
            vmerror(v, "nonfinite linear operator input or product");
}

static Obj *array_of_rank(VM *v, Val value, int rank) {
    if (!isobj(value, O_ARRAY) || A(value.o).rank != rank)
        vmerror(v, "linear operator expects a rank-%d Float64 tensor", rank);
    return value.o;
}

static int axis_product(VM *v, int a, int b) {
    uint64_t size = (uint64_t)a * b;
    if (size > ARDA_MAX_ELEMENTS)
        vmerror(v, "Kronecker axis exceeds the array limit");
    return (int)size;
}

static int bandwidth(VM *v, Val value, int n) {
    if (value.t != V_INT || value.i < 0 || value.i >= n)
        vmerror(v, "bandwidth must be an integer in [0,n)");
    return (int)value.i;
}

static Obj *construct(VM *v, int kind, Val *args) {
    Obj *a, *b = NULL;
    int rows, cols, lower = 0, upper = 0;
    switch (kind) {
    case LINOP_DIAGONAL:
        a = array_of_rank(v, args[0], 1);
        rows = cols = A(a).rows;
        break;
    case LINOP_BANDED:
        a = array_of_rank(v, args[2], 2);
        rows = cols = A(a).cols;
        lower = bandwidth(v, args[0], cols);
        upper = bandwidth(v, args[1], cols);
        if (A(a).rows != lower + upper + 1)
            vmerror(v, "band storage must have lower + upper + 1 rows");
        break;
    case LINOP_TOEPLITZ:
        a = array_of_rank(v, args[0], 1);
        b = array_of_rank(v, args[1], 1);
        rows = A(a).rows;
        cols = A(b).rows;
        if (!rows || !cols || A(a).data[0] != A(b).data[0])
            vmerror(v, "Toeplitz vectors must be nonempty with a shared first value");
        break;
    default:
        a = array_of_rank(v, args[0], 2);
        b = array_of_rank(v, args[1], 2);
        if (kind == LINOP_LOWRANK) {
            if (A(a).cols != A(b).cols)
                vmerror(v, "low-rank factors must have equal column counts");
            rows = A(a).rows;
            cols = A(b).rows;
        } else {
            rows = axis_product(v, A(a).rows, A(b).rows);
            cols = axis_product(v, A(a).cols, A(b).cols);
        }
        break;
    }
    finite_array(v, a);
    if (b)
        finite_array(v, b);
    Obj *out = newobj(v, O_LINOP);
    L(out).kind = kind;
    L(out).rows = rows;
    L(out).cols = cols;
    L(out).lower = lower;
    L(out).upper = upper;
    L(out).first = tensor_copy(v, a);
    if (b)
        L(out).second = tensor_copy(v, b);
    return out;
}

static double packed_at(const Obj *a, size_t i, size_t j) {
    return A(a).data[i * A(a).cols + j];
}

static double entry(const Obj *a, size_t i, size_t j, int transposed) {
    return transposed ? packed_at(a, j, i) : packed_at(a, i, j);
}

static double rounded(long double value) {
    if (fabsl(value) > DBL_MAX)
        return signbit(value) ? -INFINITY : INFINITY;
    return (double)value;
}

static void lowrank(const Obj *a, const Obj *b, const double *x, size_t xs, double *y, size_t ys,
                    double *scratch) {
    size_t rank = A(a).cols;
    for (size_t k = 0; k < rank; k++) {
        long double sum = 0;
        for (int j = 0; j < A(b).rows; j++)
            sum += (long double)packed_at(b, j, k) * x[j * xs];
        scratch[k] = rounded(sum);
    }
    for (int i = 0; i < A(a).rows; i++)
        y[i * ys] = rank ? rounded(linear_dot(rank, A(a).data + i * rank, scratch)) : 0;
}

static void kron(const Obj *a, const Obj *b, const double *x, size_t xs, double *y, size_t ys,
                 double *scratch, int transposed) {
    size_t ra = transposed ? A(a).cols : A(a).rows;
    size_t ca = transposed ? A(a).rows : A(a).cols;
    size_t rb = transposed ? A(b).cols : A(b).rows;
    size_t cb = transposed ? A(b).rows : A(b).cols;
    if ((uint64_t)ca * rb <= (uint64_t)ra * cb) {
        for (size_t j = 0; j < ca; j++)
            for (size_t i = 0; i < rb; i++) {
                long double sum = 0;
                for (size_t k = 0; k < cb; k++)
                    sum += (long double)entry(b, i, k, transposed) * x[(j * cb + k) * xs];
                scratch[j * rb + i] = rounded(sum);
            }
        for (size_t i = 0; i < ra; i++)
            for (size_t j = 0; j < rb; j++) {
                long double sum = 0;
                for (size_t k = 0; k < ca; k++)
                    sum += (long double)entry(a, i, k, transposed) * scratch[k * rb + j];
                y[(i * rb + j) * ys] = rounded(sum);
            }
    } else {
        for (size_t i = 0; i < ra; i++)
            for (size_t j = 0; j < cb; j++) {
                long double sum = 0;
                for (size_t k = 0; k < ca; k++)
                    sum += (long double)entry(a, i, k, transposed) * x[(k * cb + j) * xs];
                scratch[i * cb + j] = rounded(sum);
            }
        for (size_t i = 0; i < ra; i++)
            for (size_t j = 0; j < rb; j++) {
                long double sum = 0;
                for (size_t k = 0; k < cb; k++)
                    sum += (long double)entry(b, j, k, transposed) * scratch[i * cb + k];
                y[(i * rb + j) * ys] = rounded(sum);
            }
    }
}

static void banded(const Obj *o, const double *x, size_t xs, double *y, size_t ys, int transposed) {
    int n = L(o).cols, lower = L(o).lower, upper = L(o).upper;
    const Obj *bands = L(o).first;
    for (int i = 0; i < n; i++) {
        int first = i - (transposed ? upper : lower);
        int last = i + (transposed ? lower : upper);
        long double sum = 0;
        for (int j = first > 0 ? first : 0; j <= last && j < n; j++) {
            double value = transposed ? packed_at(bands, upper + j - i, i)
                                      : packed_at(bands, upper + i - j, j);
            sum += (long double)value * x[j * xs];
        }
        y[i * ys] = rounded(sum);
    }
}

static void structured_apply(const LinopContext *ctx, const double *x, size_t xs, double *y,
                             size_t ys, int transposed) {
    const Obj *o = ctx->op, *a = L(o).first, *b = L(o).second;
    switch (L(o).kind) {
    case LINOP_LOWRANK:
        lowrank(transposed ? b : a, transposed ? a : b, x, xs, y, ys, ctx->scratch);
        break;
    case LINOP_DIAGONAL:
        for (int i = 0; i < L(o).rows; i++)
            y[i * ys] = A(a).data[i] * x[i * xs];
        break;
    case LINOP_BANDED:
        banded(o, x, xs, y, ys, transposed);
        break;
    case LINOP_TOEPLITZ: {
        const Obj *col = transposed ? b : a, *row = transposed ? a : b;
        for (int i = 0; i < A(col).rows; i++) {
            long double sum = 0;
            for (int j = 0; j < A(row).rows; j++) {
                double value = i >= j ? A(col).data[i - j] : A(row).data[j - i];
                sum += (long double)value * x[j * xs];
            }
            y[i * ys] = rounded(sum);
        }
        break;
    }
    case LINOP_KRON:
        kron(a, b, x, xs, y, ys, ctx->scratch, transposed);
        break;
    }
}

static void forward(const void *ctx, const double *x, size_t xs, double *y, size_t ys) {
    structured_apply(ctx, x, xs, y, ys, 0);
}

static void transpose(const void *ctx, const double *x, size_t xs, double *y, size_t ys) {
    structured_apply(ctx, x, xs, y, ys, 1);
}

void linop_shape(VM *v, Obj *input, size_t *rows, size_t *cols) {
    if (input->type == O_LINOP) {
        *rows = L(input).rows;
        *cols = L(input).cols;
    } else if (input->type == O_SPARSE) {
        *rows = input->as.sparse.rows;
        *cols = input->as.sparse.cols;
    } else if (input->type == O_ARRAY && A(input).rank == 2) {
        *rows = A(input).rows;
        *cols = A(input).cols;
    } else if (input->type == O_TYPED_TENSOR && input->as.tensor.rank == 2 &&
               input->as.tensor.dtype == DT_F64) {
        *rows = (size_t)input->as.tensor.shape[0];
        *cols = (size_t)input->as.tensor.shape[1];
    } else {
        vmerror(v, "expected a Float64 matrix or linear operator");
    }
}

double linop_dense_at(const Obj *input, size_t i, size_t j) {
    if (input->type == O_ARRAY)
        return A(input).data[i * A(input).stride[0] + j * A(input).stride[1]];
    const TypedTensor *t = &input->as.tensor;
    ptrdiff_t at = (ptrdiff_t)t->offset + (ptrdiff_t)i * t->stride[0] + (ptrdiff_t)j * t->stride[1];
    return *(const double *)typed_element(t, at);
}

size_t linop_workspace(VM *v, Obj *input) {
    if (input->type != O_LINOP)
        return 0;
    uint64_t scratch = 0;
    if (L(input).kind == LINOP_LOWRANK)
        scratch = A(L(input).first).cols;
    else if (L(input).kind == LINOP_KRON && L(input).rows && L(input).cols) {
        const Obj *a = L(input).first, *b = L(input).second;
        uint64_t right_first = (uint64_t)A(a).cols * A(b).rows;
        uint64_t left_first = (uint64_t)A(a).rows * A(b).cols;
        scratch = right_first < left_first ? right_first : left_first;
    }
    if (scratch > ARDA_MAX_ELEMENTS)
        vmerror(v, "linear operator workspace exceeds the array limit");
    return (size_t)scratch * sizeof(double);
}

static void typed_product(const void *context, const double *x, size_t xs, double *y, size_t ys,
                          bool transpose) {
    const TypedTensor *t = context;
    int axis = transpose ? 1 : 0;
    for (int i = 0; i < t->shape[axis]; i++) {
        long double sum = 0;
        ptrdiff_t base = (ptrdiff_t)t->offset + (ptrdiff_t)i * t->stride[axis];
        for (int j = 0; j < t->shape[1 - axis]; j++) {
            const double *at =
                (const double *)typed_element(t, base + (ptrdiff_t)j * t->stride[1 - axis]);
            sum += (long double)*at * x[(size_t)j * xs];
        }
        y[(size_t)i * ys] = (double)sum;
    }
}

static void typed_forward(const void *context, const double *x, size_t xs, double *y, size_t ys) {
    typed_product(context, x, xs, y, ys, false);
}

static void typed_adjoint(const void *context, const double *x, size_t xs, double *y, size_t ys) {
    typed_product(context, x, xs, y, ys, true);
}

LinearOperator linop_open(VM *v, Obj *input, LinopContext *ctx) {
    size_t rows, cols;
    linop_shape(v, input, &rows, &cols);
    *ctx = (LinopContext){.op = input};
    if (input->type == O_ARRAY) {
        finite_array(v, input);
        return dense_operator(input);
    }
    if (input->type == O_TYPED_TENSOR) {
        TypedTensor *t = &input->as.tensor;
        for (size_t i = 0; i < t->n; i++)
            if (!isfinite(*(const double *)typed_element(t, typed_offset(t, i))))
                vmerror(v, "nonfinite linear operator input or product");
        return (LinearOperator){.rows = rows,
                                .cols = cols,
                                .context = t,
                                .apply = typed_forward,
                                .apply_transpose = typed_adjoint};
    }
    if (input->type == O_SPARSE) {
        if (!input->as.sparse.csr)
            sparse_begin(v);
        ctx->csr = sparse_as_csr(v, input);
        return sparse_operator(&ctx->csr->as.sparse);
    }
    if (L(input).kind == LINOP_POISSON)
        return (LinearOperator){.rows = rows,
                                .cols = cols,
                                .context = input,
                                .apply = poisson_operator_apply,
                                .apply_transpose = poisson_operator_apply};
    if (L(input).kind == LINOP_CARTESIAN)
        return (LinearOperator){.rows = rows,
                                .cols = cols,
                                .context = input,
                                .apply = pde_operator_apply,
                                .apply_transpose = pde_operator_apply};
    size_t scratch = linop_workspace(v, input);
    if (scratch) {
        ctx->scratch_array = array_new(v, 1, (int)(scratch / sizeof(double)), 1);
        ctx->scratch = A(ctx->scratch_array).data;
    }
    return (LinearOperator){.rows = rows,
                            .cols = cols,
                            .context = ctx,
                            .workspace_bytes = scratch,
                            .apply = forward,
                            .apply_transpose = transpose};
}

void linop_close(VM *v, LinopContext *ctx) {
    if (ctx->scratch_array)
        tensor_buffer_discard(v, A(ctx->scratch_array).buffer);
    if (ctx->csr && ctx->csr != ctx->op) {
        sparse_drop_temporary(v, ctx->csr, (Obj *)ctx->op);
        sparse_finish(v);
    }
}

void linop_apply_into(VM *v, const LinearOperator *op, Obj *x, Obj *y, int transposed) {
    size_t source = transposed ? op->rows : op->cols;
    size_t target = transposed ? op->cols : op->rows;
    size_t width = A(x).cols;
    if ((size_t)A(x).rows != source || (size_t)A(y).rows != target || (size_t)A(y).cols != width)
        vmerror(v, "linear operator product shape mismatch");
    finite_array(v, x);
    if (source && target) {
        LinearApply apply = transposed ? op->apply_transpose : op->apply;
        size_t cs = A(x).rank == 2 ? A(x).stride[1] : 0;
        for (size_t j = 0; j < width; j++)
            apply(op->context, A(x).data + j * cs, A(x).stride[0], A(y).data + j, width);
    } else if (A(y).n) {
        memset(A(y).data, 0, A(y).n * sizeof(double));
    }
    finite_array(v, y);
}

Val linop_call(VM *v, int id, int argc, Val *args) {
    int op = id - LINOP_BASE;
    if (op < 0 || (size_t)op >= sizeof names / sizeof *names)
        vmerror(v, "unknown linear operator operation");
    if (argc != arities[op] && !(op == POISSON && argc == 2))
        vmerror(v, "invalid linop.%s arity", names[op]);
    if (op == POISSON)
        return poisson_new(v, args[0], argc == 2 ? args[1] : nil());
    if (op < SHAPE)
        return objval(construct(v, op, args));
    if (args[0].t != V_OBJ)
        vmerror(v, "expected a matrix or linear operator");
    size_t rows, cols;
    linop_shape(v, args[0].o, &rows, &cols);
    if (op == INFO) {
        Obj *info = newobj(v, O_MAP);
        sparse_result_field(v, info, "rows", integer((int64_t)rows));
        sparse_result_field(v, info, "cols", integer((int64_t)cols));
        sparse_result_field(v, info, "dtype", objval(string(v, "float64")));
        sparse_result_field(v, info, "workspace_bytes",
                            integer((int64_t)linop_workspace(v, args[0].o)));
        sparse_result_field(v, info, "apply_transpose", boolean(1));
        size_t conversion = isobj(args[0], O_SPARSE) && !args[0].o->as.sparse.csr
                                ? sparse_csr_estimate(v, &args[0].o->as.sparse)
                                : 0;
        sparse_result_field(v, info, "coo_conversion_bytes", integer((int64_t)conversion));
        return objval(info);
    }
    if (op == SHAPE) {
        Obj *out = list(v);
        append(v, out, integer((int64_t)rows));
        append(v, out, integer((int64_t)cols));
        return objval(out);
    }
    if (!isobj(args[1], O_ARRAY) || (A(args[1].o).rank != 1 && A(args[1].o).rank != 2))
        vmerror(v, "linear operator product expects a vector or matrix");
    int transposed = op == ADJOINT;
    size_t source = transposed ? rows : cols, target = transposed ? cols : rows;
    if (target > ARDA_MAX_ELEMENTS)
        vmerror(v, "linear operator product exceeds the array limit");
    Obj *x = args[1].o;
    if ((size_t)A(x).rows != source)
        vmerror(v, "linear operator product shape mismatch");
    Obj *y = array_new(v, A(x).rank, (int)target, A(x).cols);
    LinopContext context;
    LinearOperator applied = linop_open(v, args[0].o, &context);
    linop_apply_into(v, &applied, x, y, transposed);
    linop_close(v, &context);
    return objval(y);
}

Obj *linop_module(VM *v) {
    return named_native_module(v, "linop", names, sizeof names / sizeof *names, LINOP_BASE);
}
