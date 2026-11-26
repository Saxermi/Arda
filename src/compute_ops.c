#include "compute_ops.h"
#include "cpu_kernels.h"

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

#if defined(_OPENMP) && !defined(ARDA_BLAS)
#include <omp.h>
#endif

static TypedTensor real_tensor(VM *v, Val value) {
    if (isobj(value, O_AD))
        vmerror_kind(v, "autodiff", "explicit compute kernels require ordinary tensors");
    TypedTensor t = typed_tensor(v, value);
    if (t.dtype != DT_F64)
        vmerror_kind(v, "dtype", "compute kernels require Float64 tensors");
    return t;
}

MatmulProblem compute_matmul_problem(VM *v, Val left, Val right) {
    TypedTensor a = real_tensor(v, left), b = real_tensor(v, right);
    if (a.rank != 2 || (b.rank != 1 && b.rank != 2) || a.shape[1] != b.shape[0])
        vmerror_kind(v, "shape", "compute matmul needs a matrix and a matching vector or matrix");
    int shape[] = {a.shape[0], b.rank == 2 ? b.shape[1] : 1};
    TypedTensor out = typed_layout(v, DT_F64, b.rank, shape);
    if (!typed_materializable(&out))
        vmerror_kind(v, "shape", "compute matmul output exceeds the tensor limit");
    return (MatmulProblem){.a = a, .b = b, .output = out};
}

int compute_tile_size(VM *v, Val value) {
    if (value.t != V_INT || value.i < 1 || value.i > COMPUTE_TILE_MAX)
        vmerror_kind(v, "domain", "tile size must be an integer from 1 to 256");
    return (int)value.i;
}

size_t compute_tiled_workspace(const MatmulProblem *p, int block) {
    if (!p->output.n || !p->a.shape[1] || p->b.rank == 1 || p->b.stride[1] == 1)
        return 0;
    int k = p->a.shape[1] < block ? p->a.shape[1] : block;
    int n = p->b.shape[1] < block ? p->b.shape[1] : block;
    return (size_t)k * n * sizeof(double);
}

static const double *base(const TypedTensor *t) {
    return (const double *)typed_element(t, (ptrdiff_t)t->offset);
}

static int span(int begin, int end, int block) {
    return end - begin < block ? end - begin : block;
}

static void tiled_vectors(const MatmulProblem *p, double *out, int first, int last) {
    const double *a = base(&p->a), *b = base(&p->b);
    ptrdiff_t stride_a = p->a.stride[1], stride_b = p->b.stride[0];
    int depth = p->a.shape[1];
    for (int i = first; i < last; i++) {
        const double *row = a + (ptrdiff_t)i * p->a.stride[0];
        double sum = 0;
        if (stride_a == 1 && stride_b == 1)
            for (int k = 0; k < depth; k++)
                sum += row[k] * b[k];
        else
            for (int k = 0; k < depth; k++)
                sum += row[(ptrdiff_t)k * stride_a] * b[(ptrdiff_t)k * stride_b];
        out[i] = sum;
    }
}

static void tiled_rows(const MatmulProblem *p, double *out, int block, double *packed, int first,
                       int last) {
    if (p->b.rank == 1) {
        tiled_vectors(p, out, first, last);
        return;
    }
    const double *a = base(&p->a), *b = base(&p->b);
    int depth = p->a.shape[1], columns = p->b.shape[1];
    for (int row = first; row < last; row += block) {
        int m = span(row, last, block);
        for (int col = 0; col < columns; col += block) {
            int n = span(col, columns, block);
            for (int inner = 0; inner < depth;) {
                int k = span(inner, depth, block);
                const double *right =
                    b + (ptrdiff_t)inner * p->b.stride[0] + (ptrdiff_t)col * p->b.stride[1];
                ptrdiff_t right_stride = p->b.stride[0];
                if (packed) {
                    for (int q = 0; q < k; q++)
                        for (int j = 0; j < n; j++)
                            packed[(size_t)q * n + j] =
                                right[(ptrdiff_t)q * right_stride + (ptrdiff_t)j * p->b.stride[1]];
                    right = packed;
                    right_stride = n;
                }
                for (int i = row; i < row + m; i++) {
                    const double *left = a + (ptrdiff_t)i * p->a.stride[0];
                    double *dst = out + (size_t)i * columns + col;
                    for (int q = 0; q < k; q++) {
                        double x = left[(ptrdiff_t)(inner + q) * p->a.stride[1]];
                        const double *src = right + (ptrdiff_t)q * right_stride;
                        for (int j = 0; j < n; j++)
                            dst[j] += x * src[j];
                    }
                }
                inner += k;
            }
        }
    }
}

int compute_tiled_team(const MatmulProblem *p, int block, int threads, bool pack) {
    if (pack || !p->output.n || !p->a.shape[1])
        return 1;
    uint64_t terms = (uint64_t)p->output.n * (unsigned)p->a.shape[1];
    int rows = p->a.shape[0];
    if (terms < (p->b.rank == 1 ? UINT64_C(1048576) : UINT64_C(8000000)))
        return 1;
    return cpu_team_size(threads, rows / block);
}

int compute_tiled_into(const MatmulProblem *p, int block, int threads, double *out,
                       double *workspace) {
    if (!p->output.n || !p->a.shape[1])
        return 1;
    int rows = p->a.shape[0];
#if defined(_OPENMP) && !defined(ARDA_BLAS)
    int team = compute_tiled_team(p, block, threads, workspace != NULL);
    if (team > 1) {
        int actual = 1, blocks = (rows + block - 1) / block;
#pragma omp parallel num_threads(team) default(none) shared(actual)                                \
    firstprivate(p, out, block, rows, blocks)
        {
            int rank = omp_get_thread_num(), size = omp_get_num_threads();
            if (!rank)
                actual = size;
            int first = (int)((size_t)blocks * rank / size) * block;
            int last = (int)((size_t)blocks * (rank + 1) / size) * block;
            tiled_rows(p, out, block, NULL, first, last < rows ? last : rows);
        }
        return actual;
    }
#else
    (void)threads;
#endif
    tiled_rows(p, out, block, workspace, 0, rows);
    return 1;
}

Val compute_tiled(VM *v, Val left, Val right, int block) {
    MatmulProblem p = compute_matmul_problem(v, left, right);
    size_t available = tensor_memory_available(v), scratch = compute_tiled_workspace(&p, block);
    if (scratch > available || p.output.n * sizeof(double) > available - scratch)
        vmerror_kind(v, "memory_budget",
                     "tiled matmul output and workspace exceed the tensor budget");
    unsigned char *work = scratch ? tensor_workspace_acquire(v, scratch) : NULL;
    if (scratch && !work)
        vmerror_kind(v, "memory_budget", "unable to reserve tiled matmul workspace");
    Obj *out = typed_new(v, DT_F64, p.output.rank, p.output.shape);
    compute_tiled_into(&p, block, v->native_threads, (double *)typed_element(&out->as.tensor, 0),
                       (double *)work);
    if (work)
        tensor_workspace_release(v, work);
    return objval(out);
}

typedef struct {
    TypedTensor tensor;
    const double *data;
} Operand;

static inline double packed_at(const Operand *operand, size_t index) {
    return operand->data[index];
}

static inline double strided_at(const Operand *operand, size_t index) {
    return *(const double *)typed_element(&operand->tensor, typed_offset(&operand->tensor, index));
}

#define EVALUATE(EXPR)                                                                             \
    if (packed) {                                                                                  \
        for (size_t i = 0; i < count; i++)                                                         \
            data[i] = EXPR(packed_at);                                                             \
    } else {                                                                                       \
        for (size_t i = 0; i < count; i++)                                                         \
            data[i] = EXPR(strided_at);                                                            \
    }
#define AX(A) (alpha * A(&inputs[0], i) + A(&inputs[1], i))
#define MUL_ADD(A) (A(&inputs[0], i) * A(&inputs[1], i) + A(&inputs[2], i))
#define PRODUCTS(A)                                                                                \
    ((A(&inputs[0], i) * A(&inputs[1], i) + A(&inputs[2], i) * A(&inputs[3], i)) + A(&inputs[4], i))

Val compute_fusion(VM *v, Fusion kind, const Val *args) {
    int operands = kind == FUSE_AXPY ? 2 : kind == FUSE_MUL_ADD ? 3 : 5;
    double alpha = kind == FUSE_AXPY ? asnum(v, args[2]) : 0;
    Operand inputs[5];
    bool packed = true;
    for (int i = 0; i < operands; i++) {
        TypedTensor t = real_tensor(v, args[i]);
        TypedTensor first = i ? inputs[0].tensor : t;
        if (t.rank != first.rank || memcmp(t.shape, first.shape, (size_t)t.rank * sizeof(int)))
            vmerror_kind(v, "shape", "fused tensor operands must have the same shape");
        inputs[i] = (Operand){.tensor = t, .data = base(&t)};
        packed &= typed_contiguous(&t);
    }
    TypedTensor first = inputs[0].tensor;
    Obj *out = typed_new(v, DT_F64, first.rank, first.shape);
    double *data = (double *)typed_element(&out->as.tensor, 0);
    size_t count = first.n;
    switch (kind) {
    case FUSE_AXPY:
        EVALUATE(AX);
        break;
    case FUSE_MUL_ADD:
        EVALUATE(MUL_ADD);
        break;
    case FUSE_PRODUCTS:
        EVALUATE(PRODUCTS);
        break;
    }
    return objval(out);
}
#undef EVALUATE
#undef AX
#undef MUL_ADD
#undef PRODUCTS

#define REDUCE(MAP)                                                                                \
    if (packed) {                                                                                  \
        for (size_t i = 0; i < t.n; i++) {                                                         \
            double x = packed_at(&input, i);                                                       \
            sum += (MAP);                                                                          \
        }                                                                                          \
    } else {                                                                                       \
        for (size_t i = 0; i < t.n; i++) {                                                         \
            double x = strided_at(&input, i);                                                      \
            sum += (MAP);                                                                          \
        }                                                                                          \
    }
Val compute_map_reduce(VM *v, Val map, Val value) {
    const char *name = asstr(v, map);
    TypedTensor t = real_tensor(v, value);
    Operand input = {.tensor = t, .data = base(&t)};
    long double sum = 0;
    bool packed = typed_contiguous(&t);
    if (!strcmp(name, "sum")) {
        REDUCE(x);
    } else if (!strcmp(name, "sum_squares")) {
        REDUCE(x * x);
    } else if (!strcmp(name, "sum_sin")) {
        REDUCE(sin(x));
    } else
        vmerror_kind(v, "domain", "unsupported compute.map_reduce map");
    return number((double)sum);
}
#undef REDUCE
