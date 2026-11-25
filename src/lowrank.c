#include "typed_tensor.h"

#include <float.h>

#define A(o) ((o)->as.array)
#define L(o) ((o)->as.lowrank)

static LowRankTensor *as_lowrank(VM *v, Val value) {
    if (!isobj(value, O_LOWRANK))
        vmerror(v, "expected a low-rank tensor");
    return &L(value.o);
}

static Obj *publish(VM *v, LowRankTensor tensor) {
    Obj *out = newobj(v, O_LOWRANK);
    L(out) = tensor;
    for (int d = 0; d < tensor.rank; d++)
        barrier(v, out, objval(tensor.factor[d]));
    if (tensor.coefficients)
        barrier(v, out, objval(tensor.coefficients));
    return out;
}

static double checked_result(VM *v, long double value) {
    if (!isfinite(value) || fabsl(value) > DBL_MAX)
        vmerror(v, "low-rank result is outside the Float64 range");
    return (double)value;
}

static TypedTensor component(VM *v, Val value, int rank) {
    TypedTensor input = typed_tensor(v, value);
    if (input.dtype != DT_F64 || (rank && input.rank != rank))
        vmerror(v, "low-rank component requires a Float64 tensor of the expected rank");
    if (!typed_materializable(&input))
        vmerror(v, "low-rank component exceeds the heap tensor limit");
    return input;
}

static long double *scratch(VM *v, size_t entries) {
    if (entries > ARDA_MAX_ELEMENTS)
        vmerror(v, "low-rank workspace exceeds the element limit");
    unsigned char *work = tensor_workspace_acquire(v, entries * sizeof(long double));
    if (!work)
        vmerror(v, "low-rank workspace exceeds the tensor memory budget or allocation failed");
    return (long double *)work;
}

static Obj *snapshot(VM *v, const TypedTensor *t) {
    Obj *out = tensor_new(v, t->rank, t->shape);
    for (size_t i = 0; i < t->n; i++) {
        double x = dtype_load(DT_F64, typed_element(t, typed_offset(t, i))).re;
        if (!isfinite(x))
            vmerror(v, "low-rank components must be finite");
        A(out).data[i] = x;
    }
    return out;
}

static Obj *construct(VM *v, LowRankKind kind, Val factors, Val coefficients) {
    if (!isobj(factors, O_LIST) || factors.o->as.list.n < 1 || factors.o->as.list.n > 8)
        vmerror(v, "low-rank tensor requires a list of 1..8 components");
    LowRankTensor t = {.kind = kind, .rank = factors.o->as.list.n};
    TypedTensor input[8], weights;
    int cp_rank = 0, bond = 1;
    size_t bytes = 0;
    if (kind == LOWRANK_TUCKER) {
        weights = component(v, coefficients, t.rank);
        bytes = weights.n * sizeof(double);
    }
    for (int d = 0; d < t.rank; d++) {
        input[d] = component(v, factors.o->as.list.a[d], kind == LOWRANK_TT ? 3 : 2);
        const int *shape = input[d].shape;
        if (kind == LOWRANK_TT) {
            if (shape[0] != bond || shape[2] < 1)
                vmerror(v, "tensor train bond ranks do not match");
            bond = shape[2];
            t.shape[d] = shape[1];
        } else {
            if (kind == LOWRANK_CP && d == 0)
                cp_rank = shape[1];
            int columns = kind == LOWRANK_CP ? cp_rank : weights.shape[d];
            if (columns < 1 || shape[1] != columns)
                vmerror(v, "low-rank factor columns do not match latent ranks");
            t.shape[d] = shape[0];
        }
        bytes += input[d].n * sizeof(double);
    }
    if (kind == LOWRANK_TT && bond != 1)
        vmerror(v, "tensor train boundary ranks must be one");
    if (kind == LOWRANK_CP) {
        bytes += (size_t)cp_rank * sizeof(double);
        if (isobj(coefficients, O_LIST)) {
            if (coefficients.o->as.list.n != cp_rank)
                vmerror(v, "CP weights must match the factor rank");
            for (int r = 0; r < cp_rank; r++)
                if (!isfinite(asnum(v, coefficients.o->as.list.a[r])))
                    vmerror(v, "CP weights must be finite");
        } else if (coefficients.t != V_NIL) {
            weights = component(v, coefficients, 1);
            if (weights.shape[0] != cp_rank)
                vmerror(v, "CP weights must match the factor rank");
        }
    }
    if (bytes > tensor_memory_available(v))
        vmerror(v, "low-rank components exceed the tensor memory budget");
    for (int d = 0; d < t.rank; d++)
        t.factor[d] = snapshot(v, &input[d]);
    if (kind == LOWRANK_TUCKER ||
        (kind == LOWRANK_CP && coefficients.t != V_NIL && !isobj(coefficients, O_LIST)))
        t.coefficients = snapshot(v, &weights);
    else if (kind == LOWRANK_CP) {
        t.coefficients = array_new(v, 1, cp_rank, 1);
        for (int r = 0; r < cp_rank; r++)
            A(t.coefficients).data[r] =
                coefficients.t == V_NIL ? 1 : asnum(v, coefficients.o->as.list.a[r]);
    }
    return publish(v, t);
}

static void coordinates(size_t index, int rank, const int *shape, int *out) {
    for (int d = rank - 1; d >= 0; d--) {
        out[d] = (int)(index % (size_t)shape[d]);
        index /= (size_t)shape[d];
    }
}

static size_t tt_width(const LowRankTensor *t) {
    size_t width = 1;
    for (int d = 0; d < t->rank; d++) {
        size_t bond = (size_t)A(t->factor[d]).shape[2];
        if (bond > width)
            width = bond;
    }
    return width;
}

static long double at(const LowRankTensor *t, const int *index, long double *work, size_t width) {
    long double sum = 0;
    if (t->kind == LOWRANK_CP) {
        size_t rank = A(t->coefficients).n;
        for (size_t r = 0; r < rank; r++) {
            long double term = A(t->coefficients).data[r];
            for (int d = 0; d < t->rank; d++)
                term *= A(t->factor[d]).data[(size_t)index[d] * rank + r];
            sum += term;
        }
    } else if (t->kind == LOWRANK_TUCKER) {
        Obj *core = t->coefficients;
        for (size_t p = 0; p < A(core).n; p++) {
            size_t idx = p;
            long double term = A(core).data[p];
            for (int d = t->rank - 1; d >= 0; d--) {
                size_t rank = (size_t)A(core).shape[d];
                term *= A(t->factor[d]).data[(size_t)index[d] * rank + idx % rank];
                idx /= rank;
            }
            sum += term;
        }
    } else {
        long double *current = work, *next = work + width;
        current[0] = 1;
        for (int d = 0; d < t->rank; d++) {
            Obj *g = t->factor[d];
            int left = A(g).shape[0], right = A(g).shape[2], rows = t->shape[d];
            for (int q = 0; q < right; q++) {
                next[q] = 0;
                for (int p = 0; p < left; p++)
                    next[q] += current[p] * A(g).data[((size_t)p * rows + index[d]) * right + q];
            }
            long double *swap = current;
            current = next;
            next = swap;
        }
        sum = current[0];
    }
    return sum;
}

static Val element(VM *v, const LowRankTensor *t, Val indices) {
    if (!isobj(indices, O_LIST) || indices.o->as.list.n != t->rank)
        vmerror(v, "expected one index per low-rank dimension");
    int index[8];
    for (int d = 0; d < t->rank; d++)
        index[d] = indexnum(v, indices.o->as.list.a[d], t->shape[d]);
    size_t width = t->kind == LOWRANK_TT ? tt_width(t) : 0;
    long double *work = width ? scratch(v, width * 2) : NULL;
    long double value = at(t, index, work, width);
    if (work)
        tensor_workspace_release(v, (unsigned char *)work);
    return number(checked_result(v, value));
}

static Obj *materialize(VM *v, const LowRankTensor *t) {
    Obj *out = tensor_new(v, t->rank, t->shape);
    if (!A(out).n)
        return out;
    size_t width = t->kind == LOWRANK_TT ? tt_width(t) : 0;
    long double *work = width ? scratch(v, width * 2) : NULL;
    for (size_t i = 0; i < A(out).n; i++) {
        int index[8];
        coordinates(i, t->rank, t->shape, index);
        long double value = at(t, index, work, width);
        if (!isfinite(value) || fabsl(value) > DBL_MAX) {
            if (work)
                tensor_workspace_release(v, (unsigned char *)work);
            vmerror(v, "low-rank result is outside the Float64 range");
        }
        A(out).data[i] = (double)value;
    }
    if (work)
        tensor_workspace_release(v, (unsigned char *)work);
    return out;
}

static long double column_dot(const Obj *a, const Obj *b, int p, int q) {
    long double sum = 0;
    for (int i = 0; i < A(a).rows; i++)
        sum += (long double)A(a).data[(size_t)i * A(a).cols + p] *
               A(b).data[(size_t)i * A(b).cols + q];
    return sum;
}

static long double cp_dot(VM *v, const LowRankTensor *a, const LowRankTensor *b) {
    long double sum = 0, magnitude = 0;
    int ra = A(a->coefficients).rows, rb = A(b->coefficients).rows;
    for (int p = 0; p < ra; p++) {
        for (int q = 0; q < rb; q++) {
            long double value =
                (long double)A(a->coefficients).data[p] * A(b->coefficients).data[q];
            for (int d = 0; d < a->rank; d++)
                value *= column_dot(a->factor[d], b->factor[d], p, q);
            sum += value;
            magnitude += fabsl(value);
        }
    }
    if (magnitude > 0 && fabsl(sum) <= 64 * LDBL_EPSILON * magnitude) {
        // Gram contributions can lose small residuals of nearly identical
        // components. For small problems re-evaluate entries without storing
        // a dense tensor. Bound the fallback work before traversing the shape.
        size_t limit = 1000000u / ((size_t)ra + rb), count = 1;
        for (int d = 0; d < a->rank; d++) {
            if (count > limit / (size_t)a->shape[d])
                vmerror(v,
                        "CP dot is cancellation dominated and exceeds the direct evaluation limit");
            count *= (size_t)a->shape[d];
        }
        sum = 0;
        for (size_t i = 0; i < count; i++) {
            int index[8];
            coordinates(i, a->rank, a->shape, index);
            sum += at(a, index, NULL, 0) * at(b, index, NULL, 0);
        }
    }
    return sum;
}

static long double tucker_dot(VM *v, const LowRankTensor *a, const LowRankTensor *b) {
    Obj *x = a->coefficients, *y = b->coefficients;
    size_t volumes = A(x).n > A(y).n ? A(x).n : A(y).n, grams = 0;
    for (int d = 0; d < a->rank; d++) {
        size_t nr = (size_t)A(x).shape[d], nc = (size_t)A(y).shape[d];
        if (nr > ARDA_MAX_ELEMENTS / nc)
            vmerror(v, "Tucker factor cross-Gram exceeds the workspace limit");
        if (nr * nc > grams)
            grams = nr * nc;
    }
    long double *ws = scratch(v, 2 * volumes + grams);
    long double *current = ws, *next = ws + volumes, *gram = next + volumes;
    for (size_t i = 0; i < A(y).n; i++)
        current[i] = A(y).data[i];
    int dims[8];
    memcpy(dims, A(y).shape, (size_t)a->rank * sizeof *dims);
    size_t total = A(y).n;
    // Mode reductions precede expansions, so every intermediate fits the
    // larger endpoint core.
    for (int expanding = 0; expanding < 2; expanding++) {
        for (int d = 0; d < a->rank; d++) {
            int na = A(x).shape[d], nb = dims[d];
            if ((na > nb) != expanding)
                continue;
            for (int i = 0; i < na; i++)
                for (int j = 0; j < nb; j++)
                    gram[(size_t)i * nb + j] = column_dot(a->factor[d], b->factor[d], i, j);
            size_t inner = 1;
            for (int j = d + 1; j < a->rank; j++)
                inner *= (size_t)dims[j];
            size_t outer = total / ((size_t)nb * inner);
            for (size_t p = 0; p < outer; p++)
                for (int i = 0; i < na; i++)
                    for (size_t q = 0; q < inner; q++) {
                        long double value = 0;
                        for (int j = 0; j < nb; j++)
                            value += gram[(size_t)i * nb + j] * current[(p * nb + j) * inner + q];
                        next[(p * na + i) * inner + q] = value;
                    }
            dims[d] = na;
            total = outer * (size_t)na * inner;
            long double *swap = current;
            current = next;
            next = swap;
        }
    }
    long double sum = 0;
    for (size_t i = 0; i < A(x).n; i++)
        sum += A(x).data[i] * current[i];
    tensor_workspace_release(v, (unsigned char *)ws);
    return sum;
}

static long double tt_dot(VM *v, const LowRankTensor *a, const LowRankTensor *b) {
    size_t transfer = 1, inner = 1;
    for (int d = 0; d < a->rank; d++) {
        const int *x = A(a->factor[d]).shape, *y = A(b->factor[d]).shape;
        if ((size_t)x[2] > ARDA_MAX_ELEMENTS / (size_t)y[2] ||
            (size_t)x[0] > ARDA_MAX_ELEMENTS / (size_t)y[2])
            vmerror(v, "tensor train transfer exceeds the workspace limit");
        size_t next = (size_t)x[2] * y[2], temp = (size_t)x[0] * y[2];
        if (next > transfer)
            transfer = next;
        if (temp > inner)
            inner = temp;
    }
    long double *ws = scratch(v, 2 * transfer + inner);
    long double *current = ws, *next = ws + transfer, *temp = next + transfer;
    current[0] = 1;
    for (int d = 0; d < a->rank; d++) {
        Obj *x = a->factor[d], *y = b->factor[d];
        int al = A(x).shape[0], ar = A(x).shape[2], bl = A(y).shape[0], br = A(y).shape[2];
        int n = a->shape[d];
        memset(next, 0, (size_t)ar * br * sizeof *next);
        for (int i = 0; i < n; i++) {
            for (int p = 0; p < al; p++)
                for (int q = 0; q < br; q++) {
                    long double value = 0;
                    for (int j = 0; j < bl; j++)
                        value +=
                            current[(size_t)p * bl + j] * A(y).data[((size_t)j * n + i) * br + q];
                    temp[(size_t)p * br + q] = value;
                }
            for (int p = 0; p < ar; p++)
                for (int q = 0; q < br; q++)
                    for (int j = 0; j < al; j++)
                        next[(size_t)p * br + q] +=
                            (long double)A(x).data[((size_t)j * n + i) * ar + p] *
                            temp[(size_t)j * br + q];
        }
        long double *swap = current;
        current = next;
        next = swap;
    }
    long double sum = current[0];
    tensor_workspace_release(v, (unsigned char *)ws);
    return sum;
}

static long double inner_product(VM *v, const LowRankTensor *a, const LowRankTensor *b) {
    if (a->kind != b->kind || a->rank != b->rank ||
        memcmp(a->shape, b->shape, (size_t)a->rank * sizeof(int)))
        vmerror(v, "low-rank dot requires matching shapes and formats");
    for (int d = 0; d < a->rank; d++)
        if (a->shape[d] == 0)
            return 0;
    switch (a->kind) {
    case LOWRANK_CP:
        return cp_dot(v, a, b);
    case LOWRANK_TUCKER:
        return tucker_dot(v, a, b);
    case LOWRANK_TT:
        return tt_dot(v, a, b);
    }
    return 0;
}

static long double vector_column(const Obj *factor, const Obj *vector, int col) {
    long double value = 0;
    for (int i = 0; i < A(vector).rows; i++)
        value += (long double)A(vector).data[i] * A(factor).data[(size_t)i * A(factor).cols + col];
    return value;
}

static Val contract_cp(VM *v, LowRankTensor out, const LowRankTensor *t, int axis, Obj *vector) {
    int r = A(t->coefficients).rows;
    if (out.rank)
        out.coefficients = array_new(v, 1, r, 1);
    long double sum = 0;
    for (int i = 0; i < r; i++) {
        long double p = vector_column(t->factor[axis], vector, i) * A(t->coefficients).data[i];
        if (out.rank)
            A(out.coefficients).data[i] = checked_result(v, p);
        else
            sum += p;
    }
    return out.rank ? objval(publish(v, out)) : number(checked_result(v, sum));
}

static Val contract_tucker(VM *v, LowRankTensor out, const LowRankTensor *t, int axis,
                           Obj *vector) {
    Obj *core = t->coefficients;
    int r = A(core).shape[axis];
    if (!out.rank) {
        long double sum = 0;
        for (int j = 0; j < r; j++)
            sum += A(core).data[j] * vector_column(t->factor[axis], vector, j);
        return number(checked_result(v, sum));
    }
    int dims[8];
    for (int d = 0, k = 0; d < t->rank; d++)
        if (d != axis)
            dims[k++] = A(core).shape[d];
    out.coefficients = tensor_new(v, out.rank, dims);
    size_t inner = A(core).stride[axis];
    long double *projection = scratch(v, (size_t)r);
    for (int j = 0; j < r; j++)
        projection[j] = vector_column(t->factor[axis], vector, j);
    for (size_t i = 0; i < A(out.coefficients).n; i++) {
        size_t off = (i / inner) * (size_t)r * inner + i % inner;
        long double sum = 0;
        for (int j = 0; j < r; j++)
            sum += A(core).data[off + (size_t)j * inner] * projection[j];
        if (!isfinite(sum) || fabsl(sum) > DBL_MAX) {
            tensor_workspace_release(v, (unsigned char *)projection);
            vmerror(v, "low-rank result is outside the Float64 range");
        }
        A(out.coefficients).data[i] = (double)sum;
    }
    tensor_workspace_release(v, (unsigned char *)projection);
    return objval(publish(v, out));
}

static long double weighted_core(Obj *core, Obj *vector, int p, int q) {
    int n = A(core).shape[1], right = A(core).shape[2];
    long double sum = 0;
    for (int i = 0; i < n; i++)
        sum += (long double)A(core).data[((size_t)p * n + i) * right + q] * A(vector).data[i];
    return sum;
}

static Val contract_train(VM *v, LowRankTensor out, const LowRankTensor *t, int axis, Obj *vector) {
    Obj *core = t->factor[axis];
    if (!out.rank)
        return number(checked_result(v, weighted_core(core, vector, 0, 0)));
    int left = A(core).shape[0], right = A(core).shape[2];
    bool next = axis < out.rank;
    int replace = next ? axis : axis - 1;
    Obj *neighbor = out.factor[replace];
    int dims[] = {next ? left : A(neighbor).shape[0], A(neighbor).shape[1],
                  next ? A(neighbor).shape[2] : right};
    Obj *mixed = tensor_new(v, 3, dims);
    if (!A(mixed).n) {
        out.factor[replace] = mixed;
        return objval(publish(v, out));
    }
    if ((size_t)left > ARDA_MAX_ELEMENTS / (size_t)right)
        vmerror(v, "tensor train contraction exceeds the workspace limit");
    long double *projected = scratch(v, (size_t)left * right);
    for (int p = 0; p < left; p++)
        for (int q = 0; q < right; q++)
            projected[(size_t)p * right + q] = weighted_core(core, vector, p, q);
    for (int p = 0; p < dims[0]; p++)
        for (int i = 0; i < dims[1]; i++)
            for (int q = 0; q < dims[2]; q++) {
                long double value = 0;
                if (next) {
                    for (int r = 0; r < right; r++)
                        value += projected[(size_t)p * right + r] *
                                 A(neighbor).data[((size_t)r * dims[1] + i) * dims[2] + q];
                } else
                    for (int r = 0; r < left; r++)
                        value +=
                            (long double)A(neighbor).data[((size_t)p * dims[1] + i) * left + r] *
                            projected[(size_t)r * right + q];
                if (!isfinite(value) || fabsl(value) > DBL_MAX) {
                    tensor_workspace_release(v, (unsigned char *)projected);
                    vmerror(v, "low-rank result is outside the Float64 range");
                }
                A(mixed).data[((size_t)p * dims[1] + i) * dims[2] + q] = (double)value;
            }
    tensor_workspace_release(v, (unsigned char *)projected);
    out.factor[replace] = mixed;
    return objval(publish(v, out));
}

static Val contract_axis(VM *v, const LowRankTensor *t, Val axis_value, Val vector_value) {
    int axis = indexnum(v, axis_value, t->rank);
    TypedTensor view = component(v, vector_value, 1);
    if (view.shape[0] != t->shape[axis])
        vmerror(v, "low-rank contraction vector length does not match the axis");
    Obj *vector = snapshot(v, &view);
    LowRankTensor out = {.kind = t->kind, .rank = t->rank - 1};
    for (int d = 0, j = 0; d < t->rank; d++)
        if (d != axis) {
            out.shape[j] = t->shape[d];
            out.factor[j++] = t->factor[d];
        }
    switch (t->kind) {
    case LOWRANK_CP:
        return contract_cp(v, out, t, axis, vector);
    case LOWRANK_TUCKER:
        return contract_tucker(v, out, t, axis, vector);
    case LOWRANK_TT:
        return contract_train(v, out, t, axis, vector);
    }
    return nil();
}

#ifdef ARDA_LAPACK
static Obj *unfold(VM *v, const Obj *source, int rows, int cols) {
    Obj *out = array_new(v, 2, rows, cols);
    memcpy(A(out).data, A(source).data, A(out).n * sizeof(double));
    return out;
}

static Obj *core_new(VM *v, int a, int n, int b) {
    int dims[] = {a, n, b};
    return tensor_new(v, 3, dims);
}

static long double frobenius(const Obj *x) {
    long double norm = 0;
    for (size_t i = 0; i < A(x).n; i++)
        norm = hypotl(norm, A(x).data[i]);
    return norm;
}

static Val with_error(VM *v, LowRankTensor out, long double tail, long double norm) {
    double relative = checked_result(v, norm > 0 ? tail / norm : 0);
    Obj *result = list(v);
    append(v, result, objval(publish(v, out)));
    append(v, result, number(relative));
    return objval(result);
}

static int retained(const Obj *s, int cap, long double cutoff, long double *discarded) {
    int rank = A(s).rows;
    const double *values = A(s).data;
    long double tail = 0;
    while (rank > 1) {
        long double next = hypotl(tail, values[rank - 1]);
        if (next > cutoff && rank <= cap)
            break;
        tail = next;
        rank--;
    }
    *discarded = hypotl(*discarded, tail);
    return rank;
}

static Obj *left_core(VM *v, const Obj *u, int left, int n, int rank) {
    Obj *g = core_new(v, left, n, rank);
    for (int i = 0; i < A(u).rows; i++)
        memcpy(A(g).data + (size_t)i * rank, A(u).data + (size_t)i * A(u).cols,
               (size_t)rank * sizeof(double));
    return g;
}

static Val tt_svd(VM *v, Val input, double tol, int cap) {
    TypedTensor format = component(v, input, 0);
    LowRankTensor out = {.rank = format.rank, .kind = LOWRANK_TT};
    memcpy(out.shape, format.shape, sizeof out.shape);
    if (!format.n) {
        for (int d = 0; d < out.rank; d++)
            out.factor[d] = core_new(v, 1, out.shape[d], 1);
        return with_error(v, out, 0, 0);
    }
    Obj *rest = snapshot(v, &format);
    long double norm = frobenius(rest), tail = 0;
    long double cutoff = out.rank > 1 ? (long double)tol * norm / sqrtl(out.rank - 1) : 0;
    int left = 1, suffix = (int)A(rest).n;
    for (int d = 0; d + 1 < out.rank; d++) {
        int n = out.shape[d];
        suffix /= n;
        Obj *matrix = unfold(v, rest, left * n, suffix);
        Obj *u, *s, *vt;
        linalg_reduced_svd(v, matrix, &u, &s, &vt);
        int rank = retained(s, cap, cutoff, &tail);
        out.factor[d] = left_core(v, u, left, n, rank);
        rest = array_new(v, 2, rank, suffix);
        for (int i = 0; i < rank; i++)
            for (int j = 0; j < suffix; j++)
                A(rest).data[(size_t)i * suffix + j] = checked_result(
                    v, (long double)A(s).data[i] * A(vt).data[(size_t)i * suffix + j]);
        left = rank;
    }
    Obj *last = core_new(v, left, out.shape[out.rank - 1], 1);
    memcpy(A(last).data, A(rest).data, A(last).n * sizeof(double));
    out.factor[out.rank - 1] = last;
    return with_error(v, out, tail, norm);
}

static Val tt_round(VM *v, LowRankTensor out, double tol, int cap) {
    if (out.kind != LOWRANK_TT)
        vmerror(v, "lowrank.round requires a tensor train");
    for (int d = 0; d < out.rank; d++)
        if (!out.shape[d]) {
            for (int j = 0; j < out.rank; j++)
                out.factor[j] = core_new(v, 1, out.shape[j], 1);
            return with_error(v, out, 0, 0);
        }
    // Sweep right to left first so the discarded singular values in the
    // later truncation sweep measure error in the ambient Frobenius norm.
    for (int d = out.rank - 1; d > 0; d--) {
        Obj *g = out.factor[d], *prev = out.factor[d - 1];
        int l = A(g).shape[0], n = A(g).shape[1], r = A(g).shape[2];
        Obj *matrix = unfold(v, g, l, n * r);
        Obj *u, *s, *vt;
        linalg_reduced_svd(v, matrix, &u, &s, &vt);
        int rank = A(s).rows;
        Obj *right = core_new(v, rank, n, r);
        memcpy(A(right).data, A(vt).data, A(right).n * sizeof(double));
        Obj *left = core_new(v, A(prev).shape[0], A(prev).shape[1], rank);
        for (size_t i = 0; i < A(prev).n / (size_t)l; i++)
            for (int j = 0; j < rank; j++) {
                long double sum = 0;
                for (int k = 0; k < l; k++)
                    sum += (long double)A(prev).data[i * l + k] * A(u).data[(size_t)k * rank + j];
                A(left).data[i * rank + j] = checked_result(v, sum * A(s).data[j]);
            }
        out.factor[d] = right;
        out.factor[d - 1] = left;
    }
    long double norm = frobenius(out.factor[0]), tail = 0;
    long double cutoff = out.rank > 1 ? (long double)tol * norm / sqrtl(out.rank - 1) : 0;
    for (int d = 0; d + 1 < out.rank; d++) {
        Obj *g = out.factor[d], *next = out.factor[d + 1];
        int l = A(g).shape[0], n = A(g).shape[1], r = A(g).shape[2];
        Obj *matrix = unfold(v, g, l * n, r);
        Obj *u, *s, *vt;
        linalg_reduced_svd(v, matrix, &u, &s, &vt);
        int rank = retained(s, cap, cutoff, &tail);
        out.factor[d] = left_core(v, u, l, n, rank);
        Obj *right = core_new(v, rank, A(next).shape[1], A(next).shape[2]);
        size_t cols = A(next).n / (size_t)r;
        for (int i = 0; i < rank; i++)
            for (size_t j = 0; j < cols; j++) {
                long double sum = 0;
                for (int k = 0; k < r; k++)
                    sum += (long double)A(vt).data[(size_t)i * r + k] *
                           A(next).data[(size_t)k * cols + j];
                A(right).data[(size_t)i * cols + j] = checked_result(v, sum * A(s).data[i]);
            }
        out.factor[d + 1] = right;
    }
    return with_error(v, out, tail, norm);
}
#endif

enum { CP, TUCKER, TT, SHAPE, AT, NORM, DOT, MATERIALIZE, CONTRACT, BACKEND, TT_SVD, ROUND, RANKS };
static const char *const names[] = {"cp",     "tucker", "tt",          "shape",    "at",
                                    "norm",   "dot",    "materialize", "contract", "backend",
                                    "tt_svd", "round",  "ranks"};
static const int arities[] = {1, 2, 1, 1, 2, 1, 2, 1, 3, 0, 2, 2, 1};
_Static_assert(sizeof names / sizeof *names == sizeof arities / sizeof *arities, "lowrank arities");

Obj *lowrank_module(VM *v) {
    return named_native_module(v, "lowrank", names, sizeof names / sizeof *names, LOWRANK_BASE);
}

Val lowrank_call(VM *v, int id, int argc, Val *args) {
    int op = id - LOWRANK_BASE;
    if (op < 0 || (size_t)op >= sizeof names / sizeof *names ||
        (argc != arities[op] && !(op == CP && argc == 2) &&
         !((op == TT_SVD || op == ROUND) && argc == 3)))
        vmerror(v, "invalid lowrank API arity");
    switch (op) {
    case CP:
        return objval(construct(v, LOWRANK_CP, args[0], argc == 2 ? args[1] : nil()));
    case TUCKER:
        return objval(construct(v, LOWRANK_TUCKER, args[1], args[0]));
    case TT:
        return objval(construct(v, LOWRANK_TT, args[0], nil()));
    case BACKEND:
#ifdef ARDA_LAPACK
        return objval(string(v, "lapack"));
#else
        return objval(string(v, "portable-c"));
#endif
    case TT_SVD:
    case ROUND:
#ifdef ARDA_LAPACK
    {
        double tol = asnum(v, args[1]);
        if (!isfinite(tol) || tol < 0 || tol > 1)
            vmerror(v, "lowrank relative tolerance must be in [0,1]");
        int cap = INT_MAX;
        if (argc == 3) {
            if (args[2].t != V_INT || args[2].i < 1 || args[2].i > INT_MAX)
                vmerror(v, "lowrank max_rank must be a positive integer");
            cap = (int)args[2].i;
        }
        return op == TT_SVD ? tt_svd(v, args[0], tol, cap)
                            : tt_round(v, *as_lowrank(v, args[0]), tol, cap);
    }
#else
        vmerror(v, "LAPACK unavailable: rebuild with LAPACK=1");
#endif
    default:
        break;
    }
    LowRankTensor *t = as_lowrank(v, args[0]);
    switch (op) {
    case SHAPE:
    case RANKS: {
        Obj *out = list(v);
        if (op == RANKS && t->kind == LOWRANK_TT)
            append(v, out, integer(1));
        for (int d = 0; d < t->rank; d++) {
            int n =
                op == SHAPE ? t->shape[d] : A(t->factor[d]).shape[t->kind == LOWRANK_TT ? 2 : 1];
            append(v, out, integer(n));
        }
        return objval(out);
    }
    case AT:
        return element(v, t, args[1]);
    case NORM: {
        long double squared = inner_product(v, t, t);
        if (!isfinite(squared))
            vmerror(v, "low-rank norm accumulation overflow");
        return number(checked_result(v, sqrtl(fmaxl(0, squared))));
    }
    case DOT:
        return number(checked_result(v, inner_product(v, t, as_lowrank(v, args[1]))));
    case MATERIALIZE:
        return objval(materialize(v, t));
    case CONTRACT:
        return contract_axis(v, t, args[1], args[2]);
    }
    vmerror(v, "unknown lowrank operation");
}
