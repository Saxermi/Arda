#include "sparse_internal.h"

#define S(o) ((o)->as.sparse)
#define A(o) ((o)->as.array)

static void csr_apply(const void *context, const double *x, size_t xs, double *y, size_t ys) {
    const Sparse *s = context;
    for (size_t row = 0; row < s->rows; row++) {
        long double sum = 0;
        for (size_t j = s->offset[row]; j < s->offset[row + 1]; j++)
            sum += (long double) s->values[j] * x[s->col[j] * xs];
        y[row * ys] = (double) sum;
    }
}

static void csr_apply_transpose(const void *context, const double *x, size_t xs, double *y, size_t ys) {
    const Sparse *s = context;
    for (size_t col = 0; col < s->cols; col++)
        y[col * ys] = 0;
    for (size_t row = 0; row < s->rows; row++)
        for (size_t j = s->offset[row]; j < s->offset[row + 1]; j++)
            y[s->col[j] * ys] += s->values[j] * x[row * xs];
}

LinearOperator sparse_operator(const Sparse *s) {
    return (LinearOperator){.rows = s->rows, .cols = s->cols, .context = s,
                            .apply = csr_apply, .apply_transpose = csr_apply_transpose};
}

static void finite_elements(VM *v, Obj *o) {
    int packed = tensor_contiguous(o);
    for (size_t i = 0; i < A(o).n; i++)
        if (!isfinite(A(o).data[packed ? i : tensor_offset(o, i)]))
            vmerror(v, "nonfinite sparse product input or output");
}

Obj *sparse_product(VM *v, Obj *matrix, Val input, int rank) {
    Sparse *s = &S(matrix);
    if (!isobj(input, O_ARRAY) || A(input.o).rank != rank || s->cols != (size_t) A(input.o).shape[0])
        vmerror(v, "sparse product shape mismatch");
    Obj *x = input.o;
    size_t width = rank == 2 ? (size_t) A(x).shape[1] : 1;
    if (s->rows > ARDA_MAX_ELEMENTS / (width ? width : 1))
        vmerror(v, "array size limit: 16000000 elements");
    finite_elements(v, x);
    Obj *y = array_new(v, rank, (int) s->rows, (int) width);
    if (!A(y).n)
        return y;
    Obj *csr = sparse_as_csr(v, matrix);
    LinearOperator op = sparse_operator(&S(csr));
    for (size_t c = 0; c < width; c++) {
        const double *column = A(x).data;
        if (rank == 2 && op.cols)
            column += c * A(x).stride[1];
        op.apply(op.context, column, A(x).stride[0], A(y).data + c, width);
    }
    sparse_drop_temporary(v, csr, matrix);
    finite_elements(v, y);
    return y;
}

Obj *sparse_transpose(VM *v, Obj *matrix) {
    Obj *csr = sparse_as_csr(v, matrix);
    const Sparse *s = &S(csr);
    Obj *out = sparse_matrix(v, s->cols, s->rows, 1);
    Sparse *t = &S(out);
    sparse_entries(v, out, s->nnz);
    for (size_t j = 0; j < s->nnz; j++)
        t->offset[(size_t) s->col[j] + 1]++;
    for (size_t row = 0; row < t->rows; row++)
        t->offset[row + 1] += t->offset[row];
    for (size_t row = 0; row < s->rows; row++) {
        for (size_t j = s->offset[row]; j < s->offset[row + 1]; j++) {
            size_t pos = t->offset[s->col[j]]++;
            t->col[pos] = (uint32_t) row;
            t->values[pos] = s->values[j];
        }
    }
    for (size_t row = t->rows; row > 0; row--)
        t->offset[row] = t->offset[row - 1];
    t->offset[0] = 0;
    sparse_drop_temporary(v, csr, matrix);
    return out;
}

static size_t merge_rows(VM *v, const Sparse *a, const Sparse *b, Sparse *out) {
    size_t used = 0;
    for (size_t row = 0; row < a->rows; row++) {
        if (out)
            out->offset[row] = used;
        size_t i = a->offset[row], j = b->offset[row];
        size_t end_a = a->offset[row + 1], end_b = b->offset[row + 1];
        while (i < end_a || j < end_b) {
            uint32_t col;
            double value;
            if (j == end_b || (i < end_a && a->col[i] < b->col[j])) {
                col = a->col[i];
                value = a->values[i++];
            } else if (i == end_a || b->col[j] < a->col[i]) {
                col = b->col[j];
                value = b->values[j++];
            } else {
                col = a->col[i];
                value = a->values[i++] + b->values[j++];
                if (!isfinite(value))
                    vmerror(v, "nonfinite sparse addition");
            }
            if (value != 0.0) {
                if (used == SIZE_MAX)
                    vmerror(v, "sparse addition size overflow");
                if (out) {
                    out->col[used] = col;
                    out->values[used] = value;
                }
                used++;
            }
        }
    }
    if (out)
        out->offset[a->rows] = used;
    return used;
}

Obj *sparse_add(VM *v, Obj *left, Obj *right) {
    if (S(left).rows != S(right).rows || S(left).cols != S(right).cols)
        vmerror(v, "sparse addition shape mismatch");
    Obj *a = sparse_as_csr(v, left);
    Obj *b = left == right ? a : sparse_as_csr(v, right);
    size_t used = merge_rows(v, &S(a), &S(b), NULL);
    Obj *out = sparse_matrix(v, S(a).rows, S(a).cols, 1);
    sparse_entries(v, out, used);
    merge_rows(v, &S(a), &S(b), &S(out));
    sparse_drop_temporary(v, a, left);
    if (b != a)
        sparse_drop_temporary(v, b, right);
    return out;
}

Obj *sparse_scale(VM *v, Obj *matrix, double factor) {
    if (factor == 0.0)
        return sparse_matrix(v, S(matrix).rows, S(matrix).cols, 1);
    Obj *csr = sparse_as_csr(v, matrix);
    if (factor == 1.0)
        return csr;
    const Sparse *s = &S(csr);
    size_t used = 0;
    for (size_t i = 0; i < s->nnz; i++) {
        double value = s->values[i] * factor;
        if (!isfinite(value))
            vmerror(v, "nonfinite sparse scaling");
        used += value != 0.0;
    }
    Obj *out = sparse_matrix(v, s->rows, s->cols, 1);
    sparse_entries(v, out, used);
    used = 0;
    for (size_t row = 0; row < s->rows; row++) {
        S(out).offset[row] = used;
        for (size_t i = s->offset[row]; i < s->offset[row + 1]; i++) {
            double value = s->values[i] * factor;
            if (value != 0.0) {
                S(out).col[used] = s->col[i];
                S(out).values[used++] = value;
            }
        }
    }
    S(out).offset[s->rows] = used;
    sparse_drop_temporary(v, csr, matrix);
    return out;
}

double sparse_csr_at(const Sparse *s, size_t row, uint32_t col) {
    size_t first = s->offset[row], last = s->offset[row + 1], end = last;
    while (first < last) {
        size_t mid = first + (last - first) / 2;
        if (s->col[mid] < col)
            first = mid + 1;
        else
            last = mid;
    }
    return first < end && s->col[first] == col ? s->values[first] : 0.0;
}

Obj *sparse_diagonal(VM *v, Obj *matrix) {
    size_t length = S(matrix).rows < S(matrix).cols ? S(matrix).rows : S(matrix).cols;
    if (length > ARDA_MAX_ELEMENTS)
        vmerror(v, "array size limit: 16000000 elements");
    Obj *csr = sparse_as_csr(v, matrix);
    Obj *out = array_new(v, 1, (int) length, 1);
    for (size_t i = 0; i < length; i++)
        A(out).data[i] = sparse_csr_at(&S(csr), i, (uint32_t) i);
    sparse_drop_temporary(v, csr, matrix);
    return out;
}

int sparse_symmetric(const Sparse *s) {
    if (s->rows != s->cols)
        return 0;
    for (uint32_t row = 0; row < s->rows; row++)
        for (size_t j = s->offset[row]; j < s->offset[(size_t) row + 1]; j++)
            if (s->col[j] != row && s->values[j] != sparse_csr_at(s, s->col[j], row))
                return 0;
    return 1;
}

void sparse_result_field(VM *v, Obj *result, const char *key, Val value) {
    tableset(v, result, &result->as.map.tab, key, value);
}

Val sparse_estimate(VM *v, Obj *matrix) {
    const Sparse *s = &S(matrix);
    size_t bytes = s->csr ? sparse_operator(s).workspace_bytes : sparse_csr_estimate(v, s);
    if (bytes > INT64_MAX)
        vmerror(v, "sparse workspace estimate out of range");
    Obj *result = newobj(v, O_MAP);
    sparse_result_field(v, result, "rows", integer(s->rows));
    sparse_result_field(v, result, "cols", integer(s->cols));
    sparse_result_field(v, result, "nnz", integer((int64_t) s->nnz));
    sparse_result_field(v, result, "matvec_workspace_bytes", integer((int64_t) bytes));
    return objval(result);
}
