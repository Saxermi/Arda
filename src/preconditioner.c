#include "solver.h"
#include "poisson.h"

#define A(o) ((o)->as.array)

size_t preconditioner_bytes(VM *v, Obj *input, size_t n, PreconditionerKind kind) {
    if (kind == PC_NONE)
        return 0;
    if (kind == PC_JACOBI)
        return n * sizeof(double);
    if (kind == PC_BLOCK_JACOBI)
        return 2 * n * sizeof(double);
    if (input->type != O_SPARSE)
        vmerror_kind(v, "type", "incomplete factorization requires sparse coefficients");
    size_t nnz = input->as.sparse.nnz;
    if (nnz > (SIZE_MAX - n * sizeof(size_t)) / sizeof(double))
        vmerror_kind(v, "memory_budget", "incomplete factors are not addressable");
    return nnz * sizeof(double) + n * sizeof(size_t);
}

static double coefficient(VM *v, Obj *op, const Sparse *s, size_t i, size_t j) {
    if (s)
        return sparse_csr_at(s, i, (uint32_t)j);
    if (op->type == O_ARRAY || op->type == O_TYPED_TENSOR)
        return linop_dense_at(op, i, j);
    if (op->as.linop.kind == LINOP_DIAGONAL)
        return i == j ? A(op->as.linop.first).data[i] : 0;
    if (op->as.linop.kind == LINOP_POISSON) {
        if (i == j)
            return poisson_diagonal(op);
        if ((i < j ? j - i : i - j) == 1)
            return -poisson_neighbor_weight(op, i < j ? i : j);
        return 0;
    }
    vmerror_kind(v, "domain", "preconditioner needs explicit coefficients or a known diagonal");
}

static void bad_pivot(VM *v) {
    vmerror_kind(v, "solver_breakdown",
                 "preconditioner has a zero, nonfinite or nonpositive pivot");
}

static void setup_diagonal(VM *v, Obj *op, Preconditioner *pc) {
    double *f = pc->factors;
    if (pc->kind == PC_JACOBI) {
        for (size_t i = 0; i < pc->n; i++) {
            double a = coefficient(v, op, pc->csr, i, i);
            if (!(a > 0) || !isfinite(1 / a))
                bad_pivot(v);
            f[i] = 1 / a;
        }
        return;
    }
    size_t i = 0;
    for (; i + 1 < pc->n; i += 2) {
        long double a = coefficient(v, op, pc->csr, i, i);
        long double b = coefficient(v, op, pc->csr, i, i + 1);
        long double c = coefficient(v, op, pc->csr, i + 1, i);
        long double d = coefficient(v, op, pc->csr, i + 1, i + 1);
        long double det = a * d - b * c;
        if (!(a > 0 && d > 0 && det > 0) || b != c || !isfinite(det))
            bad_pivot(v);
        f[2 * i] = (double)(d / det);
        f[2 * i + 1] = (double)(-b / det);
        f[2 * i + 2] = (double)(-c / det);
        f[2 * i + 3] = (double)(a / det);
        for (size_t j = 2 * i; j < 2 * i + 4; j++)
            if (!isfinite(f[j]))
                bad_pivot(v);
    }
    if (i < pc->n) {
        double last = coefficient(v, op, pc->csr, i, i);
        if (!(last > 0) || !isfinite(1 / last))
            bad_pivot(v);
        f[2 * i] = 1 / last;
    }
}

static void setup_incomplete(VM *v, Preconditioner *pc) {
    const Sparse *s = pc->csr;
    double *f = pc->factors;
    size_t *diag = pc->diagonal;
    memcpy(f, s->values, s->nnz * sizeof(double));
    for (size_t i = 0; i < pc->n; i++) {
        size_t begin = s->offset[i], end = s->offset[i + 1], middle = begin;
        while (middle < end && s->col[middle] < i)
            middle++;
        if (middle == end || s->col[middle] != i)
            bad_pivot(v);
        diag[i] = middle;
        for (size_t slot = begin; slot < middle; slot++) {
            size_t row = s->col[slot];
            f[slot] /= f[diag[row]];
            size_t col = slot + 1;
            for (size_t k = diag[row] + 1; k < s->offset[row + 1] && col < end; k++) {
                while (col < end && s->col[col] < s->col[k])
                    col++;
                if (col < end && s->col[col] == s->col[k])
                    f[col] -= f[slot] * f[k];
            }
        }
        for (size_t k = begin; k < end; k++)
            if (!isfinite(f[k]))
                bad_pivot(v);
        if (f[middle] == 0 || (pc->kind == PC_IC0 && f[middle] < 0))
            bad_pivot(v);
    }
}

Preconditioner preconditioner_setup(VM *v, Obj *input, const LinopContext *context, size_t n,
                                    PreconditionerKind kind, unsigned char *memory) {
    Preconditioner pc = {.kind = kind,
                         .csr = context->csr ? &context->csr->as.sparse : NULL,
                         .n = n,
                         .factors = (double *)memory};
    if (kind == PC_JACOBI || kind == PC_BLOCK_JACOBI)
        setup_diagonal(v, input, &pc);
    if (kind == PC_ILU0 || kind == PC_IC0) {
        pc.diagonal = (size_t *)(pc.factors + pc.csr->nnz);
        setup_incomplete(v, &pc);
    }
    return pc;
}

void preconditioner_apply(const Preconditioner *pc, const double *x, double *y) {
    const double *f = pc->factors;
    size_t n = pc->n;
    if (pc->kind == PC_NONE) {
        memcpy(y, x, n * sizeof(double));
        return;
    }
    if (pc->kind == PC_JACOBI) {
        for (size_t i = 0; i < n; i++)
            y[i] = x[i] * f[i];
        return;
    }
    if (pc->kind == PC_BLOCK_JACOBI) {
        size_t i = 0;
        for (; i + 1 < n; i += 2) {
            y[i] = f[2 * i] * x[i] + f[2 * i + 1] * x[i + 1];
            y[i + 1] = f[2 * i + 2] * x[i] + f[2 * i + 3] * x[i + 1];
        }
        if (i < n)
            y[i] = x[i] * f[2 * i];
        return;
    }
    const Sparse *s = pc->csr;
    const size_t *diag = pc->diagonal;
    for (size_t i = 0; i < n; i++) {
        long double value = x[i];
        for (size_t j = s->offset[i]; j < diag[i]; j++)
            value -= (long double)f[j] * y[s->col[j]];
        y[i] = (double)value;
    }
    if (pc->kind == PC_IC0) {
        for (size_t i = 0; i < n; i++)
            y[i] /= f[diag[i]];
        for (size_t i = n; i-- > 0;)
            for (size_t j = s->offset[i]; j < diag[i]; j++)
                y[s->col[j]] -= f[j] * y[i];
    } else {
        for (size_t i = n; i-- > 0;) {
            long double value = y[i];
            for (size_t j = diag[i] + 1; j < s->offset[i + 1]; j++)
                value -= (long double)f[j] * y[s->col[j]];
            y[i] = (double)(value / f[diag[i]]);
        }
    }
}
