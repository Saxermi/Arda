#include "solver.h"
#include <float.h>

#define A(o) ((o)->as.array)

static bool factor_float(const Obj *a, int n, float *lu, int *pivots) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double value = linop_dense_at(a, i, j);
            if (fabs(value) > FLT_MAX)
                return false;
            lu[(size_t)i * n + j] = (float)value;
        }
    for (int i = 0; i < n; i++) {
        int row = i;
        for (int j = i + 1; j < n; j++)
            if (fabsf(lu[(size_t)j * n + i]) > fabsf(lu[(size_t)row * n + i]))
                row = j;
        pivots[i] = row;
        float *pivot = lu + (size_t)i * n, *source = lu + (size_t)row * n;
        if (row != i)
            for (int j = 0; j < n; j++) {
                float x = pivot[j];
                pivot[j] = source[j];
                source[j] = x;
            }
        if (pivot[i] == 0 || !isfinite(pivot[i]))
            return false;
        for (int j = i + 1; j < n; j++) {
            float *entry = lu + (size_t)j * n;
            entry[i] /= pivot[i];
            for (int k = i + 1; k < n; k++) {
                entry[k] -= entry[i] * pivot[k];
                if (!isfinite(entry[k]))
                    return false;
            }
        }
    }
    return true;
}

static bool correction(int n, const float *lu, const int *pivots, const double *residual,
                       long double scale, float *work, double *step) {
    for (int i = 0; i < n; i++)
        work[i] = (float)(residual[i] / scale);
    for (int i = 0; i < n; i++) {
        int j = pivots[i];
        float x = work[j];
        work[j] = work[i];
        work[i] = x;
    }
    for (int i = 0; i < n; i++) {
        const float *row = lu + (size_t)i * n;
        for (int j = 0; j < i; j++)
            work[i] -= row[j] * work[j];
    }
    for (int i = n; i-- > 0;) {
        const float *row = lu + (size_t)i * n;
        for (int j = i + 1; j < n; j++)
            work[i] -= row[j] * work[j];
        work[i] /= row[i];
        long double value = work[i] * scale;
        if (!isfinite(value) || fabsl(value) > DBL_MAX)
            return false;
        step[i] = (double)value;
    }
    return true;
}

static long double accurate_residual(const Obj *a, SolverProblem *p) {
    size_t n = p->op.cols;
    for (size_t i = 0; i < n; i++) {
        long double value = p->b[i];
        for (size_t j = 0; j < n; j++)
            value -= (long double)linop_dense_at(a, i, j) * p->x[j];
        p->residual[i] = fabsl(value) <= DBL_MAX ? (double)value : INFINITY;
    }
    long double r = linear_norm(n, p->residual, 1);
    return p->rhs_norm ? r / p->rhs_norm : r;
}

Val solver_refine(VM *v, Val input, Val rhs, Val settings) {
    if (!isobj(input, O_ARRAY) || A(input.o).rank != 2 || A(input.o).rows != A(input.o).cols ||
        !A(input.o).rows)
        vmerror_kind(v, "shape", "mixed refinement requires a nonempty square numc matrix");
    Obj *a = input.o;
    SolverOptions opts = solver_options(v, settings);
    if (opts.preconditioner != PC_NONE || solver_has_options(settings, "restart") ||
        solver_has_options(settings, "basis") || solver_has_options(settings, "assume") ||
        solver_has_options(settings, "condition"))
        vmerror_kind(v, "domain", "unsupported iterative refinement option");
    if (!solver_has_options(settings, "maxiter"))
        opts.max_iterations = 10;
    int n = A(a).rows;
    size_t count = A(a).n;
    size_t bytes =
        3 * (size_t)n * sizeof(double) + (count + n) * sizeof(float) + (size_t)n * sizeof(int);
    Obj *work = solver_workspace(v, bytes, 0, (size_t)n * sizeof(double), &opts);
    double *b = (double *)work->as.buffer.data, *r = b + n, *delta = r + n;
    float *lu = (float *)(delta + n), *scratch = lu + count;
    int *pivots = (int *)(scratch + n);
    solver_read_vector(v, rhs, (size_t)n, b);
    LinopContext ctx;
    LinearOperator op = linop_open(v, a, &ctx);
    long double bn = linear_norm(n, b, 1);
    if (!isfinite(bn))
        vmerror_kind(v, "domain", "right-hand-side norm is not representable");
    Obj *x = array_new(v, 1, n, 1);
    SolverProblem p = {.vm = v,
                       .op = op,
                       .options = opts,
                       .b = b,
                       .residual = r,
                       .x = A(x).data,
                       .rhs_norm = bn,
                       .trace = opts.trace_limit ? list(v) : NULL};
    long double relative = accurate_residual(a, &p);
    const char *fallback_reason = NULL;
    size_t peak_work = bytes;
    bool singular_fallback = false;
    if (bn) {
        if (FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128 || !factor_float(a, n, lu, pivots))
            fallback_reason = "float32_factorization";
        while (!fallback_reason && relative > opts.tolerance &&
               p.iterations < opts.max_iterations) {
            long double previous = relative, scale = 0;
            for (int i = 0; i < n; i++)
                scale = fmaxl(scale, fabs(r[i]));
            if (!isfinite(scale) || !correction(n, lu, pivots, r, scale, scratch, delta) ||
                !solver_update(n, p.x, 1, delta)) {
                fallback_reason = "float32_correction";
                break;
            }
            p.iterations++;
            relative = accurate_residual(a, &p);
            solver_trace(&p, relative);
            if (!isfinite(relative) || (relative > opts.tolerance && relative >= 0.9L * previous))
                fallback_reason = "stagnation";
        }
        if (!fallback_reason && relative > opts.tolerance)
            fallback_reason = "correction_limit";
    }
    if (fallback_reason) {
        size_t fallback_bytes =
            count * sizeof(double) + 2 * (size_t)n * sizeof(double) + (size_t)n * sizeof(int);
        if (bytes > opts.workspace_limit || fallback_bytes > opts.workspace_limit - bytes)
            vmerror_kind(v, "memory_budget", "Float64 fallback exceeds solver workspace_bytes");
        if (fallback_bytes > tensor_memory_available(v))
            vmerror_kind(v, "memory_budget", "Float64 fallback exceeds the tensor budget");
        Obj *checkpoint = v->objects;
        Obj *packed_rhs = array_new(v, 1, n, 1);
        memcpy(A(packed_rhs).data, b, (size_t)n * sizeof(double));
        Obj *fallback = dense_lu_try_solve(v, a, packed_rhs);
        if (fallback)
            memcpy(p.x, A(fallback).data, (size_t)n * sizeof(double));
        else {
            singular_fallback = true;
            fallback_reason = "float64_singular";
        }
        solver_discard_since(v, checkpoint);
        peak_work += fallback_bytes;
    }
    relative = accurate_residual(a, &p);
    bool breakdown = singular_fallback || !isfinite(relative);
    p.stop = breakdown                    ? STOP_BREAKDOWN
             : relative <= opts.tolerance ? STOP_CONVERGED
                                          : STOP_STAGNATION;
    Val result = solver_result(&p, x, "refine", peak_work, relative, 0);
    Obj *map = result.o;
    sparse_result_field(v, map, "fallback", boolean(fallback_reason != NULL));
    sparse_result_field(v, map, "fallback_reason",
                        fallback_reason ? objval(string(v, fallback_reason)) : nil());
    sparse_result_field(v, map, "factor_dtype",
                        objval(string(v, fallback_reason ? "float64" : "float32")));
    linop_close(v, &ctx);
    solver_workspace_done(v, work);
    return result;
}
