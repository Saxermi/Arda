#include "solver.h"
#include <float.h>

#define A(o) ((o)->as.array)

#ifdef ARDA_LAPACK
static int build_basis(LinearOperator op, double *basis, double *work, double *h, int limit,
                       bool *breakdown) {
    size_t n = op.cols;
    for (int j = 0; j < limit; j++) {
        double *current = basis + (size_t)j * n, *column = h + (size_t)j * limit;
        op.apply(op.context, current, 1, work, 1);
        long double initial = linear_norm(n, work, 1);
        long double residual = krylov_orthogonalize(n, basis, j + 1, work, column);
        if (!isfinite(initial) || !isfinite(residual) || residual > DBL_MAX) {
            *breakdown = true;
            return j;
        }
        for (int i = 0; i <= j; i++)
            if (!isfinite(column[i])) {
                *breakdown = true;
                return j;
            }
        if (j + 1 == limit || residual <= 8 * DBL_EPSILON * initial)
            return j + 1;
        column[j + 1] = (double)residual;
        for (size_t i = 0; i < n; i++)
            current[n + i] = (double)(work[i] / residual);
    }
    return limit;
}

static double lift_pair(LinearOperator op, const double *basis, int steps, int col,
                        Obj *projected_real, Obj *projected_imag, double re, double im,
                        double *work, double *real_out, double *imag_out, int width) {
    size_t n = op.cols;
    double *x = work, *y = x + n, *ax = y + n, *ay = ax + n;
    memset(x, 0, 2 * n * sizeof(double));
    for (int j = 0; j < steps; j++) {
        const double *v = basis + (size_t)j * n;
        size_t at = (size_t)j * steps + col;
        double r = A(projected_real).data[at], i = A(projected_imag).data[at];
        for (size_t k = 0; k < n; k++) {
            x[k] += r * v[k];
            y[k] += i * v[k];
        }
    }
    long double norm = hypotl(linear_norm(n, x, 1), linear_norm(n, y, 1));
    if (!(norm > 0) || !isfinite(norm))
        return INFINITY;
    for (size_t i = 0; i < n; i++) {
        real_out[i * width] = x[i] = (double)(x[i] / norm);
        imag_out[i * width] = y[i] = (double)(y[i] / norm);
    }
    return (double)solver_eigen_residual(op, re, im, x, y, 1, ax, ay);
}
#endif

Val solver_arnoldi(VM *v, Val input, Val seed, Val wanted, Val settings) {
#ifndef ARDA_LAPACK
    (void)input;
    (void)seed;
    (void)wanted;
    (void)settings;
    vmerror_kind(v, "backend", "Arnoldi projected eigenproblem requires LAPACK=1");
#else
    Obj *object = solver_operator(v, input);
    SolverOptions opts = solver_options(v, settings);
    size_t m, n;
    linop_shape(v, object, &m, &n);
    if (!n || m != n || n > ARDA_MAX_ELEMENTS)
        vmerror_kind(v, "shape", "Arnoldi needs a nonempty square operator");
    int k = opts.basis < (int)n ? opts.basis : (int)n;
    if (wanted.t != V_INT || wanted.i < 1 || wanted.i > k ||
        (uint64_t)wanted.i > ARDA_MAX_ELEMENTS / n)
        vmerror_kind(v, "domain", "Arnoldi eigenpair count exceeds basis or output limit");
    if (solver_has_options(settings, "preconditioner") || solver_has_options(settings, "maxiter") ||
        solver_has_options(settings, "restart") || solver_has_options(settings, "trace") ||
        solver_has_options(settings, "assume") || solver_has_options(settings, "condition"))
        vmerror_kind(v, "domain", "unsupported Arnoldi option");
    size_t work_elements = ((size_t)k + 4) * n + (size_t)k * k;
    size_t projected_elements = 2 * (size_t)k * k + 6 * (size_t)k;
    if (work_elements + projected_elements > SIZE_MAX / sizeof(double))
        vmerror_kind(v, "memory_budget", "Arnoldi workspace is not addressable");
    size_t work_bytes = work_elements * sizeof(double),
           limit = (work_elements + projected_elements) * sizeof(double);
    size_t projected_bytes = projected_elements * sizeof(double),
           op_bytes = linop_workspace(v, object);
    size_t output_bytes = (2 * n + 3) * (size_t)wanted.i * sizeof(double);
    if (limit > opts.workspace_limit || op_bytes > opts.workspace_limit - limit)
        vmerror_kind(v, "memory_budget", "Arnoldi basis and projection exceed workspace_bytes");
    Obj *buffer = solver_workspace(v, work_bytes, op_bytes, output_bytes + projected_bytes, &opts);
    double *basis = (double *)buffer->as.buffer.data, *work = basis + (size_t)k * n,
           *h = work + 4 * n;
    solver_read_vector(v, seed, n, basis);
    long double norm = linear_norm(n, basis, 1);
    if (!(norm > 0) || !isfinite(norm))
        vmerror_kind(v, "domain", "Arnoldi seed must have finite nonzero norm");
    for (size_t i = 0; i < n; i++)
        basis[i] = (double)(basis[i] / norm);
    LinopContext context;
    LinearOperator op = linop_open(v, object, &context);
    bool breakdown = false;
    int steps = build_basis(op, basis, work, h, k, &breakdown), got = 0;
    if (!breakdown)
        got = steps < wanted.i ? steps : (int)wanted.i;
    Obj *wr = array_new(v, 1, got, 1), *wi = array_new(v, 1, got, 1);
    Obj *vr = array_new(v, 2, (int)n, got), *vi = array_new(v, 2, (int)n, got);
    Obj *residuals = array_new(v, 1, got, 1);
    bool good = !breakdown;
    if (got) {
        Obj *checkpoint = v->objects;
        Obj *reduced = array_new(v, 2, steps, steps);
        for (int i = 0; i < steps; i++)
            for (int j = 0; j < steps; j++)
                A(reduced).data[(size_t)i * steps + j] = h[(size_t)j * k + i];
        Obj *projected = linalg_projected_eigen(v, reduced).o;
        Val *parts = projected->as.list.a;
        double *real = A(parts[0].o).data, *imag = A(parts[1].o).data;
        bool used[128] = {0};
        for (int j = 0; j < got; j++) {
            int col = -1;
            long double magnitude = -1;
            for (int i = 0; i < steps; i++) {
                long double candidate = hypotl(real[i], imag[i]);
                if (!used[i] && candidate > magnitude) {
                    magnitude = candidate;
                    col = i;
                }
            }
            used[col] = true;
            A(wr).data[j] = real[col];
            A(wi).data[j] = imag[col];
            double residual = lift_pair(op, basis, steps, col, parts[2].o, parts[3].o, real[col],
                                        imag[col], work, A(vr).data + j, A(vi).data + j, got);
            A(residuals).data[j] = residual;
            good &= isfinite(residual) && residual <= opts.tolerance;
        }
        solver_discard_since(v, checkpoint);
    }
    bool converged = good && got == wanted.i;
    Obj *result = newobj(v, O_MAP);
    sparse_result_field(v, result, "values_real", objval(wr));
    sparse_result_field(v, result, "values_imag", objval(wi));
    sparse_result_field(v, result, "vectors_real", objval(vr));
    sparse_result_field(v, result, "vectors_imag", objval(vi));
    sparse_result_field(v, result, "residuals", objval(residuals));
    sparse_result_field(v, result, "iterations", integer(steps));
    sparse_result_field(v, result, "converged", boolean(converged));
    sparse_result_field(v, result, "breakdown", boolean(breakdown));
    sparse_result_field(v, result, "reason",
                        objval(string(v, converged        ? "converged"
                                         : breakdown      ? "numerical_breakdown"
                                         : got < wanted.i ? "invariant_subspace"
                                                          : "basis_limit")));
    sparse_result_field(v, result, "workspace_bytes", integer((int64_t)(limit + op_bytes)));
    linop_close(v, &context);
    solver_workspace_done(v, buffer);
    return objval(result);
#endif
}
