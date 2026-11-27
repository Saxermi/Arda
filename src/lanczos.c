#include "solver.h"
#include <float.h>

#define A(o) ((o)->as.array)

static bool diagonalize(int m, int pitch, double *matrix, double *vectors) {
    double scale = 0;
    for (int i = 0; i < m; i++) {
        vectors[(size_t)i * pitch + i] = 1;
        for (int j = 0; j < m; j++)
            scale = fmax(scale, fabs(matrix[(size_t)i * pitch + j]));
    }
    double threshold = DBL_EPSILON * scale;
    for (int sweep = 0; sweep < 64; sweep++) {
        bool rotated = false;
        for (int i = 0; i < m; i++) {
            for (int j = i + 1; j < m; j++) {
                double *row_i = matrix + (size_t)i * pitch, *row_j = matrix + (size_t)j * pitch;
                double entry = row_i[j];
                if (fabs(entry) <= threshold)
                    continue;
                long double tau = ((long double)row_j[j] - row_i[i]) / (2 * (long double)entry);
                double t = (double)(copysignl(1, tau) / (fabsl(tau) + hypotl(1, tau)));
                double c = 1 / hypot(1, t), s = t * c;
                row_i[i] -= t * entry;
                row_j[j] += t * entry;
                if (!isfinite(row_i[i]) || !isfinite(row_j[j]))
                    return false;
                row_i[j] = row_j[i] = 0;
                for (int k = 0; k < m; k++) {
                    if (k != i && k != j) {
                        double x = row_i[k], y = row_j[k];
                        matrix[(size_t)k * pitch + i] = row_i[k] = c * x - s * y;
                        matrix[(size_t)k * pitch + j] = row_j[k] = s * x + c * y;
                    }
                    double *v = vectors + (size_t)k * pitch;
                    double x = v[i], y = v[j];
                    v[i] = c * x - s * y;
                    v[j] = s * x + c * y;
                }
                rotated = true;
            }
        }
        if (!rotated)
            return true;
    }
    return false;
}

/* Classical symmetric Lanczos with two reorthogonalization passes. */
static int build_basis(LinearOperator op, double *q, double *w, double *t, int limit,
                       bool *breakdown) {
    size_t n = op.cols;
    double previous = 0;
    for (int j = 0; j < limit; j++) {
        double *v = q + (size_t)j * n;
        op.apply(op.context, v, 1, w, 1);
        long double norm = linear_norm(n, w, 1);
        double alpha = (double)linear_dot(n, v, w);
        if (!isfinite(norm) || !isfinite(alpha)) {
            *breakdown = true;
            return j;
        }
        t[(size_t)j * limit + j] = alpha;
        for (size_t i = 0; i < n; i++)
            w[i] -= alpha * v[i];
        if (j > 0)
            for (size_t i = 0; i < n; i++)
                w[i] -= previous * (v - n)[i];
        long double beta = krylov_orthogonalize(n, q, j + 1, w, NULL);
        if (!isfinite(beta) || beta > DBL_MAX) {
            *breakdown = true;
            return j + 1;
        }
        if (j + 1 == limit || beta <= 8 * DBL_EPSILON * norm)
            return j + 1;
        previous = (double)beta;
        t[(size_t)j * limit + j + 1] = t[(size_t)(j + 1) * limit + j] = previous;
        for (size_t i = 0; i < n; i++)
            v[n + i] = (double)(w[i] / beta);
    }
    return limit;
}

static bool ritz_vectors(LinearOperator op, int m, int basis_limit, const double *q,
                         const double *t, const double *projected, double *x, double *ax,
                         Obj *values, Obj *vectors, Obj *residuals, double tol) {
    int wanted = A(values).rows;
    size_t n = op.cols;
    bool used[128] = {0}, good = true;
    for (int k = 0; k < wanted; k++) {
        int col = -1;
        for (int j = 0; j < m; j++)
            if (!used[j] &&
                (col < 0 || t[(size_t)j * basis_limit + j] > t[(size_t)col * basis_limit + col]))
                col = j;
        used[col] = true;
        double value = t[(size_t)col * basis_limit + col];
        memset(x, 0, n * sizeof(double));
        for (int j = 0; j < m; j++) {
            const double *v = q + (size_t)j * n;
            double coef = projected[(size_t)j * basis_limit + col];
            for (size_t i = 0; i < n; i++)
                x[i] += coef * v[i];
        }
        long double norm = linear_norm(n, x, 1);
        for (size_t i = 0; i < n; i++)
            A(vectors).data[i * wanted + k] = x[i] = (double)(x[i] / norm);
        long double residual = solver_eigen_residual(op, value, 0, x, NULL, 1, ax, NULL);
        good &= isfinite(residual) && residual <= tol;
        A(values).data[k] = value;
        A(residuals).data[k] = (double)residual;
    }
    return good;
}

Val solver_lanczos(VM *v, Val input, Val seed, Val wanted, Val settings) {
    Obj *object = solver_operator(v, input);
    SolverOptions opts = solver_options(v, settings);
    size_t m, n;
    linop_shape(v, object, &m, &n);
    if (!n || m != n || n > ARDA_MAX_ELEMENTS)
        vmerror_kind(v, "shape", "Lanczos needs a nonempty square operator");
    int k = opts.basis < (int)n ? opts.basis : (int)n;
    if (wanted.t != V_INT || wanted.i < 1 || wanted.i > k ||
        (uint64_t)wanted.i > ARDA_MAX_ELEMENTS / n)
        vmerror_kind(v, "domain", "Lanczos eigenpair count exceeds basis or output limit");
    if (solver_has_options(settings, "preconditioner") || solver_has_options(settings, "maxiter") ||
        solver_has_options(settings, "restart") || solver_has_options(settings, "trace") ||
        solver_has_options(settings, "condition"))
        vmerror_kind(v, "domain", "unsupported Lanczos option");
    size_t elements = ((size_t)k + 2) * n + 2 * (size_t)k * k;
    if (elements > SIZE_MAX / sizeof(double))
        vmerror_kind(v, "memory_budget", "Lanczos workspace is not addressable");
    size_t bytes = elements * sizeof(double), op_bytes = linop_workspace(v, object);
    size_t output = (n + 2) * (size_t)wanted.i * sizeof(double);
    Obj *buffer = solver_workspace(v, bytes, op_bytes, output, &opts);
    double *q = (double *)buffer->as.buffer.data, *w = q + (size_t)k * n, *x = w + n;
    double *t = x + n, *projected = t + (size_t)k * k;
    solver_read_vector(v, seed, n, q);
    long double norm = linear_norm(n, q, 1);
    if (!(norm > 0) || !isfinite(norm))
        vmerror_kind(v, "domain", "Lanczos seed must have finite nonzero norm");
    for (size_t i = 0; i < n; i++)
        q[i] = (double)(q[i] / norm);
    LinopContext ctx;
    LinearOperator op = linop_open(v, object, &ctx);
    solver_check_symmetric(v, object, &ctx, &opts, false);
    bool breakdown = false;
    int steps = build_basis(op, q, w, t, k, &breakdown);
    bool projected_ok = !breakdown && diagonalize(steps, k, t, projected);
    int got = projected_ok ? (steps < wanted.i ? steps : (int)wanted.i) : 0;
    Obj *values = array_new(v, 1, got, 1), *vectors = array_new(v, 2, (int)n, got);
    Obj *residuals = array_new(v, 1, got, 1);
    bool good = projected_ok && ritz_vectors(op, steps, k, q, t, projected, x, w, values, vectors,
                                             residuals, opts.tolerance);
    bool converged = good && got == wanted.i;
    Obj *result = newobj(v, O_MAP);
    sparse_result_field(v, result, "values", objval(values));
    sparse_result_field(v, result, "vectors", objval(vectors));
    sparse_result_field(v, result, "residuals", objval(residuals));
    sparse_result_field(v, result, "iterations", integer(steps));
    sparse_result_field(v, result, "converged", boolean(converged));
    sparse_result_field(v, result, "breakdown", boolean(!projected_ok));
    const char *reason = converged        ? "converged"
                         : !projected_ok  ? "numerical_breakdown"
                         : got < wanted.i ? "invariant_subspace"
                                          : "basis_limit";
    sparse_result_field(v, result, "reason", objval(string(v, reason)));
    sparse_result_field(v, result, "workspace_bytes", integer((int64_t)(bytes + op_bytes)));
    linop_close(v, &ctx);
    solver_workspace_done(v, buffer);
    return objval(result);
}
