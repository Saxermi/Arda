#include "sparse_internal.h"

#define A(o) ((o)->as.array)
#define S(o) ((o)->as.sparse)
#define W(o) ((o)->as.sparse_workspace)

enum { CG_WORK_VECTORS = 5 };

Obj *sparse_workspace_new(VM *v, size_t n) {
    if (!n || n > ARDA_MAX_ELEMENTS)
        vmerror(v, "invalid sparse solver workspace length");
    if (v->bytes > SIZE_MAX - sizeof(Obj))
        vmerror(v, "sparse object accounting overflow");
    Obj *out = newobj(v, O_SPARSE_WORKSPACE);
    W(out).length = n;
    W(out).data = sparse_buffer(v, out, n, CG_WORK_VECTORS * sizeof(double), 0);
    return out;
}

static Obj *solver_vector(VM *v, Val value, size_t n) {
    if (!isobj(value, O_ARRAY) || A(value.o).rank != 1 || A(value.o).n != n)
        vmerror(v, "sparse solver vector shape mismatch");
    Obj *o = value.o;
    for (size_t i = 0; i < n; i++)
        if (!isfinite(A(o).data[i * A(o).stride[0]]))
            vmerror(v, "sparse solver vectors must be finite");
    return o;
}

/* Scale the sum of squares so that very large or small finite rhs and residual
 * vectors do not overflow or underflow merely due to squaring. */
static long double vector_norm(size_t n, const double *x, size_t stride) {
    long double scale = 0, squares = 1;
    for (size_t i = 0; i < n; i++) {
        long double value = fabsl(x[i * stride]);
        if (!isfinite(value))
            return INFINITY;
        if (value == 0)
            continue;
        if (value > scale) {
            long double ratio = scale / value;
            squares = 1 + squares * ratio * ratio;
            scale = value;
        } else {
            long double ratio = value / scale;
            squares += ratio * ratio;
        }
    }
    return scale * sqrtl(squares);
}

static double relative_norm(long double value, long double rhs_norm) {
    return (double) (rhs_norm == 0 ? value : value / rhs_norm);
}

/* The result buffer can alias the Ax workspace. */
static long double true_residual(const LinearOperator *op, Obj *rhs, const double *x, size_t xs,
                                 double *work, double *residual) {
    op->apply(op->context, x, xs, work, 1);
    for (size_t i = 0; i < op->rows; i++)
        residual[i] = A(rhs).data[i * A(rhs).stride[0]] - work[i];
    return vector_norm(op->rows, residual, 1);
}

static Val solve_result(VM *v, Obj *solution, int iterations, double residual, const char *reason, int converged) {
    Obj *report = newobj(v, O_MAP);
    sparse_result_field(v, report, "solution", objval(solution));
    sparse_result_field(v, report, "iterations", integer(iterations));
    sparse_result_field(v, report, "residual", number(residual));
    sparse_result_field(v, report, "reason", objval(string(v, reason)));
    sparse_result_field(v, report, "converged", boolean(converged));
    return objval(report);
}

static int update_step(size_t n, double alpha, double *x, double *r,
                       const double *p, const double *ap, double *next_x) {
    for (size_t i = 0; i < n; i++) {
        next_x[i] = x[i] + alpha * p[i];
        r[i] -= alpha * ap[i];
        if (!isfinite(next_x[i]) || !isfinite(r[i]))
            return 0;
    }
    memcpy(x, next_x, n * sizeof *x);
    return 1;
}

Val sparse_solve(VM *v, Obj *matrix, Val rhs, Val tol, Val maxiter, Val workspace, int jacobi) {
    size_t n = S(matrix).rows;
    if (!n || n != S(matrix).cols || n > ARDA_MAX_ELEMENTS)
        vmerror(v, "sparse CG requires a nonempty square matrix with supported vector size");
    double tolerance = sparse_finite(v, tol);
    if (tolerance <= 0 || tolerance >= 1)
        vmerror(v, "sparse CG tolerance must be in (0,1)");
    if (maxiter.t != V_INT || maxiter.i < 1 || maxiter.i > 1000000)
        vmerror(v, "sparse CG maxiter must be 1..1000000");
    Obj *b = solver_vector(v, rhs, n);
    int internal = workspace.t == V_NIL;
    if (!internal && (!isobj(workspace, O_SPARSE_WORKSPACE) || W(workspace.o).length != n))
        vmerror(v, "sparse CG workspace length mismatch");

    Obj *csr = sparse_as_csr(v, matrix);
    const Sparse *s = &S(csr);
    if (!sparse_symmetric(s))
        vmerror(v, "sparse CG requires a symmetric matrix");
    LinearOperator op = sparse_operator(s);
    for (size_t i = 0; i < n; i++)
        if (sparse_csr_at(s, i, (uint32_t) i) <= 0)
            vmerror(v, "sparse CG requires a positive diagonal");

    long double bn = vector_norm(n, A(b).data, A(b).stride[0]);
    if (!isfinite(bn))
        vmerror(v, "sparse CG rhs norm overflow");
    size_t vectors = 1 + (internal && bn != 0 ? CG_WORK_VECTORS : 0);
    sparse_budget_check(v, sparse_mul_size(v, n, vectors * sizeof(double)));
    Obj *solution = array_new(v, 1, (int) n, 1);
    if (bn == 0) {
        sparse_drop_temporary(v, csr, matrix);
        return solve_result(v, solution, 0, 0, "converged", 1);
    }
    Obj *owned = internal ? sparse_workspace_new(v, n) : workspace.o;
    double *r = W(owned).data, *z = r + n, *p = z + n, *ap = p + n, *diag = ap + n;
    double *x = A(solution).data;
    for (size_t i = 0; i < n; i++) {
        r[i] = A(b).data[i * A(b).stride[0]];
        if (jacobi)
            diag[i] = sparse_csr_at(s, i, (uint32_t) i);
    }

    int iterations = 0, restart = 1;
    long double previous = 0;
    const char *reason = "max_iterations";
    while (iterations < maxiter.i) {
        for (size_t i = 0; i < n; i++)
            z[i] = jacobi ? r[i] / diag[i] : r[i];
        long double rho = linear_dot(n, r, z);
        if (!(rho > 0) || !isfinite(rho)) {
            reason = "numerical_breakdown";
            break;
        }
        if (restart)
            memcpy(p, z, n * sizeof *p);
        else {
            double beta = (double) (rho / previous);
            for (size_t i = 0; i < n; i++)
                p[i] = z[i] + beta * p[i];
        }

        op.apply(op.context, p, 1, ap, 1);
        long double curvature = linear_dot(n, p, ap);
        if (!isfinite(curvature) || curvature <= 0) {
            reason = isfinite(curvature) ? "nonpositive_curvature" : "numerical_breakdown";
            break;
        }
        double alpha = (double) (rho / curvature);
        if (!isfinite(alpha) || alpha == 0.0 || !update_step(n, alpha, x, r, p, ap, z)) {
            reason = "numerical_breakdown";
            break;
        }
        iterations++;
        restart = 0;
        previous = rho;
        if (vector_norm(n, r, 1) <= tolerance * bn) {
            if (true_residual(&op, b, x, 1, ap, r) <= tolerance * bn)
                break;
            restart = 1;
        }
    }
    long double actual = true_residual(&op, b, x, 1, ap, r);
    double residual = relative_norm(actual, bn);
    int converged = actual <= tolerance * bn;
    if (converged)
        reason = "converged";
    else if (!isfinite(residual))
        reason = "numerical_breakdown";

    if (internal)
        sparse_drop_temporary(v, owned, NULL);
    sparse_drop_temporary(v, csr, matrix);
    return solve_result(v, solution, iterations, residual, reason, converged);
}

Val sparse_verify(VM *v, Obj *matrix, Val rhs, Val solution) {
    size_t rows = S(matrix).rows, cols = S(matrix).cols;
    Obj *b = solver_vector(v, rhs, rows), *x = solver_vector(v, solution, cols);
    long double bn = vector_norm(rows, A(b).data, A(b).stride[0]);
    if (!isfinite(bn))
        vmerror(v, "sparse rhs norm overflow");
    Obj *csr = sparse_as_csr(v, matrix);
    LinearOperator op = sparse_operator(&S(csr));
    Obj *scratch = sparse_matrix(v, 0, 0, 0);
    double *work = sparse_buffer(v, scratch, rows, sizeof *work, 0);
    S(scratch).work = work;
    long double actual = true_residual(&op, b, A(x).data, A(x).stride[0], work, work);
    sparse_drop_temporary(v, scratch, NULL);
    sparse_drop_temporary(v, csr, matrix);
    return number(relative_norm(actual, bn));
}
