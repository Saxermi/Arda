#include "solver.h"
#include "poisson.h"
#include "pde.h"
#include <float.h>

#define A(o) ((o)->as.array)

static long double max_abs(size_t n, const double *x) {
    long double largest = 0;
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(x[i]))
            return INFINITY;
        largest = fmaxl(largest, fabs(x[i]));
    }
    return largest;
}

static long double infinity_norm(Obj *op, const LinopContext *context, size_t m, size_t n) {
    long double largest = 0;
    if (context->csr) {
        const Sparse *s = &context->csr->as.sparse;
        for (size_t i = 0; i < m; i++) {
            long double rowsum = 0;
            for (size_t j = s->offset[i]; j < s->offset[i + 1]; j++)
                rowsum += fabs(s->values[j]);
            largest = fmaxl(largest, rowsum);
        }
    } else if (op->type == O_ARRAY || op->type == O_TYPED_TENSOR) {
        for (size_t i = 0; i < m; i++) {
            long double rowsum = 0;
            for (size_t j = 0; j < n; j++)
                rowsum += fabs(linop_dense_at(op, i, j));
            largest = fmaxl(largest, rowsum);
        }
    } else if (op->as.linop.kind == LINOP_DIAGONAL) {
        largest = max_abs(n, A(op->as.linop.first).data);
    } else if (op->as.linop.kind == LINOP_CARTESIAN) {
        largest = pde_operator_norm(op);
    } else if (op->as.linop.kind == LINOP_POISSON) {
        largest = poisson_diagonal(op);
        for (int i = 0; i < op->as.linop.grid_rank; i++) {
            int length = op->as.linop.extent[i];
            int neighbors = length < 3 ? length - 1 : 2;
            largest += neighbors * (long double)op->as.linop.weights[i];
        }
    } else
        return NAN;
    return largest;
}

static long double relative(long double difference, long double scale) {
    return !isfinite(scale) || !isfinite(difference) ? INFINITY
           : scale == 0                              ? difference
                                                     : difference / scale;
}

static void metric(VM *v, Obj *out, const char *key, long double value) {
    Val encoded = isnan(value) ? nil() : number(value > DBL_MAX ? INFINITY : (double)value);
    sparse_result_field(v, out, key, encoded);
}

static Val requirements(VM *v, Val method) {
    const char *name = asstr(v, method);
    bool cg = !strcmp(name, "cg"), minres = !strcmp(name, "minres");
    bool least = !strcmp(name, "lsqr") || !strcmp(name, "lsmr");
    bool general = !strcmp(name, "gmres") || !strcmp(name, "bicgstab");
    bool multigrid = !strcmp(name, "multigrid"), refine = !strcmp(name, "refine");
    bool lanczos = !strcmp(name, "lanczos"), arnoldi = !strcmp(name, "arnoldi");
    if (!(cg || minres || least || general || multigrid || refine || lanczos || arnoldi))
        vmerror_kind(v, "domain", "unknown solver method");
    Obj *map = newobj(v, O_MAP);
    sparse_result_field(v, map, "symmetric", boolean(cg || minres || multigrid || lanczos));
    sparse_result_field(v, map, "positive_definite", boolean(cg || multigrid));
    sparse_result_field(v, map, "square", boolean(!least));
    sparse_result_field(v, map, "apply_transpose", boolean(least));
    sparse_result_field(v, map, "storage",
                        objval(string(v, multigrid ? "poisson_grid"
                                         : refine  ? "dense"
                                                   : "operator")));
    sparse_result_field(v, map, "reduction", objval(string(v, "local")));
    return objval(map);
}

static void check_options(VM *v, Val settings, SolverOptions opts, int which) {
    if (solver_has_options(settings, "restart") || solver_has_options(settings, "basis") ||
        solver_has_options(settings, "trace") || solver_has_options(settings, "preconditioner") ||
        solver_has_options(settings, "assume") || solver_has_options(settings, "maxiter") ||
        (which != 0 && opts.condition))
        vmerror_kind(v, "domain", "unsupported verification option");
}

typedef struct {
    long double max, squares;
} Norm;

static void add_norm(Norm *norm, long double x) {
    if (x > norm->max) {
        long double ratio = norm->max / x;
        norm->squares = 1 + norm->squares * ratio * ratio;
        norm->max = x;
    } else if (x) {
        long double ratio = x / norm->max;
        norm->squares += ratio * ratio;
    }
}

static long double norm_value(Norm norm) {
    return norm.max * sqrtl(norm.squares);
}

/* Compute r*scale/(a*x+b) without forming an overflowing denominator or
 * residual. The common power-of-two scaling also works with binary64 long double. */
static long double backward_error(long double r, long double scale, long double a, long double x,
                                  long double b) {
    if (!isfinite(a))
        return NAN;
    int ea, ex, eb, es;
    long double ma = frexpl(a, &ea), mx = frexpl(x, &ex);
    long double mb = frexpl(b, &eb), ms = frexpl(scale, &es);
    if (!ma || !mx) {
        if (!mb)
            return r * scale;
        return ldexpl(r * ms / mb, es - eb);
    }
    int product = ea + ex, exponent = mb && eb > product ? eb : product;
    long double denom = ldexpl(ma * mx, product - exponent) + ldexpl(mb, eb - exponent);
    return ldexpl(r * ms / denom, es - exponent);
}

static double diagonal_condition(Obj *op) {
    const Obj *diag = op->as.linop.first;
    double largest = 0, smallest = INFINITY;
    for (size_t i = 0; i < A(diag).n; i++) {
        double entry = fabs(A(diag).data[i]);
        largest = fmax(largest, entry);
        smallest = fmin(smallest, entry);
    }
    return smallest ? largest / smallest : INFINITY;
}

static long double linear_check(VM *v, Obj *object, const LinopContext *ctx, LinearOperator op,
                                double *b, double *x, double *r, Val rhs, Val solution,
                                SolverOptions opts, Obj *result) {
    size_t m = op.rows, n = op.cols;
    solver_read_vector(v, rhs, m, b);
    solver_read_vector(v, solution, n, x);
    op.apply(op.context, x, 1, r, 1);
    long double bmax = max_abs(m, b), axmax = max_abs(m, r);
    long double scale = fmaxl(bmax, axmax);
    long double residual = INFINITY, absolute = INFINITY, backward = NAN;
    long double norm = infinity_norm(object, ctx, m, n);
    if (isfinite(scale)) {
        Norm delta = {0}, right = {0};
        if (!scale)
            scale = 1;
        for (size_t i = 0; i < m; i++) {
            long double lhs = b[i], ax = r[i], difference = lhs - ax;
            long double scaled =
                isfinite(difference) ? difference / scale : lhs / scale - ax / scale;
            add_norm(&delta, fabsl(scaled));
            add_norm(&right, fabsl(lhs / scale));
        }
        long double rnorm = norm_value(delta);
        absolute = rnorm * scale;
        long double bnorm = norm_value(right);
        residual = bmax ? (bnorm ? rnorm / bnorm : INFINITY) : absolute;
        backward = backward_error(delta.max, scale, norm, max_abs(n, x), bmax);
    }
    metric(v, result, "absolute_residual", absolute);
    metric(v, result, "backward_error_inf", backward);
    metric(v, result, "operator_norm_inf", norm);
    double condition = NAN;
    if (opts.condition) {
        if (object->type == O_LINOP)
            condition = diagonal_condition(object);
        else {
            Obj *checkpoint = v->objects;
            condition = dense_condition_inf(v, object);
            solver_discard_since(v, checkpoint);
        }
    }
    metric(v, result, "condition_inf", condition);
    return residual;
}

static void eigenvector(VM *v, Val value, size_t n, double *real, double *imaginary, bool complex) {
    if (isobj(value, O_MAP)) {
        Table *table = &value.o->as.map.tab;
        Entry *re = tablefind(table, "real"), *im = tablefind(table, "imag");
        if (table->n != 2 || !re || !im)
            vmerror_kind(v, "type", "complex eigenvector needs real and imag vectors");
        solver_read_vector(v, re->val, n, real);
        solver_read_vector(v, im->val, n, imaginary);
    } else {
        solver_read_vector(v, value, n, real);
        if (complex)
            memset(imaginary, 0, n * sizeof(double));
    }
}

long double solver_eigen_residual(LinearOperator op, double re, double im, const double *x,
                                  const double *y, long double xnorm, double *ax, double *ay) {
    size_t n = op.cols;
    op.apply(op.context, x, 1, ax, 1);
    if (y)
        op.apply(op.context, y, 1, ay, 1);
    long double anorm = linear_norm(n, ax, 1);
    if (y)
        anorm = hypotl(anorm, linear_norm(n, ay, 1));
    for (size_t i = 0; i < n; i++) {
        long double a = (long double)ax[i] - (long double)re * x[i];
        if (y) {
            a += (long double)im * y[i];
            long double b = (long double)ay[i] - (long double)re * y[i] - (long double)im * x[i];
            ay[i] = (double)b;
        }
        ax[i] = (double)a;
    }
    long double diff = linear_norm(n, ax, 1);
    if (y)
        diff = hypotl(diff, linear_norm(n, ay, 1));
    return relative(diff, anorm + hypotl(re, im) * xnorm);
}

static long double eigen_check(VM *v, LinearOperator op, Val eigenvalue, Val vector, double *w,
                               bool complex, Obj *out) {
    size_t n = op.cols;
    double *x = w, *ax = w + n, *y = complex ? ax + n : NULL, *ay = complex ? y + n : NULL;
    double re = isobj(eigenvalue, O_COMPLEX) ? eigenvalue.o->as.cplx.re : asnum(v, eigenvalue);
    double im = isobj(eigenvalue, O_COMPLEX) ? eigenvalue.o->as.cplx.im : 0;
    if (!isfinite(re) || !isfinite(im))
        vmerror_kind(v, "domain", "eigenvalue must be finite");
    eigenvector(v, vector, n, x, y, complex);
    long double xnorm = linear_norm(n, x, 1);
    if (complex)
        xnorm = hypotl(xnorm, linear_norm(n, y, 1));
    if (!(xnorm > 0) || !isfinite(xnorm))
        vmerror_kind(v, "domain", "eigenvector must have a finite nonzero norm");
    metric(v, out, "vector_norm", xnorm);
    return solver_eigen_residual(op, re, im, x, y, xnorm, ax, ay);
}

static long double svd_check(VM *v, LinearOperator op, Val sigma, Val left, Val right, double *w,
                             Obj *out) {
    size_t m = op.rows, n = op.cols;
    double *u = w, *x = u + m, *av = x + n, *atu = av + m;
    double s = asnum(v, sigma);
    if (!(s >= 0) || !isfinite(s))
        vmerror_kind(v, "domain", "singular value must be finite and nonnegative");
    solver_read_vector(v, left, m, u);
    solver_read_vector(v, right, n, x);
    long double unorm = linear_norm(m, u, 1), xnorm = linear_norm(n, x, 1);
    if (!(unorm > 0 && xnorm > 0) || !isfinite(unorm) || !isfinite(xnorm))
        vmerror_kind(v, "domain", "singular vectors must have finite nonzero norms");
    op.apply(op.context, x, 1, av, 1);
    op.apply_transpose(op.context, u, 1, atu, 1);
    long double lscale = linear_norm(m, av, 1) + (long double)s * unorm;
    long double rscale = linear_norm(n, atu, 1) + (long double)s * xnorm;
    for (size_t i = 0; i < m; i++)
        av[i] = (double)((long double)av[i] - (long double)s * u[i]);
    for (size_t i = 0; i < n; i++)
        atu[i] = (double)((long double)atu[i] - (long double)s * x[i]);
    long double lres = relative(linear_norm(m, av, 1), lscale);
    long double rres = relative(linear_norm(n, atu, 1), rscale);
    long double norm_error = fmaxl(fabsl(unorm - 1), fabsl(xnorm - 1));
    metric(v, out, "left_residual", lres);
    metric(v, out, "right_residual", rres);
    metric(v, out, "normalization_error", norm_error);
    return isfinite(lres) && isfinite(rres) ? fmaxl(norm_error, fmaxl(lres, rres)) : INFINITY;
}

Val solver_verify(VM *v, int which, int argc, Val *args) {
    if (which == 3)
        return requirements(v, args[0]);
    Obj *object = solver_operator(v, args[0]);
    int needed = which == 2 ? 4 : 3;
    Val settings = argc > needed ? args[needed] : nil();
    SolverOptions opts = solver_options(v, settings);
    check_options(v, settings, opts, which);
    size_t m, n;
    linop_shape(v, object, &m, &n);
    if (!m || !n || m > ARDA_MAX_ELEMENTS || n > ARDA_MAX_ELEMENTS || (which == 1 && m != n))
        vmerror_kind(v, "shape", "invalid verification operator dimensions");
    bool complex = which == 1 && (isobj(args[1], O_COMPLEX) || isobj(args[2], O_MAP));
    size_t count = which == 0 ? 2 * m + n : which == 2 ? 2 * (m + n) : (complex ? 4 : 2) * n;
    size_t bytes = count * sizeof(double), extra = 0, op_bytes = linop_workspace(v, object);
    if (opts.condition) {
        bool diagonal = object->type == O_LINOP && object->as.linop.kind == LINOP_DIAGONAL;
        if (!diagonal && (object->type != O_ARRAY || m != n || n > 128))
            vmerror_kind(v, "domain",
                         "condition estimation requires a diagonal operator or a square numc "
                         "matrix of size at most 128");
        if (!diagonal)
            extra =
                n * n * sizeof(double) + n * (sizeof(double) + sizeof(long double) + sizeof(int));
    }
    if (extra > opts.workspace_limit || bytes > opts.workspace_limit - extra ||
        op_bytes > opts.workspace_limit - extra - bytes)
        vmerror_kind(v, "memory_budget", "condition estimation exceeds solver workspace_bytes");
    Obj *buf = solver_workspace(v, bytes, op_bytes, extra, &opts);
    double *work = (double *)buf->as.buffer.data;
    LinopContext context;
    LinearOperator op = linop_open(v, object, &context);
    Obj *out = newobj(v, O_MAP);
    long double residual;
    if (which == 0)
        residual = linear_check(v, object, &context, op, work, work + m, work + m + n, args[1],
                                args[2], opts, out);
    else if (which == 1)
        residual = eigen_check(v, op, args[1], args[2], work, complex, out);
    else
        residual = svd_check(v, op, args[1], args[2], args[3], work, out);
    linop_close(v, &context);
    solver_workspace_done(v, buf);
    metric(v, out, "residual", residual);
    sparse_result_field(v, out, "verified",
                        boolean(isfinite(residual) && residual <= opts.tolerance));
    return objval(out);
}
