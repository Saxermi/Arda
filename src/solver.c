#include "solver.h"
#include "poisson.h"
#include "typed_tensor.h"

#define A(o) ((o)->as.array)

static const char *const methods[] = {"cg", "minres", "gmres", "bicgstab", "lsqr", "lsmr"};
const char *const preconditioner_names[] = {"none", "jacobi", "block_jacobi", "ilu0", "ic0"};
static const char *const reasons[] = {"max_iterations",        "converged",
                                      "least_squares",         "numerical_breakdown",
                                      "nonpositive_curvature", "stagnation"};

bool solver_has_options(Val map, const char *key) {
    return isobj(map, O_MAP) && tablefind(&map.o->as.map.tab, key) != NULL;
}

static Val option(VM *v, Val map, const char *key, int *seen) {
    if (map.t == V_NIL)
        return nil();
    if (!isobj(map, O_MAP))
        vmerror_kind(v, "type", "solver options must be a map");
    Entry *e = tablefind(&map.o->as.map.tab, key);
    *seen += e != NULL;
    return e ? e->val : nil();
}

static int option_int(VM *v, Val x, const char *name, int fallback, int min, int max) {
    if (x.t == V_NIL)
        return fallback;
    if (x.t != V_INT || x.i < min || x.i > max)
        vmerror_kind(v, "domain", "%s must be an integer in %d..%d", name, min, max);
    return (int)x.i;
}

static bool assumption(VM *v, Val map, const char *key, int *seen) {
    Val value = option(v, map, key, seen);
    if (value.t != V_NIL && value.t != V_BOOL)
        vmerror_kind(v, "type", "operator assumption %s must be Boolean", key);
    return value.t == V_BOOL && value.n;
}

SolverOptions solver_options(VM *v, Val map) {
    int seen = 0;
    Val tol = option(v, map, "tol", &seen);
    double tolerance = tol.t == V_NIL ? 1e-8 : asnum(v, tol);
    if (!(tolerance > 0 && tolerance < 1))
        vmerror_kind(v, "domain", "solver tolerance must be in (0,1)");
    SolverOptions opts = {.tolerance = tolerance};
    opts.max_iterations =
        option_int(v, option(v, map, "maxiter", &seen), "maxiter", 1000, 1, 1000000);
    opts.restart = option_int(v, option(v, map, "restart", &seen), "restart", 30, 1, 256);
    opts.trace_limit = option_int(v, option(v, map, "trace", &seen), "trace", 0, 0, 10000);
    opts.basis = option_int(v, option(v, map, "basis", &seen), "basis", 32, 1, 128);
    Val limit = option(v, map, "workspace_bytes", &seen);
    if (limit.t != V_NIL && (limit.t != V_INT || limit.i < 0 || (uint64_t)limit.i > SIZE_MAX))
        vmerror_kind(v, "domain", "solver workspace_bytes must be a nonnegative integer");
    opts.workspace_limit = limit.t == V_NIL ? 256u * 1024u * 1024u : (size_t)limit.i;
    Val pc = option(v, map, "preconditioner", &seen);
    if (pc.t != V_NIL) {
        const char *name = asstr(v, pc);
        int kind = 0;
        while (kind <= PC_IC0 && strcmp(name, preconditioner_names[kind]))
            kind++;
        if (kind > PC_IC0)
            vmerror_kind(v, "domain", "unknown solver preconditioner");
        opts.preconditioner = (PreconditionerKind)kind;
    }
    opts.condition = assumption(v, map, "condition", &seen);
    Val assume = option(v, map, "assume", &seen);
    int flags = 0;
    opts.symmetric = assumption(v, assume, "symmetric", &flags);
    opts.positive_definite = assumption(v, assume, "positive_definite", &flags);
    if (isobj(assume, O_MAP) && assume.o->as.map.tab.n != flags)
        vmerror_kind(v, "domain", "unknown operator assumption");
    if (isobj(map, O_MAP) && map.o->as.map.tab.n != seen)
        vmerror_kind(v, "domain", "unrecognized solver option");
    return opts;
}

Obj *solver_operator(VM *v, Val value) {
    if (value.t != V_OBJ)
        vmerror_kind(v, "type", "expected a linear operator");
    return value.o;
}

void solver_read_vector(VM *v, Val value, size_t length, double *out) {
    if (isobj(value, O_ARRAY) && A(value.o).rank == 1 && A(value.o).n == length) {
        for (size_t i = 0; i < length; i++) {
            out[i] = A(value.o).data[i * A(value.o).stride[0]];
            if (!isfinite(out[i]))
                vmerror_kind(v, "domain", "solver vectors must be finite");
        }
        return;
    }
    if (isobj(value, O_TYPED_TENSOR)) {
        TypedTensor t = value.o->as.tensor;
        if (t.rank == 1 && t.n == length && t.dtype == DT_F64) {
            for (size_t i = 0; i < length; i++) {
                out[i] = *(const double *)typed_element(&t, typed_offset(&t, i));
                if (!isfinite(out[i]))
                    vmerror_kind(v, "domain", "solver vectors must be finite");
            }
            return;
        }
    }
    vmerror_kind(v, "shape", "solver expects a Float64 vector of the required length");
}

void solver_check_symmetric(VM *v, Obj *input, const LinopContext *context,
                            const SolverOptions *options, bool positive) {
    size_t rows, cols;
    linop_shape(v, input, &rows, &cols);
    if (rows != cols)
        vmerror_kind(v, "shape", "symmetric solver requires a square operator");
    if (context->csr) {
        const Sparse *s = &context->csr->as.sparse;
        if (sparse_symmetric(s)) {
            if (positive)
                for (size_t i = 0; i < rows; i++)
                    if (!(sparse_csr_at(s, i, (uint32_t)i) > 0))
                        vmerror_kind(v, "domain", "CG requires a positive diagonal");
            return;
        }
    } else if (input->type == O_ARRAY || input->type == O_TYPED_TENSOR) {
        for (size_t i = 0; i < rows; i++) {
            if (positive && !(linop_dense_at(input, i, i) > 0))
                vmerror_kind(v, "domain", "CG requires a positive diagonal");
            for (size_t j = 0; j < i; j++)
                if (linop_dense_at(input, i, j) != linop_dense_at(input, j, i))
                    vmerror_kind(v, "domain", "solver requires symmetric coefficients");
        }
        return;
    } else if (input->as.linop.kind == LINOP_POISSON || input->as.linop.kind == LINOP_DIAGONAL) {
        if (positive && input->as.linop.kind == LINOP_DIAGONAL)
            for (size_t i = 0; i < rows; i++)
                if (!(A(input->as.linop.first).data[i] > 0))
                    vmerror_kind(v, "domain", "CG requires a positive diagonal");
        return;
    } else if (positive ? options->positive_definite
                        : options->symmetric || options->positive_definite) {
        return;
    }
    vmerror_kind(v, "domain",
                 "operator needs symmetry or an explicit structured-operator assumption");
}

Obj *solver_workspace(VM *v, size_t work, size_t operator_workspace, size_t output,
                      const SolverOptions *opts) {
    if (operator_workspace > opts->workspace_limit ||
        work > opts->workspace_limit - operator_workspace)
        vmerror_kind(v, "memory_budget", "solver workspace exceeds workspace_bytes");
    size_t avail = tensor_memory_available(v);
    if (operator_workspace > avail || work > avail - operator_workspace ||
        output > avail - operator_workspace - work)
        vmerror_kind(v, "memory_budget", "solver output and workspace exceed the tensor budget");
    return buffer_new(v, work);
}

void solver_workspace_done(VM *v, Obj *buffer) {
    tensor_buffer_discard(v, buffer);
}

/* Solver calls are synchronous and do not invoke user code. No native
 * buffer created after the checkpoint can escape an exception. */
void solver_discard_since(VM *v, Obj *checkpoint) {
    for (Obj *o = v->objects; o != checkpoint; o = o->next)
        if (o->type == O_BUFFER && o->as.buffer.release && !o->as.buffer.mapped)
            tensor_buffer_discard(v, o);
}

void solver_unwind(VM *v) {
    if (v->solver_checkpoint) {
        solver_discard_since(v, v->solver_checkpoint);
        v->solver_checkpoint = NULL;
    }
}

bool solver_update(size_t n, double *x, double alpha, const double *direction) {
    if (!isfinite(alpha))
        return false;
    for (size_t i = 0; i < n; i++)
        if (!isfinite(x[i] + alpha * direction[i]))
            return false;
    for (size_t i = 0; i < n; i++)
        x[i] += alpha * direction[i];
    return true;
}

long double solver_residual(SolverProblem *p) {
    p->op.apply(p->op.context, p->x, 1, p->residual, 1);
    for (size_t i = 0; i < p->op.rows; i++)
        p->residual[i] = p->b[i] - p->residual[i];
    long double r = linear_norm(p->op.rows, p->residual, 1);
    return p->rhs_norm != 0 ? r / p->rhs_norm : r;
}

long double solver_normal_residual(SolverProblem *p) {
    p->op.apply_transpose(p->op.context, p->residual, 1, p->normal, 1);
    long double r = linear_norm(p->op.cols, p->normal, 1);
    return p->normal_rhs_norm != 0 ? r / p->normal_rhs_norm : r;
}

void solver_trace(SolverProblem *p, long double estimate) {
    if (!p->trace || p->trace->as.list.n >= p->options.trace_limit)
        return;
    VM *v = p->vm;
    Obj *row = newobj(v, O_MAP);
    sparse_result_field(v, row, "iteration", integer(p->iterations));
    sparse_result_field(v, row, "estimated_residual", number((double)estimate));
    append(v, p->trace, objval(row));
}

Val solver_report(SolverProblem *p, Obj *solution, const char *method, size_t bytes) {
    long double residual = solver_residual(p);
    long double normal = p->least_squares ? solver_normal_residual(p) : 0;
    if (residual <= p->options.tolerance)
        p->stop = STOP_CONVERGED;
    else if (p->least_squares && isfinite(residual) && normal <= p->options.tolerance)
        p->stop = STOP_LEAST_SQUARES;
    else if (!isfinite(residual) || (p->least_squares && !isfinite(normal)))
        p->stop = STOP_BREAKDOWN;
    else if (p->stop == STOP_CONVERGED || p->stop == STOP_LEAST_SQUARES)
        p->stop = STOP_LIMIT;
    return solver_result(p, solution, method, bytes, residual, normal);
}

Val solver_result(SolverProblem *p, Obj *solution, const char *method, size_t bytes,
                  long double residual, long double normal) {
    bool converged = p->stop == STOP_CONVERGED || p->stop == STOP_LEAST_SQUARES;
    VM *v = p->vm;
    Obj *result = newobj(v, O_MAP);
    sparse_result_field(v, result, "solution", objval(solution));
    sparse_result_field(v, result, "iterations", integer(p->iterations));
    sparse_result_field(v, result, "residual", number((double)residual));
    sparse_result_field(v, result, "normal_residual",
                        p->least_squares ? number((double)normal) : nil());
    sparse_result_field(v, result, "converged", boolean(converged));
    sparse_result_field(v, result, "breakdown",
                        boolean(p->stop == STOP_BREAKDOWN || p->stop == STOP_CURVATURE));
    sparse_result_field(v, result, "reason", objval(string(v, reasons[p->stop])));
    sparse_result_field(v, result, "method", objval(string(v, method)));
    sparse_result_field(v, result, "preconditioner",
                        objval(string(v, preconditioner_names[p->options.preconditioner])));
    sparse_result_field(v, result, "tolerance", number(p->options.tolerance));
    sparse_result_field(v, result, "workspace_bytes", integer((int64_t)bytes));
    if (p->trace) {
        sparse_result_field(v, result, "trace", objval(p->trace));
        sparse_result_field(v, result, "trace_truncated",
                            boolean(p->iterations > p->trace->as.list.n));
    }
    return objval(result);
}

static Val solve(VM *v, SolverMethod method, Val input, Val rhs, Val settings) {
    Obj *object = solver_operator(v, input);
    SolverOptions opts = solver_options(v, settings);
    size_t m, n;
    linop_shape(v, object, &m, &n);
    bool least = method == SOLVER_LSQR || method == SOLVER_LSMR;
    if (!m || !n || n > ARDA_MAX_ELEMENTS || m > ARDA_MAX_ELEMENTS || (!least && m != n))
        vmerror_kind(v, "shape", "invalid operator dimensions for the selected solver");
    if (solver_has_options(settings, "condition") || solver_has_options(settings, "basis") ||
        (method != SOLVER_GMRES && solver_has_options(settings, "restart")))
        vmerror_kind(v, "domain", "unsupported option for selected Krylov method");
    if ((least || method == SOLVER_MINRES) && opts.preconditioner != PC_NONE)
        vmerror_kind(v, "domain", "this solver currently requires preconditioner none");
    if (method == SOLVER_CG && opts.preconditioner == PC_ILU0)
        vmerror_kind(v, "domain", "CG requires an SPD preconditioner");
    if (opts.restart > (int)n)
        opts.restart = (int)n;
    if (opts.restart > opts.max_iterations)
        opts.restart = opts.max_iterations;
    size_t work_count = 2 * m + (least ? n : 0) + krylov_elements(method, m, n, opts.restart);
    if (work_count > SIZE_MAX / sizeof(double))
        vmerror_kind(v, "memory_budget", "solver workspace is not addressable");
    size_t base_work = work_count * sizeof(double);
    size_t pc_bytes = preconditioner_bytes(v, object, n, opts.preconditioner);
    if (pc_bytes > SIZE_MAX - base_work)
        vmerror_kind(v, "memory_budget", "solver workspace is not addressable");
    size_t work_bytes = base_work + pc_bytes, op_bytes = linop_workspace(v, object);
    Obj *buffer = solver_workspace(v, work_bytes, op_bytes, n * sizeof(double), &opts);
    double *work = (double *)buffer->as.buffer.data;
    solver_read_vector(v, rhs, m, work);
    long double norm = linear_norm(m, work, 1);
    if (!isfinite(norm))
        vmerror_kind(v, "domain", "right-hand-side norm is not representable");
    LinopContext context;
    LinearOperator op = linop_open(v, object, &context);
    if (method == SOLVER_CG || method == SOLVER_MINRES || opts.preconditioner == PC_IC0)
        solver_check_symmetric(v, object, &context, &opts,
                               method == SOLVER_CG || opts.preconditioner == PC_IC0);
    Obj *x = array_new(v, 1, (int)n, 1);
    SolverProblem p = {.vm = v,
                       .op = op,
                       .options = opts,
                       .b = work,
                       .residual = work + m,
                       .normal = least ? work + 2 * m : NULL,
                       .x = A(x).data,
                       .rhs_norm = norm,
                       .trace = opts.trace_limit ? list(v) : NULL,
                       .least_squares = least};
    if (norm) {
        p.preconditioner = preconditioner_setup(v, object, &context, n, opts.preconditioner,
                                                buffer->as.buffer.data + base_work);
        if (least) {
            op.apply_transpose(op.context, p.b, 1, p.normal, 1);
            p.normal_rhs_norm = linear_norm(n, p.normal, 1);
            if (!isfinite(p.normal_rhs_norm))
                vmerror_kind(v, "domain", "normal right-hand-side norm is not representable");
        }
        krylov_run(&p, method, work + 2 * m + (least ? n : 0));
    }
    Val result = solver_report(&p, x, methods[method], work_bytes + op_bytes);
    linop_close(v, &context);
    solver_workspace_done(v, buffer);
    return result;
}

enum { SOLVE, MULTIGRID, REFINE, LANCZOS, VERIFY, VERIFY_EIGEN, VERIFY_SVD, REQUIREMENTS, ARNOLDI };
static const char *const names[] = {"solve",      "multigrid",    "refine",
                                    "lanczos",    "verify",       "verify_eigen",
                                    "verify_svd", "requirements", "arnoldi"};

Obj *solver_module(VM *v) {
    return named_native_module(v, "solver", names, sizeof names / sizeof *names, SOLVER_BASE);
}

static Val dispatch(VM *v, int id, int argc, Val *a) {
    int which = id - SOLVER_BASE;
    if (which == SOLVE && (argc == 3 || argc == 4)) {
        const char *method = asstr(v, a[0]);
        for (size_t i = 0; i < sizeof methods / sizeof *methods; i++)
            if (!strcmp(method, methods[i]))
                return solve(v, (SolverMethod)i, a[1], a[2], argc == 4 ? a[3] : nil());
        vmerror_kind(v, "domain", "unknown Krylov method");
    }
    if (which == MULTIGRID && (argc == 2 || argc == 3))
        return solver_multigrid(v, a[0], a[1], argc == 3 ? a[2] : nil());
    if (which == REFINE && (argc == 2 || argc == 3))
        return solver_refine(v, a[0], a[1], argc == 3 ? a[2] : nil());
    if (which == LANCZOS && (argc == 3 || argc == 4))
        return solver_lanczos(v, a[0], a[1], a[2], argc == 4 ? a[3] : nil());
    if (which == ARNOLDI && (argc == 3 || argc == 4))
        return solver_arnoldi(v, a[0], a[1], a[2], argc == 4 ? a[3] : nil());
    if ((which == VERIFY || which == VERIFY_EIGEN) && (argc == 3 || argc == 4))
        return solver_verify(v, which - VERIFY, argc, a);
    if (which == VERIFY_SVD && (argc == 4 || argc == 5))
        return solver_verify(v, which - VERIFY, argc, a);
    if (which == REQUIREMENTS && argc == 1)
        return solver_verify(v, 3, argc, a);
    vmerror(v, "invalid solver API arity");
}

Val solver_call(VM *v, int id, int argc, Val *a) {
    v->solver_checkpoint = v->objects;
    Val result = dispatch(v, id, argc, a);
    v->solver_checkpoint = NULL;
    return result;
}
