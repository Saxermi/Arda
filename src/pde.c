#include "pde.h"
#include "simulation_checkpoint.h"
#include "solver.h"
#include "typed_tensor.h"
#include <float.h>

#define A(o) ((o)->as.array)
#define L(o) ((o)->as.linop)

enum { DIRICHLET, NEUMANN, PERIODIC };
enum { AXES = 3, TILE = 4096 };

typedef struct {
    size_t n, length[AXES], stride[AXES];
    double weight[AXES];
    unsigned char boundary[AXES];
    int rank;
} Grid;

typedef struct {
    int axis;
    const double *lower, *upper;
} GhostFaces;

static Val entry(Val options, const char *name) {
    Entry *e = options.t == V_NIL ? NULL : tablefind(&options.o->as.map.tab, name);
    return e ? e->val : nil();
}

static int known_options(Table *t, const char *const *names, size_t count) {
    int found = 0;
    for (size_t i = 0; i < count; i++)
        found += tablefind(t, names[i]) != NULL;
    return found == t->n;
}

static Val axis_value(VM *v, Val value, int i, int rank) {
    if (!isobj(value, O_LIST))
        return value;
    if (value.o->as.list.n != rank)
        vmerror_kind(v, "shape", "PDE axis options must match the grid rank");
    return value.o->as.list.a[i];
}

static unsigned char boundary_kind(VM *v, Val value) {
    const char *name = value.t == V_NIL ? "dirichlet" : asstr(v, value);
    if (!strcmp(name, "dirichlet"))
        return DIRICHLET;
    if (!strcmp(name, "neumann"))
        return NEUMANN;
    if (!strcmp(name, "periodic"))
        return PERIODIC;
    vmerror_kind(v, "domain", "unknown PDE boundary condition");
}

static Val make_grid(VM *v, Val shape, Val spacing, Val boundary) {
    if (!isobj(shape, O_LIST) || shape.o->as.list.n < 1 || shape.o->as.list.n > AXES)
        vmerror_kind(v, "shape", "PDE grid requires one to three positive extents");
    Obj *out = newobj(v, O_LINOP);
    size_t n = 1;
    int rank = shape.o->as.list.n;
    double weight_sum = 0;
    for (int i = 0; i < rank; i++) {
        Val width = shape.o->as.list.a[i];
        if (width.t != V_INT || width.i <= 0 || (uint64_t)width.i > ARDA_MAX_ELEMENTS / n)
            vmerror_kind(v, "shape", "PDE grid size exceeds the supported element count");
        double h = asnum(v, axis_value(v, spacing, i, rank));
        if (!(h > 0) || !isfinite(h))
            vmerror_kind(v, "domain", "PDE grid spacing must be positive and finite");
        double inverse = 1 / h, weight = inverse * inverse;
        weight_sum += weight;
        if (!(weight > 0) || !isfinite(2 * weight_sum))
            vmerror_kind(v, "domain", "PDE inverse square spacing is not representable");
        n *= (size_t)width.i;
        L(out).extent[i] = (int)width.i;
        L(out).weights[i] = weight;
        L(out).boundary[i] = boundary_kind(v, axis_value(v, boundary, i, rank));
    }
    L(out).kind = LINOP_CARTESIAN;
    L(out).rows = L(out).cols = (int)n;
    L(out).grid_rank = rank;
    return objval(out);
}

static Obj *grid_object(VM *v, Val value) {
    if (!isobj(value, O_LINOP) || L(value.o).kind != LINOP_CARTESIAN)
        vmerror_kind(v, "type", "PDE requires a Cartesian grid created by pde.grid");
    return value.o;
}

bool pde_has_dirichlet(const Obj *grid) {
    for (int i = 0; i < L(grid).grid_rank; i++)
        if (L(grid).boundary[i] == DIRICHLET)
            return true;
    return false;
}

long double pde_operator_norm(const Obj *grid) {
    long double norm = 0;
    for (int i = 0; i < L(grid).grid_rank; i++) {
        int length = L(grid).extent[i];
        unsigned char bc = L(grid).boundary[i];
        int factor = 4;
        if (length == 1)
            factor = bc == DIRICHLET ? 2 : 0;
        else if (length == 2 && bc != PERIODIC)
            factor = bc == DIRICHLET ? 3 : 2;
        norm += factor * (long double)L(grid).weights[i];
    }
    return norm;
}

static void grid_strides(Grid *g) {
    size_t stride = 1;
    for (int i = g->rank; i-- > 0;) {
        g->stride[i] = stride;
        stride *= g->length[i];
    }
    g->n = stride;
}

static Grid grid_layout(const Obj *op) {
    Grid g = {.rank = L(op).grid_rank};
    for (int i = 0; i < g.rank; i++) {
        g.length[i] = (size_t)L(op).extent[i];
        g.weight[i] = L(op).weights[i];
        g.boundary[i] = L(op).boundary[i];
    }
    grid_strides(&g);
    return g;
}

/* Evaluate one axis contribution using zero Dirichlet ghosts, reflected face
 * values for zero Neumann diffusion, or periodic wrapping. */
static void stencil(const Grid *g, const double *x, size_t xs, double *y, size_t ys,
                    double diffusion, const double *courant, const GhostFaces *ghosts) {
    for (size_t start = 0; start < g->n; start += TILE) {
        size_t end = g->n - start < TILE ? g->n : start + TILE;
        for (size_t idx = start; idx < end; idx++)
            y[idx * ys] = 0;
        for (int axis = 0; axis < g->rank; axis++) {
            size_t stride = g->stride[axis], width = g->length[axis];
            unsigned char bc = g->boundary[axis];
            double coefficient = g->weight[axis] * diffusion;
            double drift = courant ? courant[axis] : 0;
            const GhostFaces *faces = ghosts && ghosts->axis == axis ? ghosts : NULL;
            if ((!coefficient && !drift) || (width == 1 && bc != DIRICHLET && !faces))
                continue;
            size_t span = width * stride, wrap = span - stride;
            for (size_t begin = start; begin < end;) {
                size_t position = begin % span;
                bool first = position < stride, last = position >= wrap;
                size_t bound = last ? span : first ? stride : wrap;
                size_t count = bound - position;
                if (count > end - begin)
                    count = end - begin;
                const double *u = x + begin * xs;
                const double *left = !first           ? u - stride * xs
                                     : bc == PERIODIC ? u + wrap * xs
                                     : bc == NEUMANN  ? u
                                                      : NULL;
                const double *right = !last            ? u + stride * xs
                                      : bc == PERIODIC ? u - wrap * xs
                                      : bc == NEUMANN  ? u
                                                       : NULL;
                if (faces && ((first && faces->lower) || (last && faces->upper))) {
                    size_t offset = (begin / span) * stride + position % stride;
                    if (first && faces->lower)
                        left = faces->lower + offset;
                    if (last && faces->upper)
                        right = faces->upper + offset;
                }
                bool closed_in = first && bc == NEUMANN, closed_out = last && bc == NEUMANN;
                double *out = y + begin * ys;
                for (size_t i = 0; i < count; i++) {
                    double a = u[i * xs];
                    double l = left ? left[i * xs] : 0, r = right ? right[i * xs] : 0;
                    double flux = coefficient ? coefficient * ((a - l) + (a - r)) : 0;
                    if (drift) {
                        double incoming = closed_in ? 0 : drift * (drift > 0 ? l : a);
                        double outgoing = closed_out ? 0 : drift * (drift > 0 ? a : r);
                        flux += outgoing - incoming;
                    }
                    out[i * ys] += flux;
                }
                begin += count;
            }
        }
    }
}

void pde_operator_apply(const void *context, const double *x, size_t xs, double *y, size_t ys) {
    Grid g = grid_layout(context);
    stencil(&g, x, xs, y, ys, 1, NULL, NULL);
}

typedef struct {
    Grid grid;
    int axis, index, parts, lower, upper;
    size_t origin, face_size;
} Subdomain;

static int partition_int(VM *v, Val value, int min, int max, const char *name) {
    if (value.t != V_INT || value.i < min || value.i > max)
        vmerror_kind(v, "domain", "PDE partition %s must be an integer in %d..%d", name, min, max);
    return (int)value.i;
}

static Subdomain subdomain(VM *v, Val *a) {
    Subdomain part = {.grid = grid_layout(grid_object(v, a[0])), .lower = -1, .upper = -1};
    part.axis = partition_int(v, a[1], 0, part.grid.rank - 1, "axis");
    int length = (int)part.grid.length[part.axis];
    part.parts = partition_int(v, a[2], 1, length, "parts");
    part.index = partition_int(v, a[3], 0, part.parts - 1, "index");
    int width = length / part.parts, extra = length % part.parts;
    part.origin = (size_t)width * part.index + (part.index < extra ? part.index : extra);
    part.grid.length[part.axis] = (size_t)(width + (part.index < extra));
    grid_strides(&part.grid);
    part.face_size = part.grid.n / part.grid.length[part.axis];
    if (part.parts > 1) {
        bool periodic = part.grid.boundary[part.axis] == PERIODIC;
        part.lower = part.index > 0 ? part.index - 1 : periodic ? part.parts - 1 : -1;
        part.upper = part.index + 1 < part.parts ? part.index + 1 : periodic ? 0 : -1;
    }
    return part;
}

static Val describe_subdomain(VM *v, Subdomain part) {
    Obj *result = newobj(v, O_MAP), *origin = list(v), *shape = list(v), *face = list(v);
    for (int i = 0; i < part.grid.rank; i++) {
        Val extent = integer((int64_t)part.grid.length[i]);
        append(v, origin, integer(i == part.axis ? (int64_t)part.origin : 0));
        append(v, shape, extent);
        if (i != part.axis)
            append(v, face, extent);
    }
    sparse_result_field(v, result, "origin", objval(origin));
    sparse_result_field(v, result, "shape", objval(shape));
    sparse_result_field(v, result, "face_shape", objval(face));
    sparse_result_field(v, result, "face_elements", integer((int64_t)part.face_size));
    sparse_result_field(v, result, "lower", part.lower < 0 ? nil() : integer(part.lower));
    sparse_result_field(v, result, "upper", part.upper < 0 ? nil() : integer(part.upper));
    return objval(result);
}

static Val subdomain_face(VM *v, Subdomain part, Val value, Val side) {
    const char *name = asstr(v, side);
    bool upper = !strcmp(name, "upper");
    if (!upper && strcmp(name, "lower"))
        vmerror_kind(v, "domain", "PDE face must be lower or upper");
    TypedTensor state = typed_tensor(v, value);
    if (state.rank != 1 || state.dtype != DT_F64 || state.n != part.grid.n)
        vmerror_kind(v, "shape", "PDE subdomain state must be a matching Float64 vector");
    Obj *out = array_new(v, 1, (int)part.face_size, 1);
    size_t stride = part.grid.stride[part.axis], span = part.grid.length[part.axis] * stride;
    size_t base = upper ? span - stride : 0;
    for (size_t i = 0; i < part.face_size; i++) {
        size_t position = (i / stride) * span + base + i % stride;
        double value;
        memcpy(&value, typed_element(&state, typed_offset(&state, position)), sizeof value);
        if (!isfinite(value))
            vmerror_kind(v, "domain", "PDE face values must be finite");
        A(out).data[i] = value;
    }
    return objval(out);
}

static Val apply_subdomain(VM *v, Subdomain part, Val state, Val lower, Val upper) {
    bool has_lower = part.lower >= 0, has_upper = part.upper >= 0;
    if (has_lower != (lower.t != V_NIL) || has_upper != (upper.t != V_NIL))
        vmerror_kind(v, "shape", "PDE ghost faces must match the subdomain neighbors");
    size_t work = part.grid.n + (has_lower + has_upper) * part.face_size;
    SolverOptions budget = {.workspace_limit = SIZE_MAX};
    size_t bytes = part.grid.n * sizeof(double);
    Obj *buffer = solver_workspace(v, work * sizeof(double), 0, bytes, &budget);
    double *input = (double *)buffer->as.buffer.data;
    solver_read_vector(v, state, part.grid.n, input);
    GhostFaces faces = {.axis = part.axis};
    double *ghost = input + part.grid.n;
    if (has_lower) {
        solver_read_vector(v, lower, part.face_size, ghost);
        faces.lower = ghost;
        ghost += part.face_size;
    }
    if (has_upper) {
        solver_read_vector(v, upper, part.face_size, ghost);
        faces.upper = ghost;
    }
    Obj *out = array_new(v, 1, (int)part.grid.n, 1);
    stencil(&part.grid, input, 1, A(out).data, 1, 1, NULL, has_lower || has_upper ? &faces : NULL);
    for (size_t i = 0; i < part.grid.n; i++)
        if (!isfinite(A(out).data[i]))
            vmerror_kind(v, "pde_nonfinite", "nonfinite subdomain stencil result");
    solver_workspace_done(v, buffer);
    return objval(out);
}

static Val poisson(VM *v, Val op, Val rhs, Val settings) {
    Obj *grid = grid_object(v, op);
    if (settings.t != V_NIL && !isobj(settings, O_MAP))
        vmerror_kind(v, "type", "PDE Poisson options must be a map");
    static const char *const names[] = {"tol", "maxiter", "trace", "workspace_bytes"};
    if (isobj(settings, O_MAP) && !known_options(&settings.o->as.map.tab, names, 4))
        vmerror_kind(v, "domain", "unknown PDE Poisson option");
    bool fixed = pde_has_dirichlet(grid);
    SolverOptions opts = solver_options(v, settings);
    size_t n = (size_t)L(grid).rows;
    Obj *options = newobj(v, O_MAP);
    for (size_t i = 0; i < 4; i++) {
        Val value = entry(settings, names[i]);
        if (value.t != V_NIL)
            sparse_result_field(v, options, names[i], value);
    }
    if (!fixed) {
        Obj *check = solver_workspace(v, n * sizeof(double), 0, 0, &opts);
        double *b = (double *)check->as.buffer.data;
        solver_read_vector(v, rhs, n, b);
        long double sum = 0, magnitude = 0;
        for (size_t i = 0; i < n; i++) {
            sum += b[i];
            magnitude += fabsl(b[i]);
        }
        if (!isfinite(sum) || !isfinite(magnitude) || fabsl(sum) > 64 * DBL_EPSILON * magnitude)
            vmerror_kind(v, "pde_compatibility",
                         "Poisson with periodic or Neumann boundaries needs a zero-mean rhs");
        solver_workspace_done(v, check);
        Obj *assume = newobj(v, O_MAP);
        sparse_result_field(v, assume, "positive_definite", boolean(1));
        sparse_result_field(v, options, "assume", objval(assume));
    }
    Obj *result = solver_solve_cg(v, op, rhs, objval(options)).o;
    if (!fixed) {
        Obj *solution = tablefind(&result->as.map.tab, "solution")->val.o;
        double *x = A(solution).data;
        long double mean = 0;
        for (size_t i = 0; i < n; i++)
            mean += (long double)x[i] / n;
        for (size_t i = 0; i < n; i++)
            x[i] -= (double)mean;
        Val verify_args[] = {op, rhs, objval(solution)};
        Obj *verified = solver_verify(v, 0, 3, verify_args).o;
        double residual = tablefind(&verified->as.map.tab, "residual")->val.n;
        bool converged = residual <= opts.tolerance;
        bool previously = tablefind(&result->as.map.tab, "converged")->val.n != 0;
        sparse_result_field(v, result, "residual", number(residual));
        sparse_result_field(v, result, "converged", boolean(converged));
        if (converged || previously) {
            sparse_result_field(v, result, "reason",
                                objval(string(v, converged ? "converged" : "gauge_residual")));
            sparse_result_field(v, result, "breakdown", boolean(0));
        }
    }
    sparse_result_field(v, result, "gauge",
                        objval(string(v, fixed ? "fixed_boundary" : "zero_mean")));
    return objval(result);
}

typedef struct {
    SolverOptions budget;
    int max_steps;
    double duration, dt;
    bool checkpoint;
} TimeGrid;

static TimeGrid time_grid(VM *v, Val time, Val step, Val settings, bool heat) {
    if (settings.t != V_NIL && !isobj(settings, O_MAP))
        vmerror_kind(v, "type", "PDE time options must be a map");
    static const char *const names[] = {"max_steps", "workspace_bytes", "checkpoint"};
    if (isobj(settings, O_MAP) && !known_options(&settings.o->as.map.tab, names, heat ? 3 : 2))
        vmerror_kind(v, "domain", "unknown PDE time option");
    double duration = asnum(v, time), dt = asnum(v, step);
    if (!(duration >= 0 && dt > 0) || !isfinite(duration) || !isfinite(dt))
        vmerror_kind(v, "domain",
                     "PDE duration and step must be finite with duration >= 0 and step > 0");
    Val cap = entry(settings, "max_steps"), bytes = entry(settings, "workspace_bytes");
    Val checkpoint = entry(settings, "checkpoint");
    if (checkpoint.t != V_NIL && checkpoint.t != V_BOOL)
        vmerror_kind(v, "type", "PDE checkpoint option must be Boolean");
    if (cap.t != V_NIL && (cap.t != V_INT || cap.i < 1 || cap.i > INT_MAX))
        vmerror_kind(v, "domain", "PDE max_steps must be a positive integer within Int32");
    int max_steps = cap.t == V_NIL ? 100000 : (int)cap.i;
    if (duration / dt > max_steps && (double)max_steps * dt < duration)
        vmerror_kind(v, "pde_limit", "PDE time step limit");
    if (bytes.t != V_NIL && (bytes.t != V_INT || bytes.i < 0 || (uint64_t)bytes.i > SIZE_MAX))
        vmerror_kind(v, "domain", "PDE workspace_bytes must be a nonnegative integer");
    SolverOptions budget = {.workspace_limit =
                                bytes.t == V_NIL ? 256u * 1024u * 1024u : (size_t)bytes.i};
    return (TimeGrid){.budget = budget,
                      .max_steps = max_steps,
                      .duration = duration,
                      .dt = dt,
                      .checkpoint = checkpoint.t == V_BOOL && checkpoint.n != 0};
}

static double next_time(VM *v, TimeGrid t, int count, double now) {
    if (count >= t.max_steps)
        vmerror_kind(v, "pde_limit", "PDE time step limit");
    double candidate = fmin(t.duration, (count + 1.0) * t.dt);
    if (!(candidate > now))
        vmerror_kind(v, "pde_limit", "PDE time step is not representable");
    return candidate;
}

static long double grid_weights(const Grid *g) {
    long double sum = 0;
    for (int i = 0; i < g->rank; i++)
        if (g->length[i] != 1 || g->boundary[i] == DIRICHLET)
            sum += g->weight[i];
    return sum;
}

static Val time_result(VM *v, Obj *x, int steps, double time, double cfl) {
    Obj *result = newobj(v, O_MAP);
    sparse_result_field(v, result, "solution", objval(x));
    sparse_result_field(v, result, "time", number(time));
    sparse_result_field(v, result, "steps", integer(steps));
    sparse_result_field(v, result, "cfl", number(cfl));
    return objval(result);
}

static void check_cfl(VM *v, long double cfl) {
    if (!isfinite(cfl) || cfl > 1 + 4 * DBL_EPSILON)
        vmerror_kind(v, "pde_stability", "PDE step exceeds the explicit CFL bound");
}

static Val transport(VM *v, Obj *grid, Val initial, Val velocity, Val diffusion, TimeGrid t) {
    Grid g = grid_layout(grid);
    double k = asnum(v, diffusion);
    if (!(k >= 0) || !isfinite(k))
        vmerror_kind(v, "domain", "diffusivity must be nonnegative and finite");
    double drift[AXES] = {0};
    long double rate = 2.0L * k * grid_weights(&g);
    if (velocity.t != V_NIL)
        for (int i = 0; i < g.rank; i++) {
            double speed = asnum(v, axis_value(v, velocity, i, g.rank));
            drift[i] = speed * sqrt(g.weight[i]);
            if (!isfinite(drift[i]))
                vmerror_kind(v, "domain", "advection rate must be finite");
            if (g.length[i] == 1 && g.boundary[i] != DIRICHLET)
                drift[i] = 0;
            rate += fabs(drift[i]);
        }
    long double cfl = t.duration ? rate * fmin(t.duration, t.dt) : 0;
    check_cfl(v, cfl);
    size_t bytes = g.n * sizeof(double);
    Obj *work = solver_workspace(v, t.duration ? bytes : 0, 0, bytes, &t.budget);
    Obj *out = array_new(v, 1, (int)g.n, 1);
    double *x = A(out).data, *next = (double *)work->as.buffer.data;
    solver_read_vector(v, initial, g.n, x);
    int count = 0;
    double time = 0;
    while (time < t.duration) {
        double end = next_time(v, t, count, time), dt = end - time;
        stencil(&g, x, 1, next, 1, k, velocity.t == V_NIL ? NULL : drift, NULL);
        for (size_t i = 0; i < g.n; i++) {
            next[i] = x[i] - dt * next[i];
            if (!isfinite(next[i]))
                vmerror_kind(v, "pde_nonfinite", "nonfinite transport state");
        }
        double *previous = x;
        x = next;
        next = previous;
        time = end;
        count++;
    }
    if (x != A(out).data)
        memcpy(A(out).data, x, bytes);
    solver_workspace_done(v, work);
    return time_result(v, out, count, time, (double)cfl);
}

static Val heat(VM *v, Obj *grid, Val initial, Val diffusivity, double start, TimeGrid t) {
    double end = start + t.duration;
    if (!isfinite(end) || (t.duration && end <= start))
        vmerror_kind(v, "pde_limit", "heat continuation time is not representable");
    Val result = transport(v, grid, initial, nil(), diffusivity, t);
    if (start != 0)
        sparse_result_field(v, result.o, "time", number(end));
    if (t.checkpoint) {
        Obj *snapshot = tensor_copy(v, entry(result, "solution").o);
        A(snapshot).readonly = true;
        Obj *record = newobj(v, O_MAP);
        sparse_result_field(v, record, "version", integer(1));
        sparse_result_field(v, record, "grid", objval(grid));
        sparse_result_field(v, record, "state", objval(snapshot));
        sparse_result_field(v, record, "time", number(end));
        sparse_result_field(v, record, "dt", number(t.dt));
        sparse_result_field(v, record, "diffusivity", number(asnum(v, diffusivity)));
        sparse_result_field(v, result.o, "checkpoint", objval(record));
    }
    return result;
}

enum {
    HEAT_TIME,
    HEAT_STEP,
    HEAT_DIFFUSIVITY,
    HEAT_RANK,
    HEAT_AXES,
    HEAT_WORDS = HEAT_AXES + 3 * AXES
};
_Static_assert((int)HEAT_WORDS <= (int)CHECKPOINT_WORDS, "heat checkpoint metadata");

static void heat_metadata(VM *v, const uint64_t words[CHECKPOINT_WORDS]) {
    double time = checkpoint_double(words[HEAT_TIME]), dt = checkpoint_double(words[HEAT_STEP]),
           diffusivity = checkpoint_double(words[HEAT_DIFFUSIVITY]);
    if (!(time >= 0 && dt > 0 && diffusivity >= 0) || !isfinite(time) || !isfinite(dt) ||
        !isfinite(diffusivity) || words[HEAT_RANK] < 1 || words[HEAT_RANK] > AXES)
        vmerror_kind(v, "checkpoint_format", "invalid heat checkpoint constants or rank");
    for (size_t i = HEAT_AXES + 3 * (size_t)words[HEAT_RANK]; i < CHECKPOINT_WORDS; i++)
        if (words[i])
            vmerror_kind(v, "checkpoint_format", "invalid reserved heat checkpoint metadata");
}

static Val heat_save(VM *v, Val path, Val record, Val budget) {
    if (!isobj(record, O_MAP))
        vmerror_kind(v, "checkpoint_format", "heat checkpoint must be a map");
    Val version = entry(record, "version");
    if (version.t != V_INT || version.i != 1)
        vmerror_kind(v, "checkpoint_format", "invalid heat checkpoint version");
    Obj *grid = grid_object(v, entry(record, "grid"));
    Val state = entry(record, "state");
    if (!isobj(state, O_ARRAY) || A(state.o).rank != 1 || A(state.o).n != (size_t)L(grid).rows)
        vmerror_kind(v, "checkpoint_format", "heat checkpoint state has the wrong shape");
    uint64_t words[CHECKPOINT_WORDS] = {
        checkpoint_real(v, entry(record, "time")),
        checkpoint_real(v, entry(record, "dt")),
        checkpoint_real(v, entry(record, "diffusivity")),
        (uint64_t)L(grid).grid_rank,
    };
    for (int i = 0; i < L(grid).grid_rank; i++) {
        words[HEAT_AXES + 3 * i] = (uint64_t)L(grid).extent[i];
        words[HEAT_AXES + 3 * i + 1] = checkpoint_real(v, number(L(grid).weights[i]));
        words[HEAT_AXES + 3 * i + 2] = L(grid).boundary[i];
    }
    heat_metadata(v, words);
    return checkpoint_save_state(v, path, CHECKPOINT_HEAT, state, words, budget);
}

static Val heat_resume(VM *v, Val path, Val duration, Val settings) {
    uint64_t words[CHECKPOINT_WORDS];
    Val state = checkpoint_load_state(v, path, CHECKPOINT_HEAT, words);
    heat_metadata(v, words);
    Obj *grid = newobj(v, O_LINOP);
    L(grid).kind = LINOP_CARTESIAN;
    L(grid).grid_rank = (int)words[HEAT_RANK];
    size_t n = 1;
    double weight_sum = 0;
    for (int i = 0; i < L(grid).grid_rank; i++) {
        uint64_t width = words[HEAT_AXES + 3 * i], boundary = words[HEAT_AXES + 3 * i + 2];
        double weight = checkpoint_double(words[HEAT_AXES + 3 * i + 1]);
        weight_sum += weight;
        if (!width || width > ARDA_MAX_ELEMENTS / n || boundary > PERIODIC || !(weight > 0) ||
            !isfinite(2 * weight_sum))
            vmerror_kind(v, "checkpoint_format", "invalid heat checkpoint grid");
        L(grid).extent[i] = (int)width;
        L(grid).weights[i] = weight;
        L(grid).boundary[i] = (unsigned char)boundary;
        n *= (size_t)width;
    }
    if (!isobj(state, O_ARRAY) || A(state.o).rank != 1 || A(state.o).n != n)
        vmerror_kind(v, "checkpoint_format", "heat checkpoint state has the wrong shape");
    L(grid).rows = L(grid).cols = (int)n;
    TimeGrid t =
        time_grid(v, duration, number(checkpoint_double(words[HEAT_STEP])), settings, true);
    Val result = heat(v, grid, state, number(checkpoint_double(words[HEAT_DIFFUSIVITY])),
                      checkpoint_double(words[HEAT_TIME]), t);
    tensor_buffer_discard(v, A(state.o).buffer);
    return result;
}

static Val wave(VM *v, Obj *grid, Val initial, Val initial_speed, Val speed, TimeGrid t) {
    Grid g = grid_layout(grid);
    double c = asnum(v, speed), squared = c * c;
    if (!(c >= 0) || !isfinite(squared))
        vmerror_kind(v, "domain", "wave speed must be nonnegative with a representable square");
    long double cfl = t.duration ? c * sqrtl(grid_weights(&g)) * fmin(t.duration, t.dt) : 0;
    check_cfl(v, cfl);
    size_t bytes = g.n * sizeof(double);
    Obj *work = solver_workspace(v, t.duration ? bytes : 0, 0, 2 * bytes, &t.budget);
    Obj *out = array_new(v, 1, (int)g.n, 1), *velocity = array_new(v, 1, (int)g.n, 1);
    double *x = A(out).data, *p = A(velocity).data, *a = (double *)work->as.buffer.data;
    solver_read_vector(v, initial, g.n, x);
    solver_read_vector(v, initial_speed, g.n, p);
    int count = 0;
    double time = 0;
    if (t.duration)
        stencil(&g, x, 1, a, 1, squared, NULL, NULL);
    while (time < t.duration) {
        double end = next_time(v, t, count, time), dt = end - time;
        for (size_t i = 0; i < g.n; i++) {
            p[i] -= (dt / 2) * a[i];
            x[i] += dt * p[i];
            if (!isfinite(x[i]))
                vmerror_kind(v, "pde_nonfinite", "nonfinite wave displacement");
        }
        stencil(&g, x, 1, a, 1, squared, NULL, NULL);
        for (size_t i = 0; i < g.n; i++) {
            p[i] -= (dt / 2) * a[i];
            if (!isfinite(p[i]))
                vmerror_kind(v, "pde_nonfinite", "nonfinite wave velocity");
        }
        time = end;
        count++;
    }
    solver_workspace_done(v, work);
    Val result = time_result(v, out, count, time, (double)cfl);
    sparse_result_field(v, result.o, "velocity", objval(velocity));
    return result;
}

enum { GRID, POISSON, HEAT, WAVE, TRANSPORT, GRID_SIZE, PARTITION, FACE, SUBDOMAIN, SAVE, RESUME };
static const char *const names[] = {"grid",
                                    "poisson",
                                    "heat",
                                    "wave",
                                    "advection_diffusion",
                                    "grid_size",
                                    "partition",
                                    "subdomain_face",
                                    "apply_subdomain",
                                    "heat_save",
                                    "heat_resume"};

const char *pde_name(int id) {
    int index = id - PDE_BASE;
    return index >= 0 && (size_t)index < sizeof names / sizeof *names ? names[index] : NULL;
}

Obj *pde_module(VM *v) {
    return named_native_module(v, "pde", names, sizeof names / sizeof *names, PDE_BASE);
}

static Val run(VM *v, int which, int argc, Val *a) {
    if (which == GRID_SIZE && argc == 1)
        return integer((int64_t)grid_object(v, a[0])->as.linop.rows);
    if (which == GRID && (argc == 2 || argc == 3))
        return make_grid(v, a[0], a[1], argc == 3 ? a[2] : nil());
    if (which == POISSON && (argc == 2 || argc == 3))
        return poisson(v, a[0], a[1], argc == 3 ? a[2] : nil());
    if (which == HEAT && (argc == 5 || argc == 6))
        return heat(v, grid_object(v, a[0]), a[1], a[2], 0,
                    time_grid(v, a[3], a[4], argc == 6 ? a[5] : nil(), true));
    if (which == WAVE && (argc == 6 || argc == 7))
        return wave(v, grid_object(v, a[0]), a[1], a[2], a[3],
                    time_grid(v, a[4], a[5], argc == 7 ? a[6] : nil(), false));
    if (which == TRANSPORT && (argc == 6 || argc == 7))
        return transport(v, grid_object(v, a[0]), a[1], a[2], a[3],
                         time_grid(v, a[4], a[5], argc == 7 ? a[6] : nil(), false));
    if (which == PARTITION && argc == 4)
        return describe_subdomain(v, subdomain(v, a));
    if (which == FACE && argc == 6)
        return subdomain_face(v, subdomain(v, a), a[4], a[5]);
    if (which == SUBDOMAIN && argc == 7)
        return apply_subdomain(v, subdomain(v, a), a[4], a[5], a[6]);
    if (which == SAVE && (argc == 2 || argc == 3))
        return heat_save(v, a[0], a[1], argc == 3 ? a[2] : nil());
    if (which == RESUME && (argc == 2 || argc == 3))
        return heat_resume(v, a[0], a[1], argc == 3 ? a[2] : nil());
    vmerror(v, "invalid PDE API arity");
}

Val pde_call(VM *v, int id, int argc, Val *args) {
    v->solver_checkpoint = v->objects;
    Val result = run(v, id - PDE_BASE, argc, args);
    v->solver_checkpoint = NULL;
    return result;
}
