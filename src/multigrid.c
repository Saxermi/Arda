#include "poisson.h"
#include "solver.h"

#define A(o) ((o)->as.array)

enum { MAX_LEVELS = 32, SMOOTH_STEPS = 3 };

typedef struct {
    int low, high;
    double low_weight, high_weight;
} Interpolation;

typedef struct {
    PoissonGrid grid;
    double *rhs, *solution, *residual;
    Interpolation *transfer[3];
    double restriction_scale;
} Level;

static int grids(VM *v, Level *level, PoissonGrid grid, size_t *vector_elements,
                 size_t *interpolation_count) {
    level[0].grid = grid;
    int count = 1;
    *vector_elements = 2 * grid.count;
    *interpolation_count = 0;
    while (grid.count > 1 && count < MAX_LEVELS) {
        PoissonGrid coarse = {.count = 1};
        double volume_ratio = 1, largest = 0;
        for (int axis = 0; axis < 3; axis++)
            if (grid.extent[axis] > 1)
                largest = fmax(largest, grid.weights[axis]);
        // With point smoothing, leave weak directions on the fine grid.
        // Coarsening the strongest directions reduces their weight on the next level.
        for (int axis = 0; axis < 3; axis++) {
            int old = grid.extent[axis];
            int next = old > 1 && grid.weights[axis] >= 0.25 * largest ? old / 2 : old;
            double ratio = (double)(next + 1) / (old + 1);
            coarse.extent[axis] = next;
            coarse.weights[axis] = grid.weights[axis] * ratio * ratio;
            coarse.diagonal += 2 * coarse.weights[axis];
            coarse.count *= (size_t)next;
            volume_ratio *= ratio;
            *interpolation_count += (size_t)old;
        }
        if (!(coarse.diagonal > 0))
            vmerror_kind(v, "domain", "coarse-grid diagonal underflow");
        level[count - 1].restriction_scale = volume_ratio;
        level[count++].grid = coarse;
        *vector_elements += 3 * coarse.count;
        grid = coarse;
    }
    return count;
}

static void setup(Level *levels, int count, double *vectors, Interpolation *transfers) {
    for (int l = 0; l < count; l++) {
        Level *level = &levels[l];
        size_t n = level->grid.count;
        level->rhs = vectors;
        level->residual = vectors + n;
        vectors += 2 * n;
        if (l) {
            level->solution = vectors;
            vectors += n;
        }
        if (l + 1 == count)
            break;
        for (int axis = 0; axis < 3; axis++) {
            level->transfer[axis] = transfers;
            int old = level->grid.extent[axis], next = levels[l + 1].grid.extent[axis];
            for (int i = 0; i < old; i++) {
                // Positions are measured in coarse-grid intervals from the Dirichlet boundary.
                int64_t numerator = ((int64_t)i + 1) * (next + 1), intervals = (int64_t)old + 1;
                int cell = (int)(numerator / intervals);
                double frac = (double)(numerator % intervals) / (double)intervals;
                transfers[i] = (Interpolation){.low = cell - 1,
                                               .high = cell == next ? -1 : cell,
                                               .low_weight = 1 - frac,
                                               .high_weight = frac};
            }
            transfers += old;
        }
    }
}

static void residual(Level *l) {
    poisson_stencil(&l->grid, l->solution, 1, l->residual, 1);
    for (size_t i = 0; i < l->grid.count; i++)
        l->residual[i] = l->rhs[i] - l->residual[i];
}

static void smooth(Level *l) {
    double factor = (2.0 / 3.0) / l->grid.diagonal;
    for (int step = 0; step < SMOOTH_STEPS; step++) {
        residual(l);
        for (size_t i = 0; i < l->grid.count; i++)
            l->solution[i] += factor * l->residual[i];
    }
}

static void transfer(Level *fine, Level *coarse, bool restrict_to_coarse) {
    int cr = coarse->grid.extent[1], cc = coarse->grid.extent[2];
    int fz = fine->grid.extent[0], fr = fine->grid.extent[1], fc = fine->grid.extent[2];
    for (int z = 0; z < fz; z++) {
        Interpolation zp = fine->transfer[0][z];
        int zids[] = {zp.low, zp.high};
        double zw[] = {zp.low_weight, zp.high_weight};
        for (int i = 0; i < fr; i++) {
            Interpolation ip = fine->transfer[1][i];
            int ids[] = {ip.low, ip.high};
            double iw[] = {ip.low_weight, ip.high_weight};
            size_t at = ((size_t)z * fr + i) * fc;
            for (int oz = 0; oz < 2; oz++) {
                if (zids[oz] < 0 || zw[oz] == 0)
                    continue;
                for (int oi = 0; oi < 2; oi++) {
                    if (ids[oi] < 0 || iw[oi] == 0)
                        continue;
                    size_t row = ((size_t)zids[oz] * cr + ids[oi]) * cc;
                    double weight = zw[oz] * iw[oi];
                    if (restrict_to_coarse) {
                        double *out = coarse->rhs + row;
                        const double *values = fine->residual + at;
                        weight *= fine->restriction_scale;
                        for (int j = 0; j < fc; j++) {
                            Interpolation t = fine->transfer[2][j];
                            double r = values[j] * weight;
                            if (t.low >= 0)
                                out[t.low] += t.low_weight * r;
                            if (t.high >= 0)
                                out[t.high] += t.high_weight * r;
                        }
                    } else {
                        const double *values = coarse->solution + row;
                        double *out = fine->solution + at;
                        for (int j = 0; j < fc; j++) {
                            Interpolation t = fine->transfer[2][j];
                            double correction = t.low >= 0 ? t.low_weight * values[t.low] : 0;
                            if (t.high >= 0)
                                correction += t.high_weight * values[t.high];
                            out[j] += weight * correction;
                        }
                    }
                }
            }
        }
    }
}

static void cycle(Level *level, int count) {
    if (count == 1) {
        level->solution[0] = level->rhs[0] / level->grid.diagonal;
        return;
    }
    Level *coarse = level + 1;
    smooth(level);
    residual(level);
    memset(coarse->rhs, 0, coarse->grid.count * sizeof(double));
    memset(coarse->solution, 0, coarse->grid.count * sizeof(double));
    transfer(level, coarse, true);
    cycle(coarse, count - 1);
    transfer(level, coarse, false);
    smooth(level);
}

Val solver_multigrid(VM *v, Val input, Val rhs, Val settings) {
    if (!isobj(input, O_LINOP) || input.o->as.linop.kind != LINOP_POISSON)
        vmerror_kind(v, "type", "multigrid needs a linop.poisson operator");
    if (input.o->as.linop.grid_rank > 3)
        vmerror_kind(v, "shape", "multigrid supports up to three grid axes");
    SolverOptions opts = solver_options(v, settings);
    if (opts.preconditioner != PC_NONE || solver_has_options(settings, "restart") ||
        solver_has_options(settings, "basis") || solver_has_options(settings, "assume") ||
        solver_has_options(settings, "condition"))
        vmerror_kind(v, "domain", "unsupported geometric multigrid option");
    if (!solver_has_options(settings, "maxiter"))
        opts.max_iterations = 100;
    Level levels[MAX_LEVELS] = {0};
    PoissonGrid grid = poisson_grid(input.o);
    size_t vectors, transfers;
    int count = grids(v, levels, grid, &vectors, &transfers);
    if (vectors > SIZE_MAX / sizeof(double) ||
        transfers > (SIZE_MAX - vectors * sizeof(double)) / sizeof(Interpolation))
        vmerror_kind(v, "memory_budget", "multigrid workspace is not addressable");
    size_t bytes = vectors * sizeof(double) + transfers * sizeof(Interpolation);
    Obj *work = solver_workspace(v, bytes, 0, grid.count * sizeof(double), &opts);
    double *data = (double *)work->as.buffer.data;
    setup(levels, count, data, (Interpolation *)(data + vectors));
    solver_read_vector(v, rhs, grid.count, levels[0].rhs);
    long double norm = linear_norm(grid.count, levels[0].rhs, 1);
    if (!isfinite(norm))
        vmerror_kind(v, "domain", "right-hand-side norm is not representable");
    Obj *x = array_new(v, 1, (int)grid.count, 1);
    levels[0].solution = A(x).data;
    LinopContext context;
    LinearOperator op = linop_open(v, input.o, &context);
    SolverProblem p = {.vm = v,
                       .op = op,
                       .options = opts,
                       .b = levels[0].rhs,
                       .x = A(x).data,
                       .residual = levels[0].residual,
                       .rhs_norm = norm,
                       .trace = opts.trace_limit ? list(v) : NULL};
    if (norm)
        while (p.iterations < opts.max_iterations) {
            cycle(levels, count);
            p.iterations++;
            long double rel = solver_residual(&p);
            solver_trace(&p, rel);
            if (!isfinite(rel) || rel <= opts.tolerance)
                break;
        }
    Val result = solver_report(&p, x, "multigrid", bytes);
    linop_close(v, &context);
    solver_workspace_done(v, work);
    return result;
}
