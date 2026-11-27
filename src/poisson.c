#include "linear_operator.h"
#include "poisson.h"

double poisson_diagonal(const Obj *op) {
    double diagonal = 0;
    for (int axis = 0; axis < op->as.linop.grid_rank; axis++)
        diagonal += 2 * op->as.linop.weights[axis];
    return diagonal;
}

double poisson_neighbor_weight(const Obj *op, size_t lower) {
    // Adjacent flattened entries can be neighbors along the last nonsingleton axis.
    for (int axis = op->as.linop.grid_rank; axis-- > 0;) {
        size_t length = (size_t)op->as.linop.extent[axis];
        if (length > 1)
            return lower % length == length - 1 ? 0 : op->as.linop.weights[axis];
    }
    return 0;
}

PoissonGrid poisson_grid(const Obj *op) {
    PoissonGrid g = {
        .extent = {1, 1, 1}, .count = (size_t)op->as.linop.rows, .diagonal = poisson_diagonal(op)};
    int rank = op->as.linop.grid_rank;
    for (int i = 0; i < rank; i++) {
        g.extent[i + 3 - rank] = op->as.linop.extent[i];
        g.weights[i + 3 - rank] = op->as.linop.weights[i];
    }
    return g;
}

Val poisson_new(VM *v, Val shape, Val weights) {
    if (!isobj(shape, O_LIST) || shape.o->as.list.n < 1 || shape.o->as.list.n > 8)
        vmerror_kind(v, "shape", "Poisson grid requires one to eight positive extents");
    int rank = shape.o->as.list.n;
    if (weights.t != V_NIL && (!isobj(weights, O_LIST) || weights.o->as.list.n != rank))
        vmerror_kind(v, "shape", "Poisson grid weights must match the dimension count");
    Obj *op = newobj(v, O_LINOP);
    op->as.linop.grid_rank = rank;
    size_t count = 1;
    for (int i = 0; i < rank; i++) {
        Val item = shape.o->as.list.a[i];
        if (item.t != V_INT || item.i < 1 || (uint64_t)item.i > ARDA_MAX_ELEMENTS / count)
            vmerror_kind(v, "shape", "Poisson grid exceeds the supported element count");
        double weight = weights.t == V_NIL ? 1 : asnum(v, weights.o->as.list.a[i]);
        if (!(weight > 0) || !isfinite(weight))
            vmerror_kind(v, "domain", "Poisson weights must be positive and finite");
        count *= (size_t)item.i;
        op->as.linop.extent[i] = (int)item.i;
        op->as.linop.weights[i] = weight;
    }
    if (!isfinite(poisson_diagonal(op)))
        vmerror_kind(v, "domain", "Poisson diagonal must be finite");
    op->as.linop.kind = LINOP_POISSON;
    op->as.linop.rows = op->as.linop.cols = (int)count;
    return objval(op);
}

void poisson_stencil(const PoissonGrid *g, const double *x, size_t xs, double *y, size_t ys) {
    int layers = g->extent[0], rows = g->extent[1], cols = g->extent[2];
    size_t plane = (size_t)rows * cols;
    double wz = g->weights[0], wy = g->weights[1], wx = g->weights[2];
    for (int z = 0; z < layers; z++) {
        for (int i = 0; i < rows; i++) {
            size_t p = (size_t)z * plane + (size_t)i * cols;
            const double *up = z ? x + (p - plane) * xs : NULL;
            const double *down = z + 1 < layers ? x + (p + plane) * xs : NULL;
            const double *prev = i ? x + (p - cols) * xs : NULL;
            const double *next = i + 1 < rows ? x + (p + cols) * xs : NULL;
            const double *row = x + p * xs;
            for (int j = 0; j < cols; j++) {
                size_t k = (size_t)j * xs;
                double value = g->diagonal * row[k];
                if (up)
                    value -= wz * up[k];
                if (down)
                    value -= wz * down[k];
                if (prev)
                    value -= wy * prev[k];
                if (next)
                    value -= wy * next[k];
                if (j)
                    value -= wx * row[k - xs];
                if (j + 1 < cols)
                    value -= wx * row[k + xs];
                y[(p + j) * ys] = value;
            }
        }
    }
}

static void poisson_stencil_nd(const Obj *op, const double *x, size_t xs, double *y, size_t ys) {
    size_t count = (size_t)op->as.linop.rows;
    double diagonal = poisson_diagonal(op);
    for (size_t i = 0; i < count; i++)
        y[i * ys] = diagonal * x[i * xs];

    // Traverse contiguous spans per axis, skipping missing Dirichlet neighbors.
    size_t stride = count;
    for (int axis = 0; axis < op->as.linop.grid_rank; axis++) {
        size_t length = (size_t)op->as.linop.extent[axis], span = stride;
        stride /= length;
        if (length == 1)
            continue;
        double weight = op->as.linop.weights[axis];
        for (size_t base = 0; base < count; base += span) {
            for (size_t i = base; i < base + span - stride; i++)
                y[i * ys] -= weight * x[(i + stride) * xs];
            for (size_t i = base + stride; i < base + span; i++)
                y[i * ys] -= weight * x[(i - stride) * xs];
        }
    }
}

void poisson_operator_apply(const void *context, const double *x, size_t xs, double *y, size_t ys) {
    const Obj *op = context;
    if (op->as.linop.grid_rank <= 3) {
        PoissonGrid grid = poisson_grid(op);
        poisson_stencil(&grid, x, xs, y, ys);
    } else {
        poisson_stencil_nd(op, x, xs, y, ys);
    }
}
