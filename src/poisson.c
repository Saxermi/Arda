#include "linear_operator.h"
#include "poisson.h"

PoissonGrid poisson_grid(const Obj *op) {
    PoissonGrid g = {.count = (size_t)op->as.linop.rows};
    for (int i = 0; i < 3; i++) {
        g.extent[i] = op->as.linop.extent[i];
        g.weights[i] = op->as.linop.weights[i];
        g.diagonal += 2 * g.weights[i];
    }
    return g;
}

Val poisson_new(VM *v, Val shape, Val weights) {
    if (!isobj(shape, O_LIST) || shape.o->as.list.n < 1 || shape.o->as.list.n > 3)
        vmerror_kind(v, "shape", "Poisson grid requires one to three positive extents");
    int rank = shape.o->as.list.n;
    if (weights.t != V_NIL && (!isobj(weights, O_LIST) || weights.o->as.list.n != rank))
        vmerror_kind(v, "shape", "Poisson grid weights must match the dimension count");
    Obj *op = newobj(v, O_LINOP);
    size_t count = 1;
    double diagonal = 0;
    for (int i = 0; i < 3; i++) {
        int n = 1;
        double weight = 0;
        if (i >= 3 - rank) {
            int index = i - (3 - rank);
            Val item = shape.o->as.list.a[index];
            if (item.t != V_INT || item.i < 1 || (uint64_t)item.i > ARDA_MAX_ELEMENTS / count)
                vmerror_kind(v, "shape", "Poisson grid exceeds the supported element count");
            n = (int)item.i;
            weight = weights.t == V_NIL ? 1 : asnum(v, weights.o->as.list.a[index]);
            if (!(weight > 0) || !isfinite(weight))
                vmerror_kind(v, "domain", "Poisson weights must be positive and finite");
        }
        count *= (size_t)n;
        diagonal += 2 * weight;
        op->as.linop.extent[i] = n;
        op->as.linop.weights[i] = weight;
    }
    if (!isfinite(diagonal))
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

void poisson_operator_apply(const void *context, const double *x, size_t xs, double *y, size_t ys) {
    PoissonGrid grid = poisson_grid(context);
    poisson_stencil(&grid, x, xs, y, ys);
}
