#ifndef ARDA_POISSON_H
#define ARDA_POISSON_H

#include "arda.h"

typedef struct {
    int extent[3];
    double weights[3];
    size_t count;
    double diagonal;
} PoissonGrid;

double poisson_diagonal(const Obj *op);
double poisson_neighbor_weight(const Obj *op, size_t lower);
/* Only for operators with at most three grid axes. */
PoissonGrid poisson_grid(const Obj *op);
Val poisson_new(VM *v, Val shape, Val weights);
void poisson_stencil(const PoissonGrid *g, const double *x, size_t xs, double *y, size_t ys);
void poisson_operator_apply(const void *context, const double *x, size_t xs, double *y, size_t ys);

#endif
