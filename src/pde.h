#ifndef ARDA_PDE_H
#define ARDA_PDE_H

#include "arda.h"

long double pde_operator_norm(const Obj *grid);
bool pde_has_dirichlet(const Obj *grid);
void pde_operator_apply(const void *context, const double *x, size_t xs, double *y, size_t ys);

#endif
