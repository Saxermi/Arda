#ifndef ARDA_LINEAR_OPERATOR_H
#define ARDA_LINEAR_OPERATOR_H

#include <stddef.h>

/* Callers supply disjoint vectors of the specified logical sizes. */
typedef void (*LinearApply)(const void *context, const double *x, size_t xs, double *y, size_t ys);

typedef struct {
    size_t rows, cols, workspace_bytes;
    const void *context;
    LinearApply apply, apply_transpose;
} LinearOperator;

long double linear_dot(size_t n, const double *a, const double *b);

#endif
