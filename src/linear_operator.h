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

struct Obj;
struct VM;
typedef struct {
    const struct Obj *op;
    struct Obj *csr;
    double *scratch;
} LinopContext;

LinearOperator dense_operator(struct Obj *matrix);
LinearOperator linop_open(struct VM *v, struct Obj *input, LinopContext *context);
void linop_close(struct VM *v, LinopContext *context);
void linop_apply_into(struct VM *v, const LinearOperator *op, struct Obj *x, struct Obj *y,
                      int transpose);

long double linear_dot(size_t n, const double *a, const double *b);

#endif
