#ifndef ARDA_SPARSE_INTERNAL_H
#define ARDA_SPARSE_INTERNAL_H

#include "arda.h"
#include "linear_operator.h"

size_t sparse_mul_size(VM *v, size_t n, size_t width);
void sparse_budget_check(VM *v, size_t bytes);
void *sparse_buffer(VM *v, Obj *o, size_t n, size_t width, int zero);
Obj *sparse_matrix(VM *v, uint32_t rows, uint32_t cols, int csr);
void sparse_entries(VM *v, Obj *o, size_t n);
void sparse_drop_temporary(VM *v, Obj *temp, Obj *original);
static inline void sparse_begin(VM *v) {
    v->sparse_checkpoint = v->objects;
    v->sparse_active = 1;
}

static inline void sparse_finish(VM *v) {
    v->sparse_active = 0;
    v->sparse_checkpoint = NULL;
}

Obj *sparse_as_csr(VM *v, Obj *o);
size_t sparse_csr_estimate(VM *v, const Sparse *s);
double sparse_csr_at(const Sparse *s, size_t row, uint32_t col);
uint32_t sparse_dimension(VM *v, Val value);
double sparse_finite(VM *v, Val value);
Obj *sparse_builder_new(VM *v, uint32_t rows, uint32_t cols, size_t capacity, size_t limit);
void sparse_builder_reserve(VM *v, Obj *o, size_t extra);
void sparse_builder_write(Obj *o, uint32_t row, uint32_t col, double value);
Obj *sparse_builder_finish(VM *v, Obj *o);

LinearOperator sparse_operator(const Sparse *matrix);
Obj *sparse_product(VM *v, Obj *matrix, Val input, int rank);
Obj *sparse_transpose(VM *v, Obj *matrix);
Obj *sparse_add(VM *v, Obj *left, Obj *right);
Obj *sparse_scale(VM *v, Obj *matrix, double factor);
Obj *sparse_diagonal(VM *v, Obj *matrix);
int sparse_symmetric(const Sparse *matrix);
void sparse_result_field(VM *v, Obj *result, const char *key, Val value);
Val sparse_estimate(VM *v, Obj *matrix);

Obj *sparse_read_mm(VM *v, const char *path);
void sparse_write_mm(VM *v, const char *path, Obj *matrix, int symmetric);

#endif
