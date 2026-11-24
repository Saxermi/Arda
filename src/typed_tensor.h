#ifndef ARDA_TYPED_TENSOR_H
#define ARDA_TYPED_TENSOR_H

#include "arda.h"

typedef enum { SC_BOOL, SC_INT, SC_REAL, SC_COMPLEX } ScalarKind;
typedef struct {
    ScalarKind kind;
    int64_t i;
    double re, im;
} TensorScalar;

typedef struct {
    const char *name;
    size_t size;
    ScalarKind kind;
} DTypeInfo;

extern const DTypeInfo dtype_info[DT_COUNT];
DType dtype_parse(VM *v, Val value);
DType dtype_promote(DType a, DType b, Op op);
TensorScalar dtype_scalar(VM *v, Val value);
Val dtype_value(VM *v, TensorScalar value);
TensorScalar dtype_load(DType type, const unsigned char *data);
void dtype_store(VM *v, DType type, unsigned char *data, TensorScalar value);
TensorScalar dtype_binary(VM *v, Op op, DType type, TensorScalar a, TensorScalar b);

TypedTensor typed_tensor(VM *v, Val value);
bool typed_layout_checked(DType dtype, int rank, const int *shape, TypedTensor *result);
TypedTensor typed_layout(VM *v, DType dtype, int rank, const int *shape);
Obj *typed_new(VM *v, DType dtype, int rank, const int *shape);
Obj *typed_wrap(VM *v, TypedTensor layout);
bool typed_contiguous(const TypedTensor *t);
ptrdiff_t typed_offset(const TypedTensor *t, size_t i);
unsigned char *typed_element(const TypedTensor *t, ptrdiff_t pos);
Val tensor_mmap(VM *v, Val path);
Val tensor_read_block(VM *v, Val path, Val start, Val count);
Val tensor_save(VM *v, Val path, Val input, Val budget);

#endif
