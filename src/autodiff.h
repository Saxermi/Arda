#ifndef ARDA_AUTODIFF_H
#define ARDA_AUTODIFF_H

#include "typed_tensor.h"

typedef enum {
    AD_LEAF,
    AD_ADD,
    AD_SUB,
    AD_MUL,
    AD_DIV,
    AD_NEG,
    AD_SIN,
    AD_COS,
    AD_EXP,
    AD_LOG,
    AD_SQRT,
    AD_TANH,
    AD_POW,
    AD_SUM,
    AD_MEAN,
    AD_RESHAPE,
    AD_PERMUTE,
    AD_VIEW,
    AD_SLICE,
    AD_AT,
    AD_MATMUL,
    AD_SPARSE_MATVEC,
    AD_CHECKPOINT,
    AD_CONSUMED
} ADOperation;

typedef struct ADCheckpoint ADCheckpoint;

typedef struct TensorAD {
    Val value, grad, left, right;
    Obj *parents[2], *pattern;
    uint64_t versions[3], batch_examples;
    ADCheckpoint *checkpoint;
    unsigned saved, visit;
    bool active, retain;
    int slot, axis, axes[8];
    ADOperation operation;
} TensorAD;

// This native entry delegates a dynamically scoped call to the VM.
#define AD_NO_GRAD_ID (AD_BASE + 13)

static inline bool ad_disabled(const VM *v) {
    return v->nf && v->frames[v->nf - 1].no_grad;
}

void ad_trace_checkpoint(VM *v, const TensorAD *node);

Val ad_sparse_matvec(VM *v, Obj *pattern, Val values, Val input);
Val ad_primal(Val value);
void ad_validate(VM *v, Val value);
Val ad_unary(VM *v, ADOperation op, Val input, Val exponent);
Val ad_record(VM *v, ADOperation op, Val value, Val left, Val right, int axis, const int *axes);

#endif
