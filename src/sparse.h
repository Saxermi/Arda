#ifndef ARDA_SPARSE_H
#define ARDA_SPARSE_H
#include <stddef.h>
#include <stdint.h>

/* Local indices are compact; future distributed global indices are separate.
 * A published matrix is immutable. Workspace is owned only during construction. */
typedef struct {
    uint32_t rows, cols;
    size_t nnz;
    int csr;
    uint32_t *row, *col;
    size_t *offset;
    double *values;
    void *work;
} Sparse;
#endif
