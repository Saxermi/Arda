#ifndef ARDA_COMPUTE_OPS_H
#define ARDA_COMPUTE_OPS_H

#include "typed_tensor.h"

enum { COMPUTE_TILE_DEFAULT = 64, COMPUTE_TILE_MAX = 256 };
typedef struct {
    TypedTensor a, b, output;
} MatmulProblem;

typedef enum { FUSE_AXPY, FUSE_MUL_ADD, FUSE_PRODUCTS } Fusion;

MatmulProblem compute_matmul_problem(VM *v, Val a, Val b);
int compute_tile_size(VM *v, Val value);
size_t compute_tiled_workspace(const MatmulProblem *p, int block);
int compute_tiled_team(const MatmulProblem *p, int block, int threads, bool pack);
int compute_tiled_into(const MatmulProblem *p, int block, int threads, double *out,
                       double *workspace);
Val compute_tiled(VM *v, Val a, Val b, int block);
Val compute_fusion(VM *v, Fusion kind, const Val *args);
Val compute_map_reduce(VM *v, Val map, Val input);

#endif
