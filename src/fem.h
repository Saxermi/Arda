#ifndef ARDA_FEM_H
#define ARDA_FEM_H

#include "solver.h"

typedef struct {
    uint32_t parent, generation;
} FemBranch;

typedef struct {
    uint32_t a, b, slot;
} FemFace;

typedef struct {
    const double *x;
    const uint32_t *cells;
    int nodes, count, dimension, arity;
    const FemBranch *tree;
    const uint32_t *leaves;
    int tree_count;
} FemMesh;

typedef struct {
    double measure, diameter, gradient[3][2];
} FemElement;

typedef struct {
    Val diffusion, reaction;
    size_t workspace;
    int max_nodes, max_elements;
    double fraction;
    bool nvb;
} FemOptions;

typedef struct {
    double *source, *boundary, *diffusion, *reaction;
} FemData;

FemOptions fem_options(VM *v, Val value, bool refinement);
Obj *fem_work(VM *v, size_t bytes, size_t output, size_t limit);
Obj *fem_mesh_wrap(VM *v, Obj *nodes, Obj *elements, int count);
FemElement fem_geometry(VM *v, FemMesh mesh, const uint32_t *cell);
FemData fem_read_data(VM *v, FemMesh mesh, Val source, Val boundary, FemOptions opts, double *work);
void fem_read_field(VM *v, Val value, size_t n, double *out, double lower, bool strict);
double fem_checked(VM *v, double value);
void fem_sort_faces(FemMesh mesh, FemFace *faces);
size_t fem_face_end(VM *v, const FemFace *faces, size_t first, size_t count);
Val fem_darcy_mixed(VM *v, FemMesh mesh, Val *args, Val settings);
uint32_t fem_component(uint32_t *parents, uint32_t node);
void fem_check_faces(VM *v, FemMesh mesh);
Val fem_estimate(VM *v, FemMesh mesh, Val *args, FemOptions opts);
Val fem_refine(VM *v, FemMesh mesh, Val *args, FemOptions opts);

#endif
