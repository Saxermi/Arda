#ifndef ARDA_FEM_H
#define ARDA_FEM_H

#include "solver.h"

typedef struct {
    const double *x;
    const uint32_t *cells;
    int nodes, count, dimension, arity;
} FemMesh;

typedef struct {
    double measure, diameter, gradient[3][2];
} FemElement;

typedef struct {
    Val diffusion, reaction;
    size_t workspace;
    int max_nodes, max_elements;
    double fraction;
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
void fem_check_faces(VM *v, FemMesh mesh);
Val fem_estimate(VM *v, FemMesh mesh, Val *args, FemOptions opts);
Val fem_refine(VM *v, FemMesh mesh, Val *args, FemOptions opts);

#endif
