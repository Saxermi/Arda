#ifndef ARDA_SOLVER_H
#define ARDA_SOLVER_H

#include "linear_operator.h"
#include "sparse_internal.h"

typedef enum {
    SOLVER_CG,
    SOLVER_MINRES,
    SOLVER_GMRES,
    SOLVER_BICGSTAB,
    SOLVER_LSQR,
    SOLVER_LSMR
} SolverMethod;
typedef enum { PC_NONE, PC_JACOBI, PC_BLOCK_JACOBI, PC_ILU0, PC_IC0 } PreconditionerKind;
typedef enum {
    STOP_LIMIT,
    STOP_CONVERGED,
    STOP_LEAST_SQUARES,
    STOP_BREAKDOWN,
    STOP_CURVATURE,
    STOP_STAGNATION
} SolverStop;

typedef struct {
    double tolerance;
    int max_iterations, restart, trace_limit, basis;
    size_t workspace_limit;
    PreconditionerKind preconditioner;
    bool symmetric, positive_definite, condition;
} SolverOptions;

typedef struct {
    PreconditionerKind kind;
    const Sparse *csr;
    size_t n;
    double *factors;
    size_t *diagonal;
} Preconditioner;

typedef struct {
    VM *vm;
    LinearOperator op;
    SolverOptions options;
    Preconditioner preconditioner;
    double *b, *x, *residual, *normal;
    long double rhs_norm, normal_rhs_norm;
    Obj *trace;
    int iterations;
    SolverStop stop;
    bool least_squares;
} SolverProblem;

extern const char *const preconditioner_names[];
SolverOptions solver_options(VM *v, Val map);
bool solver_has_options(Val map, const char *key);
Obj *solver_operator(VM *v, Val value);
void solver_read_vector(VM *v, Val value, size_t length, double *out);
void solver_check_symmetric(VM *v, Obj *input, const LinopContext *context,
                            const SolverOptions *options, bool positive);
Obj *solver_workspace(VM *v, size_t work, size_t operator_workspace, size_t output,
                      const SolverOptions *options);
void solver_workspace_done(VM *v, Obj *buffer);
void solver_discard_since(VM *v, Obj *checkpoint);
void solver_trace(SolverProblem *p, long double estimate);
long double solver_residual(SolverProblem *p);
long double solver_normal_residual(SolverProblem *p);
Val solver_report(SolverProblem *p, Obj *x, const char *method, size_t bytes);
Val solver_result(SolverProblem *p, Obj *x, const char *method, size_t bytes, long double residual,
                  long double normal);
size_t preconditioner_bytes(VM *v, Obj *input, size_t n, PreconditionerKind kind);
Preconditioner preconditioner_setup(VM *v, Obj *input, const LinopContext *context, size_t n,
                                    PreconditionerKind kind, unsigned char *memory);
void preconditioner_apply(const Preconditioner *pc, const double *x, double *y);
bool solver_update(size_t n, double *x, double alpha, const double *direction);
size_t krylov_elements(SolverMethod method, size_t m, size_t n, int restart);
long double krylov_orthogonalize(size_t n, const double *basis, int columns, double *vector,
                                 double *coefficients);
long double solver_eigen_residual(LinearOperator op, double re, double im, const double *x,
                                  const double *y, long double norm, double *ax, double *ay);
void krylov_run(SolverProblem *p, SolverMethod method, double *work);
Val solver_multigrid(VM *v, Val input, Val rhs, Val settings);
Val solver_refine(VM *v, Val input, Val rhs, Val settings);
Val solver_arnoldi(VM *v, Val input, Val seed, Val wanted, Val settings);
Val solver_lanczos(VM *v, Val input, Val seed, Val wanted, Val settings);
Val solver_verify(VM *v, int which, int argc, Val *args);

#endif
