#ifndef ARDA_FILE_COMPUTE_H
#define ARDA_FILE_COMPUTE_H

#include "arda.h"

Val compute_file_matmul(VM *v, const Val *args, int block, Val disk_budget);
Val compute_file_spmv(VM *v, const Val *args, Val block, Val disk_budget);
Val compute_file_qr_r(VM *v, Val path, Val block);
bool dense_qr_packed(double *data, int rows, int columns, double *tau);
bool dense_qr_basis(const double *packed, int rows, int columns, const double *tau, double *basis,
                    int width);
Val compute_file_rsvd_right(VM *v, int argc, const Val *args);
#ifdef ARDA_LAPACK
Val dense_projected_right_svd(VM *v, Obj *r, Obj *basis, int rank);
#endif

#endif
