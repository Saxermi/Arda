#ifndef ARDA_QUANT_KERNELS_H
#define ARDA_QUANT_KERNELS_H

#include "cpu_kernels.h"

enum { QUANT_BLOCK = 32 };

typedef void (*QuantDecode)(const unsigned char *input, bool q4, double scale, double *out);
typedef double (*QuantDot)(const double *weight, const double *input, size_t count);
typedef void (*QuantAccumulate)(double *output, const double *input, const double *weights,
                                size_t depth, size_t columns, ptrdiff_t row_stride);
typedef struct {
    QuantDecode decode;
    QuantDot dot;
    QuantAccumulate accumulate;
    const char *name;
} QuantKernels;

double quant_reference_dot(const double *weight, const double *input, size_t count);
void quant_reference_accumulate(double *output, const double *input, const double *weights,
                                size_t depth, size_t columns, ptrdiff_t row_stride);
const QuantKernels *quant_simd_kernels(unsigned features);

typedef void (*LowPrecisionMatmul)(int m, int k, int n, const uint16_t *left, const uint16_t *right,
                                   float *output);
typedef struct {
    LowPrecisionMatmul matmul;
    const char *name;
} LowpKernel;

const LowpKernel *quant_lowp_simd(unsigned features, DType dtype);

#endif
