#ifndef ARDA_CPU_KERNELS_H
#define ARDA_CPU_KERNELS_H

#include "arda.h"

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#define ARDA_X86_KERNELS 1
#endif
#if defined(__aarch64__) && defined(__ARM_NEON)
#define ARDA_ARM_KERNELS 1
#endif

enum { GEMM_AUTO, GEMM_REFERENCE, GEMM_SSE2, GEMM_AVX2, GEMM_FMA, GEMM_NEON };
typedef enum { CPU_SIN, CPU_COS, CPU_EXP } VectorMath;

enum {
    CPU_SSE2 = 1 << 0,
    CPU_AVX2 = 1 << 1,
    CPU_FMA = 1 << 2,
    CPU_NEON = 1 << 3,
    CPU_F16C = 1 << 4,
    CPU_ARM_FP16 = 1 << 5
};

unsigned cpu_detect(void);
const char *cpu_architecture(void);
int cpu_matmul_choice(const VM *v, const char *name);
const char *cpu_matmul_name(const VM *v);
const char *cpu_matmul_auto(const VM *v);
int cpu_matmul_resolve(const VM *v, int kernel, int m, int n);
const char *cpu_matmul_kernel_name(int kernel);
int cpu_matmul_block(int kernel);
int cpu_matmul_team(int threads, int m, int k, int n);
int cpu_matmul_run(const VM *v, int kernel, int threads, int m, int k, int n, const double *a,
                   const double *b, double *c);
bool cpu_parallel_available(void);
int cpu_team_size(int requested, int partitions);
static inline int cpu_thread_cap(const VM *v, int requested) {
    int cap = v->compute_thread_cap;
    return cap && cap < requested ? cap : requested;
}
static inline int cpu_parallel_team(const VM *v, int partitions) {
    return cpu_team_size(cpu_thread_cap(v, v->native_threads), partitions);
}
bool cpu_vector_math_available(const VM *v);
bool cpu_vector_unary(const VM *v, VectorMath op, size_t count, const double *in, double *out);

#endif
