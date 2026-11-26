#include "arda.h"
#include "cpu_kernels.h"

#if defined(_OPENMP) && !defined(ARDA_BLAS)
#include <omp.h>
#endif

#ifdef ARDA_BLAS
#include <cblas.h>
#endif

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

#ifndef ARDA_BLAS
#if ARDA_X86_KERNELS
#include <immintrin.h>
__attribute__((target("avx2"))) static void matmul_avx2(int m, int n, int k, const double *x,
                                                        const double *y, double *z) {
    const int block = 64;
    for (int ii = 0; ii < m; ii += block)
        for (int kk = 0; kk < k; kk += block)
            for (int jj = 0; jj < n; jj += block)
                for (int i = ii; i < m && i < ii + block; i++)
                    for (int t = kk; t < k && t < kk + block; t++) {
                        double at = x[(size_t)i * k + t];
                        __m256d aa = _mm256_set1_pd(at);
                        double *dst = z + (size_t)i * n;
                        const double *src = y + (size_t)t * n;
                        int j = jj, end = n < jj + block ? n : jj + block;
                        for (; j + 15 < end; j += 16)
                            for (int u = 0; u < 16; u += 4)
                                _mm256_storeu_pd(
                                    dst + j + u,
                                    _mm256_add_pd(_mm256_loadu_pd(dst + j + u),
                                                  _mm256_mul_pd(aa, _mm256_loadu_pd(src + j + u))));
                        for (; j + 3 < end; j += 4)
                            _mm256_storeu_pd(
                                dst + j,
                                _mm256_add_pd(_mm256_loadu_pd(dst + j),
                                              _mm256_mul_pd(aa, _mm256_loadu_pd(src + j))));
                        for (; j < end; j++)
                            dst[j] += at * src[j];
                    }
}
#endif
static void reference_matmul(unsigned features, int m, int k, int n, const double *x,
                             const double *y, double *z) {
    (void)features;
    /* Blocked row-major multiplication; innermost loop is contiguous. */
    const int block = 32;
#if ARDA_X86_KERNELS
    if (features & CPU_AVX2) {
        matmul_avx2(m, n, k, x, y, z);
        return;
    }
#endif
    for (int ii = 0; ii < m; ii += block)
        for (int kk = 0; kk < k; kk += block)
            for (int jj = 0; jj < n; jj += block)
                for (int i = ii; i < m && i < ii + block; i++)
                    for (int t = kk; t < k && t < kk + block; t++) {
                        double at = x[(size_t)i * k + t];
                        double *dst = z + (size_t)i * n;
                        const double *src = y + (size_t)t * n;
                        int j = jj, end = n < jj + block ? n : jj + block;
#if ARDA_X86_KERNELS && defined(__SSE2__)
                        __m128d aa = _mm_set1_pd(at);
                        for (; j + 7 < end; j += 8) {
                            for (int u = 0; u < 8; u += 2)
                                _mm_storeu_pd(
                                    dst + j + u,
                                    _mm_add_pd(_mm_loadu_pd(dst + j + u),
                                               _mm_mul_pd(aa, _mm_loadu_pd(src + j + u))));
                        }
                        for (; j + 1 < end; j += 2)
                            _mm_storeu_pd(dst + j,
                                          _mm_add_pd(_mm_loadu_pd(dst + j),
                                                     _mm_mul_pd(aa, _mm_loadu_pd(src + j))));
#endif
                        for (; j < end; j++)
                            dst[j] += at * src[j];
                    }
}
#endif

#ifndef ARDA_BLAS
#include "gemm_tile.h"

#if ARDA_X86_KERNELS
#define CPU_TARGET(name) __attribute__((target(name)))
#define SSE2_ACC(a, b, c) _mm_add_pd(c, _mm_mul_pd(a, b))
#define AVX2_ACC(a, b, c) _mm256_add_pd(c, _mm256_mul_pd(a, b))
TILE(gemm_sse2, CPU_TARGET("sse2"), __m128d, 2, _mm_loadu_pd, _mm_storeu_pd, _mm_set1_pd, SSE2_ACC)
TILE(gemm_avx2, CPU_TARGET("avx2"), __m256d, 4, _mm256_loadu_pd, _mm256_storeu_pd, _mm256_set1_pd,
     AVX2_ACC)
TILE(gemm_fma, CPU_TARGET("avx2,fma"), __m256d, 4, _mm256_loadu_pd, _mm256_storeu_pd,
     _mm256_set1_pd, _mm256_fmadd_pd)
#endif

#if ARDA_ARM_KERNELS
#include <arm_neon.h>
#define NEON_ACC(a, b, c) vaddq_f64(c, vmulq_f64(a, b))
TILE(gemm_neon, , float64x2_t, 2, vld1q_f64, vst1q_f64, vdupq_n_f64, NEON_ACC)
#endif
#undef TILE

typedef void (*GemmTile)(int depth, int k, int n, const double *a, const double *b, double *c);

static const struct {
    const char *name;
    unsigned required;
    int width;
    GemmTile tile;
} kernels[] = {
    [GEMM_AUTO] = {"auto", 0, 0, NULL},
    [GEMM_REFERENCE] = {"reference", 0, 0, NULL},
#if ARDA_X86_KERNELS
    [GEMM_SSE2] = {"sse2", CPU_SSE2, 4, gemm_sse2},
    [GEMM_AVX2] = {"avx2", CPU_AVX2, 8, gemm_avx2},
    [GEMM_FMA] = {"avx2-fma", CPU_AVX2 | CPU_FMA, 8, gemm_fma},
#endif
#if ARDA_ARM_KERNELS
    [GEMM_NEON] = {"neon", CPU_NEON, 4, gemm_neon},
#endif
};

static int auto_kernel(const VM *v) {
    for (int i = (int)(sizeof kernels / sizeof *kernels) - 1; i > GEMM_REFERENCE; i--)
        if (i != GEMM_FMA && kernels[i].tile &&
            (v->cpu_features & kernels[i].required) == kernels[i].required)
            return i;
    return GEMM_REFERENCE;
}

static void scalar_tile(int rows, int columns, int depth, int k, int n, const double *a,
                        const double *b, double *restrict c, bool fused) {
    for (int i = 0; i < rows; i++) {
        const double *left = a + (size_t)i * k;
        double *out = c + (size_t)i * n;
        if (fused) {
            for (int p = 0; p < depth; p++) {
                const double *right = b + (size_t)p * n;
                for (int j = 0; j < columns; j++)
                    out[j] = fma(left[p], right[j], out[j]);
            }
        } else {
            for (int p = 0; p < depth; p++) {
                const double *right = b + (size_t)p * n;
                for (int j = 0; j < columns; j++)
                    out[j] += left[p] * right[j];
            }
        }
    }
}

static int tile_end(int begin, int width, int size) {
    return size - begin < width ? size : begin + width;
}

enum { GEMM_ROWS = 32 };

static void tiled_gemm(int m, int k, int n, const double *a, const double *b, double *c,
                       int kernel) {
    int width = kernels[kernel].width;
    GemmTile tile = kernels[kernel].tile;
    bool fused = kernel == GEMM_FMA;
    for (int row = 0; row < m; row += GEMM_ROWS) {
        int end_row = tile_end(row, GEMM_ROWS, m);
        for (int column = 0; column < n; column += 64) {
            int end_column = tile_end(column, 64, n);
            for (int p = 0; p < k; p += 64) {
                int depth = tile_end(p, 64, k) - p;
                for (int i = row; i < end_row; i += 4) {
                    int rows = tile_end(i, 4, end_row) - i;
                    const double *left = a + (size_t)i * k + p;
                    double *out = c + (size_t)i * n;
                    const double *right = b + (size_t)p * n;
                    int j = column;
                    if (rows == 4)
                        for (; j <= end_column - width; j += width)
                            tile(depth, k, n, left, right + j, out + j);
                    if (j < end_column)
                        scalar_tile(rows, end_column - j, depth, k, n, left, right + j, out + j,
                                    fused);
                }
            }
        }
    }
}
#endif

int cpu_matmul_choice(const VM *v, const char *name) {
#ifdef ARDA_BLAS
    (void)v;
    return !strcmp(name, "auto") || !strcmp(name, "cblas") ? GEMM_AUTO : -1;
#else
    for (size_t i = 0; i < sizeof kernels / sizeof *kernels; i++)
        if (kernels[i].name && !strcmp(kernels[i].name, name) &&
            (v->cpu_features & kernels[i].required) == kernels[i].required)
            return (int)i;
    return -1;
#endif
}

const char *cpu_matmul_name(const VM *v) {
#ifdef ARDA_BLAS
    (void)v;
    return "cblas";
#else
    return kernels[v->matmul_kernel].name;
#endif
}

const char *cpu_matmul_auto(const VM *v) {
#ifdef ARDA_BLAS
    (void)v;
    return "cblas";
#else
    return kernels[auto_kernel(v)].name;
#endif
}

#ifndef ARDA_BLAS
static void gemm_rows(unsigned features, int kernel, int m, int k, int n, const double *a,
                      const double *b, double *c) {
    if (kernel == GEMM_REFERENCE)
        reference_matmul(features, m, k, n, a, b, c);
    else
        tiled_gemm(m, k, n, a, b, c, kernel);
}
#endif

void dense_matmul_into(VM *v, int m, int k, int n, const double *a, const double *b, double *c) {
    v->last_matmul_threads = 1;
    if (!m || !k || !n)
        return;
#ifdef ARDA_BLAS
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, a, k, b, n, 0, c, n);
#else
    int kernel = v->matmul_kernel;
    if (kernel == GEMM_AUTO) {
        kernel = auto_kernel(v);
        if (m < 4 || n < kernels[kernel].width)
            kernel = GEMM_REFERENCE;
    }
    unsigned features = v->cpu_features;
#if defined(_OPENMP)
    if ((uint64_t)m * k * n >= UINT64_C(8000000)) {
        int team = cpu_parallel_team(v, m / GEMM_ROWS);
        if (team > 1) {
            // Whole row tiles belong to one worker. No VM state is shared with the workers.
            int blocks = (m + GEMM_ROWS - 1) / GEMM_ROWS, actual = 1;
#pragma omp parallel num_threads(team) default(none) shared(actual)                                \
    firstprivate(blocks, features, kernel, m, k, n, a, b, c)
            {
                int rank = omp_get_thread_num(), size = omp_get_num_threads();
                if (!rank)
                    actual = size;
                int first = (int)((size_t)blocks * rank / size) * GEMM_ROWS;
                int last = (int)((size_t)blocks * (rank + 1) / size) * GEMM_ROWS;
                if (last > m)
                    last = m;
                gemm_rows(features, kernel, last - first, k, n, a + (size_t)first * k, b,
                          c + (size_t)first * n);
            }
            v->last_matmul_threads = actual;
            return;
        }
    }
#endif
    gemm_rows(features, kernel, m, k, n, a, b, c);
#endif
}
