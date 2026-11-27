#include "quant_kernels.h"

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

/* The four chains and reduction order are shared by the portable and SIMD dots. */
static double finish_dot(double sums[4], const double *weight, const double *input, size_t begin,
                         size_t count) {
    for (size_t i = begin; i < count; i++)
        sums[i % 4] += weight[i] * input[i];
    return (sums[0] + sums[1]) + (sums[2] + sums[3]);
}

double quant_reference_dot(const double *weight, const double *input, size_t count) {
    double sums[4] = {0};
    return finish_dot(sums, weight, input, 0, count);
}

void quant_reference_accumulate(double *output, const double *input, const double *weights,
                                size_t depth, size_t columns, ptrdiff_t row_stride) {
    for (size_t p = 0; p < depth; p++) {
        const double *row = input + (ptrdiff_t)p * row_stride;
        for (size_t col = 0; col < columns; col++)
            output[col] += weights[p] * row[col];
    }
}

/* Keep four column vectors in registers for the decoded weight block. */
#define ACCUMULATE_TILE(NAME, TARGET, VECTOR, LANES, LOAD, STORE, SPLAT, ADD, MUL)                 \
    TARGET static void NAME(double *output, const double *input, const double *weights,            \
                            size_t depth, size_t columns, ptrdiff_t row_stride) {                  \
        size_t j = 0;                                                                              \
        for (; columns - j >= 4 * LANES; j += 4 * LANES) {                                         \
            VECTOR c0 = LOAD(output + j), c1 = LOAD(output + j + LANES);                           \
            VECTOR c2 = LOAD(output + j + 2 * LANES), c3 = LOAD(output + j + 3 * LANES);           \
            for (size_t p = 0; p < depth; p++) {                                                   \
                const double *row = (input + (ptrdiff_t)p * row_stride) + j;                       \
                VECTOR weight = SPLAT(weights[p]);                                                 \
                c0 = ADD(c0, MUL(weight, LOAD(row)));                                              \
                c1 = ADD(c1, MUL(weight, LOAD(row + LANES)));                                      \
                c2 = ADD(c2, MUL(weight, LOAD(row + 2 * LANES)));                                  \
                c3 = ADD(c3, MUL(weight, LOAD(row + 3 * LANES)));                                  \
            }                                                                                      \
            STORE(output + j, c0);                                                                 \
            STORE(output + j + LANES, c1);                                                         \
            STORE(output + j + 2 * LANES, c2);                                                     \
            STORE(output + j + 3 * LANES, c3);                                                     \
        }                                                                                          \
        for (; columns - j >= LANES; j += LANES) {                                                 \
            VECTOR value = LOAD(output + j);                                                       \
            for (size_t p = 0; p < depth; p++) {                                                   \
                const double *row = (input + (ptrdiff_t)p * row_stride) + j;                       \
                value = ADD(value, MUL(SPLAT(weights[p]), LOAD(row)));                             \
            }                                                                                      \
            STORE(output + j, value);                                                              \
        }                                                                                          \
        if (j < columns)                                                                           \
            quant_reference_accumulate(output + j, input + j, weights, depth, columns - j,         \
                                       row_stride);                                                \
    }

#if ARDA_X86_KERNELS
#include <immintrin.h>

__attribute__((target("avx2"))) static inline void avx2_decode_eight(__m128i packed, int bias,
                                                                     __m256d scale, double *out) {
    __m256i codes = _mm256_sub_epi32(_mm256_cvtepu8_epi32(packed), _mm256_set1_epi32(bias));
    _mm256_storeu_pd(out, _mm256_mul_pd(scale, _mm256_cvtepi32_pd(_mm256_castsi256_si128(codes))));
    _mm256_storeu_pd(out + 4,
                     _mm256_mul_pd(scale, _mm256_cvtepi32_pd(_mm256_extracti128_si256(codes, 1))));
}

__attribute__((target("avx2"))) static void avx2_decode(const unsigned char *input, bool q4,
                                                        double scale, double *out) {
    __m128i first = _mm_loadu_si128((const __m128i_u *)input), second;
    int bias = q4 ? 8 : 128;
    if (q4) {
        __m128i mask = _mm_set1_epi8(15);
        __m128i low = _mm_and_si128(first, mask);
        __m128i high = _mm_and_si128(_mm_srli_epi16(first, 4), mask);
        first = _mm_unpacklo_epi8(low, high);
        second = _mm_unpackhi_epi8(low, high);
    } else
        second = _mm_loadu_si128((const __m128i_u *)(input + 16));
    __m256d factor = _mm256_set1_pd(scale);
    avx2_decode_eight(first, bias, factor, out);
    avx2_decode_eight(_mm_srli_si128(first, 8), bias, factor, out + 8);
    avx2_decode_eight(second, bias, factor, out + 16);
    avx2_decode_eight(_mm_srli_si128(second, 8), bias, factor, out + 24);
}

__attribute__((target("avx2"))) static double avx2_dot(const double *weight, const double *input,
                                                       size_t count) {
    __m256d sum = _mm256_setzero_pd();
    size_t i = 0;
    for (; count - i >= 4; i += 4)
        sum = _mm256_add_pd(sum,
                            _mm256_mul_pd(_mm256_loadu_pd(weight + i), _mm256_loadu_pd(input + i)));
    double sums[4];
    _mm256_storeu_pd(sums, sum);
    return finish_dot(sums, weight, input, i, count);
}

ACCUMULATE_TILE(avx2_accumulate, __attribute__((target("avx2"))), __m256d, 4, _mm256_loadu_pd,
                _mm256_storeu_pd, _mm256_set1_pd, _mm256_add_pd, _mm256_mul_pd)
#endif

#if ARDA_X86_KERNELS || ARDA_ARM_KERNELS
static float bf16_scalar(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof result);
    return result;
}
#endif

#if ARDA_X86_KERNELS
static inline __attribute__((target("avx2"))) __m256 bf16_vector(__m128i values) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(values), 16));
}

/* Compute standard FP32 accumulation from widened 16-bit input tiles. */
#define LOWP_KERNEL(NAME, ISA, SCALAR, VECTOR)                                                     \
    __attribute__((target(ISA))) static void NAME(int m, int k, int n, const uint16_t *left,       \
                                                  const uint16_t *right, float *output) {          \
        for (int row = 0; row < m; row++) {                                                        \
            float *dst = output + (size_t)row * n;                                                 \
            for (int p = 0; p < k; p++) {                                                          \
                float value = SCALAR(left[(size_t)row * k + p]);                                   \
                __m256 weight = _mm256_set1_ps(value);                                             \
                const uint16_t *src = right + (size_t)p * n;                                       \
                int col = 0;                                                                       \
                for (; n - col >= 8; col += 8) {                                                   \
                    __m128i packed = _mm_loadu_si128((const __m128i_u *)(src + col));              \
                    __m256 product = _mm256_mul_ps(weight, VECTOR(packed));                        \
                    _mm256_storeu_ps(dst + col,                                                    \
                                     _mm256_add_ps(_mm256_loadu_ps(dst + col), product));          \
                }                                                                                  \
                for (; col < n; col++)                                                             \
                    dst[col] += value * SCALAR(src[col]);                                          \
            }                                                                                      \
        }                                                                                          \
    }
LOWP_KERNEL(avx2_f16, "avx2,f16c", _cvtsh_ss, _mm256_cvtph_ps)
LOWP_KERNEL(avx2_bf16, "avx2", bf16_scalar, bf16_vector)
#undef LOWP_KERNEL
#endif

#if ARDA_ARM_KERNELS
#include <arm_neon.h>

static inline void neon_decode_four(int16x4_t codes, float64x2_t scale, double *out) {
    int32x4_t wide = vmovl_s16(codes);
    vst1q_f64(out, vmulq_f64(scale, vcvtq_f64_s64(vmovl_s32(vget_low_s32(wide)))));
    vst1q_f64(out + 2, vmulq_f64(scale, vcvtq_f64_s64(vmovl_s32(vget_high_s32(wide)))));
}

static inline void neon_decode_eight(uint8x8_t bytes, int bias, float64x2_t scale, double *out) {
    int16x8_t codes = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(bytes)), vdupq_n_s16((int16_t)bias));
    neon_decode_four(vget_low_s16(codes), scale, out);
    neon_decode_four(vget_high_s16(codes), scale, out + 4);
}

static void neon_decode(const unsigned char *input, bool q4, double scale, double *out) {
    uint8x16_t first = vld1q_u8(input), second;
    int bias = q4 ? 8 : 128;
    if (q4) {
        uint8x16_t low = vandq_u8(first, vdupq_n_u8(15));
        uint8x16_t high = vshrq_n_u8(first, 4);
        first = vzip1q_u8(low, high);
        second = vzip2q_u8(low, high);
    } else
        second = vld1q_u8(input + 16);
    float64x2_t factor = vdupq_n_f64(scale);
    neon_decode_eight(vget_low_u8(first), bias, factor, out);
    neon_decode_eight(vget_high_u8(first), bias, factor, out + 8);
    neon_decode_eight(vget_low_u8(second), bias, factor, out + 16);
    neon_decode_eight(vget_high_u8(second), bias, factor, out + 24);
}

static double neon_dot(const double *weight, const double *input, size_t count) {
    float64x2_t first = vdupq_n_f64(0), second = vdupq_n_f64(0);
    size_t i = 0;
    for (; count - i >= 4; i += 4) {
        first = vaddq_f64(first, vmulq_f64(vld1q_f64(weight + i), vld1q_f64(input + i)));
        second = vaddq_f64(second, vmulq_f64(vld1q_f64(weight + i + 2), vld1q_f64(input + i + 2)));
    }
    double sums[4];
    vst1q_f64(sums, first);
    vst1q_f64(sums + 2, second);
    return finish_dot(sums, weight, input, i, count);
}

ACCUMULATE_TILE(neon_accumulate, , float64x2_t, 2, vld1q_f64, vst1q_f64, vdupq_n_f64, vaddq_f64,
                vmulq_f64)

static inline float32x4_t neon_bf16_vector(uint16x4_t values) {
    return vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(values), 16));
}

static inline float32x4_t neon_f16_vector(uint16x4_t values) {
    return vcvt_f32_f16(vreinterpret_f16_u16(values));
}

static inline float neon_f16_scalar(uint16_t value) {
    return vgetq_lane_f32(neon_f16_vector(vdup_n_u16(value)), 0);
}

#define LOWP_NEON(NAME, SCALAR, VECTOR)                                                            \
    static void NAME(int m, int k, int n, const uint16_t *left, const uint16_t *right,             \
                     float *output) {                                                              \
        for (int row = 0; row < m; row++) {                                                        \
            float *dst = output + (size_t)row * n;                                                 \
            for (int p = 0; p < k; p++) {                                                          \
                float value = SCALAR(left[(size_t)row * k + p]);                                   \
                float32x4_t weight = vdupq_n_f32(value);                                           \
                const uint16_t *src = right + (size_t)p * n;                                       \
                int col = 0;                                                                       \
                for (; n - col >= 4; col += 4) {                                                   \
                    float32x4_t product = vmulq_f32(weight, VECTOR(vld1_u16(src + col)));          \
                    vst1q_f32(dst + col, vaddq_f32(vld1q_f32(dst + col), product));                \
                }                                                                                  \
                for (; col < n; col++)                                                             \
                    dst[col] += value * SCALAR(src[col]);                                          \
            }                                                                                      \
        }                                                                                          \
    }
LOWP_NEON(neon_f16, neon_f16_scalar, neon_f16_vector)
LOWP_NEON(neon_bf16, bf16_scalar, neon_bf16_vector)
#undef LOWP_NEON
#endif

#undef ACCUMULATE_TILE

const QuantKernels *quant_simd_kernels(unsigned features) {
    (void)features;
#if ARDA_X86_KERNELS
    static const QuantKernels avx2 = {avx2_decode, avx2_dot, avx2_accumulate, "avx2"};
    if (features & CPU_AVX2)
        return &avx2;
#endif
#if ARDA_ARM_KERNELS
    static const QuantKernels neon = {neon_decode, neon_dot, neon_accumulate, "neon"};
    if (features & CPU_NEON)
        return &neon;
#endif
    return NULL;
}

const LowpKernel *quant_lowp_simd(unsigned features, DType dtype) {
    (void)features;
    (void)dtype;
#if ARDA_X86_KERNELS
    static const LowpKernel f16 = {avx2_f16, "f16c"}, bf16 = {avx2_bf16, "avx2"};
    if (features & CPU_AVX2) {
        if (dtype == DT_BF16)
            return &bf16;
        if (dtype == DT_F16 && (features & CPU_F16C))
            return &f16;
    }
#endif
#if ARDA_ARM_KERNELS
    static const LowpKernel f16 = {neon_f16, "neon-fp16"}, bf16 = {neon_bf16, "neon"};
    if (features & CPU_NEON) {
        if (dtype == DT_BF16)
            return &bf16;
        if (dtype == DT_F16 && (features & CPU_ARM_FP16))
            return &f16;
    }
#endif
    return NULL;
}
