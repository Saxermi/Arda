#include "cpu_kernels.h"

#if defined(__aarch64__) && defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

unsigned cpu_detect(void) {
    unsigned flags = 0;
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    if (__builtin_cpu_supports("sse2"))
        flags |= CPU_SSE2;
    if (__builtin_cpu_supports("avx2"))
        flags |= CPU_AVX2;
    if (__builtin_cpu_supports("fma"))
        flags |= CPU_FMA;
#elif defined(__aarch64__) && defined(__linux__)
    if (getauxval(AT_HWCAP) & HWCAP_ASIMD)
        flags |= CPU_NEON;
#elif defined(__aarch64__) && defined(__ARM_NEON)
    flags |= CPU_NEON;
#endif
    return flags;
}

const char *cpu_architecture(void) {
#if defined(__x86_64__)
    return "x86_64";
#elif defined(__i386__)
    return "x86";
#elif defined(__aarch64__)
    return "aarch64";
#else
    return "other";
#endif
}

#if defined(ARDA_VECMATH) && defined(__x86_64__) && ARDA_X86_KERNELS
#include <immintrin.h>

/* GNU libmvec vector ABI. Special-value behavior and approximations are
 * provider-specific. Scalar tails call the ordinary libm functions. */
#define DECLARE_VEC(NAME, ATTR, VECTOR) extern ATTR VECTOR NAME(VECTOR)
#define TARGET128 __attribute__((target("sse2")))
#define TARGET256 __attribute__((target("avx2")))
DECLARE_VEC(_ZGVbN2v_sin, TARGET128, __m128d);
DECLARE_VEC(_ZGVbN2v_cos, TARGET128, __m128d);
DECLARE_VEC(_ZGVbN2v_exp, TARGET128, __m128d);
DECLARE_VEC(_ZGVdN4v_sin, TARGET256, __m256d);
DECLARE_VEC(_ZGVdN4v_cos, TARGET256, __m256d);
DECLARE_VEC(_ZGVdN4v_exp, TARGET256, __m256d);

#define APPLY_VECTOR(WIDTH, LOAD, STORE, KERNEL, TAIL)                                             \
    do {                                                                                           \
        for (; count - i >= WIDTH; i += WIDTH)                                                     \
            STORE(out + i, KERNEL(LOAD(in + i)));                                                  \
        for (; i < count; i++)                                                                     \
            out[i] = TAIL(in[i]);                                                                  \
    } while (0)

TARGET128 static void unary_sse2(VectorMath op, size_t count, const double *in, double *out) {
    size_t i = 0;
    switch (op) {
    case CPU_SIN:
        APPLY_VECTOR(2, _mm_loadu_pd, _mm_storeu_pd, _ZGVbN2v_sin, sin);
        break;
    case CPU_COS:
        APPLY_VECTOR(2, _mm_loadu_pd, _mm_storeu_pd, _ZGVbN2v_cos, cos);
        break;
    case CPU_EXP:
        APPLY_VECTOR(2, _mm_loadu_pd, _mm_storeu_pd, _ZGVbN2v_exp, exp);
        break;
    }
}

TARGET256 static void unary_avx2(VectorMath op, size_t count, const double *in, double *out) {
    size_t i = 0;
    switch (op) {
    case CPU_SIN:
        APPLY_VECTOR(4, _mm256_loadu_pd, _mm256_storeu_pd, _ZGVdN4v_sin, sin);
        break;
    case CPU_COS:
        APPLY_VECTOR(4, _mm256_loadu_pd, _mm256_storeu_pd, _ZGVdN4v_cos, cos);
        break;
    case CPU_EXP:
        APPLY_VECTOR(4, _mm256_loadu_pd, _mm256_storeu_pd, _ZGVdN4v_exp, exp);
        break;
    }
}
#endif

bool cpu_vector_math_available(const VM *v) {
#if defined(ARDA_VECMATH) && defined(__x86_64__) && ARDA_X86_KERNELS
    return (v->cpu_features & CPU_SSE2) != 0;
#else
    (void)v;
    return false;
#endif
}

bool cpu_vector_unary(const VM *v, VectorMath op, size_t count, const double *in, double *out) {
#if defined(ARDA_VECMATH) && defined(__x86_64__) && ARDA_X86_KERNELS
    if (count < 16 || !cpu_vector_math_available(v))
        return false;
    if (v->cpu_features & CPU_AVX2)
        unary_avx2(op, count, in, out);
    else
        unary_sse2(op, count, in, out);
    return true;
#else
    (void)v;
    (void)op;
    (void)count;
    (void)in;
    (void)out;
    return false;
#endif
}
