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
