#ifndef ARDA_CPU_KERNELS_H
#define ARDA_CPU_KERNELS_H

enum { CPU_SSE2 = 1 << 0, CPU_AVX2 = 1 << 1, CPU_FMA = 1 << 2, CPU_NEON = 1 << 3 };

unsigned cpu_detect(void);
const char *cpu_architecture(void);

#endif
