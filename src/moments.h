#ifndef ARDA_MOMENTS_H
#define ARDA_MOMENTS_H

#include <stdint.h>

/* Welford's recurrence, applied to samples centered at a fixed origin. */
static inline void moments_update(uint64_t count, long double value, long double *mean,
                                  long double *squares) {
    long double delta = value - *mean;
    *mean += delta / count;
    *squares += delta * (value - *mean);
}

#endif
