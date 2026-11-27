#ifndef ARDA_SORT_RECORDS_H
#define ARDA_SORT_RECORDS_H

#include <stddef.h>
#include <string.h>

/* libc qsort may allocate hidden scratch storage. An in-place heap sort
 * keeps numerical workspace within the managed budget. */
static inline void sift_records(unsigned char *base, size_t count, size_t root, size_t width,
                                int (*compare)(const void *, const void *), unsigned char *temp) {
    memcpy(temp, base + root * width, width);
    while (root < count / 2) {
        size_t child = 2 * root + 1;
        if (child + 1 < count && compare(base + child * width, base + (child + 1) * width) < 0)
            child++;
        if (compare(temp, base + child * width) >= 0)
            break;
        memcpy(base + root * width, base + child * width, width);
        root = child;
    }
    memcpy(base + root * width, temp, width);
}

static inline void sort_records(void *records, size_t count, size_t width,
                                int (*compare)(const void *, const void *), void *temp) {
    unsigned char *base = records;
    for (size_t i = count / 2; i > 0; i--)
        sift_records(base, count, i - 1, width, compare, temp);
    for (size_t i = count; i > 1; i--) {
        unsigned char *last = base + (i - 1) * width;
        memcpy(temp, last, width);
        memcpy(last, base, width);
        memcpy(base, temp, width);
        sift_records(base, i - 1, 0, width, compare, temp);
    }
}

#endif
