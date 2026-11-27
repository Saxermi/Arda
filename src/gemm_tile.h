/* The same four-row, two-vector tile for each vector ISA. Inputs can alias
 * each other. The output is a newly allocated, disjoint matrix. */
#define TILE(NAME, TARGET, VECTOR, LANES, LOAD, STORE, SPLAT, ACCUMULATE)                          \
    TARGET static void NAME(int depth, int k, int n, const double *a, const double *b,             \
                            double *restrict c) {                                                  \
        VECTOR c00 = LOAD(c), c01 = LOAD(c + LANES);                                               \
        VECTOR c10 = LOAD(c + n), c11 = LOAD(c + n + LANES);                                       \
        VECTOR c20 = LOAD(c + 2 * (size_t)n), c21 = LOAD(c + 2 * (size_t)n + LANES);               \
        VECTOR c30 = LOAD(c + 3 * (size_t)n), c31 = LOAD(c + 3 * (size_t)n + LANES);               \
        for (int p = 0; p < depth; p++) {                                                          \
            const double *row = b + (size_t)p * n;                                                 \
            VECTOR b0 = LOAD(row), b1 = LOAD(row + LANES);                                         \
            VECTOR aa = SPLAT(a[p]);                                                               \
            c00 = ACCUMULATE(aa, b0, c00);                                                         \
            c01 = ACCUMULATE(aa, b1, c01);                                                         \
            aa = SPLAT(a[k + p]);                                                                  \
            c10 = ACCUMULATE(aa, b0, c10);                                                         \
            c11 = ACCUMULATE(aa, b1, c11);                                                         \
            aa = SPLAT(a[2 * (size_t)k + p]);                                                      \
            c20 = ACCUMULATE(aa, b0, c20);                                                         \
            c21 = ACCUMULATE(aa, b1, c21);                                                         \
            aa = SPLAT(a[3 * (size_t)k + p]);                                                      \
            c30 = ACCUMULATE(aa, b0, c30);                                                         \
            c31 = ACCUMULATE(aa, b1, c31);                                                         \
        }                                                                                          \
        STORE(c, c00);                                                                             \
        STORE(c + LANES, c01);                                                                     \
        STORE(c + n, c10);                                                                         \
        STORE(c + n + LANES, c11);                                                                 \
        STORE(c + 2 * (size_t)n, c20);                                                             \
        STORE(c + 2 * (size_t)n + LANES, c21);                                                     \
        STORE(c + 3 * (size_t)n, c30);                                                             \
        STORE(c + 3 * (size_t)n + LANES, c31);                                                     \
    }
