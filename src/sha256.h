#ifndef ARDA_SHA256_H
#define ARDA_SHA256_H

#include <stddef.h>
#include <stdint.h>

enum { SHA256_BYTES = 32 };
typedef struct {
    uint32_t state[8];
    uint64_t bytes;
    unsigned char block[64];
} Sha256;

Sha256 sha256_begin(void);
void sha256_add(Sha256 *hash, const void *data, size_t bytes);
void sha256_finish(Sha256 *hash, unsigned char digest[SHA256_BYTES]);

#endif
