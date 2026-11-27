#include "sha256.h"

#include <string.h>

// SHA-256 compression, following FIPS 180-4 sections 4 and 6.
static const uint32_t constants[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

Sha256 sha256_begin(void) {
    return (Sha256){{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
                     0x1f83d9ab, 0x5be0cd19},
                    0,
                    {0}};
}

static uint32_t rotate(uint32_t x, unsigned count) {
    return (x >> count) | (x << (32 - count));
}

static void compress(Sha256 *hash, const unsigned char *block) {
    uint32_t w[64];
    for (unsigned i = 0; i < 16; i++) {
        const unsigned char *p = block + 4 * i;
        w[i] = (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3];
    }
    for (unsigned i = 16; i < 64; i++) {
        uint32_t a = w[i - 15], b = w[i - 2];
        uint32_t s0 = rotate(a, 7) ^ rotate(a, 18) ^ (a >> 3);
        uint32_t s1 = rotate(b, 17) ^ rotate(b, 19) ^ (b >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = hash->state[0], b = hash->state[1], c = hash->state[2], d = hash->state[3];
    uint32_t e = hash->state[4], f = hash->state[5], g = hash->state[6], h = hash->state[7];
    for (unsigned i = 0; i < 64; i++) {
        uint32_t s1 = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
        uint32_t choose = (e & f) ^ (~e & g);
        uint32_t t1 = h + s1 + choose + constants[i] + w[i];
        uint32_t s0 = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + s0 + majority;
    }
    const uint32_t end[] = {a, b, c, d, e, f, g, h};
    for (unsigned i = 0; i < 8; i++)
        hash->state[i] += end[i];
}

void sha256_add(Sha256 *hash, const void *data, size_t bytes) {
    const unsigned char *p = data;
    size_t buffered = hash->bytes % 64;
    hash->bytes += bytes;
    if (buffered && bytes) {
        size_t take = bytes < 64 - buffered ? bytes : 64 - buffered;
        memcpy(hash->block + buffered, p, take);
        if (buffered + take < 64)
            return;
        compress(hash, hash->block);
        p += take;
        bytes -= take;
    }
    while (bytes >= 64) {
        compress(hash, p);
        p += 64;
        bytes -= 64;
    }
    if (bytes)
        memcpy(hash->block, p, bytes);
}

void sha256_finish(Sha256 *hash, unsigned char digest[SHA256_BYTES]) {
    uint64_t bits = hash->bytes * 8;
    size_t used = hash->bytes % 64;
    unsigned char pad[64] = {0x80}, length[8];
    sha256_add(hash, pad, used < 56 ? 56 - used : 120 - used);
    for (unsigned i = 0; i < 8; i++)
        length[7 - i] = (unsigned char)(bits >> (8 * i));
    sha256_add(hash, length, sizeof length);
    for (unsigned i = 0; i < SHA256_BYTES; i++)
        digest[i] = (unsigned char)(hash->state[i / 4] >> (24 - 8 * (i % 4)));
}
