#ifndef ARDA_ACTIVATION_STORE_H
#define ARDA_ACTIVATION_STORE_H

#include "typed_tensor.h"

enum { AD_READ_SLOTS = 3 };

typedef struct ADOffloadPlan {
    size_t references, limit, block_bytes, live_bytes, peak_bytes, live_files;
    uint64_t written_bytes, read_bytes, read_chunks, written_files;
    char directory[];
} ADOffloadPlan;

typedef struct {
    const Buffer *buffers[AD_READ_SLOTS];
    size_t count, block_bytes;
} ADReadSpec;

typedef struct ADReadCache {
    size_t slot_bytes, slots, next;
    struct {
        const Buffer *buffer;
        size_t offset;
    } entries[AD_READ_SLOTS];
} ADReadCache;

ADOffloadPlan *offload_plan(VM *v, Val value);
Val offload_plan_new(VM *v, Val directory, Val budget, Val block);
Val offload_stats(VM *v, const ADOffloadPlan *plan);
void offload_plan_release(ADOffloadPlan *plan);
void offload_file_release(Buffer *buffer);
Val offload_snapshot(VM *v, ADOffloadPlan *plan, const TypedTensor *input);

void offload_add_read(ADReadSpec *spec, const Buffer *buffer);
size_t offload_read_bytes(ADReadSpec spec);
void offload_read_start(VM *v, ADReadCache *cache, ADReadSpec spec);
void offload_read_finish(VM *v);
double offload_read_float64(VM *v, const Buffer *buffer, size_t byte_offset);
bool offload_read_chunk(Buffer const *buffer, size_t byte_offset, unsigned char *dst, size_t bytes);
void offload_record_replay(VM *v, size_t elements, size_t operations);

#endif
