#include "arda.h"

static void free_buffer(void *allocation, size_t bytes) {
    (void)bytes;
    free(allocation);
}

size_t tensor_memory_unlimited(void) {
    return (uintmax_t)SIZE_MAX > INT64_MAX ? (size_t)INT64_MAX : SIZE_MAX;
}

size_t tensor_memory_available(VM *v) {
    TensorMemory *m = &v->tensor_memory;
    return m->limit - m->heap - m->reserved;
}

static void update_peak(TensorMemory *m) {
    size_t total = m->heap + m->reserved + m->cached;
    if (total > m->peak)
        m->peak = total;
}

size_t tensor_pool_trim(VM *v) {
    TensorMemory *m = &v->tensor_memory;
    size_t bytes = m->cached;
    free(m->pool);
    m->pool = NULL;
    m->cached = 0;
    return bytes;
}

unsigned char *tensor_workspace_acquire(VM *v, size_t bytes) {
    TensorMemory *m = &v->tensor_memory;
    if (m->reserved || !bytes || bytes > tensor_memory_available(v))
        return NULL;
    unsigned char *memory;
    if (m->cached == bytes) {
        memory = m->pool;
        m->pool = NULL;
        m->cached = 0;
    } else {
        tensor_pool_trim(v);
        memory = malloc(bytes);
        if (!memory)
            return NULL;
    }
    m->reserved = bytes;
    update_peak(m);
    return memory;
}

void tensor_workspace_release(VM *v, unsigned char *memory) {
    TensorMemory *m = &v->tensor_memory;
    size_t bytes = m->reserved;
    m->reserved = 0;
    if (bytes <= 8u * 1024 * 1024) {
        m->pool = memory;
        m->cached = bytes;
    } else
        free(memory);
}

void tensor_buffer_release(VM *v, Buffer *buffer) {
    TensorMemory *m = &v->tensor_memory;
    if (buffer->mapped)
        m->mapped -= buffer->allocation_bytes;
    else
        m->heap -= buffer->bytes;
    buffer->release(buffer->allocation, buffer->allocation_bytes);
}

void tensor_bump_version(VM *v, Obj *buffer) {
    if (buffer->as.buffer.version == UINT64_MAX)
        vmerror_kind(v, "mutation", "tensor buffer version exhausted");
    buffer->as.buffer.version++;
}

Obj *buffer_new(VM *v, size_t bytes) {
    if (bytes > PTRDIFF_MAX)
        vmerror(v, "tensor buffer exceeds addressable size");
    size_t available = tensor_memory_available(v);
    if (bytes > available)
        vmerror_kind(v, "memory_budget", "tensor memory budget exceeded");
    TensorMemory *m = &v->tensor_memory;
    if (m->cached > available - bytes)
        tensor_pool_trim(v);
    Obj *buffer = newobj(v, O_BUFFER);
    void *data = mem(bytes);
    buffer->as.buffer = (Buffer){.data = data,
                                 .allocation = data,
                                 .bytes = bytes,
                                 .allocation_bytes = bytes,
                                 .alignment = _Alignof(max_align_t),
                                 .release = free_buffer};
    account(v, buffer, 0, bytes);
    m->heap += bytes;
    update_peak(m);
    return buffer;
}
