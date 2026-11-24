#include "arda.h"

static void free_buffer(void *allocation, size_t bytes) {
    (void)bytes;
    free(allocation);
}

Obj *buffer_new(VM *v, size_t bytes) {
    if (bytes > PTRDIFF_MAX)
        vmerror(v, "tensor buffer exceeds addressable size");
    Obj *buffer = newobj(v, O_BUFFER);
    void *data = mem(bytes);
    buffer->as.buffer = (Buffer){.data = data,
                                 .allocation = data,
                                 .bytes = bytes,
                                 .allocation_bytes = bytes,
                                 .alignment = _Alignof(max_align_t),
                                 .release = free_buffer};
    account(v, buffer, 0, bytes);
    return buffer;
}
