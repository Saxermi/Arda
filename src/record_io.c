#include "record_io.h"

uint64_t record_u64(const unsigned char *p) {
    uint64_t value = 0;
    for (unsigned i = 0; i < 8; i++)
        value |= (uint64_t)p[i] << (8 * i);
    return value;
}

void record_put_u64(unsigned char *p, uint64_t word) {
    for (unsigned i = 0; i < 8; i++)
        p[i] = (unsigned char)(word >> (8 * i));
}

bool record_flush(RecordOutput *out) {
    if (out->used && !tensor_file_write(out->fd, out->buffer, out->used))
        return false;
    out->used = 0;
    return true;
}

bool record_emit(RecordOutput *out, const void *data, size_t bytes) {
    if (bytes > out->limit - out->size) {
        errno = EFBIG;
        return false;
    }
    out->size += bytes;
    if (out->fd < 0 || !bytes)
        return true;
    if (out->checksummed)
        sha256_add(&out->hash, data, bytes);
    if (bytes > sizeof out->buffer - out->used && !record_flush(out))
        return false;
    if (bytes >= sizeof out->buffer)
        return tensor_file_write(out->fd, data, bytes);
    memcpy(out->buffer + out->used, data, bytes);
    out->used += bytes;
    return true;
}

bool record_text(RecordOutput *out, const char *text) {
    return record_emit(out, text, strlen(text));
}

size_t record_budget(VM *v, Val value, size_t limit) {
    if (value.t == V_NIL)
        return limit;
    if (value.t != V_INT || value.i < 0 || (uint64_t)value.i > limit)
        vmerror_kind(v, "disk_budget", "metadata byte limit is outside the allowed range");
    return (size_t)value.i;
}

_Noreturn void record_error(VM *v, const char *operation, int error) {
    if (error == EFBIG)
        vmerror_kind(v, "disk_budget", "%s exceeds the disk budget", operation);
    if (error == EINVAL)
        vmerror_kind(v, "metadata", "%s requires bounded finite scalar/list/map metadata",
                     operation);
    vmerror_kind(v, "io", "%s failed: %s", operation, strerror(error));
}
