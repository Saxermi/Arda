#include "typed_tensor.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum { HEADER_SIZE = 128, IO_BLOCK = 65536 };
static const unsigned char magic[] = "ARDATEN1";

static uint64_t get_u64(const unsigned char *p) {
    uint64_t value = 0;
    for (unsigned i = 0; i < 8; i++)
        value |= (uint64_t)p[i] << (8 * i);
    return value;
}

static void put_u64(unsigned char *p, uint64_t value) {
    for (unsigned i = 0; i < 8; i++)
        p[i] = (unsigned char)(value >> (8 * i));
}

static void require_little_endian(VM *v) {
    const uint16_t one = 1;
    if (*(const unsigned char *)&one != 1)
        vmerror(v, "tensor files currently require a little-endian host");
}

static bool read_exact(int fd, void *data, size_t bytes, off_t offset) {
    unsigned char *p = data;
    while (bytes) {
        size_t size = bytes > IO_BLOCK ? IO_BLOCK : bytes;
        ssize_t n = pread(fd, p, size, offset);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            return false;
        bytes -= (size_t)n;
        p += n;
        offset += n;
    }
    return true;
}

static bool write_exact(int fd, const void *data, size_t bytes) {
    const unsigned char *p = data;
    while (bytes) {
        ssize_t n = write(fd, p, bytes);
        if (n < 0 && errno == EINTR)
            continue;
        if (!n)
            errno = EIO;
        if (n <= 0)
            return false;
        bytes -= (size_t)n;
        p += n;
    }
    return true;
}

static bool decode_header(const unsigned char *header, off_t size, TypedTensor *tensor) {
    if (memcmp(header, magic, sizeof magic - 1) || size < HEADER_SIZE ||
        (uintmax_t)size > PTRDIFF_MAX || header[8] < 1 || header[8] > DT_COUNT || header[9] < 1 ||
        header[9] > 8 || header[10] != 1 || get_u64(header + 16) != HEADER_SIZE)
        return false;
    for (int i = 11; i < 16; i++)
        if (header[i])
            return false;
    for (int i = 96; i < HEADER_SIZE; i++)
        if (header[i])
            return false;
    DType dtype = (DType)(header[8] - 1);
    int rank = header[9], shape[8];
    for (int i = 0; i < 8; i++) {
        uint64_t dim = get_u64(header + 32 + 8 * i);
        if (dim > INT_MAX || (i >= rank && dim))
            return false;
        shape[i] = (int)dim;
    }
    if (!typed_layout_checked(dtype, rank, shape, tensor))
        return false;
    uint64_t stored_bytes = get_u64(header + 24);
    return stored_bytes == tensor->n * dtype_info[dtype].size &&
           stored_bytes == (uintmax_t)size - HEADER_SIZE;
}

static int open_tensor(VM *v, const char *path, TypedTensor *tensor, size_t *size) {
    require_little_endian(v);
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0)
        vmerror(v, "cannot open tensor file '%s': %s", path, strerror(errno));
    struct stat info;
    unsigned char header[HEADER_SIZE];
    if (fstat(fd, &info) || !S_ISREG(info.st_mode) || !read_exact(fd, header, sizeof header, 0) ||
        !decode_header(header, info.st_size, tensor)) {
        close(fd);
        vmerror(v, "invalid tensor file header, shape, endian marker or length");
    }
    *size = (size_t)info.st_size;
    return fd;
}

static void release_mapping(void *allocation, size_t size) {
    munmap(allocation, size);
}

Val tensor_mmap(VM *v, Val path_value) {
    const char *path = asstr(v, path_value);
    TypedTensor t;
    size_t size;
    int fd = open_tensor(v, path, &t, &size);
    void *mapping = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    int mapped_errno = errno;
    close(fd);
    if (mapping == MAP_FAILED)
        vmerror(v, "cannot map tensor file '%s': %s", path, strerror(mapped_errno));
    t.buffer = newobj(v, O_BUFFER);
    t.buffer->as.buffer = (Buffer){.data = (unsigned char *)mapping + HEADER_SIZE,
                                   .allocation = mapping,
                                   .bytes = size - HEADER_SIZE,
                                   .allocation_bytes = size,
                                   .alignment = _Alignof(max_align_t),
                                   .release = release_mapping,
                                   .readonly = true};
    return objval(typed_wrap(v, t));
}

Val tensor_read_block(VM *v, Val path_value, Val start, Val count) {
    const char *path = asstr(v, path_value);
    if (start.t != V_INT || start.i < 0 || count.t != V_INT || count.i < 0 ||
        count.i > ARDA_MAX_ELEMENTS)
        vmerror(v, "block access requires a nonnegative start and count <= 16000000");
    TypedTensor source;
    size_t size;
    int fd = open_tensor(v, path, &source, &size);
    if ((uint64_t)start.i > source.n || (uint64_t)count.i > source.n - (size_t)start.i) {
        close(fd);
        vmerror(v, "tensor block is outside the file");
    }
    int n = (int)count.i;
    Obj *out = typed_new(v, source.dtype, 1, &n);
    size_t width = dtype_info[source.dtype].size;
    bool ok = read_exact(fd, typed_element(&out->as.tensor, 0), (size_t)n * width,
                         (off_t)(HEADER_SIZE + (size_t)start.i * width));
    close(fd);
    if (!ok)
        vmerror(v, "tensor block read was interrupted by an I/O error or truncated data");
    return objval(out);
}

static bool write_payload(int fd, const TypedTensor *t) {
    unsigned char chunk[IO_BLOCK];
    size_t width = dtype_info[t->dtype].size, per_chunk = sizeof chunk / width;
    bool contiguous = typed_contiguous(t);
    for (size_t start = 0; start < t->n;) {
        size_t count = t->n - start;
        if (count > per_chunk)
            count = per_chunk;
        const unsigned char *data;
        if (contiguous)
            data = typed_element(t, (ptrdiff_t)(t->offset + start));
        else {
            for (size_t i = 0; i < count; i++)
                memcpy(chunk + i * width, typed_element(t, typed_offset(t, start + i)), width);
            data = chunk;
        }
        if (!write_exact(fd, data, count * width))
            return false;
        start += count;
    }
    return true;
}

Val tensor_save(VM *v, Val path_value, Val input, Val budget) {
    require_little_endian(v);
    const char *path = asstr(v, path_value);
    TypedTensor t = typed_tensor(v, input);
    size_t bytes = t.n * dtype_info[t.dtype].size;
    if (bytes > PTRDIFF_MAX - HEADER_SIZE)
        vmerror(v, "tensor file exceeds addressable size");
    size_t total = HEADER_SIZE + bytes;
    if (budget.t != V_NIL && (budget.t != V_INT || budget.i < 0 || (uint64_t)budget.i < total))
        vmerror(v, "tensor file exceeds disk budget or budget is invalid");

    unsigned char header[HEADER_SIZE] = {0};
    memcpy(header, magic, sizeof magic - 1);
    header[8] = (unsigned char)t.dtype + 1;
    header[9] = (unsigned char)t.rank;
    header[10] = 1;
    put_u64(header + 16, HEADER_SIZE);
    put_u64(header + 24, bytes);
    for (int i = 0; i < t.rank; i++)
        put_u64(header + 32 + 8 * i, (uint64_t)t.shape[i]);

    static const char suffix[] = ".tmp.XXXXXX";
    size_t length = strlen(path);
    if (!length || length > SIZE_MAX - sizeof suffix)
        vmerror(v, "invalid tensor file path");
    char *temporary = mem(length + sizeof suffix);
    memcpy(temporary, path, length);
    memcpy(temporary + length, suffix, sizeof suffix);
    int fd = mkstemp(temporary);
    if (fd < 0) {
        int cause = errno;
        free(temporary);
        vmerror(v, "cannot create temporary tensor file: %s", strerror(cause));
    }
    bool ok = write_exact(fd, header, sizeof header) && write_payload(fd, &t) && fsync(fd) == 0;
    int cause = ok ? 0 : errno;
    if (close(fd) && ok) {
        ok = false;
        cause = errno;
    }
    if (ok && rename(temporary, path)) {
        ok = false;
        cause = errno;
    }
    if (!ok)
        unlink(temporary);
    free(temporary);
    if (!ok)
        vmerror(v, "cannot publish tensor file: %s", strerror(cause));
    return nil();
}
