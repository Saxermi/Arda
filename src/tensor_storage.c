#include "activation_store.h"
#include "moments.h"

#include <fcntl.h>
#include <float.h>
#include <pthread.h>
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
        if (!n)
            errno = EIO;
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

static bool write_at(int fd, const unsigned char *data, size_t bytes, off_t offset) {
    while (bytes) {
        ssize_t n = pwrite(fd, data, bytes, offset);
        if (n < 0 && errno == EINTR)
            continue;
        if (!n)
            errno = EIO;
        if (n <= 0)
            return false;
        bytes -= (size_t)n;
        data += n;
        offset += n;
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
    if (size > tensor_memory_unlimited() - v->tensor_memory.mapped) {
        close(fd);
        vmerror(v, "tensor mapped byte count overflow");
    }
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
                                   .readonly = true,
                                   .mapped = true};
    v->tensor_memory.mapped += size;
    return objval(typed_wrap(v, t));
}

static Obj *read_flat(VM *v, int fd, DType dtype, size_t start, int count) {
    Obj *out = typed_new(v, dtype, 1, &count);
    size_t width = dtype_info[dtype].size;
    return read_exact(fd, typed_element(&out->as.tensor, 0), (size_t)count * width,
                      (off_t)(HEADER_SIZE + start * width))
               ? out
               : NULL;
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
    if ((uint64_t)count.i * dtype_info[source.dtype].size > tensor_memory_available(v)) {
        close(fd);
        vmerror(v, "tensor memory budget exceeded");
    }
    Obj *out = read_flat(v, fd, source.dtype, (size_t)start.i, (int)count.i);
    close(fd);
    if (!out)
        vmerror(v, "tensor block read failed or data was truncated");
    return objval(out);
}

static int block_elements(VM *v, Val value) {
    if (value.t != V_INT || value.i < 1 || value.i > ARDA_MAX_ELEMENTS)
        vmerror(v, "tensor block size must be 1..16000000 elements");
    return (int)value.i;
}

Val tensor_blocks(VM *v, Val path_value, Val block) {
    const char *path = asstr(v, path_value);
    int count = block_elements(v, block);
    TypedTensor t;
    size_t size;
    int fd = open_tensor(v, path, &t, &size);
    Obj *out = newobj(v, O_TENSOR_BLOCKS);
    out->as.tensor_blocks.fd = fd;
    out->as.tensor_blocks.elements = t.n;
    out->as.tensor_blocks.block = count;
    out->as.tensor_blocks.dtype = t.dtype;
    return objval(out);
}

void tensor_blocks_close(Obj *blocks) {
    if (blocks->as.tensor_blocks.fd >= 0) {
        close(blocks->as.tensor_blocks.fd);
        blocks->as.tensor_blocks.fd = -1;
    }
}

Val tensor_block_next(VM *v, Obj *blocks, size_t *cursor) {
    int fd = blocks->as.tensor_blocks.fd;
    if (fd < 0)
        vmerror(v, "tensor blocks are closed");
    size_t remaining = blocks->as.tensor_blocks.elements - *cursor;
    if (!remaining)
        return nil();
    size_t block = (size_t)blocks->as.tensor_blocks.block;
    int count = (int)(remaining < block ? remaining : block);
    Obj *out = read_flat(v, fd, blocks->as.tensor_blocks.dtype, *cursor, count);
    if (!out)
        vmerror(v, "tensor block read failed or data was truncated");
    *cursor += (size_t)count;
    return objval(out);
}

static bool tile_fits(const TypedTensor *source, int rank, const int *origin, const int *shape) {
    if (source->rank != rank)
        return false;
    for (int d = 0; d < rank; d++)
        if (origin[d] > source->shape[d] || shape[d] > source->shape[d] - origin[d])
            return false;
    return true;
}

static off_t tile_row(const TypedTensor *file, const TypedTensor *tile, const int *origin,
                      size_t row) {
    size_t offset = (size_t)origin[file->rank - 1];
    for (int d = file->rank - 2; d >= 0; d--) {
        size_t length = (size_t)tile->shape[d];
        offset += ((size_t)origin[d] + row % length) * (size_t)file->stride[d];
        row /= length;
    }
    return (off_t)(HEADER_SIZE + offset * dtype_info[file->dtype].size);
}

Val tensor_read_tile(VM *v, Val path_value, Val origin_value, Val shape_value) {
    const char *path = asstr(v, path_value);
    int origin[8], shape[8];
    int rank = typed_dimensions(v, origin_value, origin);
    if (typed_dimensions(v, shape_value, shape) != rank)
        vmerror(v, "tensor tile origin and shape ranks differ");
    TypedTensor file, layout;
    size_t size;
    int fd = open_tensor(v, path, &file, &size);
    if (!tile_fits(&file, rank, origin, shape) ||
        !typed_layout_checked(file.dtype, rank, shape, &layout)) {
        close(fd);
        vmerror(v, "tensor tile is outside the file");
    }
    size_t width = dtype_info[file.dtype].size;
    if (!typed_materializable(&layout) || layout.n * width > tensor_memory_available(v)) {
        close(fd);
        vmerror(v, "tensor tile exceeds the allocation limit or memory budget");
    }
    Obj *out = typed_new(v, file.dtype, rank, shape);
    size_t row_length = (size_t)shape[rank - 1];
    for (size_t i = 0; i < layout.n; i += row_length) {
        unsigned char *dst = typed_element(&out->as.tensor, (ptrdiff_t)i);
        if (!read_exact(fd, dst, row_length * width,
                        tile_row(&file, &layout, origin, i / row_length))) {
            close(fd);
            vmerror(v, "tensor tile read failed or data was truncated");
        }
    }
    close(fd);
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

static bool write_tensor(int fd, const void *context) {
    const TypedTensor *t = context;
    unsigned char header[HEADER_SIZE] = {0};
    memcpy(header, magic, sizeof magic - 1);
    header[8] = (unsigned char)t->dtype + 1;
    header[9] = (unsigned char)t->rank;
    header[10] = 1;
    put_u64(header + 16, HEADER_SIZE);
    put_u64(header + 24, t->n * dtype_info[t->dtype].size);
    for (int i = 0; i < t->rank; i++)
        put_u64(header + 32 + 8 * i, (uint64_t)t->shape[i]);
    return write_exact(fd, header, sizeof header) && write_payload(fd, t);
}

bool offload_read_chunk(const Buffer *buffer, size_t byte_offset, unsigned char *dst,
                        size_t bytes) {
    return read_exact(buffer->spill_fd, dst, bytes, (off_t)(HEADER_SIZE + byte_offset));
}

Val offload_snapshot(VM *v, ADOffloadPlan *plan, const TypedTensor *input) {
    require_little_endian(v);
    if (input->n > (PTRDIFF_MAX - HEADER_SIZE) / sizeof(double))
        vmerror_kind(v, "disk_budget", "activation file exceeds addressable storage");
    size_t bytes = input->n * sizeof(double);
    if (bytes + HEADER_SIZE > plan->limit - plan->live_bytes)
        vmerror_kind(v, "disk_budget", "activation file would exceed its offload disk budget");
    size_t total = bytes + HEADER_SIZE;
    if (total > tensor_memory_unlimited() - v->tensor_memory.mapped)
        vmerror_kind(v, "memory_budget", "activation mapping byte count overflow");

    TypedTensor out = typed_layout(v, DT_F64, input->rank, input->shape);
    static const char suffix[] = "/.arda-activation.XXXXXX";
    size_t length = strlen(plan->directory);
    if (length > SIZE_MAX - sizeof suffix)
        vmerror_kind(v, "io", "activation directory path is too long");
    char *path = mem(length + sizeof suffix);
    memcpy(path, plan->directory, length);
    memcpy(path + length, suffix, sizeof suffix);
    int fd = mkstemp(path);
    if (fd < 0) {
        int cause = errno;
        free(path);
        vmerror_kind(v, "io", "cannot create activation file: %s", strerror(cause));
    }
    if (unlink(path)) {
        int cause = errno;
        close(fd);
        unlink(path);
        free(path);
        vmerror_kind(v, "io", "cannot unlink activation file: %s", strerror(cause));
    }
    free(path);
    if (fcntl(fd, F_SETFD, FD_CLOEXEC) < 0 || !write_tensor(fd, input)) {
        int cause = errno;
        close(fd);
        vmerror_kind(v, "io", "cannot write activation file: %s", strerror(cause));
    }
    void *mapping = mmap(NULL, total, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) {
        int cause = errno;
        close(fd);
        vmerror_kind(v, "io", "cannot map activation file: %s", strerror(cause));
    }

    out.buffer = newobj(v, O_BUFFER);
    out.readonly = true;
    out.buffer->as.buffer = (Buffer){.data = (unsigned char *)mapping + HEADER_SIZE,
                                     .allocation = mapping,
                                     .bytes = bytes,
                                     .allocation_bytes = total,
                                     .alignment = _Alignof(max_align_t),
                                     .release = release_mapping,
                                     .readonly = true,
                                     .mapped = true,
                                     .offload = plan,
                                     .spill_fd = fd};
    plan->references++;
    plan->live_bytes += total;
    plan->live_files++;
    if (plan->live_bytes > plan->peak_bytes)
        plan->peak_bytes = plan->live_bytes;
    plan->written_bytes =
        total > (uint64_t)INT64_MAX - plan->written_bytes ? INT64_MAX : plan->written_bytes + total;
    if (plan->written_files < INT64_MAX)
        plan->written_files++;
    v->tensor_memory.mapped += total;
    return objval(typed_wrap(v, out));
}

/* The writer uses ordinary error returns so the temporary is always removed
 * and the caller can release any open descriptor and workspace before throwing. */
static int publish_file(const char *path, bool (*writer)(int, const void *), const void *context) {
    static const char suffix[] = ".tmp.XXXXXX";
    size_t length = strlen(path);
    if (!length || length > SIZE_MAX - sizeof suffix)
        return EINVAL;
    char *temporary = mem(length + sizeof suffix);
    memcpy(temporary, path, length);
    memcpy(temporary + length, suffix, sizeof suffix);
    int fd = mkstemp(temporary);
    if (fd < 0) {
        int cause = errno;
        free(temporary);
        return cause;
    }
    bool ok = writer(fd, context) && fsync(fd) == 0;
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
    return cause;
}

static bool disk_budget_ok(Val budget, size_t total) {
    return budget.t == V_NIL || (budget.t == V_INT && budget.i >= 0 && (uint64_t)budget.i >= total);
}

Val tensor_save(VM *v, Val path_value, Val input, Val budget) {
    require_little_endian(v);
    const char *path = asstr(v, path_value);
    TypedTensor t = typed_tensor(v, input);
    size_t bytes = t.n * dtype_info[t.dtype].size;
    if (bytes > PTRDIFF_MAX - HEADER_SIZE)
        vmerror(v, "tensor file exceeds addressable size");
    if (!disk_budget_ok(budget, HEADER_SIZE + bytes))
        vmerror(v, "tensor file exceeds disk budget or budget is invalid");
    int cause = publish_file(path, write_tensor, &t);
    if (cause)
        vmerror(v, "cannot publish tensor file: %s", strerror(cause));
    return nil();
}

typedef struct {
    const TypedTensor *file, *tile;
    const int *origin;
    int source;
    size_t file_size, scratch_size;
    unsigned char *scratch;
} TileWrite;

static bool write_tile_snapshot(int fd, const void *context) {
    const TileWrite *write = context;
    for (size_t pos = 0; pos < write->file_size;) {
        size_t count = write->file_size - pos;
        if (count > write->scratch_size)
            count = write->scratch_size;
        if (!read_exact(write->source, write->scratch, count, (off_t)pos) ||
            !write_exact(fd, write->scratch, count))
            return false;
        pos += count;
    }
    const TypedTensor *tile = write->tile;
    size_t width = dtype_info[tile->dtype].size;
    size_t per_chunk = write->scratch_size / width;
    size_t row_length = (size_t)tile->shape[tile->rank - 1];
    bool contiguous = typed_contiguous(tile);
    for (size_t row = 0; row < tile->n / row_length; row++) {
        off_t row_offset = tile_row(write->file, tile, write->origin, row);
        for (size_t col = 0; col < row_length;) {
            size_t count = row_length - col;
            if (count > per_chunk)
                count = per_chunk;
            size_t start = row * row_length + col;
            const unsigned char *data;
            if (contiguous)
                data = typed_element(tile, (ptrdiff_t)(tile->offset + start));
            else {
                for (size_t i = 0; i < count; i++)
                    memcpy(write->scratch + i * width,
                           typed_element(tile, typed_offset(tile, start + i)), width);
                data = write->scratch;
            }
            if (!write_at(fd, data, count * width, row_offset + (off_t)(col * width)))
                return false;
            col += count;
        }
    }
    return true;
}

Val tensor_write_tile(VM *v, Val path_value, Val origin_value, Val input, Val budget) {
    const char *path = asstr(v, path_value);
    TypedTensor tile = typed_tensor(v, input), file;
    int origin[8];
    int rank = typed_dimensions(v, origin_value, origin);
    if (rank != tile.rank)
        vmerror(v, "tensor tile origin and input ranks differ");
    size_t size;
    int fd = open_tensor(v, path, &file, &size);
    if (tile.dtype != file.dtype || !tile_fits(&file, rank, origin, tile.shape)) {
        close(fd);
        vmerror(v, "tensor tile dtype differs or tile is outside the file");
    }
    if (!disk_budget_ok(budget, size)) {
        close(fd);
        vmerror(v, "tensor file exceeds disk budget or budget is invalid");
    }
    if (!tile.n) {
        close(fd);
        return nil();
    }
    size_t bytes = size < IO_BLOCK ? size : IO_BLOCK;
    unsigned char *scratch = tensor_workspace_acquire(v, bytes);
    if (!scratch) {
        close(fd);
        vmerror(v, "unable to reserve tensor tile workspace within the memory budget");
    }
    TileWrite write = {.file = &file,
                       .tile = &tile,
                       .origin = origin,
                       .source = fd,
                       .file_size = size,
                       .scratch_size = bytes,
                       .scratch = scratch};
    int cause = publish_file(path, write_tile_snapshot, &write);
    close(fd);
    tensor_workspace_release(v, scratch);
    if (cause)
        vmerror(v, "cannot publish tensor tile: %s", strerror(cause));
    return nil();
}

static void report(VM *v, Obj *map, const char *key, Val value) {
    tableset(v, map, &map->as.map.tab, key, value);
}

static long double real_sample(TensorScalar x) {
    return x.kind <= SC_INT ? (long double)x.i : (long double)x.re;
}

static long double centered_sample(TensorScalar x, TensorScalar origin) {
    if (x.kind > SC_INT)
        return (long double)x.re - origin.re;
    // Subtract as magnitude first so nearby Int64s stay distinct even when
    // long double has the same significand width as double.
    return x.i >= origin.i ? (long double)((uint64_t)x.i - (uint64_t)origin.i)
                           : -(long double)((uint64_t)origin.i - (uint64_t)x.i);
}

typedef struct {
    TensorScalar origin;
    uint64_t count;
    long double mean, squares;
} StreamMoments;

static bool accumulate_block(StreamMoments *m, DType dtype, const unsigned char *data, size_t n) {
    size_t width = dtype_info[dtype].size;
    for (size_t i = 0; i < n; i++) {
        TensorScalar x = dtype_load(dtype, data + i * width);
        if (x.kind == SC_REAL && !isfinite(x.re))
            return false;
        if (!m->count)
            m->origin = x;
        moments_update(++m->count, centered_sample(x, m->origin), &m->mean, &m->squares);
    }
    return isfinite(m->mean) && isfinite(m->squares);
}

typedef enum { READ_IDLE, READ_PENDING, READ_DONE, READ_STOP } ReadState;
typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t changed;
    pthread_t thread;
    int fd;
    ReadState state;
    unsigned char *buffer;
    off_t offset;
    size_t bytes;
    bool ok;
} StreamReader;

static void *stream_worker(void *context) {
    StreamReader *r = context;
    pthread_mutex_lock(&r->lock);
    while (r->state != READ_STOP) {
        if (r->state != READ_PENDING) {
            pthread_cond_wait(&r->changed, &r->lock);
            continue;
        }
        unsigned char *buffer = r->buffer;
        size_t bytes = r->bytes;
        off_t offset = r->offset;
        pthread_mutex_unlock(&r->lock);
        bool ok = read_exact(r->fd, buffer, bytes, offset);
        pthread_mutex_lock(&r->lock);
        r->ok = ok;
        r->state = READ_DONE;
        pthread_cond_signal(&r->changed);
    }
    pthread_mutex_unlock(&r->lock);
    return NULL;
}

static int stream_start(StreamReader *r) {
    int error = pthread_mutex_init(&r->lock, NULL);
    if (error)
        return error;
    error = pthread_cond_init(&r->changed, NULL);
    if (!error) {
        error = pthread_create(&r->thread, NULL, stream_worker, r);
        if (!error)
            return 0;
        pthread_cond_destroy(&r->changed);
    }
    pthread_mutex_destroy(&r->lock);
    return error;
}

static void stream_request(StreamReader *r, unsigned char *buffer, size_t bytes, off_t offset) {
    pthread_mutex_lock(&r->lock);
    r->buffer = buffer;
    r->bytes = bytes;
    r->offset = offset;
    r->state = READ_PENDING;
    pthread_cond_signal(&r->changed);
    pthread_mutex_unlock(&r->lock);
}

static bool stream_wait(StreamReader *r) {
    pthread_mutex_lock(&r->lock);
    while (r->state == READ_PENDING)
        pthread_cond_wait(&r->changed, &r->lock);
    bool ok = r->ok;
    pthread_mutex_unlock(&r->lock);
    return ok;
}

static void stream_stop(StreamReader *r) {
    stream_wait(r);
    pthread_mutex_lock(&r->lock);
    r->state = READ_STOP;
    pthread_cond_signal(&r->changed);
    pthread_mutex_unlock(&r->lock);
    pthread_join(r->thread, NULL);
    pthread_cond_destroy(&r->changed);
    pthread_mutex_destroy(&r->lock);
}

static const char *stream_moments(int fd, const TypedTensor *file, StreamMoments *m,
                                  unsigned char *scratch, size_t count, bool prefetch) {
    static const char read_error[] = "tensor stream read failed or data was truncated";
    static const char moments_error[] =
        "stream_stats requires finite values and representable moments";
    size_t width = dtype_info[file->dtype].size;
    StreamReader reader = {.fd = fd};
    if (prefetch && stream_start(&reader))
        return "unable to start stream prefetch worker";

    const char *error = NULL;
    unsigned char *current = scratch;
    for (size_t pos = 0; pos < file->n;) {
        size_t batch = file->n - pos < count ? file->n - pos : count;
        size_t next = pos + batch, remaining = file->n - next;
        size_t next_count = remaining < count ? remaining : count;
        bool ok = prefetch && pos
                      ? stream_wait(&reader)
                      : read_exact(fd, current, batch * width, (off_t)(HEADER_SIZE + pos * width));
        if (!ok) {
            error = read_error;
            break;
        }
        unsigned char *other = current == scratch ? scratch + count * width : scratch;
        if (prefetch && next_count)
            stream_request(&reader, other, next_count * width, (off_t)(HEADER_SIZE + next * width));
        if (!accumulate_block(m, file->dtype, current, batch)) {
            error = moments_error;
            break;
        }
        if (prefetch)
            current = other;
        pos = next;
    }
    if (prefetch)
        stream_stop(&reader);
    return error;
}

Val tensor_stream_stats(VM *v, Val path_value, Val block, bool plan_only, bool prefetch) {
    const char *path = asstr(v, path_value);
    size_t count = block.t == V_NIL ? 65536 : (size_t)block_elements(v, block);
    TypedTensor file;
    size_t size;
    int fd = open_tensor(v, path, &file, &size);
    if (dtype_info[file.dtype].kind == SC_COMPLEX) {
        close(fd);
        vmerror(v, "stream_stats requires a real tensor dtype");
    }
    if (count > file.n)
        count = file.n;
    prefetch &= file.n > count;
    size_t width = dtype_info[file.dtype].size, bytes = count * width * (prefetch ? 2u : 1u);
    if (plan_only) {
        close(fd);
        size_t available = tensor_memory_available(v);
        Obj *out = newobj(v, O_MAP);
        report(v, out, "elements", integer((int64_t)file.n));
        report(v, out, "block_elements", integer((int64_t)count));
        report(v, out, "workspace_bytes", integer((int64_t)bytes));
        report(v, out, "available_bytes", integer((int64_t)available));
        report(v, out, "output_bytes", integer(0));
        report(v, out, "fits", boolean(bytes <= available));
        report(v, out, "prefetch", boolean(prefetch));
        report(v, out, "io_bytes", integer((int64_t)(file.n * width)));
        return objval(out);
    }
    unsigned char *scratch = bytes ? tensor_workspace_acquire(v, bytes) : NULL;
    if (bytes && !scratch) {
        close(fd);
        vmerror_kind(v, "memory_budget",
                     "unable to reserve streaming workspace within the memory budget");
    }
    StreamMoments m = {0};
    const char *error = stream_moments(fd, &file, &m, scratch, count, prefetch);
    close(fd);
    if (scratch)
        tensor_workspace_release(v, scratch);
    if (error)
        vmerror(v, "%s", error);
    long double mean = real_sample(m.origin) + m.mean;
    long double variance = m.count ? m.squares / m.count : 0;
    if (fabsl(mean) > DBL_MAX || variance > DBL_MAX)
        vmerror(v, "streaming moments exceed the Float64 range");
    Obj *out = newobj(v, O_MAP);
    report(v, out, "count", integer((int64_t)m.count));
    report(v, out, "mean", m.count ? number((double)mean) : nil());
    report(v, out, "variance", m.count ? number((double)fmaxl(0, variance)) : nil());
    return objval(out);
}
