#include "activation_store.h"
#include "moments.h"
#include "record_io.h"
#include "simulation_checkpoint.h"
#include "tensor_file.h"

#include <fcntl.h>
#include <float.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum { HEADER_SIZE = 128, IO_BLOCK = 65536 };
static const unsigned char magic[] = "ARDATEN1";
static const char temporary_suffix[] = ".tmp.XXXXXX";

typedef struct {
    TypedTensor layout;
    int fd;
    size_t size;
    char *path, *temporary;
} TensorOutput;

static TensorOutput *active_output(VM *v, Val value) {
    if (!isobj(value, O_TENSOR_OUTPUT))
        vmerror_kind(v, "type", "expected a tensor output handle");
    TensorOutput *output = (TensorOutput *)(value.o + 1);
    if (output->fd < 0)
        vmerror_kind(v, "io", "tensor output is closed");
    return output;
}

void tensor_file_require_little_endian(VM *v) {
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

bool tensor_file_write(int fd, const void *data, size_t bytes) {
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
        header[9] > 8 || header[10] != 1 || record_u64(header + 16) != HEADER_SIZE)
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
        uint64_t dim = record_u64(header + 32 + 8 * i);
        if (dim > INT_MAX || (i >= rank && dim))
            return false;
        shape[i] = (int)dim;
    }
    if (!typed_layout_checked(dtype, rank, shape, tensor))
        return false;
    uint64_t stored_bytes = record_u64(header + 24);
    return stored_bytes == tensor->n * dtype_info[dtype].size &&
           stored_bytes == (uintmax_t)size - HEADER_SIZE;
}

int tensor_file_open(const char *path, TensorFile *file) {
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0)
        return errno;
    struct stat info;
    unsigned char header[HEADER_SIZE];
    if (fstat(fd, &info) || !S_ISREG(info.st_mode) || !read_exact(fd, header, sizeof header, 0) ||
        !decode_header(header, info.st_size, &file->layout)) {
        close(fd);
        return EINVAL;
    }
    file->bytes = (size_t)info.st_size;
    file->fd = fd;
    return 0;
}

static int open_tensor(VM *v, const char *path, TypedTensor *tensor, size_t *size) {
    tensor_file_require_little_endian(v);
    TensorFile file;
    int cause = tensor_file_open(path, &file);
    if (cause == EINVAL)
        vmerror(v, "invalid tensor file header, shape, endian marker or length");
    if (cause)
        vmerror(v, "cannot open tensor file '%s': %s", path, strerror(cause));
    *tensor = file.layout;
    *size = file.bytes;
    return file.fd;
}

bool tensor_file_read(const TensorFile *file, size_t start, size_t count, void *data) {
    if (start > file->layout.n || count > file->layout.n - start) {
        errno = EINVAL;
        return false;
    }
    size_t width = dtype_info[file->layout.dtype].size;
    return read_exact(file->fd, data, count * width, (off_t)(HEADER_SIZE + start * width));
}

bool tensor_file_patch(int fd, const TypedTensor *layout, size_t start, size_t count,
                       const void *data) {
    if (start > layout->n || count > layout->n - start) {
        errno = EINVAL;
        return false;
    }
    size_t width = dtype_info[layout->dtype].size;
    return write_at(fd, data, count * width, (off_t)(HEADER_SIZE + start * width));
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
    if (read_exact(fd, typed_element(&out->as.tensor, 0), (size_t)count * width,
                   (off_t)(HEADER_SIZE + start * width)))
        return out;
    tensor_buffer_discard(v, out->as.tensor.buffer);
    return NULL;
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
    out->as.tensor_blocks.rank = t.rank;
    memcpy(out->as.tensor_blocks.shape, t.shape, sizeof t.shape);
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
    int origin[8], shape[8];
    int rank = typed_dimensions(v, origin_value, origin);
    if (typed_dimensions(v, shape_value, shape) != rank)
        vmerror(v, "tensor tile origin and shape ranks differ");
    TypedTensor file, layout;
    bool close_source = !isobj(path_value, O_TENSOR_OUTPUT);
    int fd;
    if (close_source) {
        size_t size;
        fd = open_tensor(v, asstr(v, path_value), &file, &size);
    } else {
        TensorOutput *output = active_output(v, path_value);
        file = output->layout;
        fd = output->fd;
    }
    if (!tile_fits(&file, rank, origin, shape) ||
        !typed_layout_checked(file.dtype, rank, shape, &layout)) {
        if (close_source)
            close(fd);
        vmerror(v, "tensor tile is outside the file");
    }
    size_t width = dtype_info[file.dtype].size;
    if (!typed_materializable(&layout) || layout.n * width > tensor_memory_available(v)) {
        if (close_source)
            close(fd);
        vmerror(v, "tensor tile exceeds the allocation limit or memory budget");
    }
    Obj *out = typed_new(v, file.dtype, rank, shape);
    size_t row_length = (size_t)shape[rank - 1];
    for (size_t i = 0; i < layout.n; i += row_length) {
        unsigned char *dst = typed_element(&out->as.tensor, (ptrdiff_t)i);
        if (!read_exact(fd, dst, row_length * width,
                        tile_row(&file, &layout, origin, i / row_length))) {
            if (close_source)
                close(fd);
            tensor_buffer_discard(v, out->as.tensor.buffer);
            vmerror(v, "tensor tile read failed or data was truncated");
        }
    }
    if (close_source)
        close(fd);
    return objval(out);
}

static bool write_payload(int fd, const TypedTensor *t, Sha256 *hash) {
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
        if (hash)
            sha256_add(hash, data, count * width);
        if (!tensor_file_write(fd, data, count * width))
            return false;
        start += count;
    }
    return true;
}

static bool write_header(int fd, const TypedTensor *t) {
    unsigned char header[HEADER_SIZE] = {0};
    memcpy(header, magic, sizeof magic - 1);
    header[8] = (unsigned char)t->dtype + 1;
    header[9] = (unsigned char)t->rank;
    header[10] = 1;
    record_put_u64(header + 16, HEADER_SIZE);
    record_put_u64(header + 24, t->n * dtype_info[t->dtype].size);
    for (int i = 0; i < t->rank; i++)
        record_put_u64(header + 32 + 8 * i, (uint64_t)t->shape[i]);
    return tensor_file_write(fd, header, sizeof header);
}

static bool write_tensor(int fd, const void *context) {
    return write_header(fd, context) && write_payload(fd, context, NULL);
}

bool offload_read_chunk(const Buffer *buffer, size_t byte_offset, unsigned char *dst,
                        size_t bytes) {
    return read_exact(buffer->spill_fd, dst, bytes, (off_t)(HEADER_SIZE + byte_offset));
}

Val offload_snapshot(VM *v, ADOffloadPlan *plan, const TypedTensor *input) {
    tensor_file_require_little_endian(v);
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
int tensor_publish_file(const char *path, bool (*writer)(int, const void *), const void *context) {
    size_t length = strlen(path);
    if (!length || length > SIZE_MAX - sizeof temporary_suffix)
        return EINVAL;
    char *temporary = mem(length + sizeof temporary_suffix);
    memcpy(temporary, path, length);
    memcpy(temporary + length, temporary_suffix, sizeof temporary_suffix);
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

bool tensor_file_fits_budget(Val budget, size_t total) {
    return budget.t == V_NIL || (budget.t == V_INT && budget.i >= 0 && (uint64_t)budget.i >= total);
}

bool tensor_file_size(const TypedTensor *layout, size_t *size) {
    size_t width = dtype_info[layout->dtype].size;
    if (layout->n > (PTRDIFF_MAX - HEADER_SIZE) / width)
        return false;
    *size = HEADER_SIZE + layout->n * width;
    off_t length = (off_t)*size;
    return length >= 0 && (uintmax_t)length == (uintmax_t)*size;
}

typedef struct {
    const TypedTensor *layout;
    bool (*writer)(int, const void *);
    const void *context;
} PayloadWrite;

static bool write_stream(int fd, const void *context) {
    const PayloadWrite *stream = context;
    return write_header(fd, stream->layout) && stream->writer(fd, stream->context);
}

int tensor_publish_payload(const char *path, const TypedTensor *layout,
                           bool (*writer)(int, const void *), const void *context) {
    PayloadWrite stream = {.layout = layout, .writer = writer, .context = context};
    return tensor_publish_file(path, write_stream, &stream);
}

void tensor_output_abort(Obj *object) {
    TensorOutput *output = (TensorOutput *)(object + 1);
    if (output->fd >= 0) {
        close(output->fd);
        output->fd = -1;
        unlink(output->temporary);
    }
}

Val tensor_open_output(VM *v, Val path_value, Val shape, Val type, Val budget) {
    tensor_file_require_little_endian(v);
    const char *path = asstr(v, path_value);
    int dimensions[8];
    int rank = typed_dimensions(v, shape, dimensions);
    TypedTensor layout = typed_layout(v, dtype_parse(v, type), rank, dimensions);
    size_t size;
    if (!tensor_file_size(&layout, &size))
        vmerror_kind(v, "shape", "tensor output exceeds the addressable file size");
    if (!tensor_file_fits_budget(budget, size))
        vmerror_kind(v, "disk_budget",
                     "tensor output exceeds the disk budget or budget is invalid");
    size_t length = strlen(path);
    if (!length || length > (SIZE_MAX - sizeof(TensorOutput) - sizeof temporary_suffix - 1) / 2)
        vmerror_kind(v, "io", "invalid tensor output path");
    Obj *object = newobj_extra(v, O_TENSOR_OUTPUT,
                               sizeof(TensorOutput) + 2 * length + 1 + sizeof temporary_suffix);
    TensorOutput *output = (TensorOutput *)(object + 1);
    *output = (TensorOutput){.layout = layout, .size = size, .fd = -1};
    output->path = (char *)(output + 1);
    output->temporary = output->path + length + 1;
    memcpy(output->path, path, length + 1);
    memcpy(output->temporary, path, length);
    memcpy(output->temporary + length, temporary_suffix, sizeof temporary_suffix);
    output->fd = mkstemp(output->temporary);
    if (output->fd < 0)
        vmerror_kind(v, "io", "cannot create tensor output: %s", strerror(errno));
    if (fcntl(output->fd, F_SETFD, FD_CLOEXEC) < 0 || !write_header(output->fd, &layout) ||
        ftruncate(output->fd, (off_t)size)) {
        int cause = errno;
        tensor_output_abort(object);
        vmerror_kind(v, "io", "cannot initialize tensor output: %s", strerror(cause));
    }
    return objval(object);
}

Val tensor_commit_output(VM *v, Val value) {
    TensorOutput *output = active_output(v, value);
    if (fsync(output->fd)) {
        int cause = errno;
        tensor_output_abort(value.o);
        vmerror_kind(v, "io", "cannot synchronize tensor output: %s", strerror(cause));
    }
    int fd = output->fd;
    output->fd = -1;
    if (close(fd) || rename(output->temporary, output->path)) {
        int cause = errno;
        unlink(output->temporary);
        vmerror_kind(v, "io", "cannot publish tensor output: %s", strerror(cause));
    }
    return nil();
}

Val tensor_write_block(VM *v, Val value, Val start, Val input) {
    TensorOutput *output = active_output(v, value);
    TypedTensor tile = typed_tensor(v, input);
    if (start.t != V_INT || start.i < 0 || (uint64_t)start.i > output->layout.n ||
        tile.n > output->layout.n - (size_t)start.i || tile.dtype != output->layout.dtype ||
        !typed_contiguous(&tile))
        vmerror_kind(v, "shape",
                     "output block requires a matching dtype, valid range and packed data");
    if (tile.n && !tensor_file_patch(output->fd, &output->layout, (size_t)start.i, tile.n,
                                     typed_element(&tile, (ptrdiff_t)tile.offset))) {
        int cause = errno;
        tensor_output_abort(value.o);
        vmerror_kind(v, "io", "cannot write tensor output block: %s", strerror(cause));
    }
    return nil();
}

Val tensor_truncate_output(VM *v, Val value, Val count) {
    TensorOutput *output = active_output(v, value);
    if (output->layout.rank != 1 || count.t != V_INT || count.i < 0 ||
        (uint64_t)count.i > output->layout.n)
        vmerror_kind(v, "shape", "output truncation requires a shorter vector length");
    int length = (int)count.i;
    TypedTensor layout = typed_layout(v, output->layout.dtype, 1, &length);
    size_t size;
    if (!tensor_file_size(&layout, &size))
        vmerror_kind(v, "shape", "truncated output size is invalid");
    if (lseek(output->fd, 0, SEEK_SET) < 0 || !write_header(output->fd, &layout) ||
        ftruncate(output->fd, (off_t)size)) {
        int cause = errno;
        tensor_output_abort(value.o);
        vmerror_kind(v, "io", "cannot truncate tensor output: %s", strerror(cause));
    }
    output->layout = layout;
    output->size = size;
    return nil();
}

enum { CHECKPOINT_DIMS = 8, CHECKPOINT_HEADER = 96 + 8 * CHECKPOINT_WORDS, CHECKPOINT_INT64 = 9 };
static const unsigned char legacy_checkpoint_magic[] = "ARDACKP1";
static const unsigned char checkpoint_magic[] = "ARDACKP2";

typedef struct {
    unsigned char header[CHECKPOINT_HEADER];
    TypedTensor tensor;
    unsigned char scalar[8];
} CheckpointWrite;

static bool checkpoint_finite(const TypedTensor *tensor) {
    bool packed = typed_contiguous(tensor);
    for (size_t i = 0; i < tensor->n; i++) {
        ptrdiff_t position = packed ? (ptrdiff_t)(tensor->offset + i) : typed_offset(tensor, i);
        double value;
        memcpy(&value, typed_element(tensor, position), sizeof value);
        if (!isfinite(value))
            return false;
    }
    return true;
}

static bool write_checkpoint(int fd, const void *context) {
    const CheckpointWrite *record = context;
    Sha256 hash = sha256_begin();
    unsigned char digest[SHA256_BYTES];
    sha256_add(&hash, record->header, sizeof record->header);
    if (!tensor_file_write(fd, record->header, sizeof record->header))
        return false;
    if (record->tensor.rank) {
        if (!write_payload(fd, &record->tensor, &hash))
            return false;
    } else {
        sha256_add(&hash, record->scalar, sizeof record->scalar);
        if (!tensor_file_write(fd, record->scalar, sizeof record->scalar))
            return false;
    }
    sha256_finish(&hash, digest);
    return tensor_file_write(fd, digest, sizeof digest);
}

Val checkpoint_save_state(VM *v, Val path, CheckpointKind kind, Val state,
                          const uint64_t words[CHECKPOINT_WORDS], Val budget) {
    tensor_file_require_little_endian(v);
    const char *filename = asstr(v, path);
    CheckpointWrite record = {0};
    size_t bytes = sizeof record.scalar;
    int format = 0;
    if (isobj(state, O_ARRAY)) {
        record.tensor = typed_tensor(v, state);
        if (!typed_materializable(&record.tensor))
            vmerror_kind(v, "checkpoint_format", "checkpoint tensor exceeds the heap limit");
        bytes = record.tensor.n * sizeof(double);
        format = record.tensor.rank;
        for (int i = 0; i < format; i++)
            record_put_u64(record.header + 32 + 8 * i, (uint64_t)record.tensor.shape[i]);
    } else if (state.t == V_INT) {
        format = CHECKPOINT_INT64;
        record_put_u64(record.scalar, (uint64_t)state.i);
    } else if (state.t == V_NUM) {
        record_put_u64(record.scalar, checkpoint_real(v, state));
    } else {
        vmerror_kind(v, "checkpoint_format", "checkpoint state must be a scalar or numc array");
    }
    if (!tensor_file_fits_budget(budget, CHECKPOINT_HEADER + bytes + SHA256_BYTES))
        vmerror_kind(v, "disk_budget", "checkpoint exceeds the disk budget or budget is invalid");
    if (record.tensor.rank && !checkpoint_finite(&record.tensor))
        vmerror_kind(v, "checkpoint_format", "checkpoint state must be finite");
    memcpy(record.header, checkpoint_magic, sizeof checkpoint_magic - 1);
    record_put_u64(record.header + 8, kind);
    record_put_u64(record.header + 16, (uint64_t)format);
    record_put_u64(record.header + 24, bytes);
    for (int i = 0; i < CHECKPOINT_WORDS; i++)
        record_put_u64(record.header + 96 + 8 * i, words[i]);
    int cause = tensor_publish_file(filename, write_checkpoint, &record);
    if (cause)
        vmerror_kind(v, "io", "cannot publish checkpoint: %s", strerror(cause));
    return nil();
}

static bool decode_checkpoint(const unsigned char *header, off_t size, CheckpointKind kind,
                              int *format, TypedTensor *layout, bool *checksummed) {
    uint64_t type = record_u64(header + 16);
    *checksummed = !memcmp(header, checkpoint_magic, sizeof checkpoint_magic - 1);
    if ((!*checksummed &&
         memcmp(header, legacy_checkpoint_magic, sizeof legacy_checkpoint_magic - 1)) ||
        record_u64(header + 8) != (uint64_t)kind || type > CHECKPOINT_INT64 ||
        size < CHECKPOINT_HEADER)
        return false;
    *format = (int)type;
    int rank = type <= CHECKPOINT_DIMS ? *format : 0;
    int shape[CHECKPOINT_DIMS];
    for (int i = 0; i < CHECKPOINT_DIMS; i++) {
        uint64_t extent = record_u64(header + 32 + 8 * i);
        if (extent > ARDA_MAX_ELEMENTS || (i >= rank && extent))
            return false;
        shape[i] = (int)extent;
    }
    if (rank &&
        (!typed_layout_checked(DT_F64, rank, shape, layout) || !typed_materializable(layout)))
        return false;
    size_t payload = (rank ? layout->n : 1) * sizeof(double);
    size_t trailer = *checksummed ? SHA256_BYTES : 0;
    return record_u64(header + 24) == payload &&
           (uintmax_t)size == CHECKPOINT_HEADER + payload + trailer;
}

Val checkpoint_load_state(VM *v, Val path, CheckpointKind kind, uint64_t words[CHECKPOINT_WORDS]) {
    tensor_file_require_little_endian(v);
    const char *filename = asstr(v, path);
    int fd = open(filename, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0)
        vmerror_kind(v, "io", "cannot open checkpoint: %s", strerror(errno));
    struct stat file;
    unsigned char header[CHECKPOINT_HEADER];
    int format;
    bool checksummed;
    TypedTensor layout = {0};
    if (fstat(fd, &file) || !S_ISREG(file.st_mode) || !read_exact(fd, header, sizeof header, 0) ||
        !decode_checkpoint(header, file.st_size, kind, &format, &layout, &checksummed)) {
        close(fd);
        vmerror_kind(v, "checkpoint_format", "invalid checkpoint type, layout or length");
    }
    for (int i = 0; i < CHECKPOINT_WORDS; i++)
        words[i] = record_u64(header + 96 + 8 * i);
    size_t bytes = (layout.rank ? layout.n : 1) * sizeof(double);
    if (layout.rank && bytes > tensor_memory_available(v)) {
        close(fd);
        vmerror_kind(v, "memory_budget", "checkpoint state exceeds the tensor budget");
    }
    unsigned char scalar[8];
    Obj *out = layout.rank ? tensor_new(v, layout.rank, layout.shape) : NULL;
    void *data = out ? (void *)out->as.array.data : scalar;
    bool ok = read_exact(fd, data, bytes, CHECKPOINT_HEADER);
    if (ok && checksummed) {
        unsigned char stored[SHA256_BYTES], digest[SHA256_BYTES];
        Sha256 hash = sha256_begin();
        sha256_add(&hash, header, sizeof header);
        sha256_add(&hash, data, bytes);
        sha256_finish(&hash, digest);
        ok = read_exact(fd, stored, sizeof stored, (off_t)(CHECKPOINT_HEADER + bytes)) &&
             !memcmp(stored, digest, sizeof stored);
    }
    close(fd);
    if (out) {
        TypedTensor tensor = typed_tensor(v, objval(out));
        if (ok && checkpoint_finite(&tensor))
            return objval(out);
        tensor_buffer_discard(v, out->as.array.buffer);
    } else if (ok) {
        uint64_t bits = record_u64(scalar);
        if (format == CHECKPOINT_INT64)
            return integer(bits <= INT64_MAX ? (int64_t)bits
                                             : INT64_MIN + (int64_t)(bits - (UINT64_C(1) << 63)));
        double value = checkpoint_double(bits);
        if (isfinite(value))
            return number(value);
    }
    vmerror_kind(v, "checkpoint_format",
                 "checkpoint state is truncated, nonfinite or has an invalid checksum");
}

Val tensor_save(VM *v, Val path_value, Val input, Val budget) {
    tensor_file_require_little_endian(v);
    const char *path = asstr(v, path_value);
    TypedTensor t = typed_tensor(v, input);
    size_t bytes = t.n * dtype_info[t.dtype].size;
    if (bytes > PTRDIFF_MAX - HEADER_SIZE)
        vmerror(v, "tensor file exceeds addressable size");
    if (!tensor_file_fits_budget(budget, HEADER_SIZE + bytes))
        vmerror(v, "tensor file exceeds disk budget or budget is invalid");
    int cause = tensor_publish_file(path, write_tensor, &t);
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

static bool write_tile_data(int fd, const TileWrite *write) {
    const TypedTensor *tile = write->tile;
    size_t width = dtype_info[tile->dtype].size;
    size_t row_length = (size_t)tile->shape[tile->rank - 1];
    bool contiguous = typed_contiguous(tile);
    size_t per_chunk = write->scratch ? write->scratch_size / width : row_length;
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

static bool write_tile_snapshot(int fd, const void *context) {
    const TileWrite *write = context;
    for (size_t pos = 0; pos < write->file_size;) {
        size_t count = write->file_size - pos;
        if (count > write->scratch_size)
            count = write->scratch_size;
        if (!read_exact(write->source, write->scratch, count, (off_t)pos) ||
            !tensor_file_write(fd, write->scratch, count))
            return false;
        pos += count;
    }
    return write_tile_data(fd, write);
}

static Val write_output_tile(VM *v, Val value, const int *origin, const TypedTensor *tile,
                             Val budget) {
    TensorOutput *output = active_output(v, value);
    if (tile->dtype != output->layout.dtype ||
        !tile_fits(&output->layout, tile->rank, origin, tile->shape))
        vmerror_kind(v, "shape", "tensor tile dtype differs or tile is outside the output");
    if (!tensor_file_fits_budget(budget, output->size))
        vmerror_kind(v, "disk_budget",
                     "tensor output exceeds the disk budget or budget is invalid");
    if (!tile->n)
        return nil();
    size_t scratch_size = 0;
    unsigned char *scratch = NULL;
    if (!typed_contiguous(tile)) {
        size_t width = dtype_info[tile->dtype].size;
        size_t row_bytes = (size_t)tile->shape[tile->rank - 1] * width;
        scratch_size = row_bytes < IO_BLOCK ? row_bytes : IO_BLOCK;
        size_t available = tensor_memory_available(v);
        if (scratch_size > available)
            scratch_size = available - available % width;
        if (!scratch_size || !(scratch = tensor_workspace_acquire(v, scratch_size)))
            vmerror_kind(v, "memory_budget", "unable to reserve output tile workspace");
    }
    TileWrite write = {.file = &output->layout,
                       .tile = tile,
                       .origin = origin,
                       .scratch_size = scratch_size,
                       .scratch = scratch};
    bool ok = write_tile_data(output->fd, &write);
    int cause = errno;
    if (scratch)
        tensor_workspace_release(v, scratch);
    if (!ok) {
        tensor_output_abort(value.o);
        vmerror_kind(v, "io", "cannot write tensor output tile: %s", strerror(cause));
    }
    return nil();
}

Val tensor_write_tile(VM *v, Val path_value, Val origin_value, Val input, Val budget) {
    TypedTensor tile = typed_tensor(v, input), file;
    int origin[8];
    int rank = typed_dimensions(v, origin_value, origin);
    if (rank != tile.rank)
        vmerror(v, "tensor tile origin and input ranks differ");
    if (isobj(path_value, O_TENSOR_OUTPUT))
        return write_output_tile(v, path_value, origin, &tile, budget);
    const char *path = asstr(v, path_value);
    size_t size;
    int fd = open_tensor(v, path, &file, &size);
    if (tile.dtype != file.dtype || !tile_fits(&file, rank, origin, tile.shape)) {
        close(fd);
        vmerror(v, "tensor tile dtype differs or tile is outside the file");
    }
    if (!tensor_file_fits_budget(budget, size)) {
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
    int cause = tensor_publish_file(path, write_tile_snapshot, &write);
    close(fd);
    tensor_workspace_release(v, scratch);
    if (cause)
        vmerror(v, "cannot publish tensor tile: %s", strerror(cause));
    return nil();
}

static void report(VM *v, Obj *map, const char *key, Val value) {
    tableset(v, map, &map->as.map.tab, key, value);
}

Val tensor_file_info(VM *v, Val source) {
    TypedTensor layout = {0};
    if (isobj(source, O_TENSOR_BLOCKS)) {
        Obj *blocks = source.o;
        if (blocks->as.tensor_blocks.fd < 0)
            vmerror_kind(v, "io", "tensor blocks are closed");
        layout.rank = blocks->as.tensor_blocks.rank;
        memcpy(layout.shape, blocks->as.tensor_blocks.shape, sizeof layout.shape);
        layout.dtype = blocks->as.tensor_blocks.dtype;
        layout.n = blocks->as.tensor_blocks.elements;
    } else {
        size_t size;
        close(open_tensor(v, asstr(v, source), &layout, &size));
    }
    Obj *out = newobj(v, O_MAP);
    Obj *shape = newobj(v, O_LIST);
    report(v, out, "shape", objval(shape));
    for (int i = 0; i < layout.rank; i++)
        append(v, shape, integer(layout.shape[i]));
    report(v, out, "elements", integer((int64_t)layout.n));
    report(v, out, "dtype", objval(string(v, dtype_info[layout.dtype].name)));
    return objval(out);
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
