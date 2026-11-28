#include "experiment.h"
#include "metadata.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

enum { MANIFEST_HEADER = 16 };
static const unsigned char manifest_magic[8] = "ARDAEXP1";

typedef struct {
    Val manifest;
    size_t payload, limit;
} WriteManifest;

static bool write_manifest(int fd, const void *context) {
    const WriteManifest *job = context;
    unsigned char header[MANIFEST_HEADER], digest[SHA256_BYTES];
    memcpy(header, manifest_magic, sizeof manifest_magic);
    record_put_u64(header + 8, job->payload);
    RecordOutput out = {.fd = fd, .limit = job->limit, .hash = sha256_begin(), .checksummed = true};
    if (!record_emit(&out, header, sizeof header) || !metadata_binary(&out, job->manifest))
        return false;
    sha256_finish(&out.hash, digest);
    out.checksummed = false;
    return record_emit(&out, digest, sizeof digest) && record_flush(&out);
}

static Val save(VM *v, Val path, Val value, Val options) {
    const char *file = asstr(v, path);
    if (!isobj(value, O_MAP))
        vmerror_kind(v, "type", "experiment manifest must be a map");
    size_t limit = record_budget(v, options, METADATA_BYTES);
    const size_t framing = MANIFEST_HEADER + SHA256_BYTES;
    RecordOutput count = {.fd = -1, .limit = limit};
    if (!record_emit(&count, NULL, framing) || !metadata_binary(&count, value))
        record_error(v, "experiment manifest", errno);
    WriteManifest job = {.manifest = value, .payload = count.size - framing, .limit = limit};
    int cause = tensor_publish_file(file, write_manifest, &job);
    if (cause)
        record_error(v, "experiment manifest publication", cause);
    return integer((int64_t)count.size);
}

static int input_file(VM *v, Val path, uint64_t limit, uint64_t *size) {
    const char *file = asstr(v, path);
    int fd = open(file, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0)
        vmerror_kind(v, "io", "cannot open experiment input: %s", strerror(errno));
    struct stat st;
    if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size < 0) {
        close(fd);
        vmerror_kind(v, "io", "experiment input must be a regular file");
    }
    if ((uintmax_t)st.st_size > limit) {
        close(fd);
        vmerror_kind(v, "disk_budget", "experiment input exceeds the byte limit");
    }
    *size = (uint64_t)st.st_size;
    return fd;
}

static bool read_bytes(int fd, unsigned char *out, size_t bytes) {
    while (bytes) {
        ssize_t read_count = read(fd, out, bytes);
        if (read_count < 0 && errno == EINTR)
            continue;
        if (read_count <= 0)
            return false;
        bytes -= (size_t)read_count;
        out += read_count;
    }
    return true;
}

static bool at_end(int fd) {
    unsigned char extra;
    ssize_t read_count;
    do {
        read_count = read(fd, &extra, 1);
    } while (read_count < 0 && errno == EINTR);
    return read_count == 0;
}

static Val load(VM *v, Val filename, Val budget) {
    size_t limit = record_budget(v, budget, METADATA_BYTES);
    uint64_t bytes;
    int fd = input_file(v, filename, limit, &bytes);
    size_t size = (size_t)bytes, framing = MANIFEST_HEADER + SHA256_BYTES;
    if (size <= framing || size > tensor_memory_available(v)) {
        close(fd);
        if (size <= framing)
            vmerror_kind(v, "manifest_format", "experiment manifest is truncated");
        vmerror_kind(v, "memory_budget", "experiment manifest exceeds the tensor budget");
    }
    Obj *scratch = buffer_new(v, size);
    const unsigned char *data = scratch->as.buffer.data;
    bool ok = read_bytes(fd, scratch->as.buffer.data, size) && at_end(fd);
    close(fd);
    Val result;
    if (ok && !memcmp(data, manifest_magic, sizeof manifest_magic) &&
        record_u64(data + 8) == size - framing) {
        Sha256 hash = sha256_begin();
        unsigned char digest[SHA256_BYTES];
        sha256_add(&hash, data, size - SHA256_BYTES);
        sha256_finish(&hash, digest);
        ok = !memcmp(digest, data + size - SHA256_BYTES, SHA256_BYTES) &&
             metadata_decode(v, data + MANIFEST_HEADER, size - framing, &result) &&
             isobj(result, O_MAP);
    } else {
        ok = false;
    }
    tensor_buffer_discard(v, scratch);
    if (!ok)
        vmerror_kind(v, "manifest_format", "invalid manifest version, checksum or metadata");
    return result;
}

static Val fingerprint(VM *v, Val path, Val budget) {
    if (budget.t != V_INT || budget.i < 0 || (uint64_t)budget.i > UINT64_MAX / 8)
        vmerror_kind(v, "disk_budget", "fingerprint needs a nonnegative explicit byte limit");
    uint64_t size;
    int fd = input_file(v, path, (uint64_t)budget.i, &size);
    Sha256 hash = sha256_begin();
    unsigned char block[8192], digest[SHA256_BYTES];
    bool ok = true;
    for (uint64_t remaining = size; remaining;) {
        size_t length = remaining < sizeof block ? (size_t)remaining : sizeof block;
        if (!read_bytes(fd, block, length)) {
            ok = false;
            break;
        }
        sha256_add(&hash, block, length);
        remaining -= length;
    }
    ok = ok && at_end(fd);
    close(fd);
    if (!ok)
        vmerror_kind(v, "io", "experiment input changed size or became unreadable");
    sha256_finish(&hash, digest);
    char hex[2 * SHA256_BYTES + 1];
    for (int i = 0; i < SHA256_BYTES; i++)
        snprintf(hex + 2 * i, 3, "%02x", (unsigned)digest[i]);
    return objval(string(v, hex));
}

enum { SAVE, LOAD, FINGERPRINT };
static const char *const names[] = {"save", "load", "fingerprint"};

Obj *experiment_module(VM *v) {
    return named_native_module(v, "experiment", names, sizeof names / sizeof *names,
                               EXPERIMENT_BASE);
}

Val experiment_call(VM *v, int id, int argc, Val *args) {
    vm_require_file_access(v);
    switch (id - EXPERIMENT_BASE) {
    case SAVE:
        if (argc == 2 || argc == 3)
            return save(v, args[0], args[1], argc == 3 ? args[2] : nil());
        break;
    case LOAD:
        if (argc == 1 || argc == 2)
            return load(v, args[0], argc == 2 ? args[1] : nil());
        break;
    case FINGERPRINT:
        if (argc == 2)
            return fingerprint(v, args[0], args[1]);
        break;
    }
    vmerror_kind(v, "arity", "invalid experiment API arity");
}
