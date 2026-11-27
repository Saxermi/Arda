#include "file_compute.h"
#include "tensor_file.h"

#include <float.h>
#include <limits.h>
#include <unistd.h>

static int smaller(int a, int b) {
    return a < b ? a : b;
}

static void close_inputs(TensorFile *inputs, int count) {
    for (int i = 0; i < count; i++)
        close(inputs[i].fd);
}

static void open_inputs(VM *v, int count, const char *const *paths, TensorFile *inputs) {
    tensor_file_require_little_endian(v);
    for (int i = 0; i < count; i++) {
        int cause = tensor_file_open(paths[i], inputs + i);
        if (cause) {
            close_inputs(inputs, i);
            vmerror_kind(v, cause == EINVAL ? "file_format" : "io",
                         "cannot open tensor input '%s': %s", paths[i], strerror(cause));
        }
    }
}

static void write_result(VM *v, const char *path, TensorFile *inputs, int count,
                         const TypedTensor *layout, size_t scratch, Val budget, void **workspace,
                         bool (*writer)(int, const void *), const void *context) {
    size_t size;
    if (!tensor_file_size(layout, &size) || !tensor_file_fits_budget(budget, size)) {
        close_inputs(inputs, count);
        vmerror_kind(v, "disk_budget", "file result exceeds the disk budget or supported size");
    }
    *workspace = scratch ? tensor_workspace_acquire(v, scratch) : NULL;
    if (scratch && !*workspace) {
        close_inputs(inputs, count);
        vmerror_kind(v, "memory_budget", "file computation exceeds the workspace budget");
    }
    int cause = tensor_publish_payload(path, layout, writer, context);
    close_inputs(inputs, count);
    if (*workspace)
        tensor_workspace_release(v, *workspace);
    if (cause)
        vmerror_kind(v,
                     cause == EDOM     ? "domain"
                     : cause == ERANGE ? "numeric"
                                       : "io",
                     "file result failed: %s", strerror(cause));
}

typedef struct {
    TensorFile input[2];
    TypedTensor output;
    int block;
    void *workspace;
} FileMatmul;

static bool read_rectangle(const TensorFile *input, int row, int col, int rows, int cols,
                           double *packed) {
    size_t width = (size_t)input->layout.shape[1];
    for (int i = 0; i < rows; i++)
        if (!tensor_file_read(input, ((size_t)row + i) * width + col, (size_t)cols,
                              packed + (size_t)i * cols))
            return false;
    return true;
}

static bool matmul_output(int fd, const void *context) {
    const FileMatmul *job = context;
    const TensorFile *a = job->input, *b = a + 1;
    const TypedTensor *out = &job->output;
    int n = out->shape[1], m = out->shape[0], k = a->layout.shape[1], block = job->block;
    if (!out->n)
        return true;
    if (!k) {
        size_t size;
        return tensor_file_size(out, &size) && ftruncate(fd, (off_t)size) == 0;
    }
    size_t inner = (size_t)smaller(k, block), rows = (size_t)smaller(m, block);
    size_t cols = (size_t)smaller(n, block);
    double *left = job->workspace, *right = left + rows * inner, *tile = right + inner * cols;
    for (int row = 0, height; row < m; row += height) {
        height = smaller(m - row, block);
        for (int col = 0, width; col < n; col += width) {
            width = smaller(n - col, block);
            size_t elements = (size_t)height * width;
            memset(tile, 0, elements * sizeof *tile);
            for (int start = 0, depth; start < k; start += depth) {
                depth = smaller(k - start, block);
                if (!read_rectangle(a, row, start, height, depth, left) ||
                    !read_rectangle(b, start, col, depth, width, right))
                    return false;
                for (int i = 0; i < height; i++)
                    for (int p = 0; p < depth; p++) {
                        double x = left[(size_t)i * depth + p];
                        for (int j = 0; j < width; j++)
                            tile[(size_t)i * width + j] += x * right[(size_t)p * width + j];
                    }
            }
            for (size_t i = 0; i < elements; i++)
                if (!isfinite(tile[i])) {
                    errno = ERANGE;
                    return false;
                }
            for (int i = 0; i < height; i++)
                if (!tensor_file_patch(fd, out, ((size_t)row + i) * n + col, (size_t)width,
                                       tile + (size_t)i * width))
                    return false;
        }
    }
    return true;
}

Val compute_file_matmul(VM *v, const Val *args, int block, Val disk_budget) {
    const char *paths[] = {asstr(v, args[0]), asstr(v, args[1])};
    const char *destination = asstr(v, args[2]);
    FileMatmul job = {.block = block};
    open_inputs(v, 2, paths, job.input);
    const TypedTensor *a = &job.input[0].layout, *b = &job.input[1].layout;
    int dims[] = {a->shape[0], b->shape[1]};
    if (a->dtype != DT_F64 || b->dtype != DT_F64 || a->rank != 2 || b->rank != 2 ||
        a->shape[1] != b->shape[0] || !typed_layout_checked(DT_F64, 2, dims, &job.output)) {
        close_inputs(job.input, 2);
        vmerror_kind(v, "shape", "file_matmul requires compatible Float64 matrix files");
    }
    size_t r = (size_t)smaller(dims[0], block), c = (size_t)smaller(dims[1], block);
    size_t k = (size_t)smaller(a->shape[1], block);
    size_t scratch = job.output.n && k ? (r * k + k * c + r * c) * sizeof(double) : 0;
    write_result(v, destination, job.input, 2, &job.output, scratch, disk_budget, &job.workspace,
                 matmul_output, &job);
    return nil();
}

typedef struct {
    TensorFile input[4];
    TypedTensor output;
    size_t block;
    void *workspace;
} FileSpmv;

static bool csr_row(const FileSpmv *job, size_t row, int64_t *end) {
    return tensor_file_read(job->input, row, 1, end);
}

static bool spmv_output(int fd, const void *context) {
    const FileSpmv *job = context;
    const TensorFile *indices = job->input + 1, *values = job->input + 2, *x = job->input + 3;
    size_t block = job->block;
    int64_t *columns = job->workspace;
    double *entries = (double *)(columns + block), *vector = entries + block;
    int64_t begin;
    if (!csr_row(job, 0, &begin))
        return false;
    if (begin != 0) {
        errno = EDOM;
        return false;
    }
    size_t cached = 0, valid = 0;
    for (size_t row = 0; row < job->output.n; row++) {
        int64_t end, last = -1;
        long double sum = 0;
        if (!csr_row(job, row + 1, &end))
            return false;
        if (end < begin || (uint64_t)end > values->layout.n) {
            errno = EDOM;
            return false;
        }
        for (size_t pos = (size_t)begin; pos < (size_t)end;) {
            size_t count = (size_t)end - pos;
            if (count > block)
                count = block;
            if (!tensor_file_read(indices, pos, count, columns) ||
                !tensor_file_read(values, pos, count, entries))
                return false;
            for (size_t i = 0; i < count; i++) {
                int64_t col = columns[i];
                if (col <= last || (uint64_t)col >= x->layout.n || !isfinite(entries[i])) {
                    errno = EDOM;
                    return false;
                }
                last = col;
                size_t index = (size_t)col;
                if (index < cached || index - cached >= valid) {
                    cached = index - index % block;
                    valid = x->layout.n - cached;
                    if (valid > block)
                        valid = block;
                    if (!tensor_file_read(x, cached, valid, vector))
                        return false;
                }
                double input = vector[index - cached];
                if (!isfinite(input)) {
                    errno = EDOM;
                    return false;
                }
                sum += (long double)entries[i] * input;
            }
            pos += count;
        }
        if (!isfinite(sum) || fabsl(sum) > DBL_MAX) {
            errno = ERANGE;
            return false;
        }
        double y = (double)sum;
        if (!tensor_file_write(fd, &y, sizeof y))
            return false;
        begin = end;
    }
    if ((uint64_t)begin != values->layout.n) {
        errno = EDOM;
        return false;
    }
    return true;
}

static int workspace_block(VM *v, Val value, int fallback) {
    if (value.t == V_NIL)
        return fallback;
    if (value.t != V_INT || value.i < 1 || value.i > 1048576)
        vmerror_kind(v, "domain", "file block size must be an integer from 1 to 1048576");
    return (int)value.i;
}

Val compute_file_spmv(VM *v, const Val *args, Val block_value, Val disk_budget) {
    const char *paths[] = {asstr(v, args[0]), asstr(v, args[1]), asstr(v, args[2]),
                           asstr(v, args[3])};
    const char *destination = asstr(v, args[4]);
    FileSpmv job = {.block = (size_t)workspace_block(v, block_value, 4096)};
    open_inputs(v, 4, paths, job.input);
    const TypedTensor *rowptr = &job.input[0].layout, *indices = &job.input[1].layout;
    const TypedTensor *values = &job.input[2].layout, *x = &job.input[3].layout;
    int rows = rowptr->shape[0] - 1;
    if (rowptr->rank != 1 || indices->rank != 1 || values->rank != 1 || x->rank != 1 ||
        rowptr->dtype != DT_I64 || indices->dtype != DT_I64 || values->dtype != DT_F64 ||
        x->dtype != DT_F64 || values->n != indices->n ||
        !typed_layout_checked(DT_F64, 1, &rows, &job.output)) {
        close_inputs(job.input, 4);
        vmerror_kind(v, "shape",
                     "file_spmv requires CSR Int64 indices and Float64 values and vector");
    }
    size_t useful = values->n > x->n ? values->n : x->n;
    if (job.block > useful)
        job.block = useful ? useful : 1;
    size_t scratch = job.block * (sizeof(int64_t) + 2 * sizeof(double));
    write_result(v, destination, job.input, 4, &job.output, scratch, disk_budget, &job.workspace,
                 spmv_output, &job);
    return nil();
}

static bool retain_upper(double *packed, int rows, int n) {
    if (!dense_qr_packed(packed, rows, n, NULL))
        return false;
    for (int i = 0; i < n; i++)
        memset(packed + (size_t)i * n, 0, (size_t)i * sizeof(double));
    return true;
}

Val compute_file_qr_r(VM *v, Val path, Val block_value) {
    const char *paths[] = {asstr(v, path)};
    int block = workspace_block(v, block_value, 256);
    TensorFile input;
    open_inputs(v, 1, paths, &input);
    int m = input.layout.shape[0], n = input.layout.shape[1];
    int dims[] = {n, n};
    TypedTensor result;
    if (input.layout.dtype != DT_F64 || input.layout.rank != 2 || n == 0 || m < n || block < n ||
        !typed_layout_checked(DT_F64, 2, dims, &result) || !typed_materializable(&result)) {
        close_inputs(&input, 1);
        vmerror_kind(v, "shape",
                     "file_qr_r requires a tall Float64 matrix and block rows >= columns");
    }
    block = smaller(block, m);
    size_t capacity = (size_t)n + block;
    size_t output_bytes = result.n * sizeof(double);
    size_t available = tensor_memory_available(v);
    if (output_bytes > available || capacity > (available - output_bytes) / sizeof(double) / n) {
        close_inputs(&input, 1);
        vmerror_kind(v, "memory_budget", "file QR result and workspace exceed the tensor budget");
    }
    size_t bytes = capacity * n * sizeof(double);
    double *packed = (double *)tensor_workspace_acquire(v, bytes);
    if (!packed) {
        close_inputs(&input, 1);
        vmerror_kind(v, "memory_budget", "unable to reserve file QR workspace");
    }
    int cause = 0;
    int retained = 0;
    for (int row = 0, count; row < m; row += count) {
        count = smaller(block, m - row);
        size_t start = (size_t)retained * n, length = (size_t)count * n;
        if (!tensor_file_read(&input, (size_t)row * n, length, packed + start)) {
            cause = errno;
            break;
        }
        for (size_t i = start; i < start + length; i++)
            if (!isfinite(packed[i])) {
                cause = EDOM;
                break;
            }
        if (cause)
            break;
        if (!retain_upper(packed, retained + count, n)) {
            cause = ERANGE;
            break;
        }
        retained = n;
    }
    close_inputs(&input, 1);
    if (cause) {
        tensor_workspace_release(v, (unsigned char *)packed);
        vmerror_kind(v,
                     cause == EDOM     ? "domain"
                     : cause == ERANGE ? "numeric"
                                       : "io",
                     "file QR failed: %s", strerror(cause));
    }
    Obj *output = array_new(v, 2, n, n);
    memcpy(output->as.array.data, packed, output_bytes);
    tensor_workspace_release(v, (unsigned char *)packed);
    return objval(output);
}

#ifdef ARDA_LAPACK
static bool read_finite_row(const TensorFile *input, int i, double *row) {
    int n = input->layout.shape[1];
    if (!tensor_file_read(input, (size_t)i * n, (size_t)n, row))
        return false;
    for (int j = 0; j < n; j++)
        if (!isfinite(row[j])) {
            errno = EDOM;
            return false;
        }
    return true;
}

static bool sketch_row_space(const TensorFile *input, int width, int64_t seed, double *packed,
                             double *row, double *probe, double *basis) {
    int m = input->layout.shape[0], n = input->layout.shape[1];
    RandomState rng;
    random_seed(&rng, seed, 0);
    memset(packed, 0, (size_t)n * width * sizeof(double));
    for (int i = 0; i < m; i++) {
        if (!read_finite_row(input, i, row))
            return false;
        for (int k = 0; k < width; k++)
            probe[k] = (double)random_normal_unit(&rng);
        for (int j = 0; j < n; j++)
            for (int k = 0; k < width; k++)
                packed[(size_t)j * width + k] += row[j] * probe[k];
    }
    for (size_t i = 0; i < (size_t)n * width; i++)
        if (!isfinite(packed[i])) {
            errno = ERANGE;
            return false;
        }
    if (!dense_qr_packed(packed, n, width, probe) ||
        !dense_qr_basis(packed, n, width, probe, basis, width)) {
        errno = ERANGE;
        return false;
    }
    return true;
}

static bool project_row_space(const TensorFile *input, int width, int block, const double *basis,
                              double *packed, double *row) {
    int m = input->layout.shape[0], n = input->layout.shape[1], retained = 0;
    for (int i = 0, count; i < m; i += count) {
        count = smaller(m - i, block);
        for (int p = 0; p < count; p++) {
            if (!read_finite_row(input, i + p, row))
                return false;
            double *projected = packed + (size_t)(retained + p) * width;
            memset(projected, 0, (size_t)width * sizeof(double));
            for (int j = 0; j < n; j++)
                for (int k = 0; k < width; k++)
                    projected[k] += row[j] * basis[(size_t)j * width + k];
            for (int k = 0; k < width; k++)
                if (!isfinite(projected[k])) {
                    errno = ERANGE;
                    return false;
                }
        }
        if (!retain_upper(packed, retained + count, width)) {
            errno = ERANGE;
            return false;
        }
        retained = width;
    }
    return true;
}
#endif

Val compute_file_rsvd_right(VM *v, int argc, const Val *args) {
#ifndef ARDA_LAPACK
    (void)argc;
    (void)args;
    vmerror(v, "LAPACK unavailable: rebuild with LAPACK=1");
#else
    const char *paths[] = {asstr(v, args[0])};
    Val target = args[1], extra = argc > 2 ? args[2] : integer(8);
    Val seed = argc > 3 ? args[3] : integer(0);
    int block = workspace_block(v, argc > 4 ? args[4] : nil(), 256);
    if (target.t != V_INT || target.i < 1 || extra.t != V_INT || extra.i < 0 || seed.t != V_INT)
        vmerror_kind(v, "domain",
                     "file randomized SVD requires positive rank, nonnegative oversampling and an "
                     "integer seed");
    TensorFile input;
    open_inputs(v, 1, paths, &input);
    int m = input.layout.shape[0], n = input.layout.shape[1], limit = smaller(m, n);
    int rank = target.i <= limit ? (int)target.i : 0;
    int width = rank + (extra.i < limit - rank ? (int)extra.i : limit - rank);
    if (input.layout.dtype != DT_F64 || input.layout.rank != 2 || !rank || block < width ||
        (uint64_t)n * width > ARDA_MAX_ELEMENTS) {
        close_inputs(&input, 1);
        vmerror_kind(v, "shape",
                     "file randomized SVD needs a Float64 matrix and an admissible sketch");
    }
    block = smaller(block, m);
    size_t basis_elements = (size_t)n * width, factor_elements = (size_t)width * width;
    uint64_t panel_elements = ((uint64_t)block + width) * width;
    uint64_t packed_elements = panel_elements > basis_elements ? panel_elements : basis_elements;
    uint64_t work_elements = (uint64_t)n + width + packed_elements;
    size_t available_elements = tensor_memory_available(v) / sizeof(double);
    if (work_elements + basis_elements + factor_elements > available_elements) {
        close_inputs(&input, 1);
        vmerror_kind(v, "memory_budget", "file randomized SVD exceeds the tensor budget");
    }
    double *packed = (double *)tensor_workspace_acquire(v, (size_t)work_elements * sizeof(double));
    if (!packed) {
        close_inputs(&input, 1);
        vmerror_kind(v, "memory_budget", "unable to reserve the file randomized SVD workspace");
    }
    Obj *basis = array_new(v, 2, n, width), *r = array_new(v, 2, width, width);
    double *row = packed + packed_elements, *probe = row + n, *q = basis->as.array.data;
    bool ok = sketch_row_space(&input, width, seed.i, packed, row, probe, q) &&
              project_row_space(&input, width, block, q, packed, row);
    int cause = errno;
    if (ok)
        memcpy(r->as.array.data, packed, factor_elements * sizeof(double));
    close_inputs(&input, 1);
    tensor_workspace_release(v, (unsigned char *)packed);
    if (!ok)
        vmerror_kind(v,
                     cause == EDOM     ? "domain"
                     : cause == ERANGE ? "numeric"
                                       : "io",
                     "file randomized SVD failed: %s", strerror(cause));
    return dense_projected_right_svd(v, r, basis, rank);
#endif
}
