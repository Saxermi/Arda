#include "quant_kernels.h"
#include "typed_tensor.h"

#include <float.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

_Static_assert(sizeof(float) == 4, "quantization requires a four-byte Float32 scale");

typedef enum { AUTO, REFERENCE, SIMD } KernelMode;

static size_t code_bytes(bool q4) {
    return q4 ? QUANT_BLOCK / 2 : QUANT_BLOCK;
}

static size_t block_bytes(bool q4) {
    return sizeof(float) + code_bytes(q4);
}

static const unsigned char *raw(const TypedTensor *t, ptrdiff_t offset) {
    return typed_element(t, offset);
}

static double tensor_number(const TypedTensor *t, ptrdiff_t offset) {
    return dtype_load(t->dtype, raw(t, offset)).re;
}

static TypedTensor dense_input(VM *v, Val value) {
    if (isobj(value, O_AD))
        vmerror_kind(v, "autodiff", "quantized kernels do not record an autodiff tape");
    return typed_tensor(v, value);
}

static const QuantizedMatrix *packed_input(VM *v, Val value) {
    if (!isobj(value, O_QUANTIZED))
        vmerror_kind(v, "type", "expected a quantized matrix");
    return &value.o->as.quantized;
}

static void put_field(VM *v, Obj *map, const char *key, Val value) {
    tableset(v, map, &map->as.map.tab, key, value);
}

static Val info(VM *v, const QuantizedMatrix *q) {
    Obj *map = newobj(v, O_MAP), *shape = list(v);
    size_t blocks = (size_t)q->rows * q->blocks_per_row;
    append(v, shape, integer(q->rows));
    append(v, shape, integer(q->columns));
    put_field(v, map, "shape", objval(shape));
    put_field(v, map, "encoding", objval(string(v, q->q4 ? "q4" : "q8")));
    put_field(v, map, "block_size", integer(QUANT_BLOCK));
    put_field(v, map, "blocks", integer((int64_t)blocks));
    put_field(v, map, "code_bytes", integer((int64_t)(blocks * code_bytes(q->q4))));
    put_field(v, map, "scale_bytes", integer((int64_t)(blocks * sizeof(float))));
    put_field(v, map, "bytes", integer((int64_t)q->buffer->as.buffer.bytes));
    return objval(map);
}

static Val backends(VM *v) {
    Obj *map = newobj(v, O_MAP);
    const QuantKernels *q = quant_simd_kernels(v->cpu_features);
    const LowpKernel *f16 = quant_lowp_simd(v->cpu_features, DT_F16);
    const LowpKernel *bf16 = quant_lowp_simd(v->cpu_features, DT_BF16);
    put_field(v, map, "packed", objval(string(v, q ? q->name : "reference")));
    put_field(v, map, "float16", objval(string(v, f16 ? f16->name : "reference")));
    put_field(v, map, "bfloat16", objval(string(v, bf16 ? bf16->name : "reference")));
    return objval(map);
}

typedef enum { ENCODED, NONFINITE, SCALE_OVERFLOW } EncodeResult;

static EncodeResult encode_block(unsigned char *block, const TypedTensor *input, ptrdiff_t offset,
                                 size_t count, bool q4) {
    double values[QUANT_BLOCK], maximum = 0;
    for (size_t i = 0; i < count; i++) {
        values[i] = tensor_number(input, offset + (ptrdiff_t)i * input->stride[1]);
        if (!isfinite(values[i]))
            return NONFINITE;
        double magnitude = fabs(values[i]);
        if (magnitude > maximum)
            maximum = magnitude;
    }
    int limit = q4 ? 7 : 127;
    long double ideal = (long double)maximum / limit;
    if (ideal > FLT_MAX)
        return SCALE_OVERFLOW;
    float scale = (float)ideal;
    if (maximum && (!scale || (long double)scale < ideal))
        scale = nextafterf(scale, INFINITY);
    memcpy(block, &scale, sizeof scale);
    unsigned char *codes = block + sizeof scale;
    memset(codes, q4 ? 0x88 : 0x80, code_bytes(q4));
    if (scale == 0)
        return ENCODED;
    for (size_t i = 0; i < count; i++) {
        double rounded = round(values[i] / scale);
        int code = rounded > limit ? limit : rounded < -limit ? -limit : (int)rounded;
        if (q4) {
            unsigned shift = (unsigned)(i % 2) * 4;
            unsigned mask = 15u << shift;
            codes[i / 2] = (unsigned char)((codes[i / 2] & ~mask) | (unsigned)(code + 8) << shift);
        } else
            codes[i] = (unsigned char)(code + 128);
    }
    return ENCODED;
}

static Val quantize(VM *v, Val value, const char *format) {
    bool q4 = !strcmp(format, "q4");
    if (!q4 && strcmp(format, "q8"))
        vmerror_kind(v, "domain", "quantization encoding must be q4 or q8");
    TypedTensor t = dense_input(v, value);
    if (t.rank != 2 || !typed_materializable(&t))
        vmerror_kind(v, "shape", "quantization requires a rank-two matrix within the tensor limit");
    if (dtype_info[t.dtype].kind != SC_REAL)
        vmerror_kind(v, "dtype", "quantization requires a real floating-point tensor");

    size_t blocks = ((size_t)t.shape[1] + QUANT_BLOCK - 1) / QUANT_BLOCK;
    size_t stride = block_bytes(q4);
    size_t bytes = (size_t)t.shape[0] * blocks * stride;
    Obj *buffer = buffer_new(v, bytes);
    unsigned char *block = buffer->as.buffer.data;
    for (int row = 0; row < t.shape[0]; row++)
        for (size_t index = 0; index < blocks; index++, block += stride) {
            size_t begin = index * QUANT_BLOCK, count = (size_t)t.shape[1] - begin;
            if (count > QUANT_BLOCK)
                count = QUANT_BLOCK;
            ptrdiff_t offset =
                (ptrdiff_t)t.offset + (ptrdiff_t)row * t.stride[0] + (ptrdiff_t)begin * t.stride[1];
            EncodeResult result = encode_block(block, &t, offset, count, q4);
            if (result != ENCODED) {
                tensor_buffer_discard(v, buffer);
                if (result == NONFINITE)
                    vmerror_kind(v, "nonfinite", "quantization requires finite weights");
                vmerror_kind(v, "overflow", "quantization scale exceeds Float32 range");
            }
        }
    buffer->as.buffer.readonly = true;
    Obj *out = newobj(v, O_QUANTIZED);
    out->as.quantized = (QuantizedMatrix){.buffer = buffer,
                                          .rows = t.shape[0],
                                          .columns = t.shape[1],
                                          .blocks_per_row = blocks,
                                          .q4 = q4};
    barrier(v, out, objval(buffer));
    return objval(out);
}

static void decode_block(const unsigned char *block, size_t count, bool q4, double *output,
                         const QuantKernels *kernels) {
    float scale;
    memcpy(&scale, block, sizeof scale);
    const unsigned char *codes = block + sizeof scale;
    if (kernels && count == QUANT_BLOCK) {
        kernels->decode(codes, q4, scale, output);
        return;
    }
    if (q4)
        for (size_t i = 0; i < count; i++) {
            unsigned shift = (unsigned)(i % 2) * 4;
            int code = (int)((codes[i / 2] >> shift) & 15) - 8;
            output[i] = (double)scale * code;
        }
    else
        for (size_t i = 0; i < count; i++)
            output[i] = (double)scale * ((int)codes[i] - 128);
}

static size_t block_count(const QuantizedMatrix *q, size_t begin) {
    size_t left = (size_t)q->columns - begin;
    return left < QUANT_BLOCK ? left : QUANT_BLOCK;
}

static Val dequantize(VM *v, const QuantizedMatrix *q) {
    int shape[2] = {q->rows, q->columns};
    Obj *out = typed_new(v, DT_F64, 2, shape);
    double *data = (double *)typed_element(&out->as.tensor, 0);
    const unsigned char *block = q->buffer->as.buffer.data;
    size_t stride = block_bytes(q->q4);
    for (int row = 0; row < q->rows; row++)
        for (size_t index = 0; index < q->blocks_per_row; index++, block += stride) {
            size_t begin = index * QUANT_BLOCK;
            decode_block(block, block_count(q, begin), q->q4,
                         data + (size_t)row * q->columns + begin, NULL);
        }
    return objval(out);
}

static KernelMode mode_value(VM *v, const Val *value) {
    if (!value)
        return AUTO;
    const char *mode = asstr(v, *value);
    if (!strcmp(mode, "auto"))
        return AUTO;
    if (!strcmp(mode, "reference"))
        return REFERENCE;
    if (!strcmp(mode, "simd"))
        return SIMD;
    vmerror_kind(v, "domain", "kernel mode must be auto, reference, or simd");
}

static void multiply_vector(const QuantizedMatrix *q, const TypedTensor *input,
                            const QuantKernels *kernels, double *out) {
    const unsigned char *block = q->buffer->as.buffer.data;
    QuantDot dot = kernels ? kernels->dot : quant_reference_dot;
    size_t stride = block_bytes(q->q4);
    for (int row = 0; row < q->rows; row++) {
        double sum = 0;
        for (size_t index = 0; index < q->blocks_per_row; index++, block += stride) {
            double weights[QUANT_BLOCK], gathered[QUANT_BLOCK];
            size_t begin = index * QUANT_BLOCK, count = block_count(q, begin);
            ptrdiff_t offset = (ptrdiff_t)input->offset + (ptrdiff_t)begin * input->stride[0];
            decode_block(block, count, q->q4, weights, kernels);
            const double *values;
            if (input->stride[0] == 1)
                values = (const double *)raw(input, offset);
            else {
                for (size_t i = 0; i < count; i++)
                    gathered[i] = tensor_number(input, offset + (ptrdiff_t)i * input->stride[0]);
                values = gathered;
            }
            sum += dot(weights, values, count);
        }
        out[row] = sum;
    }
}

static void multiply_matrix(const QuantizedMatrix *q, const TypedTensor *input,
                            const QuantKernels *kernels, double *out) {
    const unsigned char *block = q->buffer->as.buffer.data;
    QuantAccumulate accumulate = kernels ? kernels->accumulate : quant_reference_accumulate;
    size_t stride = block_bytes(q->q4), n = (size_t)input->shape[1];
    for (int row = 0; row < q->rows; row++) {
        double *dst = out + (size_t)row * n;
        for (size_t index = 0; index < q->blocks_per_row; index++, block += stride) {
            double weights[QUANT_BLOCK];
            size_t begin = index * QUANT_BLOCK, count = block_count(q, begin);
            ptrdiff_t offset = (ptrdiff_t)input->offset + (ptrdiff_t)begin * input->stride[0];
            decode_block(block, count, q->q4, weights, kernels);
            if (input->stride[1] == 1) {
                accumulate(dst, (const double *)raw(input, offset), weights, count, n,
                           input->stride[0]);
            } else {
                for (size_t i = 0; i < count; i++) {
                    ptrdiff_t row_offset = offset + (ptrdiff_t)i * input->stride[0];
                    for (size_t col = 0; col < n; col++)
                        dst[col] +=
                            weights[i] *
                            tensor_number(input, row_offset + (ptrdiff_t)col * input->stride[1]);
                }
            }
        }
    }
}

static Val packed_product(VM *v, const QuantizedMatrix *q, Val value, bool vector,
                          KernelMode mode) {
    TypedTensor x = dense_input(v, value);
    if (x.rank != (vector ? 1 : 2) || x.shape[0] != q->columns)
        vmerror_kind(v, "shape", "quantized product requires a matching inner dimension");
    if (x.dtype != DT_F64)
        vmerror_kind(v, "dtype", "quantized products require Float64 activations");
    const QuantKernels *kernels =
        mode != REFERENCE && x.stride[x.rank - 1] == 1 ? quant_simd_kernels(v->cpu_features) : NULL;
    if (mode == SIMD && !kernels)
        vmerror_kind(v, "backend",
                     "packed SIMD requires supported CPU and unit input column stride");
    int shape[2] = {q->rows, vector ? 1 : x.shape[1]};
    Obj *out = typed_new(v, DT_F64, vector ? 1 : 2, shape);
    if (!out->as.tensor.n || !q->columns)
        return objval(out);

    double *data = (double *)typed_element(&out->as.tensor, 0);
    if (vector)
        multiply_vector(q, &x, kernels, data);
    else
        multiply_matrix(q, &x, kernels, data);
    return objval(out);
}

static void lowp_reference(const TypedTensor *a, const TypedTensor *b, float *out) {
    int k = a->shape[1], n = b->shape[1];
    for (int row = 0; row < a->shape[0]; row++) {
        float *dst = out + (size_t)row * n;
        ptrdiff_t a_row = (ptrdiff_t)a->offset + (ptrdiff_t)row * a->stride[0];
        for (int p = 0; p < k; p++) {
            float weight = (float)tensor_number(a, a_row + (ptrdiff_t)p * a->stride[1]);
            ptrdiff_t b_row = (ptrdiff_t)b->offset + (ptrdiff_t)p * b->stride[0];
            for (int col = 0; col < n; col++) {
                float value = (float)tensor_number(b, b_row + (ptrdiff_t)col * b->stride[1]);
                dst[col] += weight * value;
            }
        }
    }
}

static Val lowp_matmul(VM *v, Val left, Val right, KernelMode mode) {
    TypedTensor a = dense_input(v, left), b = dense_input(v, right);
    if (a.rank != 2 || b.rank != 2 || a.shape[1] != b.shape[0])
        vmerror_kind(v, "shape", "low-precision matmul requires matching rank-two inputs");
    if ((a.dtype != DT_F16 && a.dtype != DT_BF16) || b.dtype != a.dtype)
        vmerror_kind(v, "dtype", "low-precision matmul requires matching Float16 or BFloat16");
    const LowpKernel *simd = mode != REFERENCE && typed_contiguous(&a) && typed_contiguous(&b)
                                 ? quant_lowp_simd(v->cpu_features, a.dtype)
                                 : NULL;
    if (mode == SIMD && !simd)
        vmerror_kind(v, "backend",
                     "low-precision SIMD requires supported CPU and contiguous inputs");
    int shape[2] = {a.shape[0], b.shape[1]};
    Obj *out = typed_new(v, DT_F32, 2, shape);
    if (!out->as.tensor.n || !a.shape[1])
        return objval(out);

    float *data = (float *)typed_element(&out->as.tensor, 0);
    if (simd)
        simd->matmul(shape[0], a.shape[1], shape[1], (const uint16_t *)raw(&a, (ptrdiff_t)a.offset),
                     (const uint16_t *)raw(&b, (ptrdiff_t)b.offset), data);
    else
        lowp_reference(&a, &b, data);
    return objval(out);
}

enum { QUANTIZE, DEQUANTIZE, INFO, GEMV, GEMM, LOWP_MATMUL, BACKENDS, OPS };
static const char *const names[] = {"quantize", "dequantize",  "info",    "gemv",
                                    "gemm",     "lowp_matmul", "backends"};
static const int required[] = {2, 1, 1, 2, 2, 2, 0};
_Static_assert(sizeof names / sizeof *names == OPS && sizeof required / sizeof *required == OPS,
               "quant dispatch");

Val quant_call(VM *v, int id, int argc, Val *args) {
    int op = id - QUANT_BASE;
    if (op < 0 || op >= OPS ||
        (argc != required[op] && !(op >= GEMV && op <= LOWP_MATMUL && argc == 3)))
        vmerror_kind(v, "arity", "invalid quant module call");
    switch (op) {
    case QUANTIZE:
        return quantize(v, args[0], asstr(v, args[1]));
    case DEQUANTIZE:
        return dequantize(v, packed_input(v, args[0]));
    case INFO:
        return info(v, packed_input(v, args[0]));
    case GEMV:
    case GEMM:
        return packed_product(v, packed_input(v, args[0]), args[1], op == GEMV,
                              mode_value(v, argc == 3 ? args + 2 : NULL));
    case LOWP_MATMUL:
        return lowp_matmul(v, args[0], args[1], mode_value(v, argc == 3 ? args + 2 : NULL));
    default:
        return backends(v);
    }
}

Obj *quant_module(VM *v) {
    return named_native_module(v, "quant", names, OPS, QUANT_BASE);
}
