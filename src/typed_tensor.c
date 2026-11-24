#include "typed_tensor.h"

static int is_tensor(Val x) {
    return isobj(x, O_TYPED_TENSOR) || isobj(x, O_ARRAY);
}

TypedTensor typed_tensor(VM *v, Val x) {
    if (isobj(x, O_TYPED_TENSOR))
        return x.o->as.tensor;
    if (!isobj(x, O_ARRAY))
        vmerror(v, "expected tensor or Float64 array");
    TypedTensor t = {.buffer = x.o->as.array.buffer,
                     .offset = x.o->as.array.offset,
                     .rank = x.o->as.array.rank,
                     .n = x.o->as.array.n,
                     .dtype = DT_F64,
                     .readonly = x.o->as.array.readonly};
    for (int i = 0; i < t.rank; i++) {
        t.shape[i] = x.o->as.array.shape[i];
        t.stride[i] = (ptrdiff_t)x.o->as.array.stride[i];
    }
    return t;
}

TypedTensor typed_layout(VM *v, DType dtype, int rank, const int *shape) {
    if (rank < 1 || rank > 8)
        vmerror(v, "tensor rank must be 1..8");
    TypedTensor t = {.rank = rank, .dtype = dtype};
    size_t span = 1, capacity = PTRDIFF_MAX / dtype_info[dtype].size;
    bool empty = false;
    for (int i = rank - 1; i >= 0; i--) {
        if (shape[i] < 0)
            vmerror(v, "invalid tensor dimension");
        t.shape[i] = shape[i];
        t.stride[i] = (ptrdiff_t)span;
        size_t length = shape[i] ? (size_t)shape[i] : 1;
        if (span > capacity / length)
            vmerror(v, "tensor layout exceeds addressable size");
        span *= length;
        empty |= shape[i] == 0;
    }
    t.n = empty ? 0 : span;
    return t;
}

static void materializable(VM *v, const TypedTensor *t) {
    size_t span = 1;
    for (int i = 0; i < t->rank; i++) {
        size_t length = t->shape[i] ? (size_t)t->shape[i] : 1;
        if (span > ARDA_MAX_ELEMENTS / length)
            vmerror(v, "tensor allocation limit: 16000000 elements; use block access");
        span *= length;
    }
}

Obj *typed_wrap(VM *v, TypedTensor layout) {
    Obj *obj = newobj(v, O_TYPED_TENSOR);
    obj->as.tensor = layout;
    barrier(v, obj, objval(layout.buffer));
    return obj;
}

Obj *typed_new(VM *v, DType dtype, int rank, const int *shape) {
    TypedTensor t = typed_layout(v, dtype, rank, shape);
    materializable(v, &t);
    t.buffer = buffer_new(v, t.n * dtype_info[dtype].size);
    return typed_wrap(v, t);
}

ptrdiff_t typed_offset(const TypedTensor *t, size_t i) {
    ptrdiff_t offset = (ptrdiff_t)t->offset;
    for (int d = t->rank - 1; d >= 0; d--) {
        size_t length = (size_t)t->shape[d];
        offset += (ptrdiff_t)(i % length) * t->stride[d];
        i /= length;
    }
    return offset;
}

unsigned char *typed_element(const TypedTensor *t, ptrdiff_t pos) {
    return t->buffer->as.buffer.data + (size_t)pos * dtype_info[t->dtype].size;
}

static TensorScalar element(const TypedTensor *t, ptrdiff_t pos) {
    return dtype_load(t->dtype, typed_element(t, pos));
}

static void store(VM *v, const TypedTensor *t, ptrdiff_t pos, TensorScalar x) {
    dtype_store(v, t->dtype, typed_element(t, pos), x);
}

static int contiguous(const TypedTensor *t) {
    if (!t->n)
        return 1;
    ptrdiff_t stride = 1;
    for (int i = t->rank - 1; i >= 0; i--) {
        if (t->shape[i] > 1 && t->stride[i] != stride)
            return 0;
        stride *= t->shape[i];
    }
    return 1;
}

static const Obj *list_input(VM *v, Val x) {
    if (!isobj(x, O_LIST))
        vmerror(v, "expected a list");
    return x.o;
}

static int get_shape(VM *v, Val x, int *shape) {
    const Obj *obj = list_input(v, x);
    int rank = obj->as.list.n;
    if (rank < 1 || rank > 8)
        vmerror(v, "tensor rank must be 1..8");
    for (int i = 0; i < rank; i++) {
        Val dim = obj->as.list.a[i];
        if (dim.t != V_INT || dim.i < 0 || dim.i > INT_MAX)
            vmerror(v, "invalid tensor dimension");
        shape[i] = (int)dim.i;
    }
    return rank;
}

static Val shape_value(VM *v, const TypedTensor *t) {
    Obj *out = list(v);
    for (int i = 0; i < t->rank; i++)
        append(v, out, integer(t->shape[i]));
    return objval(out);
}

static ptrdiff_t index_offset(VM *v, const TypedTensor *t, Val x) {
    const Obj *indices = list_input(v, x);
    if (indices->as.list.n != t->rank)
        vmerror(v, "expected one index per tensor dimension");
    ptrdiff_t pos = (ptrdiff_t)t->offset;
    for (int i = 0; i < t->rank; i++)
        pos += indexnum(v, indices->as.list.a[i], t->shape[i]) * t->stride[i];
    return pos;
}

static Obj *cast(VM *v, const TypedTensor *t, DType dtype) {
    Obj *out = typed_new(v, dtype, t->rank, t->shape);
    TypedTensor *dst = &out->as.tensor;
    if (dtype == t->dtype && contiguous(t)) {
        memcpy(typed_element(dst, 0), typed_element(t, (ptrdiff_t)t->offset),
               t->n * dtype_info[dtype].size);
    } else
        for (size_t i = 0; i < t->n; i++)
            store(v, dst, (ptrdiff_t)i, element(t, typed_offset(t, i)));
    return out;
}

static Obj *to_numc(VM *v, const TypedTensor *t) {
    Obj *out = tensor_new(v, t->rank, t->shape);
    for (size_t i = 0; i < t->n; i++)
        dtype_store(v, DT_F64, (unsigned char *)(out->as.array.data + i),
                    element(t, typed_offset(t, i)));
    return out;
}

static Val tolist(VM *v, const TypedTensor *t, int axis, ptrdiff_t pos) {
    Obj *out = list(v);
    for (int i = 0; i < t->shape[axis]; i++) {
        ptrdiff_t next = pos + i * t->stride[axis];
        Val value =
            axis + 1 == t->rank ? dtype_value(v, element(t, next)) : tolist(v, t, axis + 1, next);
        append(v, out, value);
    }
    return objval(out);
}

static TypedTensor strided_view(VM *v, const TypedTensor *src, Val shape, Val strides, Val offset) {
    int dims[8];
    int rank = get_shape(v, shape, dims);
    const Obj *steps = list_input(v, strides);
    if (!contiguous(src) || steps->as.list.n != rank || offset.t != V_INT || offset.i < 0 ||
        (uint64_t)offset.i > src->n)
        vmerror(v, "strided view requires a contiguous source and valid offset and strides");
    TypedTensor t = typed_layout(v, src->dtype, rank, dims);
    t.buffer = src->buffer;
    t.offset = src->offset;
    t.readonly = true;
    if (t.n && (uint64_t)offset.i == src->n)
        vmerror(v, "strided view is outside the source");
    size_t before = (size_t)offset.i, after = t.n ? src->n - before - 1 : 0;
    if (t.n)
        t.offset += before;
    for (int d = 0; d < rank; d++) {
        Val stride = steps->as.list.a[d];
        if (stride.t != V_INT || stride.i > PTRDIFF_MAX || stride.i < -PTRDIFF_MAX)
            vmerror(v, "invalid tensor stride");
        size_t span = (size_t)(stride.i < 0 ? -stride.i : stride.i);
        size_t n = t.n ? (size_t)(t.shape[d] - 1) : 0;
        size_t *available = stride.i < 0 ? &before : &after;
        if (n && span > *available / n)
            vmerror(v, "strided view is outside the source");
        *available -= n * span;
        t.stride[d] = n ? (ptrdiff_t)stride.i : 0;
    }
    return t;
}

static Obj *permute(VM *v, TypedTensor t, Val axes) {
    const Obj *list = list_input(v, axes);
    if (list->as.list.n != t.rank)
        vmerror(v, "permutation requires one axis per dimension");
    TypedTensor out = t;
    unsigned seen = 0;
    for (int i = 0; i < t.rank; i++) {
        int j = indexnum(v, list->as.list.a[i], t.rank);
        if (seen & (1u << j))
            vmerror(v, "duplicate permutation axis");
        seen |= 1u << j;
        out.shape[i] = t.shape[j];
        out.stride[i] = t.stride[j];
    }
    return typed_wrap(v, out);
}

static DType scalar_dtype(TensorScalar s) {
    switch (s.kind) {
    case SC_BOOL:
        return DT_BOOL;
    case SC_INT:
        return DT_I64;
    case SC_REAL:
        return DT_F64;
    default:
        return DT_C128;
    }
}

static int broadcast(VM *v, int x, int y) {
    if (x != y && x != 1 && y != 1)
        vmerror(v, "tensor shape mismatch (broadcast)");
    return x == 1 ? y : x;
}

static void broadcast_shape(VM *v, const TypedTensor *a, const TypedTensor *b, int rank,
                            int *dims) {
    for (int d = 0; d < rank; d++) {
        int ad = d - rank + a->rank, bd = d - rank + b->rank;
        dims[d] = broadcast(v, ad < 0 ? 1 : a->shape[ad], bd < 0 ? 1 : b->shape[bd]);
    }
}

static ptrdiff_t broadcast_offset(const TypedTensor *t, const int *dims, int rank, size_t pos) {
    ptrdiff_t offset = (ptrdiff_t)t->offset;
    for (int i = t->rank - 1; i >= 0; i--) {
        size_t dim = (size_t)dims[i + rank - t->rank];
        if (t->shape[i] > 1)
            offset += (ptrdiff_t)(pos % dim) * t->stride[i];
        pos /= dim;
    }
    return offset;
}

Val typed_tensor_binary(VM *v, Op op, Val x, Val y) {
    if (op != OP_ADD && op != OP_SUB && op != OP_MUL && op != OP_DIV)
        vmerror(v, "unsupported tensor arithmetic operator");
    TypedTensor a = {0}, b = {0};
    TensorScalar av = {0}, bv = {0};
    if (is_tensor(x))
        a = typed_tensor(v, x);
    else {
        av = dtype_scalar(v, x);
        a.dtype = scalar_dtype(av);
    }
    if (is_tensor(y))
        b = typed_tensor(v, y);
    else {
        bv = dtype_scalar(v, y);
        b.dtype = scalar_dtype(bv);
    }
    int rank = a.rank > b.rank ? a.rank : b.rank, dims[8];
    broadcast_shape(v, &a, &b, rank, dims);
    DType dtype = dtype_promote(a.dtype, b.dtype, op);
    Obj *out = typed_new(v, dtype, rank, dims);
    TypedTensor *dst = &out->as.tensor;
    for (size_t i = 0; i < dst->n; i++) {
        TensorScalar p = a.rank ? element(&a, broadcast_offset(&a, dims, rank, i)) : av;
        TensorScalar q = b.rank ? element(&b, broadcast_offset(&b, dims, rank, i)) : bv;
        store(v, dst, (ptrdiff_t)i, dtype_binary(v, op, dtype, p, q));
    }
    return objval(out);
}

static DType accumulator(DType dtype) {
    switch (dtype_info[dtype].kind) {
    case SC_BOOL:
    case SC_INT:
        return DT_I64;
    case SC_REAL:
        return DT_F64;
    default:
        return DT_C128;
    }
}

static TensorScalar inner(VM *v, const TypedTensor *a, const TypedTensor *b, ptrdiff_t ai,
                          ptrdiff_t bi, int da, int db, DType type) {
    TensorScalar sum = {.kind = dtype_info[type].kind};
    for (int i = 0; i < a->shape[da]; i++) {
        TensorScalar p = element(a, ai + i * a->stride[da]);
        TensorScalar q = element(b, bi + i * b->stride[db]);
        sum = dtype_binary(v, OP_ADD, type, sum, dtype_binary(v, OP_MUL, type, p, q));
    }
    return sum;
}

static TypedTensor without_axis(TypedTensor t, int axis) {
    for (int i = axis; i + 1 < t.rank; i++) {
        t.shape[i] = t.shape[i + 1];
        t.stride[i] = t.stride[i + 1];
    }
    t.rank--;
    return t;
}

static Obj *matmul(VM *v, TypedTensor a, TypedTensor b) {
    if (a.rank < 2 || b.rank < 2 || a.shape[a.rank - 1] != b.shape[b.rank - 2])
        vmerror(v, "tensor matmul requires compatible matrix axes");
    int da = a.rank - 1, db = b.rank - 2;
    int rank = a.rank > b.rank ? a.rank : b.rank, dims[8];
    TypedTensor batches_a = a, batches_b = b;
    batches_a.rank -= 2;
    batches_b.rank -= 2;
    broadcast_shape(v, &batches_a, &batches_b, rank - 2, dims);
    int m = a.shape[da - 1], n = b.shape[db + 1];
    dims[rank - 2] = m;
    dims[rank - 1] = n;
    DType type = accumulator(dtype_promote(a.dtype, b.dtype, OP_MUL));
    Obj *out = typed_new(v, type, rank, dims);
    TypedTensor *dst = &out->as.tensor;
    for (size_t i = 0; i < dst->n; i++) {
        size_t row = i / (size_t)n % (size_t)m, col = i % (size_t)n;
        size_t batch = i / ((size_t)m * n);
        ptrdiff_t ai =
            broadcast_offset(&batches_a, dims, rank - 2, batch) + (ptrdiff_t)row * a.stride[da - 1];
        ptrdiff_t bi =
            broadcast_offset(&batches_b, dims, rank - 2, batch) + (ptrdiff_t)col * b.stride[db + 1];
        store(v, dst, (ptrdiff_t)i, inner(v, &a, &b, ai, bi, da, db, type));
    }
    return out;
}

static Val tensordot(VM *v, TypedTensor a, TypedTensor b, int da, int db) {
    if (a.shape[da] != b.shape[db])
        vmerror(v, "tensor contraction dimensions differ");
    TypedTensor left = without_axis(a, da), right = without_axis(b, db);
    int rank = left.rank + right.rank, dims[8];
    DType type = accumulator(dtype_promote(a.dtype, b.dtype, OP_MUL));
    if (!rank)
        return dtype_value(
            v, inner(v, &a, &b, (ptrdiff_t)a.offset, (ptrdiff_t)b.offset, da, db, type));
    if (rank > 8)
        vmerror(v, "tensor contraction output rank exceeds 8");
    memcpy(dims, left.shape, (size_t)left.rank * sizeof(int));
    memcpy(dims + left.rank, right.shape, (size_t)right.rank * sizeof(int));
    Obj *out = typed_new(v, type, rank, dims);
    TypedTensor *dst = &out->as.tensor;
    if (!dst->n)
        return objval(out);
    size_t width = 1;
    for (int i = 0; i < right.rank; i++)
        width *= (size_t)right.shape[i];
    for (size_t i = 0; i < dst->n; i++)
        store(v, dst, (ptrdiff_t)i,
              inner(v, &a, &b, typed_offset(&left, i / width), typed_offset(&right, i % width), da,
                    db, type));
    return objval(out);
}

static const char *einsum_labels(VM *v, const char *spec, const TypedTensor *t, int *sizes,
                                 ptrdiff_t *strides) {
    unsigned seen = 0;
    for (int d = 0; d < t->rank; d++) {
        if (*spec < 'a' || *spec > 'z')
            vmerror(v, "einsum requires one lowercase label per input axis");
        int label = *spec++ - 'a';
        if (seen & (1u << label))
            vmerror(v, "einsum diagonal labels are unsupported");
        seen |= 1u << label;
        if (sizes[label] >= 0 && sizes[label] != t->shape[d])
            vmerror(v, "einsum dimensions differ for a shared label");
        sizes[label] = t->shape[d];
        strides[label] = t->stride[d];
    }
    return spec;
}

static void einsum_offsets(size_t index, int rank, const int *labels, const int *sizes,
                           const ptrdiff_t *astride, const ptrdiff_t *bstride, ptrdiff_t *ai,
                           ptrdiff_t *bi) {
    for (int d = rank - 1; d >= 0; d--) {
        int label = labels[d];
        size_t length = (size_t)sizes[label];
        ptrdiff_t coordinate = (ptrdiff_t)(index % length);
        index /= length;
        *ai += coordinate * astride[label];
        *bi += coordinate * bstride[label];
    }
}

static Val einsum(VM *v, const char *spec, TypedTensor a, TypedTensor b) {
    int sizes[26], output[8], reduced[26], dims[8];
    ptrdiff_t astride[26] = {0}, bstride[26] = {0};
    for (int i = 0; i < 26; i++)
        sizes[i] = -1;
    spec = einsum_labels(v, spec, &a, sizes, astride);
    if (*spec != ',')
        vmerror(v, "einsum requires two operand label lists");
    spec = einsum_labels(v, spec + 1, &b, sizes, bstride);
    if (strncmp(spec, "->", 2))
        vmerror(v, "einsum requires an explicit output");
    spec += 2;
    unsigned seen = 0;
    int rank = 0, nreduce = 0;
    while (*spec) {
        if (*spec < 'a' || *spec > 'z' || rank == 8)
            vmerror(v, "einsum output requires at most eight lowercase labels");
        int label = *spec++ - 'a';
        if (sizes[label] < 0 || (seen & (1u << label)))
            vmerror(v, "invalid or duplicate einsum output label");
        seen |= 1u << label;
        output[rank] = label;
        dims[rank++] = sizes[label];
    }
    size_t terms = 1;
    for (int i = 0; i < 26; i++)
        if (sizes[i] >= 0 && !(seen & (1u << i))) {
            reduced[nreduce++] = i;
            if (!sizes[i])
                terms = 0;
        }
    for (int d = 0; terms && d < nreduce; d++) {
        size_t length = (size_t)sizes[reduced[d]];
        if (terms > SIZE_MAX / length)
            vmerror(v, "einsum contraction size overflow");
        terms *= length;
    }
    DType dtype = accumulator(dtype_promote(a.dtype, b.dtype, OP_MUL));
    Obj *out = rank ? typed_new(v, dtype, rank, dims) : NULL;
    size_t outputs = out ? out->as.tensor.n : 1;
    for (size_t i = 0; i < outputs; i++) {
        ptrdiff_t abase = (ptrdiff_t)a.offset, bbase = (ptrdiff_t)b.offset;
        einsum_offsets(i, rank, output, sizes, astride, bstride, &abase, &bbase);
        TensorScalar sum = {.kind = dtype_info[dtype].kind};
        for (size_t term = 0; term < terms; term++) {
            ptrdiff_t ai = abase, bi = bbase;
            einsum_offsets(term, nreduce, reduced, sizes, astride, bstride, &ai, &bi);
            TensorScalar product = dtype_binary(v, OP_MUL, dtype, element(&a, ai), element(&b, bi));
            sum = dtype_binary(v, OP_ADD, dtype, sum, product);
        }
        if (!out)
            return dtype_value(v, sum);
        store(v, &out->as.tensor, (ptrdiff_t)i, sum);
    }
    return objval(out);
}

static TensorScalar sum_axis(VM *v, const TypedTensor *t, int axis, ptrdiff_t base, DType dtype) {
    TensorScalar s = {.kind = dtype_info[dtype].kind};
    for (int i = 0; i < t->shape[axis]; i++)
        s = dtype_binary(v, OP_ADD, dtype, s, element(t, base + i * t->stride[axis]));
    return s;
}

static Val sum(VM *v, const TypedTensor *t, int axis) {
    DType dtype = accumulator(t->dtype);
    if (t->rank == 1)
        return dtype_value(v, sum_axis(v, t, axis, (ptrdiff_t)t->offset, dtype));
    TypedTensor layout = without_axis(*t, axis);
    Obj *result = typed_new(v, dtype, layout.rank, layout.shape);
    TypedTensor *dst = &result->as.tensor;
    for (size_t i = 0; i < dst->n; i++)
        store(v, dst, (ptrdiff_t)i, sum_axis(v, t, axis, typed_offset(&layout, i), dtype));
    return objval(result);
}

static size_t flat_index(VM *v, Val x, size_t length) {
    if (x.t != V_INT || x.i < 0 || (uint64_t)x.i >= length)
        vmerror(v, "tensor flat index outside bounds");
    return (size_t)x.i;
}

static Obj *gather(VM *v, const TypedTensor *t, Val indices) {
    const Obj *list = list_input(v, indices);
    Obj *out = typed_new(v, t->dtype, 1, &list->as.list.n);
    for (int i = 0; i < list->as.list.n; i++) {
        size_t index = flat_index(v, list->as.list.a[i], t->n);
        store(v, &out->as.tensor, i, element(t, typed_offset(t, index)));
    }
    return out;
}

static Obj *scatter(VM *v, const TypedTensor *t, Val indices, Val values) {
    const Obj *list = list_input(v, indices), *vals = list_input(v, values);
    if (list->as.list.n != vals->as.list.n)
        vmerror(v, "tensor scatter requires matching index and value lists");
    Obj *out = cast(v, t, t->dtype);
    for (int i = 0; i < list->as.list.n; i++) {
        size_t index = flat_index(v, list->as.list.a[i], t->n);
        store(v, &out->as.tensor, (ptrdiff_t)index, dtype_scalar(v, vals->as.list.a[i]));
    }
    return out;
}

static Obj *mask(VM *v, const TypedTensor *t, const TypedTensor *flags) {
    if (flags->dtype != DT_BOOL || flags->rank != t->rank ||
        memcmp(t->shape, flags->shape, (size_t)t->rank * sizeof(*t->shape)))
        vmerror(v, "tensor mask requires a bool tensor with the same shape");
    materializable(v, t);
    int selected = 0;
    for (size_t i = 0; i < t->n; i++)
        selected += element(flags, typed_offset(flags, i)).i != 0;
    Obj *out = typed_new(v, t->dtype, 1, &selected);
    ptrdiff_t next = 0;
    for (size_t i = 0; i < t->n; i++)
        if (element(flags, typed_offset(flags, i)).i)
            store(v, &out->as.tensor, next++, element(t, typed_offset(t, i)));
    return out;
}

static Obj *fft(VM *v, const TypedTensor *t, bool inverse) {
    if (t->rank != 1 || !t->n || t->n > 1048576 || (t->n & (t->n - 1)))
        vmerror(v, "tensor FFT requires vector length a power of two <= 1048576");
    int n = t->shape[0];
    Obj *pairs = array_new(v, 2, n, 2);
    for (int i = 0; i < n; i++) {
        TensorScalar value = element(t, typed_offset(t, (size_t)i));
        pairs->as.array.data[2 * i] = value.kind <= SC_INT ? (double)value.i : value.re;
        pairs->as.array.data[2 * i + 1] = value.im;
    }
    Val args[] = {objval(pairs)};
    Obj *result = calculus_call(v, CALC_BASE + (inverse ? 3 : 2), 1, args).o;
    DType dtype = dtype_promote(t->dtype, DT_C64, OP_ADD);
    Obj *out = typed_new(v, dtype, 1, &n);
    for (int i = 0; i < n; i++) {
        TensorScalar value = {.kind = SC_COMPLEX,
                              .re = result->as.array.data[2 * i],
                              .im = result->as.array.data[2 * i + 1]};
        store(v, &out->as.tensor, i, value);
    }
    return out;
}

enum {
    ARRAY,
    ZEROS,
    CAST,
    DTYPE,
    SHAPE,
    BYTES,
    TOLIST,
    AT,
    PUT,
    VIEW,
    PERMUTE,
    TO_NUMC,
    READONLY,
    MATMUL,
    SUM,
    TENSORDOT,
    GATHER,
    SCATTER,
    MASK,
    FFT,
    IFFT,
    EINSUM
};
static const char *const names[] = {"array",    "zeros",  "cast", "dtype",     "shape",   "bytes",
                                    "tolist",   "at",     "put",  "view",      "permute", "to_numc",
                                    "readonly", "matmul", "sum",  "tensordot", "gather",  "scatter",
                                    "mask",     "fft",    "ifft", "einsum"};
static const int arities[] = {2, 2, 2, 1, 1, 1, 1, 2, 3, 4, 2, 1, 1, 2, 2, 4, 2, 3, 2, 1, 1, 3};
_Static_assert(sizeof names / sizeof *names == sizeof arities / sizeof *arities, "tensor arities");

Val typed_tensor_call(VM *v, int id, int argc, Val *args) {
    int op = id - TYPED_TENSOR_BASE;
    if (op < 0 || (size_t)op >= sizeof names / sizeof *names ||
        (argc != arities[op] && !(op == ARRAY && argc == 3)))
        vmerror(v, "invalid tensor API arity");
    if (op == EINSUM)
        return einsum(v, asstr(v, args[0]), typed_tensor(v, args[1]), typed_tensor(v, args[2]));
    if (op == ARRAY || op == ZEROS) {
        int dims[8], rank;
        DType dtype = dtype_parse(v, args[1]);
        if (op == ZEROS) {
            rank = get_shape(v, args[0], dims);
            return objval(typed_new(v, dtype, rank, dims));
        }
        const Obj *values = list_input(v, args[0]);
        dims[0] = values->as.list.n;
        rank = argc == 3 ? get_shape(v, args[2], dims) : 1;
        TypedTensor layout = typed_layout(v, dtype, rank, dims);
        if (layout.n != (size_t)values->as.list.n)
            vmerror(v, "tensor flat data length and shape differ");
        Obj *out = typed_new(v, dtype, rank, dims);
        for (int i = 0; i < values->as.list.n; i++)
            store(v, &out->as.tensor, i, dtype_scalar(v, values->as.list.a[i]));
        return objval(out);
    }

    TypedTensor t = typed_tensor(v, args[0]);
    switch (op) {
    case CAST:
        return objval(cast(v, &t, dtype_parse(v, args[1])));
    case DTYPE:
        return objval(string(v, dtype_info[t.dtype].name));
    case SHAPE:
        return shape_value(v, &t);
    case BYTES:
        return integer((int64_t)(t.n * dtype_info[t.dtype].size));
    case TOLIST:
        materializable(v, &t);
        return tolist(v, &t, 0, (ptrdiff_t)t.offset);
    case AT:
        return dtype_value(v, element(&t, index_offset(v, &t, args[1])));
    case PUT:
        if (t.readonly || t.buffer->as.buffer.readonly)
            vmerror(v, "tensor is read-only");
        store(v, &t, index_offset(v, &t, args[1]), dtype_scalar(v, args[2]));
        return nil();
    case VIEW:
        return objval(typed_wrap(v, strided_view(v, &t, args[1], args[2], args[3])));
    case READONLY:
        t.readonly = true;
        return objval(typed_wrap(v, t));
    case PERMUTE:
        return objval(permute(v, t, args[1]));
    case TO_NUMC:
        return objval(to_numc(v, &t));
    case MATMUL:
        return objval(matmul(v, t, typed_tensor(v, args[1])));
    case SUM:
        return sum(v, &t, indexnum(v, args[1], t.rank));
    case TENSORDOT: {
        TypedTensor rhs = typed_tensor(v, args[1]);
        return tensordot(v, t, rhs, indexnum(v, args[2], t.rank), indexnum(v, args[3], rhs.rank));
    }
    case GATHER:
        return objval(gather(v, &t, args[1]));
    case SCATTER:
        return objval(scatter(v, &t, args[1], args[2]));
    case MASK: {
        TypedTensor flags = typed_tensor(v, args[1]);
        return objval(mask(v, &t, &flags));
    }
    case FFT:
    case IFFT:
        return objval(fft(v, &t, op == IFFT));
    default:
        vmerror(v, "unknown tensor operation");
    }
}

Obj *typed_tensor_module(VM *v) {
    return named_native_module(v, "tensor", names, sizeof names / sizeof *names, TYPED_TENSOR_BASE);
}
