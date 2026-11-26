#include "typed_tensor.h"
#include "cpu_kernels.h"

static Val budget_value(const TensorMemory *m) {
    return m->limited ? integer((int64_t)m->limit) : nil();
}

static Val memory_budget(VM *v, int argc, Val *args) {
    TensorMemory *m = &v->tensor_memory;
    if (!argc)
        return budget_value(m);
    Val value = args[0];
    size_t limit = tensor_memory_unlimited();
    if (value.t != V_NIL) {
        if (value.t != V_INT || value.i < 0 || (uint64_t)value.i > SIZE_MAX)
            vmerror(v, "tensor memory budget must be a nonnegative byte count or mun");
        limit = (size_t)value.i;
    }
    if (limit < m->heap + m->reserved)
        vmerror(v, "tensor memory budget is below live buffers and reservations");
    if (m->cached > limit - m->heap - m->reserved)
        tensor_pool_trim(v);
    m->limit = limit;
    m->limited = value.t != V_NIL;
    return budget_value(m);
}

static void field(VM *v, Obj *map, const char *key, Val value) {
    tableset(v, map, &map->as.map.tab, key, value);
}

static Val memory_stats(VM *v, int argc, Val *args) {
    TypedTensor t = {0};
    if (argc)
        t = typed_tensor(v, args[0]);
    size_t vm_bytes = v->bytes;
    const TensorMemory *m = &v->tensor_memory;
    Obj *map = newobj(v, O_MAP);
    field(v, map, "vm_heap_bytes", integer((int64_t)vm_bytes));
    field(v, map, "tensor_heap_bytes", integer((int64_t)m->heap));
    field(v, map, "mapped_bytes", integer((int64_t)m->mapped));
    field(v, map, "workspace_reserved_bytes", integer((int64_t)m->reserved));
    field(v, map, "workspace_cached_bytes", integer((int64_t)m->cached));
    field(v, map, "peak_budgeted_bytes", integer((int64_t)m->peak));
    field(v, map, "budget_bytes", budget_value(m));
    if (argc) {
        const Buffer *buffer = &t.buffer->as.buffer;
        field(v, map, "logical_bytes", integer((int64_t)(t.n * dtype_info[t.dtype].size)));
        field(v, map, "backing_bytes", integer((int64_t)buffer->bytes));
        field(v, map, "storage", objval(string(v, buffer->mapped ? "mapped" : "heap")));
    }
    return objval(map);
}

static Val cpu_info(VM *v) {
    Obj *map = newobj(v, O_MAP);
    field(v, map, "architecture", objval(string(v, cpu_architecture())));
    field(v, map, "sse2", boolean(!!(v->cpu_features & CPU_SSE2)));
    field(v, map, "avx2", boolean(!!(v->cpu_features & CPU_AVX2)));
    field(v, map, "fma", boolean(!!(v->cpu_features & CPU_FMA)));
    field(v, map, "neon", boolean(!!(v->cpu_features & CPU_NEON)));
    field(v, map, "matmul_kernel", objval(string(v, cpu_matmul_name(v))));
    field(v, map, "auto_matmul", objval(string(v, cpu_matmul_auto(v))));
    field(v, map, "vector_math_available", boolean(cpu_vector_math_available(v)));
    return objval(map);
}

static Val matmul_kernel(VM *v, int argc, Val *args) {
    if (argc) {
        int mode = cpu_matmul_choice(v, asstr(v, args[0]));
        if (mode < 0)
            vmerror_kind(v, "backend", "requested Float64 matmul kernel is unavailable");
        v->matmul_kernel = mode;
    }
    return objval(string(v, cpu_matmul_name(v)));
}

static Val vector_math(VM *v, int argc, Val *args) {
    if (argc) {
        if (args[0].t != V_BOOL)
            vmerror_kind(v, "type", "compute.vector_math expects a Boolean");
        if (args[0].n && !cpu_vector_math_available(v))
            vmerror_kind(v, "backend", "vector math is unavailable in this build or on this CPU");
        v->vector_math = args[0].n != 0;
    }
    return boolean(v->vector_math);
}

enum {
    MEMORY_STATS,
    MEMORY_BUDGET,
    TRIM_POOL,
    PLAN,
    STREAM_STATS,
    CPU_INFO,
    MATMUL_KERNEL,
    VECTOR_MATH,
    COMPUTE_COUNT
};
static const char *const names[] = {"memory_stats",  "memory_budget", "trim_pool",
                                    "plan",          "stream_stats",  "cpu",
                                    "matmul_kernel", "vector_math"};
_Static_assert(sizeof names / sizeof *names == COMPUTE_COUNT, "compute names");

Obj *compute_module(VM *v) {
    return named_native_module(v, "compute", names, sizeof names / sizeof *names, COMPUTE_BASE);
}

Val compute_call(VM *v, int id, int argc, Val *args) {
    switch (id - COMPUTE_BASE) {
    case MATMUL_KERNEL:
        if (argc <= 1)
            return matmul_kernel(v, argc, args);
        break;
    case VECTOR_MATH:
        if (argc <= 1)
            return vector_math(v, argc, args);
        break;
    case CPU_INFO:
        if (!argc)
            return cpu_info(v);
        break;
    case MEMORY_STATS:
        if (argc <= 1)
            return memory_stats(v, argc, args);
        break;
    case MEMORY_BUDGET:
        if (argc <= 1)
            return memory_budget(v, argc, args);
        break;
    case TRIM_POOL:
        if (!argc)
            return integer((int64_t)tensor_pool_trim(v));
        break;
    case PLAN:
        if (argc == 2 || argc == 3) {
            if (strcmp(asstr(v, args[0]), "stream_stats"))
                vmerror(v, "compute.plan currently supports stream_stats");
            return tensor_stream_stats(v, args[1], argc == 3 ? args[2] : nil(), true);
        }
        break;
    case STREAM_STATS:
        if (argc == 1 || argc == 2)
            return tensor_stream_stats(v, args[0], argc == 2 ? args[1] : nil(), false);
        break;
    }
    vmerror(v, "invalid compute API arity");
}
