#include "compute_plan.h"
#include "compute_topology.h"
#include "cpu_kernels.h"
#include "file_compute.h"

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
    field(v, map, "f16c", boolean(!!(v->cpu_features & CPU_F16C)));
    field(v, map, "matmul_kernel", objval(string(v, cpu_matmul_name(v))));
    field(v, map, "auto_matmul", objval(string(v, cpu_matmul_auto(v))));
    field(v, map, "vector_math_available", boolean(cpu_vector_math_available(v)));
    field(v, map, "native_threads_available", boolean(cpu_parallel_available()));
    field(v, map, "last_packed_matmul_threads", integer(v->last_matmul_threads));
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

static Val native_threads(VM *v, int argc, Val *args) {
    if (argc) {
        if (args[0].t != V_INT || args[0].i < 1 || args[0].i > 256)
            vmerror_kind(v, "domain", "compute.threads expects an integer from 1 to 256");
        if (args[0].i != 1 && !cpu_parallel_available())
            vmerror_kind(v, "backend", "native parallel kernels require OPENMP=1 without BLAS");
        v->native_threads = (int)args[0].i;
    }
    return integer(v->native_threads);
}

static bool prefetch_option(VM *v, Val value) {
    if (value.t == V_NIL)
        return false;
    if (!isobj(value, O_MAP))
        vmerror_kind(v, "type", "stream options must be a map");
    Table *table = &value.o->as.map.tab;
    Entry *prefetch = tablefind(table, "prefetch");
    if (table->n != (prefetch != NULL) || (prefetch && prefetch->val.t != V_BOOL))
        vmerror_kind(v, "type", "stream options accept a Boolean prefetch field");
    return prefetch && prefetch->val.n != 0;
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
    THREADS,
    TILED,
    AXPY,
    MUL_ADD,
    FUSED_PRODUCTS,
    MAP_REDUCE,
    EXPLAIN,
    EXECUTE,
    LAST_RUN,
    COMPARE,
    TOPOLOGY,
    AFFINITY,
    FIRST_TOUCH,
    FILE_MATMUL,
    FILE_SPMV,
    FILE_QR_R,
    FILE_RSVD_RIGHT,
    COMPUTE_COUNT
};
static const char *const names[] = {
    "memory_stats", "memory_budget", "trim_pool",      "plan",       "stream_stats",
    "cpu",          "matmul_kernel", "vector_math",    "threads",    "tiled_matmul",
    "axpy",         "mul_add",       "fused_products", "map_reduce", "explain",
    "execute",      "last_run",      "compare",        "topology",   "affinity",
    "first_touch",  "file_matmul",   "file_spmv",      "file_qr_r",  "file_rsvd_right"};
_Static_assert(sizeof names / sizeof *names == COMPUTE_COUNT, "compute names");

Obj *compute_module(VM *v) {
    return named_native_module(v, "compute", names, sizeof names / sizeof *names, COMPUTE_BASE);
}

Val compute_call(VM *v, int id, int argc, Val *args) {
    switch (id - COMPUTE_BASE) {
    case FILE_MATMUL:
        if (argc >= 3 && argc <= 5)
            return compute_file_matmul(
                v, args, argc > 3 ? compute_tile_size(v, args[3]) : COMPUTE_TILE_DEFAULT,
                argc > 4 ? args[4] : nil());
        break;
    case FILE_SPMV:
        if (argc >= 5 && argc <= 7)
            return compute_file_spmv(v, args, argc > 5 ? args[5] : nil(),
                                     argc > 6 ? args[6] : nil());
        break;
    case FILE_QR_R:
        if (argc == 1 || argc == 2)
            return compute_file_qr_r(v, args[0], argc > 1 ? args[1] : nil());
        break;
    case FILE_RSVD_RIGHT:
        if (argc >= 2 && argc <= 5)
            return compute_file_rsvd_right(v, argc, args);
        break;
    case TILED:
        if (argc == 2 || argc == 3)
            return compute_tiled(v, args[0], args[1],
                                 argc == 3 ? compute_tile_size(v, args[2]) : COMPUTE_TILE_DEFAULT);
        break;
    case AXPY:
    case MUL_ADD:
    case FUSED_PRODUCTS:
        if (argc == (id - COMPUTE_BASE == FUSED_PRODUCTS ? 5 : 3))
            return compute_fusion(v, (Fusion)(id - COMPUTE_BASE - AXPY), args);
        break;
    case MAP_REDUCE:
        if (argc == 2)
            return compute_map_reduce(v, args[0], args[1]);
        break;
    case THREADS:
        if (argc <= 1)
            return native_threads(v, argc, args);
        break;
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
        if (argc >= 2 && argc <= 4) {
            const char *operation = asstr(v, args[0]);
            if (!strcmp(operation, "matmul") && argc >= 3)
                return compute_plan(v, args[1], args[2], argc == 4 ? args[3] : nil());
            if (!strcmp(operation, "stream_stats"))
                return tensor_stream_stats(v, args[1], argc >= 3 ? args[2] : nil(), true,
                                           prefetch_option(v, argc == 4 ? args[3] : nil()));
            vmerror(v, "compute.plan supports matmul and stream_stats");
        }
        break;
    case EXPLAIN:
        if (argc == 1)
            return compute_explain(v, args[0]);
        break;
    case EXECUTE:
        if (argc == 3)
            return compute_execute(v, args[0], args[1], args[2]);
        break;
    case COMPARE:
        if (argc == 2 || argc == 3)
            return compute_compare(v, args[0], args[1], argc == 3 ? args[2] : nil());
        break;
    case TOPOLOGY:
        if (!argc)
            return compute_topology(v);
        break;
    case AFFINITY:
        if (argc <= 1)
            return compute_affinity(v, argc, args);
        break;
    case FIRST_TOUCH:
        if (argc == 1)
            return compute_first_touch(v, args[0]);
        break;
    case LAST_RUN:
        if (!argc)
            return compute_last_run(v);
        break;
    case STREAM_STATS:
        if (argc >= 1 && argc <= 3)
            return tensor_stream_stats(v, args[0], argc >= 2 ? args[1] : nil(), false,
                                       prefetch_option(v, argc == 3 ? args[2] : nil()));
        break;
    }
    vmerror(v, "invalid compute API arity");
}
