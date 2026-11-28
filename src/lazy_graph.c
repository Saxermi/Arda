#include "lazy_graph.h"
#include "tensor_file.h"

#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

enum { GRAPH_NODES = 128, GRAPH_DEPTH = 64, DEFAULT_BLOCK = 8192, MAX_BLOCK = 1048576 };
typedef enum { INPUT, FILE_INPUT, ADD, SUB, MUL, DIV, NEG, SIN, EXP, SUM } Operation;
typedef enum { FUSED, TILED, FULL, CANDIDATES } Kernel;
typedef enum { GENERAL, MUL_ADD, SINE_DECAY } Recipe;
typedef enum { DONE, CHANGED, READ_ERROR, CANCELLED } Status;

typedef struct GraphNode {
    Operation op;
    int depth;
    Obj *a, *b;
    /* The leading input owns the shape. The DAG keeps that node alive. */
    const TypedTensor *shape;
    union {
        struct {
            TypedTensor tensor;
            uint64_t version;
        } memory;
        struct {
            TensorFile file;
            struct timespec modified;
        } disk;
    } input;
} GraphNode;

typedef struct {
    Obj *object;
    int a, b, last, slot;
} Step;

typedef struct {
    size_t block, workspace;
    double ram, seconds;
    bool enabled, fits;
} Candidate;

typedef struct {
    double flops, specials, ram, disk, overhead;
    size_t block;
    int kernel;
} Options;

typedef struct {
    Obj *root;
    TypedTensor shape;
    int count, slots, fused_slots, operands[3];
    Recipe recipe;
    Kernel selected;
    bool reduction, file, mapped;
    size_t output;
    double flops, specials, file_bytes;
    Candidate candidates[CANDIDATES];
    Step steps[GRAPH_NODES];
} Graph;

typedef struct {
    Obj *report;
    const char *algorithm;
} GraphRun;

static GraphNode *node(Obj *object) {
    return (GraphNode *)(object + 1);
}

static GraphNode *as_node(VM *v, Val value) {
    if (!isobj(value, O_LAZY_NODE))
        vmerror_kind(v, "type", "expected a lazy graph node");
    return node(value.o);
}

static Graph *as_plan(VM *v, Val value) {
    if (!isobj(value, O_LAZY_PLAN))
        vmerror_kind(v, "type", "expected a lazy graph plan");
    return (Graph *)(value.o + 1);
}

static void put(VM *v, Obj *map, const char *key, Val value) {
    tableset(v, map, &map->as.map.tab, key, value);
}

static Val label(VM *v, const char *text) {
    return objval(string(v, text));
}

static Val bytes(size_t n) {
    return integer((int64_t)n);
}

static size_t min_size(size_t a, size_t b) {
    return a < b ? a : b;
}

static double seconds(void) {
    struct timespec t;
    return clock_gettime(CLOCK_MONOTONIC, &t) ? NAN : (double)t.tv_sec + 1e-9 * t.tv_nsec;
}

static Obj *new_node(VM *v, Operation op) {
    Obj *object = newobj_extra(v, O_LAZY_NODE, sizeof(GraphNode));
    *node(object) = (GraphNode){.op = op, .depth = 1};
    return object;
}

static Val input(VM *v, Val value) {
    TypedTensor t = typed_tensor(v, value);
    if (t.dtype != DT_F64)
        vmerror_kind(v, "dtype", "lazy graph inputs require Float64");
    Obj *object = new_node(v, INPUT);
    GraphNode *n = node(object);
    n->input.memory.tensor = t;
    n->input.memory.version = t.buffer->as.buffer.version;
    n->shape = &n->input.memory.tensor;
    barrier(v, object, objval(t.buffer));
    return objval(object);
}

static Val file_input(VM *v, Val path) {
    vm_require_file_access(v);
    tensor_file_require_little_endian(v);
    const char *filename = asstr(v, path);
    Obj *object = new_node(v, FILE_INPUT);
    GraphNode *n = node(object);
    n->input.disk.file.fd = -1;
    TensorFile file;
    int error = tensor_file_open(filename, &file);
    if (error)
        vmerror_kind(v, error == EINVAL ? "file_format" : "io", "cannot open lazy tensor input");
    struct stat st;
    if (file.layout.dtype != DT_F64 || fstat(file.fd, &st) || (uintmax_t)st.st_size != file.bytes) {
        close(file.fd);
        vmerror_kind(v, file.layout.dtype != DT_F64 ? "dtype" : "io",
                     "lazy file inputs require a stable Float64 tensor file");
    }
    n->input.disk.file = file;
    n->input.disk.modified = st.st_mtim;
    n->shape = &n->input.disk.file.layout;
    return objval(object);
}

static bool same_shape(const TypedTensor *a, const TypedTensor *b) {
    return a->rank == b->rank && !memcmp(a->shape, b->shape, sizeof(int) * (size_t)a->rank);
}

static Val operation(VM *v, Operation op, Val left, Val right) {
    GraphNode *a = as_node(v, left);
    GraphNode *b = op <= DIV ? as_node(v, right) : NULL;
    if (a->op == SUM || (b && b->op == SUM))
        vmerror_kind(v, "shape", "sum must be the root of a lazy graph");
    if (b && !same_shape(a->shape, b->shape))
        vmerror_kind(v, "shape", "lazy tensor operands must have identical shapes");
    int depth = 1 + (b && b->depth > a->depth ? b->depth : a->depth);
    if (depth > GRAPH_DEPTH)
        vmerror_kind(v, "limit", "lazy graph depth limit exceeded");
    Obj *object = new_node(v, op);
    GraphNode *n = node(object);
    n->depth = depth;
    n->a = left.o;
    n->b = b ? right.o : NULL;
    n->shape = a->shape;
    barrier(v, object, left);
    if (b)
        barrier(v, object, right);
    return objval(object);
}

static int sort_node(VM *v, Graph *g, Obj *object) {
    if (!object)
        return -1;
    for (int i = 0; i < g->count; i++)
        if (g->steps[i].object == object)
            return i;
    GraphNode *n = node(object);
    int a = sort_node(v, g, n->a), b = sort_node(v, g, n->b);
    if (g->count == GRAPH_NODES)
        vmerror_kind(v, "limit", "lazy graph exceeds 128 distinct nodes");
    int i = g->count++;
    g->steps[i] = (Step){object, a, b, i, -1};
    if (a >= 0)
        g->steps[a].last = i;
    if (b >= 0)
        g->steps[b].last = i;
    return i;
}

static bool leaf(Operation op) {
    return op == INPUT || op == FILE_INPUT;
}

static bool packed(const GraphNode *n) {
    return n->op == INPUT && typed_contiguous(&n->input.memory.tensor);
}

static void find_recipe(Graph *g) {
    const Step *root = &g->steps[g->count - 1];
    if (node(root->object)->op == ADD) {
        const Step *product = &g->steps[root->a];
        if (node(product->object)->op == MUL && leaf(node(g->steps[root->b].object)->op) &&
            leaf(node(g->steps[product->a].object)->op) &&
            leaf(node(g->steps[product->b].object)->op)) {
            g->recipe = MUL_ADD;
            g->operands[0] = product->a;
            g->operands[1] = product->b;
            g->operands[2] = root->b;
        }
    } else if (g->reduction) {
        const Step *product = &g->steps[root->a];
        if (node(product->object)->op != MUL)
            return;
        const Step *s = &g->steps[product->a], *e = &g->steps[product->b];
        if (node(s->object)->op != SIN || node(e->object)->op != EXP)
            return;
        const Step *neg = &g->steps[e->a];
        if (node(neg->object)->op == NEG && neg->a == s->a &&
            leaf(node(g->steps[s->a].object)->op)) {
            g->recipe = SINE_DECAY;
            g->operands[0] = s->a;
        }
    }
}

static void analyze(VM *v, Val root, Graph *g) {
    GraphNode *n = as_node(v, root);
    g->root = root.o;
    g->shape = *n->shape;
    g->shape.buffer = NULL;
    g->reduction = n->op == SUM;
    if (!g->reduction && !typed_materializable(&g->shape))
        vmerror_kind(v, "shape",
                     "graph output exceeds the heap tensor limit; reduce before materializing");
    g->output = g->reduction ? 0 : g->shape.n * sizeof(double);
    sort_node(v, g, root.o);
    find_recipe(g);
    int free_list[GRAPH_NODES], available = 0;
    for (int i = 0; i < g->count; i++) {
        Step *s = &g->steps[i];
        GraphNode *current = node(s->object);
        if (current->op == FILE_INPUT) {
            g->file = true;
            g->file_bytes += (double)g->shape.n * sizeof(double);
        }
        if (current->op == INPUT && current->input.memory.tensor.buffer->as.buffer.mapped) {
            g->mapped = true;
            g->file_bytes += (double)g->shape.n * sizeof(double);
        }
        if (current->op == SIN || current->op == EXP)
            g->specials += (double)g->shape.n;
        else if (!leaf(current->op))
            g->flops += (double)g->shape.n;
        if (i < g->count - 1 && !packed(current))
            s->slot = available ? free_list[--available] : g->slots++;
        /* Release slots only after the consuming step has received its own. */
        int children[] = {s->a, s->b};
        for (int j = 0; j < 2; j++) {
            int child = children[j];
            if (child >= 0 && (j == 0 || child != s->a))
                if (g->steps[child].last == i && g->steps[child].slot >= 0)
                    free_list[available++] = g->steps[child].slot;
        }
        if (leaf(current->op) && !packed(current))
            g->fused_slots++;
    }
}

static double rate(VM *v, Val value, bool zero) {
    double x = asnum(v, value);
    if (!isfinite(x) || (zero ? x < 0 : x <= 0))
        vmerror_kind(v, "domain", "graph rates must be finite and positive");
    return x;
}

static void calibration(VM *v, Val value, Options *o) {
    if (!isobj(value, O_MAP))
        vmerror_kind(v, "type", "graph calibration must be a map");
    const Table *t = &value.o->as.map.tab;
    for (int i = 0; i < t->cap; i++) {
        const Entry *e = &t->a[i];
        if (!e->key)
            continue;
        if (!strcmp(e->key, "flops_per_second"))
            o->flops = rate(v, e->val, false);
        else if (!strcmp(e->key, "specials_per_second"))
            o->specials = rate(v, e->val, false);
        else if (!strcmp(e->key, "ram_bytes_per_second"))
            o->ram = rate(v, e->val, false);
        else if (!strcmp(e->key, "disk_bytes_per_second"))
            o->disk = rate(v, e->val, false);
        else if (!strcmp(e->key, "overhead_seconds"))
            o->overhead = rate(v, e->val, true);
        else
            vmerror_kind(v, "domain", "unknown graph calibration field");
    }
}

static Options options(VM *v, Val value) {
    Options o = {.block = DEFAULT_BLOCK, .kernel = -1};
    if (value.t == V_NIL)
        return o;
    if (!isobj(value, O_MAP))
        vmerror_kind(v, "type", "graph options must be a map");
    const Table *t = &value.o->as.map.tab;
    for (int i = 0; i < t->cap; i++) {
        const Entry *e = &t->a[i];
        if (!e->key)
            continue;
        if (!strcmp(e->key, "block_elements")) {
            if (e->val.t != V_INT || e->val.i < 1 || e->val.i > MAX_BLOCK)
                vmerror_kind(v, "domain", "graph block_elements must be 1..1048576");
            o.block = (size_t)e->val.i;
        } else if (!strcmp(e->key, "kernel")) {
            const char *s = asstr(v, e->val);
            if (!strcmp(s, "fused"))
                o.kernel = FUSED;
            else if (!strcmp(s, "tiled"))
                o.kernel = TILED;
            else if (!strcmp(s, "in_ram"))
                o.kernel = FULL;
            else if (strcmp(s, "auto"))
                vmerror_kind(v, "domain", "unknown graph kernel");
        } else if (!strcmp(e->key, "calibration")) {
            calibration(v, e->val, &o);
        } else {
            vmerror_kind(v, "domain", "unknown graph option");
        }
    }
    return o;
}

static size_t blocks(size_t n, size_t size) {
    return n ? 1 + (n - 1) / size : 0;
}

static double logical_ram(const Graph *g, bool fused) {
    double streams = 0;
    for (int i = 0; i < g->count; i++) {
        const Step *s = &g->steps[i];
        const GraphNode *n = node(s->object);
        if (leaf(n->op))
            streams += n->op == FILE_INPUT ? 1 : !packed(n) || i == g->count - 1 ? 2 : 0;
        else if (!fused)
            streams += (s->a >= 0) + (s->b >= 0) + (n->op != SUM);
    }
    if (fused)
        streams += g->recipe == MUL_ADD ? 4 : 1;
    return streams * (double)g->shape.n * sizeof(double);
}

static Candidate candidate(const Graph *g, const Options *o, Kernel k, size_t available) {
    size_t n = g->shape.n;
    Candidate c = {.enabled = k != FUSED || g->recipe != GENERAL, .seconds = NAN};
    if (!c.enabled)
        return c;
    size_t step = (size_t)(k == FUSED ? g->fused_slots : g->slots) * sizeof(double);
    c.block = k == FULL ? n : min_size(n, o->block);
    if (k != FULL && n && step && available >= g->output) {
        size_t affordable = (available - g->output) / step;
        c.block = min_size(c.block, affordable ? affordable : 1);
    }
    c.workspace = step && c.block > SIZE_MAX / step ? SIZE_MAX : step * c.block;
    c.fits = g->output <= available && c.workspace <= available - g->output;
    c.ram = logical_ram(g, k == FUSED);
    if ((!g->flops || o->flops) && (!g->specials || o->specials) && (!c.ram || o->ram) &&
        (!g->file_bytes || o->disk)) {
        double cpu =
            (g->flops ? g->flops / o->flops : 0) + (g->specials ? g->specials / o->specials : 0);
        double ram = c.ram ? c.ram / o->ram : 0;
        double disk = g->file_bytes ? g->file_bytes / o->disk : 0;
        c.seconds = fmax(cpu, fmax(ram, disk)) + (double)blocks(n, c.block) * o->overhead;
    }
    return c;
}

static void choose(VM *v, Graph *g, Options o) {
    for (int i = 0; i < CANDIDATES; i++)
        g->candidates[i] = candidate(g, &o, (Kernel)i, tensor_memory_available(v));
    if (o.kernel >= 0) {
        Candidate c = g->candidates[o.kernel];
        if (!c.enabled)
            vmerror_kind(v, "backend", "requested fused recipe does not match the graph");
        if (!c.fits)
            vmerror_kind(v, "memory_budget",
                         "requested graph plan exceeds available tensor memory");
        g->selected = (Kernel)o.kernel;
        return;
    }
    Kernel order[] = {FUSED, TILED, FULL};
    if (!g->file && g->shape.n <= o.block) {
        order[1] = FULL;
        order[2] = TILED;
    }
    int best = -1;
    for (int i = 0; i < CANDIDATES; i++) {
        Kernel k = order[i];
        Candidate c = g->candidates[k];
        if (c.enabled && c.fits &&
            (best < 0 || (isfinite(c.seconds) && c.seconds < g->candidates[best].seconds)))
            best = k;
    }
    if (best < 0)
        vmerror_kind(v, "memory_budget", "no graph plan fits the available tensor memory");
    g->selected = (Kernel)best;
}

Val graph_plan(VM *v, Val root, Val config) {
    Options o = options(v, config);
    Graph g = {0};
    analyze(v, root, &g);
    choose(v, &g, o);
    Obj *object = newobj_extra(v, O_LAZY_PLAN, sizeof g);
    *(Graph *)(object + 1) = g;
    barrier(v, object, root);
    return objval(object);
}

static const char *kernel_name(Kernel k) {
    return k == FUSED ? "fused" : k == TILED ? "tiled" : "in_ram";
}

static const char *storage_name(const Graph *g) {
    return g->file ? "file_blocks" : g->mapped ? "mapped" : "heap";
}

static const char *algorithm(const Graph *g) {
    if (g->selected == FUSED)
        return g->recipe == MUL_ADD ? "Fused multiply and add without FMA"
                                    : "Fused sine decay with ordered reduction";
    return g->selected == FULL ? "Whole-array topological schedule"
                               : "Tiled topological schedule with reusable buffers";
}

static Val estimate(double seconds_estimate) {
    return isfinite(seconds_estimate) ? number(seconds_estimate) : nil();
}

static void candidate_fields(VM *v, Obj *map, const Graph *g, Kernel k) {
    Candidate c = g->candidates[k];
    put(v, map, "kernel", label(v, kernel_name(k)));
    put(v, map, "available", boolean(c.enabled));
    put(v, map, "fits", boolean(c.fits));
    put(v, map, "block_elements", bytes(c.block));
    put(v, map, "workspace_bytes", c.workspace <= PTRDIFF_MAX ? bytes(c.workspace) : nil());
    put(v, map, "logical_ram_bytes", number(c.ram));
    put(v, map, "estimated_seconds", estimate(c.seconds));
}

Val graph_explain(VM *v, Val plan) {
    const Graph *g = as_plan(v, plan);
    Obj *report = newobj(v, O_MAP), *all = list(v);
    candidate_fields(v, report, g, g->selected);
    put(v, report, "algorithm", label(v, algorithm(g)));
    put(v, report, "storage", label(v, storage_name(g)));
    put(v, report, "nodes", integer(g->count));
    put(v, report, "elements", bytes(g->shape.n));
    put(v, report, "output_bytes", bytes(g->output));
    put(v, report, "logical_file_bytes", number(g->file_bytes));
    put(v, report, "flops", number(g->flops));
    put(v, report, "special_ops", number(g->specials));
    put(v, report, "deterministic", boolean(true));
    put(v, report, "candidates", objval(all));
    for (int i = 0; i < CANDIDATES; i++) {
        Obj *row = newobj(v, O_MAP);
        candidate_fields(v, row, g, (Kernel)i);
        append(v, all, objval(row));
    }
    return objval(report);
}

static bool unchanged(const Graph *g) {
    for (int i = 0; i < g->count; i++) {
        const GraphNode *n = node(g->steps[i].object);
        if (n->op == INPUT &&
            n->input.memory.version != n->input.memory.tensor.buffer->as.buffer.version)
            return false;
        if (n->op == FILE_INPUT) {
            struct stat st;
            if (fstat(n->input.disk.file.fd, &st) ||
                (uintmax_t)st.st_size != n->input.disk.file.bytes ||
                st.st_mtim.tv_sec != n->input.disk.modified.tv_sec ||
                st.st_mtim.tv_nsec != n->input.disk.modified.tv_nsec)
                return false;
        }
    }
    return true;
}

static bool read_input(const GraphNode *n, size_t start, size_t count, double *output) {
    if (n->op == FILE_INPUT)
        return tensor_file_read(&n->input.disk.file, start, count, output);
    const TypedTensor *t = &n->input.memory.tensor;
    if (packed(n)) {
        memcpy(output, typed_element(t, (ptrdiff_t)(t->offset + start)), count * sizeof(double));
    } else {
        for (size_t i = 0; i < count; i++)
            memcpy(output + i, typed_element(t, typed_offset(t, start + i)), sizeof(double));
    }
    return true;
}

static const double *borrow(const GraphNode *n, size_t start) {
    const TypedTensor *t = &n->input.memory.tensor;
    return (const double *)typed_element(t, (ptrdiff_t)(t->offset + start));
}

static void pointwise(Operation op, size_t count, double *out, const double *a, const double *b) {
#define APPLY(name, expr)                                                                          \
    case name:                                                                                     \
        for (size_t j = 0; j < count; j++)                                                         \
            out[j] = (expr);                                                                       \
        break
    switch (op) {
        APPLY(ADD, a[j] + b[j]);
        APPLY(SUB, a[j] - b[j]);
        APPLY(MUL, a[j] * b[j]);
        APPLY(DIV, a[j] / b[j]);
        APPLY(NEG, -a[j]);
        APPLY(SIN, sin(a[j]));
        APPLY(EXP, exp(a[j]));
    default:
        break;
    }
#undef APPLY
}

static bool scheduled_block(const Graph *g, size_t start, size_t count, size_t stride, double *work,
                            double *out, long double *sum) {
    const double *values[GRAPH_NODES];
    for (int i = 0; i < g->count; i++) {
        const Step *s = &g->steps[i];
        const GraphNode *n = node(s->object);
        if (n->op == SUM) {
            for (size_t j = 0; j < count; j++)
                *sum += values[s->a][j];
        } else if (packed(n) && s->slot < 0 && i != g->count - 1) {
            values[i] = borrow(n, start);
        } else {
            double *dest = s->slot < 0 ? out : work + (size_t)s->slot * stride;
            if (leaf(n->op)) {
                if (!read_input(n, start, count, dest))
                    return false;
            } else {
                pointwise(n->op, count, dest, values[s->a], s->b < 0 ? NULL : values[s->b]);
            }
            values[i] = dest;
        }
    }
    return true;
}

static bool fused_block(const Graph *g, size_t start, size_t count, size_t stride, double *work,
                        double *out, long double *sum) {
    const double *values[GRAPH_NODES];
    int slot = 0;
    for (int i = 0; i < g->count; i++) {
        const GraphNode *n = node(g->steps[i].object);
        if (!leaf(n->op))
            continue;
        if (packed(n)) {
            values[i] = borrow(n, start);
        } else {
            double *dest = work + (size_t)slot++ * stride;
            if (!read_input(n, start, count, dest))
                return false;
            values[i] = dest;
        }
    }
    const double *a = values[g->operands[0]];
    if (g->recipe == MUL_ADD) {
        const double *b = values[g->operands[1]], *c = values[g->operands[2]];
        for (size_t j = 0; j < count; j++)
            out[j] = a[j] * b[j] + c[j];
    } else {
        for (size_t j = 0; j < count; j++) {
            double product = sin(a[j]) * exp(-a[j]);
            *sum += product;
        }
    }
    return true;
}

static Val run_result(VM *v, Val value, const Graph *g, Candidate c, double elapsed,
                      int trace_limit) {
    Obj *report = newobj(v, O_MAP), *trace = list(v);
    size_t total = blocks(g->shape.n, c.block);
    size_t recorded = min_size(total, (size_t)trace_limit);
    for (size_t i = 0; i < recorded; i++) {
        Obj *event = newobj(v, O_MAP);
        put(v, event, "start", bytes(i * c.block));
        put(v, event, "elements", bytes(min_size(c.block, g->shape.n - i * c.block)));
        append(v, trace, objval(event));
    }
    put(v, report, "value", value);
    put(v, report, "algorithm", label(v, algorithm(g)));
    candidate_fields(v, report, g, g->selected);
    put(v, report, "workspace_peak_bytes", bytes(c.workspace));
    put(v, report, "output_bytes", bytes(g->output));
    put(v, report, "logical_file_bytes", number(g->file_bytes));
    put(v, report, "elapsed_seconds", estimate(elapsed));
    put(v, report, "estimate_error_seconds", estimate(elapsed - c.seconds));
    put(v, report, "blocks", bytes(total));
    put(v, report, "trace", objval(trace));
    put(v, report, "trace_omitted", bytes(total - recorded));
    return graph_make_run(v, report, algorithm(g));
}

Val graph_make_run(VM *v, Obj *report, const char *name) {
    Obj *run = newobj_extra(v, O_LAZY_RUN, sizeof(GraphRun));
    *(GraphRun *)(run + 1) = (GraphRun){report, name};
    barrier(v, run, objval(report));
    return objval(run);
}

Val graph_execute(VM *v, Val plan, int trace_limit) {
    const Graph *g = as_plan(v, plan);
    if (g->file || g->mapped)
        vm_require_file_access(v);
    if (!unchanged(g))
        vmerror_kind(v, "mutation", "lazy input changed after graph construction");
    Candidate c = g->candidates[g->selected];
    size_t available = tensor_memory_available(v);
    if (g->output > available || c.workspace > available - g->output)
        vmerror_kind(v, "memory_budget", "graph plan exceeds the current tensor memory budget");
    Obj *out = g->reduction ? NULL : typed_new(v, DT_F64, g->shape.rank, g->shape.shape);
    double *data = out ? (double *)typed_element(&out->as.tensor, 0) : NULL;
    unsigned char *work = c.workspace ? tensor_workspace_acquire(v, c.workspace) : NULL;
    if (c.workspace && !work) {
        if (out)
            tensor_buffer_discard(v, out->as.tensor.buffer);
        vmerror_kind(v, "memory_budget", "cannot reserve graph workspace");
    }
    double start_time = trace_limit >= 0 ? seconds() : NAN;
    long double sum = 0;
    Status status = DONE;
    for (size_t start = 0; start < g->shape.n;) {
        if (v->interrupted && *v->interrupted) {
            status = CANCELLED;
            break;
        }
        size_t count = min_size(c.block, g->shape.n - start);
        bool ok = g->selected == FUSED ? fused_block(g, start, count, c.block, (double *)work,
                                                     out ? data + start : NULL, &sum)
                                       : scheduled_block(g, start, count, c.block, (double *)work,
                                                         out ? data + start : NULL, &sum);
        if (!ok) {
            status = READ_ERROR;
            break;
        }
        start += count;
    }
    double elapsed = trace_limit >= 0 ? seconds() - start_time : NAN;
    if (status == DONE && v->interrupted && *v->interrupted)
        status = CANCELLED;
    if (status == DONE && g->file && !unchanged(g))
        status = CHANGED;
    if (work)
        tensor_workspace_release(v, work);
    if (status != DONE) {
        if (out)
            tensor_buffer_discard(v, out->as.tensor.buffer);
        const char *code = status == CANCELLED ? "interrupt"
                           : status == CHANGED ? "mutation"
                                               : "io";
        vmerror_kind(v, code, "lazy execution did not complete");
    }
    Val value = out ? objval(out) : number((double)sum);
    return trace_limit < 0 ? value : run_result(v, value, g, c, elapsed, trace_limit);
}

Val graph_run_info(VM *v, Val value) {
    if (!isobj(value, O_LAZY_RUN))
        vmerror_kind(v, "type", "expected a lazy execution run");
    return objval(((GraphRun *)(value.o + 1))->report);
}

const char *graph_run_algorithm(Val value) {
    return ((GraphRun *)(value.o + 1))->algorithm;
}

void graph_trace_object(VM *v, Obj *object) {
    if (object->type == O_LAZY_PLAN) {
        markobj(v, ((Graph *)(object + 1))->root);
    } else if (object->type == O_LAZY_RUN) {
        markobj(v, ((GraphRun *)(object + 1))->report);
    } else {
        GraphNode *n = node(object);
        markobj(v, n->a);
        markobj(v, n->b);
        if (n->op == INPUT)
            markobj(v, n->input.memory.tensor.buffer);
    }
}

void graph_release_object(Obj *object) {
    if (node(object)->op == FILE_INPUT && node(object)->input.disk.file.fd >= 0)
        close(node(object)->input.disk.file.fd);
}

Obj *lazy_module(VM *v) {
    const char *const names[] = {"input", "file", "add", "sub", "mul",        "div",
                                 "neg",   "sin",  "exp", "sum", "materialize"};
    return named_native_module(v, "lazy", names, sizeof names / sizeof *names, LAZY_BASE);
}

Val lazy_call(VM *v, int id, int argc, Val *args) {
    int op = id - LAZY_BASE;
    if (op == SUM + 1 && (argc == 1 || argc == 2))
        return graph_execute(v, graph_plan(v, args[0], argc == 2 ? args[1] : nil()), -1);
    if (op < INPUT || op > SUM || argc != (op >= ADD && op <= DIV ? 2 : 1))
        vmerror_kind(v, "arity", "invalid lazy operation or argument count");
    if (op == INPUT)
        return input(v, args[0]);
    if (op == FILE_INPUT)
        return file_input(v, args[0]);
    return operation(v, (Operation)op, args[0], argc == 2 ? args[1] : nil());
}
