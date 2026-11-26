#include "compute_plan.h"
#include "cpu_kernels.h"
#include <time.h>

enum { KERNEL_TILED = -1, TUNE_CACHE_SIZE = 8 };
enum { AXIS_ROWS, AXIS_DEPTH, AXIS_COLUMNS };

typedef struct {
    int dimensions[3], rhs_rank;
    ptrdiff_t strides_a[2], strides_b[2];
    unsigned mapped;
} ProblemShape;

typedef struct {
    double flops, ram, disk, overhead;
} Calibration;

typedef struct {
    ProblemShape shape;
    Calibration rates;
    uint64_t flops;
    double logical_bytes, file_bytes;
    size_t workspace, output;
    int kernel, block, threads;
    double measured;
    bool cache_hit;
} Plan;

typedef struct {
    int kernel, block, threads;
    size_t workspace;
    bool valid, empty;
} Run;

typedef struct {
    ProblemShape shape;
    int threads, kernel, block;
    double seconds;
} TuneEntry;

struct ComputeState {
    Run last;
    TuneEntry cache[TUNE_CACHE_SIZE];
    unsigned next;
};

static struct ComputeState *compute_state(VM *v) {
    if (!v->compute_state)
        v->compute_state = mem(sizeof *v->compute_state);
    return v->compute_state;
}

static void field(VM *v, Obj *map, const char *name, Val value) {
    tableset(v, map, &map->as.map.tab, name, value);
}

static Table *options_table(VM *v, Val options) {
    if (options.t == V_NIL)
        return NULL;
    if (!isobj(options, O_MAP))
        vmerror_kind(v, "type", "compute options must be a map");
    return &options.o->as.map.tab;
}

static Val option(Table *table, const char *name, int *seen) {
    Entry *entry = table ? tablefind(table, name) : NULL;
    *seen += entry != NULL;
    return entry ? entry->val : nil();
}

static void check_options(VM *v, Table *table, int seen) {
    if (table && table->n != seen)
        vmerror_kind(v, "domain", "unrecognized compute option");
}

static double rate(VM *v, Val value, const char *label, bool allow_zero) {
    if (value.t == V_NIL)
        return 0;
    double number = asnum(v, value);
    if (!isfinite(number) || number < 0 || (!number && !allow_zero))
        vmerror_kind(v, "domain", "invalid compute calibration or tolerance for %s", label);
    return number;
}

static Calibration calibration(VM *v, Val options) {
    Table *table = options_table(v, options);
    int seen = 0;
    Calibration rates;
    rates.flops = rate(v, option(table, "flops_per_second", &seen), "flops_per_second", false);
    rates.ram =
        rate(v, option(table, "ram_bytes_per_second", &seen), "ram_bytes_per_second", false);
    rates.disk =
        rate(v, option(table, "disk_bytes_per_second", &seen), "disk_bytes_per_second", false);
    rates.overhead = rate(v, option(table, "overhead_seconds", &seen), "overhead_seconds", true);
    check_options(v, table, seen);
    return rates;
}

static ProblemShape problem_shape(const MatmulProblem *p) {
    return (ProblemShape){
        .dimensions = {p->a.shape[0], p->a.shape[1], p->b.rank == 2 ? p->b.shape[1] : 1},
        .rhs_rank = p->b.rank,
        .strides_a = {p->a.stride[0], p->a.stride[1]},
        .strides_b = {p->b.stride[0], p->b.rank == 2 ? p->b.stride[1] : 0},
        .mapped = (unsigned)p->a.buffer->as.buffer.mapped |
                  ((unsigned)p->b.buffer->as.buffer.mapped << 1)};
}

static bool same_shape(const ProblemShape *a, const ProblemShape *b) {
    return a->rhs_rank == b->rhs_rank && a->mapped == b->mapped &&
           !memcmp(a->dimensions, b->dimensions, sizeof a->dimensions) &&
           !memcmp(a->strides_a, b->strides_a, sizeof a->strides_a) &&
           !memcmp(a->strides_b, b->strides_b, sizeof a->strides_b);
}

static bool packed_gemm(const MatmulProblem *p) {
    return p->b.rank == 2 && typed_contiguous(&p->a) && typed_contiguous(&p->b) &&
           typed_materializable(&p->a) && typed_materializable(&p->b);
}

static int requested_threads(VM *v, Val value) {
    if (value.t == V_NIL)
        return v->native_threads;
    if (value.t != V_INT || value.i < 1 || value.i > 256)
        vmerror_kind(v, "domain", "compute threads must be an integer from 1 to 256");
    if (value.i != 1 && !cpu_parallel_available())
        vmerror_kind(v, "backend", "native parallel kernels require OPENMP=1 without BLAS");
    return (int)value.i;
}

static void checked_workspace(VM *v, size_t output, size_t workspace) {
    size_t available = tensor_memory_available(v);
    if (workspace > available || output > available - workspace)
        vmerror_kind(v, "memory_budget", "compute output and workspace exceed the tensor budget");
}

static double *reserve(VM *v, size_t bytes) {
    if (!bytes)
        return NULL;
    unsigned char *memory = tensor_workspace_acquire(v, bytes);
    if (!memory)
        vmerror_kind(v, "memory_budget", "unable to reserve compute workspace");
    return (double *)memory;
}

static double *output_data(Obj *out) {
    return (double *)typed_element(&out->as.tensor, 0);
}

static Run run_into(const MatmulProblem *p, VM *v, const Plan *plan, double *out,
                    double *workspace) {
    const int *d = plan->shape.dimensions;
    bool empty = !plan->output || !d[AXIS_DEPTH];
    int threads = 1;
    if (!empty) {
        if (plan->kernel == KERNEL_TILED)
            threads = compute_tiled_into(p, plan->block, plan->threads, out,
                                         plan->workspace ? workspace : NULL);
        else
            threads =
                cpu_matmul_run(v, plan->kernel, plan->threads, d[AXIS_ROWS], d[AXIS_DEPTH],
                               d[AXIS_COLUMNS], (const double *)typed_element(&p->a, p->a.offset),
                               (const double *)typed_element(&p->b, p->b.offset), out);
    }
    return (Run){.kernel = plan->kernel,
                 .block = plan->block,
                 .threads = threads,
                 .workspace = plan->workspace,
                 .valid = true,
                 .empty = empty};
}

static double wall_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts))
        return NAN;
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int compare_time(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double trial(VM *v, const MatmulProblem *p, const Plan *plan, double *out, double *work) {
    enum { SAMPLES = 3 };
    double times[SAMPLES];
    memset(out, 0, plan->output);
    run_into(p, v, plan, out, work);
    for (int i = 0; i < SAMPLES; i++) {
        double start = wall_seconds();
        memset(out, 0, plan->output);
        run_into(p, v, plan, out, work);
        times[i] = wall_seconds() - start;
        if (!isfinite(times[i]))
            return NAN;
    }
    qsort(times, SAMPLES, sizeof *times, compare_time);
    return times[SAMPLES / 2];
}

static void tune(VM *v, const MatmulProblem *p, Plan *plan) {
    if (!plan->flops || plan->flops > UINT64_C(128000000) || plan->shape.mapped)
        vmerror_kind(v, "domain",
                     "compute tuning needs a nonempty heap problem of at most 64M terms");
    struct ComputeState *state = compute_state(v);
    for (unsigned i = 0; i < TUNE_CACHE_SIZE; i++) {
        const TuneEntry *entry = &state->cache[i];
        if (entry->threads == plan->threads && same_shape(&plan->shape, &entry->shape)) {
            plan->kernel = entry->kernel;
            plan->block = entry->block;
            plan->measured = entry->seconds;
            plan->cache_hit = true;
            return;
        }
    }

    // Strict tiled kernels and the non-contracting native default are comparable candidates.
    Plan candidates[3] = {*plan, *plan, *plan};
    int count = 2;
    for (int i = 0; i < 2; i++) {
        candidates[i].kernel = KERNEL_TILED;
        candidates[i].block = i ? COMPUTE_TILE_DEFAULT : 32;
        candidates[i].workspace = compute_tiled_workspace(p, candidates[i].block);
    }
#ifndef ARDA_BLAS
    if (plan->kernel != KERNEL_TILED)
        count++;
#endif
    size_t workspace = 0;
    for (int i = 0; i < count; i++)
        if (candidates[i].workspace > workspace)
            workspace = candidates[i].workspace;
    checked_workspace(v, plan->output, workspace);
    double *work = reserve(v, workspace);
    Obj *out = typed_new(v, DT_F64, p->output.rank, p->output.shape);
    double best = INFINITY;
    for (int i = 0; i < count; i++) {
        double measured = trial(v, p, &candidates[i], output_data(out), work);
        if (measured < best) {
            best = measured;
            *plan = candidates[i];
        }
    }
    tensor_buffer_discard(v, out->as.tensor.buffer);
    if (work)
        tensor_workspace_release(v, (unsigned char *)work);
    if (!isfinite(best))
        vmerror_kind(v, "backend", "monotonic clock unavailable for compute tuning");
    plan->measured = best;
    state->cache[state->next++ % TUNE_CACHE_SIZE] =
        (TuneEntry){plan->shape, plan->threads, plan->kernel, plan->block, best};
}

static Plan make_plan(VM *v, const MatmulProblem *p, Val options) {
    Table *table = options_table(v, options);
    int seen = 0;
    Val kernel_option = option(table, "kernel", &seen);
    Val block_option = option(table, "block", &seen);
    Val thread_option = option(table, "threads", &seen);
    Val tune_option = option(table, "tune", &seen);
    Val rates = option(table, "calibration", &seen);
    check_options(v, table, seen);
    if (tune_option.t != V_NIL && tune_option.t != V_BOOL)
        vmerror_kind(v, "type", "compute tune must be a Boolean");
    bool autotune = tune_option.t == V_BOOL && tune_option.n != 0;
    const char *kernel = kernel_option.t == V_NIL ? "auto" : asstr(v, kernel_option);
    bool automatic = !strcmp(kernel, "auto");
    bool tiled = !strcmp(kernel, "tiled");
    if (autotune && (!automatic || block_option.t != V_NIL))
        vmerror_kind(v, "domain", "compute tuning requires an automatic kernel and block");

    Plan plan = {.shape = problem_shape(p),
                 .rates = calibration(v, rates),
                 .output = p->output.n * sizeof(double),
                 .threads = requested_threads(v, thread_option),
                 .kernel = KERNEL_TILED,
                 .measured = NAN};
    const int *d = plan.shape.dimensions;
    if (automatic) {
#ifndef ARDA_BLAS
        if (block_option.t == V_NIL && packed_gemm(p))
            plan.kernel = cpu_matmul_resolve(v, GEMM_AUTO, d[AXIS_ROWS], d[AXIS_COLUMNS]);
#endif
    } else if (!tiled) {
        if (!packed_gemm(p))
            vmerror_kind(v, "shape", "requested compute backend needs packed matrix inputs");
        int choice = cpu_matmul_choice(v, kernel);
        if (choice < 0)
            vmerror_kind(v, "backend", "requested compute kernel is unavailable");
        plan.kernel = cpu_matmul_resolve(v, choice, d[AXIS_ROWS], d[AXIS_COLUMNS]);
        if (block_option.t != V_NIL)
            vmerror_kind(v, "domain", "custom block size requires the tiled kernel");
#ifdef ARDA_BLAS
        if (thread_option.t != V_NIL)
            vmerror_kind(v, "domain", "configure CBLAS threads through the provider");
#endif
    }
    uint64_t terms = (uint64_t)p->output.n * (unsigned)d[AXIS_DEPTH];
    plan.flops = 2 * terms;
    plan.logical_bytes = (double)plan.output;
    if (terms) {
        double left = (double)p->a.n * sizeof(double), right = (double)p->b.n * sizeof(double);
        plan.logical_bytes += left + right;
        plan.file_bytes =
            ((plan.shape.mapped & 1) ? left : 0) + ((plan.shape.mapped & 2) ? right : 0);
    }
    plan.block =
        plan.kernel == KERNEL_TILED
            ? (block_option.t == V_NIL ? COMPUTE_TILE_DEFAULT : compute_tile_size(v, block_option))
            : cpu_matmul_block(plan.kernel);
    if (autotune)
        tune(v, p, &plan);
    plan.workspace = plan.kernel == KERNEL_TILED ? compute_tiled_workspace(p, plan.block) : 0;
    return plan;
}

static const Plan *as_plan(VM *v, Val value) {
    if (!isobj(value, O_COMPUTE_PLAN))
        vmerror_kind(v, "type", "expected a compute plan");
    return (const Plan *)(value.o + 1);
}

static const char *kernel_name(int kernel) {
    return kernel == KERNEL_TILED ? "tiled" : cpu_matmul_kernel_name(kernel);
}

static Val thread_field(int count) {
    return count ? integer(count) : nil();
}

static Val run_report(VM *v, Run run) {
    Obj *map = newobj(v, O_MAP);
    field(v, map, "kernel", objval(string(v, run.empty ? "none" : kernel_name(run.kernel))));
    field(v, map, "block", run.block && !run.empty ? integer(run.block) : nil());
    field(v, map, "threads", thread_field(run.threads));
    field(v, map, "workspace_bytes", integer((int64_t)run.workspace));
    return objval(map);
}

Val compute_plan(VM *v, Val a, Val b, Val options) {
    MatmulProblem problem = compute_matmul_problem(v, a, b);
    Plan plan = make_plan(v, &problem, options);
    Obj *obj = newobj_extra(v, O_COMPUTE_PLAN, sizeof plan);
    *(Plan *)(obj + 1) = plan;
    return objval(obj);
}

static Val estimate(const Plan *p, bool include_overhead) {
    const Calibration *r = &p->rates;
    if ((p->flops && !r->flops) || (p->logical_bytes && !r->ram) || (p->file_bytes && !r->disk))
        return nil();
    double compute = p->flops ? p->flops / r->flops : 0;
    double ram = p->logical_bytes ? p->logical_bytes / r->ram : 0;
    double disk = p->file_bytes ? p->file_bytes / r->disk : 0;
    double seconds = fmax(compute, fmax(ram, disk)) + (include_overhead ? r->overhead : 0);
    return isfinite(seconds) ? number(seconds) : nil();
}

Val compute_explain(VM *v, Val value) {
    const Plan *p = as_plan(v, value);
    const int *d = p->shape.dimensions;
    int team = p->kernel == KERNEL_TILED
                   ? compute_tiled_team(d[AXIS_ROWS], p->shape.rhs_rank, p->flops / 2, p->block,
                                        p->threads, p->workspace != 0)
                   : cpu_matmul_team(p->threads, d[AXIS_ROWS], d[AXIS_DEPTH], d[AXIS_COLUMNS]);
    Obj *map = newobj(v, O_MAP);
    field(v, map, "kernel", objval(string(v, kernel_name(p->kernel))));
    field(v, map, "block", p->block ? integer(p->block) : nil());
    field(v, map, "requested_threads", team ? integer(p->threads) : nil());
    field(v, map, "estimated_threads", thread_field(team));
    field(v, map, "workspace_bytes", integer((int64_t)p->workspace));
    field(v, map, "output_bytes", integer((int64_t)p->output));
    field(v, map, "flops", integer((int64_t)p->flops));
    field(v, map, "logical_ram_bytes", number(p->logical_bytes));
    field(v, map, "logical_file_bytes", number(p->file_bytes));
    field(v, map, "lower_bound_seconds", estimate(p, false));
    field(v, map, "estimated_seconds", estimate(p, true));
    field(v, map, "measured_seconds", isfinite(p->measured) ? number(p->measured) : nil());
    field(v, map, "cache_hit", boolean(p->cache_hit));
    return objval(map);
}

Val compute_execute(VM *v, Val value, Val a, Val b) {
    const Plan *plan = as_plan(v, value);
    MatmulProblem p = compute_matmul_problem(v, a, b);
    ProblemShape shape = problem_shape(&p);
    if (!same_shape(&shape, &plan->shape))
        vmerror_kind(v, "shape", "compute plan does not match operand shape, strides or storage");
    checked_workspace(v, plan->output, plan->workspace);
    double *workspace = reserve(v, plan->workspace);
    Obj *out = typed_new(v, DT_F64, p.output.rank, p.output.shape);
    Run run = run_into(&p, v, plan, output_data(out), workspace);
    if (workspace)
        tensor_workspace_release(v, (unsigned char *)workspace);
    compute_state(v)->last = run;
    return objval(out);
}

Val compute_last_run(VM *v) {
    return v->compute_state && v->compute_state->last.valid ? run_report(v, v->compute_state->last)
                                                            : nil();
}
