#include "compute_plan.h"
#include "compute_topology.h"
#include "cpu_kernels.h"
#include <float.h>
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
    int measured_threads;
} Plan;

typedef struct {
    int kernel, block, threads;
    size_t workspace;
    bool valid, empty;
} Run;

typedef struct {
    uint64_t affinity[COMPUTE_AFFINITY_WORDS];
    unsigned eligible, features;
    int teams[3], native_kernel;
} TuneResources;

typedef struct {
    ProblemShape shape;
    TuneResources resources;
    int candidate;
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

static bool fits(size_t available, size_t output, size_t workspace) {
    return output <= available && workspace <= available - output;
}

static void checked_workspace(VM *v, size_t output, size_t workspace) {
    if (!fits(tensor_memory_available(v), output, workspace))
        vmerror_kind(v, "memory_budget", "compute output and workspace exceed the tensor budget");
}

static double *reserve(VM *v, size_t bytes, Obj *first, Obj *second) {
    if (!bytes)
        return NULL;
    unsigned char *memory = tensor_workspace_acquire(v, bytes);
    if (!memory) {
        tensor_buffer_discard(v, first->as.tensor.buffer);
        if (second)
            tensor_buffer_discard(v, second->as.tensor.buffer);
        vmerror_kind(v, "memory_budget", "unable to reserve compute workspace");
    }
    return (double *)memory;
}

static int plan_team(VM *v, const Plan *p) {
    const int *d = p->shape.dimensions;
    int threads = cpu_thread_cap(v, p->threads);
    return p->kernel == KERNEL_TILED
               ? compute_tiled_team(d[AXIS_ROWS], p->shape.rhs_rank, p->flops / 2, p->block,
                                    threads, p->workspace != 0)
               : cpu_matmul_team(threads, d[AXIS_ROWS], d[AXIS_DEPTH], d[AXIS_COLUMNS]);
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
            threads = compute_tiled_into(p, plan->block, cpu_thread_cap(v, plan->threads), out,
                                         plan->workspace ? workspace : NULL);
        else
            threads = cpu_matmul_run(v, plan->kernel, cpu_thread_cap(v, plan->threads),
                                     d[AXIS_ROWS], d[AXIS_DEPTH], d[AXIS_COLUMNS],
                                     (const double *)typed_element(&p->a, p->a.offset),
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

static double timed_run(VM *v, const MatmulProblem *p, const Plan *plan, double *out, double *work,
                        Run *report) {
    double start = wall_seconds();
    memset(out, 0, plan->output);
    *report = run_into(p, v, plan, out, work);
    return wall_seconds() - start;
}

static double trial(VM *v, const MatmulProblem *p, Plan *plan, double *out, double *work,
                    bool *stable) {
    enum { SAMPLES = 3 };
    double times[SAMPLES];
    memset(out, 0, plan->output);
    plan->measured_threads = run_into(p, v, plan, out, work).threads;
    for (int i = 0; i < SAMPLES; i++) {
        Run report;
        times[i] = timed_run(v, p, plan, out, work, &report);
        if (report.threads != plan->measured_threads) {
            *stable = false;
            plan->measured_threads = 0;
        }
        if (!isfinite(times[i]))
            return NAN;
    }
    qsort(times, SAMPLES, sizeof *times, compare_time);
    return times[SAMPLES / 2];
}

static void tune(VM *v, const MatmulProblem *p, Plan *plan, size_t outputs) {
    if (!plan->flops || plan->flops > UINT64_C(128000000) || plan->shape.mapped)
        vmerror_kind(v, "domain",
                     "compute tuning needs a nonempty heap problem of at most 64M terms");
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
    TuneResources resources = {.features = v->cpu_features, .native_kernel = plan->kernel};
    size_t workspace = 0, available = tensor_memory_available(v);
    bool parallel = false;
    for (int i = 0; i < count; i++) {
        if (!fits(available, outputs, candidates[i].workspace))
            continue;
        resources.eligible |= 1u << i;
        resources.teams[i] = plan_team(v, &candidates[i]);
        parallel |= resources.teams[i] > 1;
        if (candidates[i].workspace > workspace)
            workspace = candidates[i].workspace;
    }
    if (!resources.eligible)
        vmerror_kind(v, "memory_budget", "no compute tuning candidate fits the tensor budget");
    bool cacheable =
        (!parallel || !cpu_dynamic_teams()) && compute_tuning_placement(resources.affinity);
    struct ComputeState *state = compute_state(v);
    for (unsigned i = 0; cacheable && i < TUNE_CACHE_SIZE; i++) {
        const TuneEntry *entry = &state->cache[i];
        const TuneResources *old = &entry->resources;
        if (old->eligible == resources.eligible && old->native_kernel == resources.native_kernel &&
            old->features == resources.features &&
            !memcmp(old->teams, resources.teams, sizeof old->teams) &&
            !memcmp(old->affinity, resources.affinity, sizeof old->affinity) &&
            same_shape(&entry->shape, &plan->shape)) {
            *plan = candidates[entry->candidate];
            plan->measured = entry->seconds;
            plan->measured_threads = resources.teams[entry->candidate];
            plan->cache_hit = true;
            return;
        }
    }

    Obj *out = typed_new(v, DT_F64, p->output.rank, p->output.shape);
    double *work = reserve(v, workspace, out, NULL);
    double best = INFINITY;
    int winner = -1;
    for (int i = 0; i < count; i++) {
        if (!(resources.eligible & (1u << i)))
            continue;
        double measured = trial(v, p, &candidates[i], output_data(out), work, &cacheable);
        cacheable &= candidates[i].measured_threads == resources.teams[i];
        if (measured < best) {
            best = measured;
            winner = i;
        }
    }
    if (work)
        tensor_workspace_release(v, (unsigned char *)work);
    tensor_buffer_discard(v, out->as.tensor.buffer);
    if (winner < 0)
        vmerror_kind(v, "backend", "monotonic clock unavailable for compute tuning");
    *plan = candidates[winner];
    plan->measured = best;
    if (cacheable)
        state->cache[state->next++ % TUNE_CACHE_SIZE] =
            (TuneEntry){plan->shape, resources, winner, best};
}

static Plan make_plan(VM *v, const MatmulProblem *p, Val options, unsigned outputs) {
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
        tune(v, p, &plan, outputs * plan.output);
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
    Plan plan = make_plan(v, &problem, options, 1);
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
    int team = plan_team(v, p);
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
    field(v, map, "measured_threads", thread_field(p->measured_threads));
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
    Obj *out = typed_new(v, DT_F64, p.output.rank, p.output.shape);
    double *workspace = reserve(v, plan->workspace, out, NULL);
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

enum { COMPARE_MAX_REPEATS = 9 };

static int sample_count(VM *v, Val value, int fallback, int minimum, int maximum) {
    if (value.t == V_NIL)
        return fallback;
    if (value.t != V_INT || value.i < minimum || value.i > maximum)
        vmerror_kind(v, "domain", "compute comparison sample count is outside its bounds");
    return (int)value.i;
}

static double tolerance(VM *v, Val value, double fallback, const char *name) {
    return value.t == V_NIL ? fallback : rate(v, value, name, true);
}

typedef struct {
    Plan plan;
    Obj *result;
    double samples[COMPARE_MAX_REPEATS];
    Run last;
} Comparison;

static Val comparison_report(VM *v, Comparison *c, int repeats) {
    qsort(c->samples, (size_t)repeats, sizeof *c->samples, compare_time);
    double middle = c->samples[repeats / 2];
    if (!(repeats % 2))
        middle = (middle + c->samples[repeats / 2 - 1]) / 2;
    Obj *map = run_report(v, c->last).o;
    field(v, map, "median_seconds", number(middle));
    return objval(map);
}

static bool check_outputs(const double *a, const double *b, size_t count, double atol, double rtol,
                          long double *max_error) {
    bool equivalent = true;
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(a[i]) || !isfinite(b[i])) {
            *max_error = INFINITY;
            equivalent = false;
            continue;
        }
        long double x = a[i], y = b[i], error = fabsl(x - y);
        long double scale = fmaxl(fabsl(x), fabsl(y));
        *max_error = fmaxl(*max_error, error);
        if (error <= atol)
            continue;
        long double relative = isfinite(error) ? error / scale : fabsl(x / scale - y / scale);
        if (relative > (long double)rtol + (long double)atol / scale)
            equivalent = false;
    }
    return equivalent;
}

Val compute_compare(VM *v, Val a, Val b, Val options) {
    Table *table = options_table(v, options);
    int seen = 0;
    Val first = option(table, "first", &seen), second = option(table, "second", &seen);
    int warmups = sample_count(v, option(table, "warmup", &seen), 1, 0, 5);
    int repeats = sample_count(v, option(table, "repeats", &seen), 5, 1, COMPARE_MAX_REPEATS);
    double atol = tolerance(v, option(table, "atol", &seen), 1e-12, "atol");
    double rtol = tolerance(v, option(table, "rtol", &seen), 1e-10, "rtol");
    check_options(v, table, seen);
    MatmulProblem p = compute_matmul_problem(v, a, b);
    Comparison sides[2] = {0};
    sides[0].plan = make_plan(v, &p, first, 2);
    sides[1].plan = make_plan(v, &p, second, 2);
    size_t workspace = sides[0].plan.workspace > sides[1].plan.workspace ? sides[0].plan.workspace
                                                                         : sides[1].plan.workspace;
    size_t output = sides[0].plan.output;
    checked_workspace(v, 2 * output, workspace);
    for (int i = 0; i < 2; i++)
        sides[i].result = typed_new(v, DT_F64, p.output.rank, p.output.shape);
    double *work = reserve(v, workspace, sides[0].result, sides[1].result);
    bool clock_ok = true;
    for (int trial = 0; trial < warmups + repeats; trial++) {
        for (int order = 0; order < 2; order++) {
            Comparison *side = &sides[(trial + order) % 2];
            double *out = output_data(side->result);
            if (trial < warmups) {
                memset(out, 0, output);
                run_into(&p, v, &side->plan, out, work);
            } else {
                double seconds = timed_run(v, &p, &side->plan, out, work, &side->last);
                side->samples[trial - warmups] = seconds;
                clock_ok &= isfinite(seconds);
            }
        }
    }
    long double max_error = 0;
    bool equivalent = check_outputs(output_data(sides[0].result), output_data(sides[1].result),
                                    p.output.n, atol, rtol, &max_error);
    for (int i = 0; i < 2; i++)
        tensor_buffer_discard(v, sides[i].result->as.tensor.buffer);
    if (work)
        tensor_workspace_release(v, (unsigned char *)work);
    if (!clock_ok)
        vmerror_kind(v, "backend", "monotonic clock unavailable for compute comparison");
    Obj *map = newobj(v, O_MAP);
    field(v, map, "first", comparison_report(v, &sides[0], repeats));
    field(v, map, "second", comparison_report(v, &sides[1], repeats));
    field(v, map, "equivalent", boolean(equivalent));
    field(v, map, "max_abs_error", max_error <= DBL_MAX ? number((double)max_error) : nil());
    field(v, map, "warmup", integer(warmups));
    field(v, map, "repeats", integer(repeats));
    field(v, map, "atol", number(atol));
    field(v, map, "rtol", number(rtol));
    return objval(map);
}
