#include "simulation_checkpoint.h"
#include "solver.h"

_Static_assert(sizeof(double) == sizeof(uint64_t), "checkpoint Float64 layout");

uint64_t checkpoint_real(VM *v, Val value) {
    double real = asnum(v, value);
    if (value.t == V_INT && (real >= 0x1p63 || (int64_t)real != value.i))
        vmerror_kind(v, "checkpoint_format",
                     "checkpoint integer is not exactly representable as Float64");
    if (!isfinite(real))
        vmerror_kind(v, "checkpoint_format", "checkpoint number must be finite");
    uint64_t bits;
    memcpy(&bits, &real, sizeof bits);
    return bits;
}

double checkpoint_double(uint64_t word) {
    double result;
    memcpy(&result, &word, sizeof result);
    return result;
}

enum { TIME, STEP, RTOL, ATOL, MAXITER, RK45_WORDS };

static Val required(VM *v, Val record, const char *name) {
    Entry *field = tablefind(&record.o->as.map.tab, name);
    if (!field)
        vmerror_kind(v, "checkpoint_format", "RK45 checkpoint is missing %s", name);
    return field->val;
}

static void validate_rk45(VM *v, const uint64_t words[CHECKPOINT_WORDS]) {
    for (int i = 0; i < MAXITER; i++)
        if (!isfinite(checkpoint_double(words[i])) ||
            (i != TIME && checkpoint_double(words[i]) <= 0))
            vmerror_kind(v, "checkpoint_format", "invalid RK45 time, step or tolerance");
    if (!words[MAXITER] || words[MAXITER] > INT64_MAX)
        vmerror_kind(v, "checkpoint_format", "invalid RK45 step limit");
    for (int i = RK45_WORDS; i < CHECKPOINT_WORDS; i++)
        if (words[i])
            vmerror_kind(v, "checkpoint_format", "invalid reserved RK45 metadata");
}

static Val rk45_save(VM *v, Val path, Val record, Val budget) {
    if (!isobj(record, O_MAP))
        vmerror_kind(v, "checkpoint_format", "RK45 checkpoint must be a map");
    Val version = required(v, record, "version"), cap = required(v, record, "maxiter");
    if (version.t != V_INT || version.i != 1 || cap.t != V_INT || cap.i <= 0)
        vmerror_kind(v, "checkpoint_format", "invalid RK45 checkpoint version or step limit");
    uint64_t words[CHECKPOINT_WORDS] = {
        checkpoint_real(v, required(v, record, "time")),
        checkpoint_real(v, required(v, record, "step")),
        checkpoint_real(v, required(v, record, "rtol")),
        checkpoint_real(v, required(v, record, "atol")),
        (uint64_t)cap.i,
    };
    validate_rk45(v, words);
    return checkpoint_save_state(v, path, CHECKPOINT_RK45, required(v, record, "state"), words,
                                 budget);
}

static Val rk45_load(VM *v, Val path) {
    uint64_t words[CHECKPOINT_WORDS];
    Val state = checkpoint_load_state(v, path, CHECKPOINT_RK45, words);
    validate_rk45(v, words);
    Obj *record = newobj(v, O_MAP);
    sparse_result_field(v, record, "version", integer(1));
    sparse_result_field(v, record, "state", state);
    sparse_result_field(v, record, "time", number(checkpoint_double(words[TIME])));
    sparse_result_field(v, record, "step", number(checkpoint_double(words[STEP])));
    sparse_result_field(v, record, "rtol", number(checkpoint_double(words[RTOL])));
    sparse_result_field(v, record, "atol", number(checkpoint_double(words[ATOL])));
    sparse_result_field(v, record, "maxiter", integer((int64_t)words[MAXITER]));
    return objval(record);
}

static const char *const names[] = {"rk45_save", "rk45_load"};

Obj *checkpoint_module(VM *v) {
    return named_native_module(v, "_checkpoint", names, sizeof names / sizeof *names,
                               CHECKPOINT_BASE);
}

Val checkpoint_call(VM *v, int id, int argc, Val *args) {
    if (id == CHECKPOINT_BASE && (argc == 2 || argc == 3))
        return rk45_save(v, args[0], args[1], argc == 3 ? args[2] : nil());
    if (id == CHECKPOINT_BASE + 1 && argc == 1) {
        v->solver_checkpoint = v->objects;
        Val record = rk45_load(v, args[0]);
        v->solver_checkpoint = NULL;
        return record;
    }
    vmerror_kind(v, "arity", "invalid ODE checkpoint API arity");
}
