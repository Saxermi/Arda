#include "arda.h"
#include <float.h>

#define A(o) ((o)->as.array)

enum { GENERATOR, UINT32, UNIFORM, NORMAL, INTEGER, POISSON, SHUFFLE, CHOICE };
static const char *const names[] = {
    "generator", "uint32", "uniform", "normal", "integer", "poisson", "shuffle", "choice"
};
static const int arities[] = {1, 1, 3, 3, 3, 2, 2, 2};
_Static_assert(sizeof names / sizeof *names == sizeof arities / sizeof *arities,
               "random arity table");

/* PCG XSH-RR over a 64-bit linear congruential state. */
static uint32_t next32(RandomState *rng) {
    uint64_t previous = rng->state;
    rng->state = previous * UINT64_C(6364136223846793005) + rng->increment;
    uint32_t mixed = (uint32_t) ((previous ^ (previous >> 18)) >> 27);
    unsigned rotation = (unsigned) (previous >> 59);
    return (mixed >> rotation) | (mixed << ((32 - rotation) & 31));
}

static uint64_t next64(RandomState *rng) {
    uint64_t high = next32(rng);
    return (high << 32) | next32(rng);
}

static uint64_t bounded(RandomState *rng, uint64_t width) {
    uint64_t reject_below = (UINT64_C(0) - width) % width;
    uint64_t bits;
    do {
        bits = next64(rng);
    } while (bits < reject_below);
    return bits % width;
}

static int64_t signed_bits(uint64_t value) {
    if (value <= INT64_MAX)
        return (int64_t) value;
    return -1 - (int64_t) (UINT64_MAX - value);
}

static double unit(RandomState *rng) {
    uint64_t high = next32(rng) >> 5;
    uint64_t low = next32(rng) >> 6;
    return ((high << 26) | low) * 0x1p-53;
}

static double positive_unit(RandomState *rng) {
    double result;
    do {
        result = unit(rng);
    } while (result == 0);
    return result;
}

void random_seed(RandomState *rng, int64_t seed, uint64_t stream) {
    *rng = (RandomState){.increment = (stream << 1) | 1};
    next32(rng);
    rng->state += (uint64_t) seed;
    next32(rng);
}

static Val generator(VM *v, int argc, Val *args) {
    if (args[0].t != V_INT)
        vmerror(v, "random seed must be an integer");
    if (argc == 2 && (args[1].t != V_INT || args[1].i < 0))
        vmerror(v, "random stream must be a nonnegative integer");
    uint64_t stream = argc == 2 ? (uint64_t) args[1].i : 0;
    Obj *out = newobj(v, O_RNG);
    random_seed(&out->as.rng, args[0].i, stream);
    return objval(out);
}

static int sample_count(VM *v, Val value) {
    if (value.t != V_INT || value.i < 0 || value.i > ARDA_MAX_ELEMENTS)
        vmerror(v, "sample count must be a nonnegative integer within the array limit");
    return (int) value.i;
}

static double finite_parameter(VM *v, Val value) {
    double number = asnum(v, value);
    if (!isfinite(number))
        vmerror(v, "random parameters must be finite");
    return number;
}

static double uniform(RandomState *rng, double low, double high) {
    long double u = unit(rng);
    double result = (double) ((1 - u) * low + u * high);
    return result < high ? result : nextafter(high, low);
}

long double random_normal_unit(RandomState *rng) {
    long double radius = sqrtl(-2 * logl(positive_unit(rng)));
    long double angle = 6.283185307179586476925286766559L * unit(rng);
    return radius * cosl(angle);
}

static double normal(VM *v, RandomState *rng, double mean, double deviation) {
    long double result = mean + deviation * random_normal_unit(rng);
    if (!isfinite(result) || fabsl(result) > DBL_MAX)
        vmerror(v, "normal sample is outside the Float64 range");
    return (double) result;
}

typedef struct {
    long double mean, a, b, accept_scale, squeeze, zero_mass, log_mean;
} PoissonParameters;

static PoissonParameters poisson_parameters(double mean) {
    PoissonParameters p = {.mean = mean};
    if (mean < 10) {
        p.zero_mass = expl(-mean);
    } else {
        p.b = 0.931L + 2.53L * sqrtl(mean);
        p.a = -0.059L + 0.02483L * p.b;
        p.accept_scale = 1.1239L + 1.1328L / (p.b - 3.4L);
        p.squeeze = 0.9277L - 3.6224L / (p.b - 2);
        p.log_mean = logl(mean);
    }
    return p;
}

static long double poisson_log_mass(PoissonParameters p, long double k) {
    if (k < 16)
        return k * p.log_mean - p.mean - lgammal(k + 1);
    /* Rearrange the Stirling expansion so the leading log terms cancel
     * analytically, which keeps the test useful at large means. */
    long double difference = p.mean - k, inv = 1 / k, square = inv * inv;
    long double correction = inv * (1.0L / 12 - square * (1.0L / 360 - square *
                             (1.0L / 1260 - square * (1.0L / 1680 - square / 1188))));
    return (k + 0.5L) * log1pl(difference / k) - difference - 0.5L * p.log_mean -
           0.918938533204672741780329736406L - correction;
}

static int64_t poisson(RandomState *rng, PoissonParameters p) {
    if (p.mean == 0)
        return 0;
    if (p.mean < 10) {
        long double product = 1;
        int64_t k = 0;
        while ((product *= positive_unit(rng)) > p.zero_mass)
            k++;
        return k;
    }
    /* Hörmann's transformed rejection in the large-mean regime. */
    for (;;) {
        long double u = positive_unit(rng) - 0.5L;
        long double v = positive_unit(rng);
        long double distance = 0.5L - fabsl(u);
        long double k = floorl((2 * p.a / distance + p.b) * u + p.mean + 0.43L);
        if (k < 0 || k >= 0x1p63L)
            continue;
        if (distance >= 0.07L && v <= p.squeeze)
            return (int64_t) k;
        long double envelope = p.a / (distance * distance) + p.b;
        if (logl(v * p.accept_scale / envelope) <= poisson_log_mass(p, k))
            return (int64_t) k;
    }
}

static Val draw_samples(VM *v, RandomState *rng, int op, int argc, Val *args) {
    int batch = argc > arities[op];
    int count = batch ? sample_count(v, args[argc - 1]) : 1;
    double first = 0, second = 0;
    uint64_t lower = 0, width = 0;
    PoissonParameters parameters = {0};
    if (op == INTEGER) {
        if (args[1].t != V_INT || args[2].t != V_INT || args[1].i >= args[2].i)
            vmerror(v, "random integer requires integer bounds with low < high");
        lower = (uint64_t) args[1].i;
        width = (uint64_t) args[2].i - lower;
    } else {
        first = finite_parameter(v, args[1]);
        if (op == POISSON) {
            if (first < 0 || first > 1e8)
                vmerror(v, "Poisson mean must be in [0, 1e8]");
            parameters = poisson_parameters(first);
        } else {
            second = finite_parameter(v, args[2]);
            if (op == UNIFORM && first >= second)
                vmerror(v, "uniform requires low < high");
            if (op == NORMAL && second < 0)
                vmerror(v, "normal deviation must be nonnegative");
        }
    }
    int discrete = op == INTEGER || op == POISSON;
    Obj *out = batch ? (discrete ? list(v) : array_new(v, 1, count, 1)) : NULL;
    for (int i = 0; i < count; i++) {
        Val sample;
        switch (op) {
        case UNIFORM: sample = number(uniform(rng, first, second)); break;
        case NORMAL: sample = number(normal(v, rng, first, second)); break;
        case INTEGER: sample = integer(signed_bits(lower + bounded(rng, width))); break;
        default: sample = integer(poisson(rng, parameters)); break;
        }
        if (!batch)
            return sample;
        if (discrete)
            append(v, out, sample);
        else
            A(out).data[i] = sample.n;
    }
    return objval(out);
}

static int sequence_length(VM *v, Val input) {
    if (isobj(input, O_LIST))
        return input.o->as.list.n;
    if (isobj(input, O_ARRAY) && A(input.o).rank == 1)
        return A(input.o).rows;
    vmerror(v, "random choice and shuffle expect a list or a vector");
}

static Val sequence_value(Val input, int index) {
    Obj *object = input.o;
    if (object->type == O_LIST)
        return object->as.list.a[index];
    return number(A(object).data[(size_t) index * A(object).stride[0]]);
}

static Val choice(VM *v, RandomState *rng, Val input, int argc, Val *args) {
    int size = sequence_length(v, input);
    if (!size)
        vmerror(v, "random choice requires a nonempty sequence");
    if (argc == 2)
        return sequence_value(input, (int) bounded(rng, size));
    int count = sample_count(v, args[2]);
    int from_list = input.o->type == O_LIST;
    Obj *out = from_list ? list(v) : array_new(v, 1, count, 1);
    for (int i = 0; i < count; i++) {
        Val selected = sequence_value(input, (int) bounded(rng, size));
        if (from_list)
            append(v, out, selected);
        else
            A(out).data[i] = selected.n;
    }
    return objval(out);
}

static Val shuffle(VM *v, RandomState *rng, Val input) {
    int size = sequence_length(v, input);
    if (input.o->type == O_LIST) {
        Obj *out = list(v);
        for (int i = 0; i < size; i++)
            append(v, out, input.o->as.list.a[i]);
        for (int i = size - 1; i > 0; i--) {
            int j = (int) bounded(rng, (uint64_t) i + 1);
            Val saved = out->as.list.a[i];
            out->as.list.a[i] = out->as.list.a[j];
            out->as.list.a[j] = saved;
        }
        return objval(out);
    }
    Obj *out = tensor_copy(v, input.o);
    double *data = A(out).data;
    for (int i = size - 1; i > 0; i--) {
        int j = (int) bounded(rng, (uint64_t) i + 1);
        double saved = data[i];
        data[i] = data[j];
        data[j] = saved;
    }
    return objval(out);
}

Val random_call(VM *v, int id, int argc, Val *args) {
    int op = id - RANDOM_BASE;
    if (op < 0 || (size_t) op >= sizeof names / sizeof *names)
        vmerror(v, "unknown random operation");
    if (argc != arities[op] && !((op != UINT32 && op != SHUFFLE) && argc == arities[op] + 1))
        vmerror(v, "invalid random.%s arity", names[op]);
    if (op == GENERATOR)
        return generator(v, argc, args);
    if (!isobj(args[0], O_RNG))
        vmerror(v, "expected a random generator");

    /* Publish the new state only after a successful call. */
    RandomState state = args[0].o->as.rng;
    Val result;
    switch (op) {
    case UINT32: result = integer(next32(&state)); break;
    case SHUFFLE: result = shuffle(v, &state, args[1]); break;
    case CHOICE: result = choice(v, &state, args[1], argc, args); break;
    default: result = draw_samples(v, &state, op, argc, args); break;
    }
    args[0].o->as.rng = state;
    return result;
}

Obj *random_module(VM *v) {
    return named_native_module(v, "random", names, sizeof names / sizeof *names, RANDOM_BASE);
}
