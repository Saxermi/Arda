#include "typed_tensor.h"

#include <float.h>

typedef struct {
    TypedTensor values;
    bool factor;
} Covariance;

static double entry(const TypedTensor *t, size_t i) {
    double x;
    memcpy(&x, typed_element(t, typed_offset(t, i)), sizeof x);
    return x;
}

static double finite(VM *v, long double x) {
    if (!isfinite(x) || fabsl(x) > DBL_MAX) {
        gcfull(v);
        vmerror_kind(v, "nonfinite", "uncertainty result is outside finite Float64 range");
    }
    return (double)x;
}

static TypedTensor real_tensor(VM *v, Val value) {
    TypedTensor t = typed_tensor(v, value);
    if (t.dtype != DT_F64)
        vmerror_kind(v, "dtype", "uncertainty requires Float64 tensors");
    return t;
}

static Covariance covariance(VM *v, Val value, const TypedTensor *point) {
    if (point->rank != 1)
        vmerror_kind(v, "shape", "covariance requires a rank-one input point");
    if (!isobj(value, O_MAP) || value.o->as.map.tab.n != 1)
        vmerror_kind(v, "shape", "covariance must have exactly one field, stddev or factor");
    Table *fields = &value.o->as.map.tab;
    Entry *f = tablefind(fields, "factor"), *s = tablefind(fields, "stddev");
    if (!f && !s)
        vmerror_kind(v, "shape", "covariance must have a stddev vector or a factor matrix");
    TypedTensor t = real_tensor(v, (f ? f : s)->val);
    if (t.rank != (f ? 2 : 1) || t.shape[0] != point->shape[0])
        vmerror_kind(v, "shape", "covariance must match the input vector length");
    for (size_t i = 0; i < t.n; i++) {
        double x = finite(v, entry(&t, i));
        if (s && x < 0)
            vmerror_kind(v, "domain", "standard deviations must be nonnegative");
    }
    return (Covariance){.values = t, .factor = f != NULL};
}

static long double project_column(const TypedTensor *factor, const TypedTensor *direction,
                                  size_t column) {
    size_t width = (size_t)factor->shape[1];
    long double value = 0;
    for (size_t i = 0; i < direction->n; i++)
        value += (long double)entry(factor, i * width + column) * entry(direction, i);
    return value;
}

static void field(VM *v, Obj *map, const char *name, Val value) {
    tableset(v, map, &map->as.map.tab, name, value);
}

static Val spread(VM *v, Covariance c, const TypedTensor *direction) {
    long double stddev = 0;
    if (c.factor && direction->n) {
        for (int j = 0; j < c.values.shape[1]; j++)
            stddev = hypotl(stddev, project_column(&c.values, direction, (size_t)j));
    } else {
        for (size_t i = 0; i < direction->n; i++)
            stddev = hypotl(stddev, (long double)entry(&c.values, i) * entry(direction, i));
    }
    double sigma = finite(v, stddev), variance = finite(v, stddev * stddev);
    Obj *out = newobj(v, O_MAP);
    field(v, out, "stddev", number(sigma));
    field(v, out, "variance", number(variance));
    return objval(out);
}

static void require_workspace(VM *v, size_t output, size_t scratch) {
    size_t available = tensor_memory_available(v);
    if (scratch > available || output > available - scratch)
        vmerror_kind(v, "memory_budget", "uncertainty workspace exceeds the tensor memory budget");
}

static Val transform(VM *v, Covariance c, const TypedTensor *point, RandomState *rng) {
    size_t width = c.factor && point->n ? (size_t)c.values.shape[1] : 0;
    size_t scratch = width * sizeof(long double);
    require_workspace(v, point->n * sizeof(double), scratch);
    Obj *out = typed_new(v, DT_F64, 1, point->shape);
    double *result = (double *)typed_element(&out->as.tensor, 0);
    if (!point->n)
        return objval(out);
    if (!c.factor) {
        for (size_t i = 0; i < point->n; i++) {
            long double x = entry(point, i), s = entry(&c.values, i);
            result[i] = finite(v, rng ? x + s * random_normal_unit(rng) : (s * x) * s);
        }
        return objval(out);
    }

    Obj *workspace = buffer_new(v, scratch);
    long double *coefficients = (long double *)workspace->as.buffer.data;
    for (size_t j = 0; j < width; j++)
        coefficients[j] = rng ? random_normal_unit(rng) : project_column(&c.values, point, j);
    for (size_t i = 0; i < point->n; i++) {
        long double value = rng ? entry(point, i) : 0;
        for (size_t j = 0; j < width; j++)
            value += entry(&c.values, i * width + j) * coefficients[j];
        result[i] = finite(v, value);
    }
    tensor_buffer_discard(v, workspace);
    return objval(out);
}

static Val relative(VM *v, Val output, Val input, Val derivative) {
    double y = asnum(v, output);
    if (y == 0)
        return nil();
    if (input.t == V_NUM || input.t == V_INT) {
        double x = asnum(v, input);
        return x == 0 ? nil() : number(finite(v, (long double)x * asnum(v, derivative) / y));
    }
    TypedTensor x = real_tensor(v, input), gradient = real_tensor(v, derivative);
    if (x.rank != gradient.rank ||
        memcmp(x.shape, gradient.shape, (size_t)x.rank * sizeof *x.shape))
        vmerror_kind(v, "shape",
                     "relative sensitivity requires matching input and gradient shapes");
    require_workspace(v, x.n * sizeof(double), x.n);
    Obj *values = typed_new(v, DT_F64, x.rank, x.shape);
    Obj *defined = typed_new(v, DT_BOOL, x.rank, x.shape);
    unsigned char *mask = defined->as.tensor.buffer->as.buffer.data;
    double *result = (double *)typed_element(&values->as.tensor, 0);
    for (size_t i = 0; i < x.n; i++) {
        double xi = entry(&x, i);
        mask[i] = xi != 0;
        result[i] = mask[i] ? finite(v, (long double)xi * entry(&gradient, i) / y) : 0;
    }
    Obj *report = newobj(v, O_MAP);
    field(v, report, "values", objval(values));
    field(v, report, "defined", objval(defined));
    return objval(report);
}

enum { APPLY, SPREAD, MULTIPLY, DRAW, RELATIVE };
static const char *const names[] = {"apply", "spread", "multiply", "draw", "relative"};
_Static_assert(UNCERTAINTY_BASE + APPLY == UNCERTAINTY_APPLY_ID, "private callback ID");

Val uncertainty_call(VM *v, int id, int argc, Val *args) {
    int op = id - UNCERTAINTY_BASE;
    if (op < SPREAD || op > RELATIVE || argc != (op >= DRAW ? 3 : 2))
        vmerror(v, "invalid uncertainty call");
    if (op == RELATIVE)
        return relative(v, args[0], args[1], args[2]);

    TypedTensor point = real_tensor(v, args[1]);
    Covariance c = covariance(v, args[0], &point);
    for (size_t i = 0; i < point.n; i++)
        finite(v, entry(&point, i));
    if (op == SPREAD)
        return spread(v, c, &point);
    if (op == DRAW && !isobj(args[2], O_RNG))
        vmerror_kind(v, "type", "Monte Carlo requires a random generator");
    return transform(v, c, &point, op == DRAW ? &args[2].o->as.rng : NULL);
}

Obj *uncertainty_module(VM *v) {
    return named_native_module(v, "_uncertainty", names, sizeof names / sizeof *names,
                               UNCERTAINTY_BASE);
}
