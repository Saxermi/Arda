#include "typed_tensor.h"

#include <float.h>

#define Q(o) ((o)->as.quantity)
#define A(o) ((o)->as.array)
enum { LENGTH, MASS, TIME, CURRENT, TEMPERATURE, AMOUNT, LUMINOUS, DIMENSIONS };
enum { MAX_UNIT_EXPONENT = 32 };

typedef struct {
    const char *name;
    Unit unit;
} KnownUnit;

static const Unit dimensionless = {.scale = 1};
static const KnownUnit known[] = {
    {"1", {.scale = 1}},
    {"%", {.scale = .01}},
    {"rad", {.scale = 1}},
    {"m", {.scale = 1, .dimension = {[LENGTH] = 1}}},
    {"mm", {.scale = .001, .dimension = {[LENGTH] = 1}}},
    {"cm", {.scale = .01, .dimension = {[LENGTH] = 1}}},
    {"km", {.scale = 1000, .dimension = {[LENGTH] = 1}}},
    {"g", {.scale = .001, .dimension = {[MASS] = 1}}},
    {"kg", {.scale = 1, .dimension = {[MASS] = 1}}},
    {"mg", {.scale = 1e-6, .dimension = {[MASS] = 1}}},
    {"s", {.scale = 1, .dimension = {[TIME] = 1}}},
    {"ms", {.scale = .001, .dimension = {[TIME] = 1}}},
    {"us", {.scale = 1e-6, .dimension = {[TIME] = 1}}},
    {"ns", {.scale = 1e-9, .dimension = {[TIME] = 1}}},
    {"min", {.scale = 60, .dimension = {[TIME] = 1}}},
    {"h", {.scale = 3600, .dimension = {[TIME] = 1}}},
    {"A", {.scale = 1, .dimension = {[CURRENT] = 1}}},
    {"K", {.scale = 1, .dimension = {[TEMPERATURE] = 1}, .absolute = true}},
    {"degC", {.scale = 1, .offset = 273.15, .dimension = {[TEMPERATURE] = 1}, .absolute = true}},
    {"degF",
     {.scale = 5.0 / 9, .offset = 459.67, .dimension = {[TEMPERATURE] = 1}, .absolute = true}},
    {"deltaK", {.scale = 1, .dimension = {[TEMPERATURE] = 1}}},
    {"deltaC", {.scale = 1, .dimension = {[TEMPERATURE] = 1}}},
    {"deltaF", {.scale = 5.0 / 9, .dimension = {[TEMPERATURE] = 1}}},
    {"mol", {.scale = 1, .dimension = {[AMOUNT] = 1}}},
    {"cd", {.scale = 1, .dimension = {[LUMINOUS] = 1}}},
    {"Hz", {.scale = 1, .dimension = {[TIME] = -1}}},
    {"N", {.scale = 1, .dimension = {[LENGTH] = 1, [MASS] = 1, [TIME] = -2}}},
    {"Pa", {.scale = 1, .dimension = {[LENGTH] = -1, [MASS] = 1, [TIME] = -2}}},
    {"J", {.scale = 1, .dimension = {[LENGTH] = 2, [MASS] = 1, [TIME] = -2}}},
    {"W", {.scale = 1, .dimension = {[LENGTH] = 2, [MASS] = 1, [TIME] = -3}}},
    {"V", {.scale = 1, .dimension = {[LENGTH] = 2, [MASS] = 1, [TIME] = -3, [CURRENT] = -1}}},
    {"L", {.scale = .001, .dimension = {[LENGTH] = 3}}},
};

static const Unit *lookup(const char *text, size_t len) {
    for (size_t i = 0; i < sizeof known / sizeof *known; i++)
        if (strlen(known[i].name) == len && !memcmp(text, known[i].name, len))
            return &known[i].unit;
    return NULL;
}

static void add_dimensions(VM *v, Unit *a, const Unit *b, int power) {
    for (int i = 0; i < DIMENSIONS; i++) {
        int exponent = a->dimension[i] + power * b->dimension[i];
        if (exponent < -MAX_UNIT_EXPONENT || exponent > MAX_UNIT_EXPONENT)
            vmerror(v, "unit dimension exponent exceeds the supported range");
        a->dimension[i] = (int8_t)exponent;
    }
}

static Unit parse_unit(VM *v, const char *text) {
    const Unit *named = lookup(text, strlen(text));
    if (named)
        return *named;
    Unit result = dimensionless;
    const char *p = text;
    int sign = 1;
    for (;;) {
        const char *start = p;
        while ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') || (*p >= '0' && *p <= '9') ||
               *p == '%')
            p++;
        const Unit *term = lookup(start, (size_t)(p - start));
        if (!term)
            vmerror(v, "unknown unit symbol or invalid unit expression");
        if (term->absolute && term->offset != 0)
            vmerror(v, "offset temperature units cannot appear in compound units");
        int power = 1;
        if (*p == '^') {
            p++;
            int exponent_sign = 1;
            if (*p == '-' || *p == '+')
                exponent_sign = *p++ == '-' ? -1 : 1;
            if (*p < '0' || *p > '9')
                vmerror(v, "invalid unit exponent");
            power = 0;
            do {
                power = 10 * power + *p++ - '0';
                if (power > MAX_UNIT_EXPONENT)
                    vmerror(v, "unit exponent exceeds the supported range");
            } while (*p >= '0' && *p <= '9');
            power *= exponent_sign;
        }
        add_dimensions(v, &result, term, sign * power);
        result.scale *= pow(term->scale, sign * power);
        if (result.scale <= 0 || !isfinite(result.scale))
            vmerror(v, "unit scale is outside the Float64 range");
        if (!*p)
            return result;
        if (*p != '*' && *p != '/')
            vmerror(v, "invalid unit expression");
        sign = *p++ == '/' ? -1 : 1;
    }
}

static bool same_dimensions(const Unit *a, const Unit *b) {
    return !memcmp(a->dimension, b->dimension, sizeof a->dimension);
}

static Obj *canonical_symbol(VM *v, const Unit *unit) {
    if (unit->absolute)
        return string(v, "K");
    const char *base[] = {"m", "kg", "s", "A", "deltaK", "mol", "cd"};
    char symbol[128] = {0};
    size_t length = 0;
    for (int d = 0; d < DIMENSIONS; d++) {
        int exponent = unit->dimension[d];
        if (!exponent)
            continue;
        int used =
            snprintf(symbol + length, sizeof symbol - length, "%s%s", length ? "*" : "", base[d]);
        length += (size_t)used;
        if (exponent != 1) {
            used = snprintf(symbol + length, sizeof symbol - length, "^%d", exponent);
            length += (size_t)used;
        }
    }
    return string(v, length ? symbol : "1");
}

static Quantity *quantity(VM *v, Val value) {
    if (!isobj(value, O_QUANTITY))
        vmerror(v, "expected a scientific quantity");
    return &Q(value.o);
}

static bool tensor_value(Val value) {
    return isobj(value, O_ARRAY) || isobj(value, O_TYPED_TENSOR);
}

static double finite_result(VM *v, long double value) {
    if (!isfinite(value) || fabsl(value) > DBL_MAX)
        vmerror(v, "quantity value must be finite and within the Float64 range");
    return (double)value;
}

static double convert_scalar(VM *v, long double x, const Unit *unit, bool inverse) {
    long double value = inverse ? x / unit->scale - unit->offset : (x + unit->offset) * unit->scale;
    if (!inverse && unit->absolute && value < 0)
        vmerror(v, "absolute temperature must be at least zero kelvin");
    return finite_result(v, value);
}

static Val transform(VM *v, Val value, const Unit *unit, bool inverse) {
    if (!tensor_value(value)) {
        long double scalar = value.t == V_INT ? (long double)value.i : asnum(v, value);
        return number(convert_scalar(v, scalar, unit, inverse));
    }
    TypedTensor t = typed_tensor(v, value);
    if (t.dtype != DT_F64)
        vmerror(v, "quantity tensors require Float64 storage");
    Obj *out = tensor_new(v, t.rank, t.shape);
    for (size_t i = 0; i < t.n; i++) {
        double x = dtype_load(DT_F64, typed_element(&t, typed_offset(&t, i))).re;
        A(out).data[i] = convert_scalar(v, x, unit, inverse);
    }
    return objval(out);
}

static bool finite_quantity_value(VM *v, Val value, bool absolute) {
    if (isobj(value, O_ARRAY)) {
        for (size_t i = 0; i < A(value.o).n; i++)
            if (!isfinite(A(value.o).data[i]) || (absolute && A(value.o).data[i] < 0))
                return false;
        return true;
    }
    double scalar = asnum(v, value);
    return isfinite(scalar) && (!absolute || scalar >= 0);
}

static Val publish(VM *v, Val value, Unit unit, Obj *symbol) {
    if (!finite_quantity_value(v, value, unit.absolute))
        vmerror(v, unit.absolute ? "invalid absolute temperature result"
                                 : "nonfinite quantity result");
    Obj *out = newobj(v, O_QUANTITY);
    Q(out) = (Quantity){.value = value, .symbol = symbol, .unit = unit};
    barrier(v, out, value);
    barrier(v, out, objval(symbol));
    return objval(out);
}

static Unit product_unit(VM *v, const Unit *a, const Unit *b, Op op) {
    if (a->absolute || b->absolute)
        vmerror(v, "absolute temperature cannot be multiplied or divided");
    Unit result = dimensionless;
    add_dimensions(v, &result, a, 1);
    add_dimensions(v, &result, b, op == OP_DIV ? -1 : 1);
    return result;
}

static void additive_unit(VM *v, const Quantity *a, const Quantity *b, Op op, Unit *unit,
                          Obj **symbol) {
    if (!same_dimensions(&a->unit, &b->unit))
        vmerror(v, "quantity dimensions do not match");
    if (a->unit.absolute && b->unit.absolute) {
        if (op != OP_SUB)
            vmerror(v, "two absolute temperatures cannot be added");
        *unit = dimensionless;
        unit->dimension[TEMPERATURE] = 1;
        *symbol = canonical_symbol(v, unit);
    } else if (b->unit.absolute) {
        if (op == OP_SUB)
            vmerror(v, "cannot subtract an absolute temperature from an interval");
        *unit = b->unit;
        *symbol = b->symbol;
    } else {
        *unit = a->unit;
        *symbol = a->symbol;
    }
}

static Quantity operand(VM *v, Val value) {
    if (isobj(value, O_QUANTITY))
        return Q(value.o);
    return (Quantity){.value = value, .unit = dimensionless, .symbol = string(v, "1")};
}

static Val arithmetic(VM *v, Op op, Val a, Val b) {
    if (isobj(a, O_ARRAY) || isobj(b, O_ARRAY))
        return tensor_binary(v, op, a, b);
    long double x = a.t == V_INT ? (long double)a.i : asnum(v, a);
    long double y = b.t == V_INT ? (long double)b.i : asnum(v, b);
    if (op == OP_DIV && y == 0)
        vmerror(v, "quantity division by zero");
    return number(finite_result(v, op == OP_ADD   ? x + y
                                   : op == OP_SUB ? x - y
                                   : op == OP_MUL ? x * y
                                                  : x / y));
}

Val units_binary(VM *v, Op op, Val left, Val right) {
    if (op != OP_ADD && op != OP_SUB && op != OP_MUL && op != OP_DIV)
        vmerror(v, "unsupported quantity operator");
    Quantity a = operand(v, left), b = operand(v, right);
    Unit unit;
    Obj *symbol;
    if (op == OP_ADD || op == OP_SUB)
        additive_unit(v, &a, &b, op, &unit, &symbol);
    else {
        unit = product_unit(v, &a.unit, &b.unit, op);
        symbol = canonical_symbol(v, &unit);
    }
    return publish(v, arithmetic(v, op, a.value, b.value), unit, symbol);
}

static Val quantity_sum(VM *v, Quantity *q, int argc, Val *args) {
    if (q->unit.absolute)
        vmerror(v, "absolute temperatures cannot be summed");
    Val result = q->value;
    if (argc == 2) {
        if (!isobj(q->value, O_ARRAY))
            vmerror(v, "quantity axis sum requires a tensor");
        TypedTensor t = typed_tensor(v, q->value);
        result = typed_sum_axis(v, &t, indexnum(v, args[1], t.rank), true);
    } else if (isobj(q->value, O_ARRAY)) {
        Obj *x = q->value.o;
        long double sum = 0;
        for (size_t i = 0; i < A(x).n; i++)
            sum += A(x).data[i];
        result = number(finite_result(v, sum));
    }
    return publish(v, result, q->unit, q->symbol);
}

enum { QUANTITY, VALUE, CONVERT, UNIT, DIMENSION, MATMUL, SUM };
static const char *const names[] = {"quantity",  "value",  "convert", "unit",
                                    "dimension", "matmul", "sum"};
static const int arities[] = {2, 1, 2, 1, 1, 2, 1};
_Static_assert(sizeof names / sizeof *names == sizeof arities / sizeof *arities, "units arities");

Obj *units_module(VM *v) {
    return named_native_module(v, "units", names, sizeof names / sizeof *names, UNITS_BASE);
}

Val units_call(VM *v, int id, int argc, Val *args) {
    int op = id - UNITS_BASE;
    if (op < 0 || (size_t)op >= sizeof names / sizeof *names ||
        (argc != arities[op] && !((op == SUM || op == VALUE) && argc == 2)))
        vmerror(v, "invalid units API arity");
    if (op == QUANTITY) {
        Obj *symbol = isobj(args[1], O_STRING) ? args[1].o : NULL;
        Unit unit = parse_unit(v, asstr(v, args[1]));
        return publish(v, transform(v, args[0], &unit, false), unit, symbol);
    }
    Quantity *q = quantity(v, args[0]);
    switch (op) {
    case UNIT:
        return objval(q->symbol);
    case DIMENSION: {
        Obj *out = list(v);
        for (int d = 0; d < DIMENSIONS; d++)
            append(v, out, integer(q->unit.dimension[d]));
        return objval(out);
    }
    case VALUE:
    case CONVERT: {
        Unit unit = q->unit;
        if (argc == 2) {
            unit = parse_unit(v, asstr(v, args[1]));
            if (!same_dimensions(&unit, &q->unit) || unit.absolute != q->unit.absolute)
                vmerror(v, "quantity conversion requires matching dimensions and temperature kind");
        }
        return op == VALUE ? transform(v, q->value, &unit, true)
                           : publish(v, q->value, unit, args[1].o);
    }
    case MATMUL: {
        Quantity *b = quantity(v, args[1]);
        Unit unit = product_unit(v, &q->unit, &b->unit, OP_MUL);
        if (!isobj(q->value, O_ARRAY) || !isobj(b->value, O_ARRAY))
            vmerror(v, "quantity matmul requires two Float64 tensors");
        Obj *result = dense_matmul(v, q->value.o, b->value.o);
        return publish(v, objval(result), unit, canonical_symbol(v, &unit));
    }
    case SUM:
        return quantity_sum(v, q, argc, args);
    }
    vmerror(v, "unknown units operation");
}
