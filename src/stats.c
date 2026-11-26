#include "arda.h"
#include "moments.h"
#include <float.h>

#define A(o) ((o)->as.array)

enum {
    SUM,
    CUMSUM,
    MEDIAN,
    QUANTILE,
    COVARIANCE,
    CORRELATION,
    REGRESSION,
    HISTOGRAM,
    ARGMIN,
    ARGMAX,
    CLIP,
    ONLINE,
    UPDATE,
    SUMMARY,
    VARIANCE,
    STDDEV,
    NORMAL_PDF,
    NORMAL_CDF,
    NORMAL_SF,
    EXPONENTIAL_PDF,
    EXPONENTIAL_CDF,
    EXPONENTIAL_SF
};
static const char *const names[] = {"sum",
                                    "cumsum",
                                    "median",
                                    "quantile",
                                    "covariance",
                                    "correlation",
                                    "regression",
                                    "histogram",
                                    "argmin",
                                    "argmax",
                                    "clip",
                                    "online",
                                    "update",
                                    "summary",
                                    "variance",
                                    "stddev",
                                    "normal_pdf",
                                    "normal_cdf",
                                    "normal_sf",
                                    "exponential_pdf",
                                    "exponential_cdf",
                                    "exponential_sf"};
static const int arities[] = {1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 3, 0, 2, 1, 1, 1, 3, 3, 3, 2, 2, 2};
_Static_assert(sizeof names / sizeof *names == sizeof arities / sizeof *arities,
               "stats arity table");

static double finite_number(VM *v, Val input) {
    double value = asnum(v, input);
    if (!isfinite(value))
        vmerror(v, "statistics require finite inputs");
    return value;
}

static double representable(VM *v, long double value) {
    if (!isfinite(value) || fabsl(value) > DBL_MAX)
        vmerror(v, "statistic is outside the Float64 range");
    return (double)value;
}

static double element(const Obj *x, int i) {
    return A(x).data[(size_t)i * A(x).stride[0]];
}

static Obj *vector(VM *v, Val value) {
    if (!isobj(value, O_ARRAY) || A(value.o).rank != 1)
        vmerror(v, "statistics expect a Float64 vector");
    Obj *x = value.o;
    for (int i = 0; i < A(x).rows; i++)
        if (!isfinite(element(x, i)))
            vmerror(v, "statistics require finite inputs");
    return x;
}

static Val compensated_sum(VM *v, Obj *x, int cumulative) {
    int n = A(x).rows;
    Obj *out = cumulative ? array_new(v, 1, n, 1) : NULL;
    long double sum = 0, correction = 0;
    for (int i = 0; i < n; i++) {
        long double value = element(x, i), next = sum + value;
        correction += fabsl(sum) >= fabsl(value) ? (sum - next) + value : (value - next) + sum;
        sum = next;
        if (cumulative)
            A(out).data[i] = representable(v, sum + correction);
    }
    return cumulative ? objval(out) : number(representable(v, sum + correction));
}

static int compare_double(const void *lhs, const void *rhs) {
    double a = *(const double *)lhs, b = *(const double *)rhs;
    return (a > b) - (a < b);
}

static Val quantile(VM *v, Obj *x, double q) {
    if (q < 0 || q > 1)
        vmerror(v, "quantile probability must be in [0, 1]");
    int n = A(x).rows;
    if (!n)
        vmerror(v, "quantile requires a nonempty vector");
    Obj *copy = tensor_copy(v, x);
    double *values = A(copy).data;
    qsort(values, (size_t)n, sizeof *values, compare_double);
    long double rank = (n - 1) * (long double)q;
    int lower = (int)rank, upper = (int)ceill(rank);
    long double fraction = rank - lower;
    return number(representable(v, (1 - fraction) * values[lower] + fraction * values[upper]));
}

static Val extreme_index(VM *v, Obj *x, int maximum) {
    int n = A(x).rows, index = 0;
    if (!n)
        vmerror(v, "argmin and argmax require a nonempty vector");
    for (int i = 1; i < n; i++) {
        double candidate = element(x, i), best = element(x, index);
        if (maximum ? candidate > best : candidate < best)
            index = i;
    }
    return integer(index);
}

static Val histogram(VM *v, Obj *x, Obj *edges) {
    int bins = A(edges).rows - 1;
    if (bins < 1)
        vmerror(v, "histogram requires at least two edges");
    for (int i = 0; i < bins; i++)
        if (element(edges, i) >= element(edges, i + 1))
            vmerror(v, "histogram edges must be strictly increasing");
    Obj *out = array_new(v, 1, bins, 1);
    double first = element(edges, 0), last = element(edges, bins);
    for (int i = 0; i < A(x).rows; i++) {
        double value = element(x, i);
        if (value < first || value > last)
            continue;
        int low = 0, high = bins;
        while (high - low > 1) {
            int middle = low + (high - low) / 2;
            if (value < element(edges, middle))
                high = middle;
            else
                low = middle;
        }
        A(out).data[low]++;
    }
    return objval(out);
}

static Val clip(VM *v, Val input, double low, double high) {
    if (!isobj(input, O_ARRAY))
        vmerror(v, "clip expects a Float64 tensor");
    if (low > high)
        vmerror(v, "clip requires low <= high");
    Obj *out = tensor_copy(v, input.o);
    for (size_t i = 0; i < A(out).n; i++) {
        double value = A(out).data[i];
        if (!isfinite(value))
            vmerror(v, "statistics require finite inputs");
        A(out).data[i] = fmin(fmax(value, low), high);
    }
    return objval(out);
}

static double vector_scale(const Obj *x) {
    double scale = 0;
    for (int i = 0; i < A(x).rows; i++)
        scale = fmax(scale, fabs(element(x, i)));
    return scale == 0 ? 1 : scale;
}

static long double centered(double value, double origin, double scale) {
    long double delta = (long double)value - origin;
    if (isfinite(delta))
        return delta / scale;
    return (long double)value / scale - (long double)origin / scale;
}

static Val dispersion(VM *v, Obj *x, Val correction, int standard_deviation) {
    int n = A(x).rows;
    if (correction.t != V_INT || correction.i < 0 || correction.i >= n)
        vmerror(v, "variance and stddev require 0 <= integer ddof < vector length");
    double origin = element(x, 0), scale = vector_scale(x);
    long double mean = 0, squares = 0;
    for (int i = 0; i < n; i++) {
        long double value = centered(element(x, i), origin, scale), delta = value - mean;
        mean += delta / (i + 1);
        squares += delta * (value - mean);
    }
    long double deviation = sqrtl(squares / (n - correction.i)) * scale;
    return number(representable(v, standard_deviation ? deviation : deviation * deviation));
}

static Val distribution(VM *v, int op, Val *args) {
    double x = finite_number(v, args[0]), parameter = finite_number(v, args[1]);
    if (op <= NORMAL_SF) {
        double deviation = finite_number(v, args[2]);
        if (deviation <= 0)
            vmerror(v, "normal deviation must be positive");
        long double z = centered(x, parameter, deviation);
        if (op == NORMAL_PDF) {
            const long double log_sqrt_two_pi = 0.91893853320467274178032973640561764L;
            return number(representable(v, expl(-z * z / 2 - logl(deviation) - log_sqrt_two_pi)));
        }
        long double tail = z / sqrtl(2);
        return number((double)(erfcl(op == NORMAL_CDF ? -tail : tail) / 2));
    }
    if (parameter <= 0)
        vmerror(v, "exponential rate must be positive");
    if (x < 0)
        return number(op == EXPONENTIAL_SF ? 1 : 0);
    long double t = (long double)x * parameter;
    if (op == EXPONENTIAL_PDF)
        return number((double)fminl(parameter, expl(logl(parameter) - t)));
    return number((double)(op == EXPONENTIAL_CDF ? -expm1l(-t) : expl(-t)));
}

static Val paired_statistics(VM *v, Obj *x, Obj *y, int op) {
    int n = A(x).rows;
    if (n < 2 || A(y).rows != n)
        vmerror(v, "paired statistics require equal vector lengths of at least two");
    double sx = vector_scale(x), sy = vector_scale(y);
    double ox = element(x, 0), oy = element(y, 0);
    long double mx = 0, my = 0, xx = 0, yy = 0, xy = 0;
    for (int i = 1; i < n; i++) {
        long double value_x = centered(element(x, i), ox, sx);
        long double value_y = centered(element(y, i), oy, sy);
        long double dx = value_x - mx, dy = value_y - my;
        mx += dx / (i + 1);
        my += dy / (i + 1);
        xx += dx * (value_x - mx);
        yy += dy * (value_y - my);
        xy += dx * (value_y - my);
    }
    if (op == COVARIANCE) {
        int ex, ey;
        double mantissa_x = frexp(sx, &ex), mantissa_y = frexp(sy, &ey);
        return number(representable(v, ldexpl(xy / (n - 1) * mantissa_x * mantissa_y, ex + ey)));
    }
    if (op == CORRELATION) {
        if (!(xx > 0) || !(yy > 0))
            vmerror(v, "correlation requires nonconstant vectors");
        long double result = xy / (sqrtl(xx) * sqrtl(yy));
        return number((double)fminl(1, fmaxl(-1, result)));
    }
    if (!(xx > 0))
        vmerror(v, "regression requires a nonconstant predictor");
    long double slope = (xy / xx) * sy / sx;
    long double intercept = (oy + my * sy) - slope * (ox + mx * sx);
    double fitted_slope = representable(v, slope), fitted_intercept = representable(v, intercept);
    Obj *out = list(v);
    append(v, out, number(fitted_slope));
    append(v, out, number(fitted_intercept));
    return objval(out);
}

typedef struct {
    int64_t count;
    long double origin, mean, squares;
} Moments;

static Val online_moments(VM *v, int op, Val *args) {
    if (op == ONLINE)
        return objval(newobj_extra(v, O_MOMENTS, sizeof(Moments)));
    if (!isobj(args[0], O_MOMENTS))
        vmerror(v, "expected an online moments accumulator");
    /* The tail allocation holds bytes. Copy through a naturally aligned
     * local so long double does not change the layout of every VM object. */
    void *storage = args[0].o + 1;
    Moments state;
    memcpy(&state, storage, sizeof state);
    if (op == UPDATE) {
        long double value = finite_number(v, args[1]);
        if (state.count == INT64_MAX)
            vmerror(v, "online moments count overflow");
        if (!state.count)
            state.origin = value;
        value -= state.origin;
        moments_update((uint64_t)++state.count, value, &state.mean, &state.squares);
        if (!isfinite(state.mean) || !isfinite(state.squares))
            vmerror(v, "online moments overflow");
        memcpy(storage, &state, sizeof state);
        return nil();
    }
    Obj *out = list(v);
    append(v, out, integer(state.count));
    if (state.count) {
        append(v, out, number(representable(v, state.origin + state.mean)));
        append(v, out, number(representable(v, state.squares / state.count)));
    } else {
        append(v, out, nil());
        append(v, out, nil());
    }
    return objval(out);
}

Val stats_call(VM *v, int id, int argc, Val *args) {
    int op = id - STATS_BASE;
    if (op < 0 || (size_t)op >= sizeof names / sizeof *names)
        vmerror(v, "unknown stats operation");
    if (argc != arities[op] && !((op == VARIANCE || op == STDDEV) && argc == 2))
        vmerror(v, "invalid stats.%s arity", names[op]);
    if (op >= NORMAL_PDF)
        return distribution(v, op, args);
    if (op >= ONLINE && op <= SUMMARY)
        return online_moments(v, op, args);
    if (op == CLIP)
        return clip(v, args[0], finite_number(v, args[1]), finite_number(v, args[2]));

    Obj *x = vector(v, args[0]);
    switch (op) {
    case VARIANCE:
    case STDDEV:
        return dispersion(v, x, argc == 2 ? args[1] : integer(0), op == STDDEV);
    case SUM:
    case CUMSUM:
        return compensated_sum(v, x, op == CUMSUM);
    case MEDIAN:
        return quantile(v, x, 0.5);
    case QUANTILE:
        return quantile(v, x, finite_number(v, args[1]));
    case COVARIANCE:
    case CORRELATION:
    case REGRESSION:
        return paired_statistics(v, x, vector(v, args[1]), op);
    case HISTOGRAM:
        return histogram(v, x, vector(v, args[1]));
    default:
        return extreme_index(v, x, op == ARGMAX);
    }
}

Obj *stats_module(VM *v) {
    return named_native_module(v, "stats", names, sizeof names / sizeof *names, STATS_BASE);
}
