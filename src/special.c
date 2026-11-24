#include "arda.h"
#include <float.h>

enum { GAMMA, LGAMMA, BETA, LBETA, BESSEL_J, BESSEL_Y, ERF, ERFC };
static const char *const names[] = {"gamma",    "lgamma",   "beta", "lbeta",
                                    "bessel_j", "bessel_y", "erf",  "erfc"};
static const int arities[] = {1, 1, 2, 2, 2, 2, 1, 1};
_Static_assert(sizeof names / sizeof *names == sizeof arities / sizeof *arities,
               "special function arity table");

static long double finite_argument(VM *v, Val value) {
    double x = asnum(v, value);
    if (!isfinite(x))
        vmerror(v, "special functions require finite arguments");
    return value.t == V_INT ? (long double)value.i : x;
}

/* Stirling's log-gamma remainder (DLMF 5.11.1), used only for x >= 16. */
static long double stirling_tail(long double x) {
    long double inv = 1 / x, z = inv * inv;
    long double series = 1.0L / 156;
    series = -691.0L / 360360 + z * series;
    series = 1.0L / 1188 + z * series;
    series = -1.0L / 1680 + z * series;
    series = 1.0L / 1260 + z * series;
    series = -1.0L / 360 + z * series;
    return inv * (1.0L / 12 + z * series);
}

static long double log_beta(long double x, long double y) {
    long double a = fminl(x, y), b = fmaxl(x, y);
    if (b < 16)
        return lgammal(a) + lgammal(b) - lgammal(a + b);

    // Expand the large terms together to avoid subtracting nearly equal log-gammas.
    long double ratio = a / b;
    long double log_scale = log1pl(ratio);
    long double scaled = (a - 0.5L) * log_scale + b * log_scale;
    long double correction = stirling_tail(b) - stirling_tail(a + b);
    if (a < 16)
        return lgammal(a) - a * logl(b) + a - scaled + correction;
    return (a - 0.5L) * logl(ratio) - 0.5L * logl(b) - scaled +
           0.91893853320467274178032973640561764L + stirling_tail(a) + correction;
}

Val special_call(VM *v, int id, int argc, Val *args) {
    int op = id - SPECIAL_BASE;
    if (op < 0 || (size_t)op >= sizeof names / sizeof *names || argc != arities[op])
        vmerror(v, "invalid special function arity");

    long double x = finite_argument(v, args[0]), result;
    switch (op) {
    case GAMMA:
    case LGAMMA:
        if (x <= 0 && truncl(x) == x)
            vmerror(v, "gamma has a pole at nonpositive integers");
        result = op == GAMMA ? tgammal(x) : lgammal(x);
        break;
    case BETA:
    case LBETA: {
        long double y = finite_argument(v, args[1]);
        if (x <= 0 || y <= 0)
            vmerror(v, "beta requires positive arguments");
        result = log_beta(x, y);
        if (op == BETA)
            result = expl(result);
        break;
    }
    case BESSEL_J:
    case BESSEL_Y: {
        if (args[0].t != V_INT || args[0].i < -10000 || args[0].i > 10000)
            vmerror(v, "Bessel order must be an integer in [-10000, 10000]");
        double z = (double)finite_argument(v, args[1]);
        if (op == BESSEL_Y && z <= 0)
            vmerror(v, "Bessel Y requires a positive argument");
        int n = (int)args[0].i;
        result = op == BESSEL_J ? jn(n, z) : yn(n, z);
        break;
    }
    case ERF:
        result = erf((double)x);
        break;
    case ERFC:
        result = erfc((double)x);
        break;
    default:
        vmerror(v, "unknown special function");
    }
    if (!isfinite(result) || fabsl(result) > DBL_MAX)
        vmerror(v, "special function result is outside the Float64 range");
    return number((double)result);
}

Obj *special_module(VM *v) {
    return named_native_module(v, "special", names, sizeof names / sizeof *names, SPECIAL_BASE);
}
