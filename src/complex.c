#include "arda.h"
#include <complex.h>
#include <float.h>

enum { NEW, REAL, IMAG, CONJ, ABS, ARG, EXP, LOG, SQRT, SIN, COS, TAN, POW };

static const char *const names[] = {"new", "real", "imag", "conj", "abs", "arg", "exp",
                                    "log", "sqrt", "sin",  "cos",  "tan", "pow"};

Val complex_value(VM *v, double re, double im) {
    Obj *object = newobj(v, O_COMPLEX);
    object->as.cplx.re = re;
    object->as.cplx.im = im;
    return objval(object);
}

static double complex as_complex(VM *v, Val value) {
    double parts[2] = {0, 0};
    if (isobj(value, O_COMPLEX)) {
        parts[0] = value.o->as.cplx.re;
        parts[1] = value.o->as.cplx.im;
    } else {
        parts[0] = asnum(v, value);
    }
    /* C11 defines the two-component representation. Building with + I*
     * would change signed zeros and can contaminate the real part with NaN. */
    double complex result;
    memcpy(&result, parts, sizeof result);
    return result;
}

static double component(long double value) {
    if (fabsl(value) > DBL_MAX)
        return signbit(value) ? -INFINITY : INFINITY;
    return (double)value;
}

Val complex_binary(VM *v, Op op, Val a, Val b) {
    long double complex x = as_complex(v, a), y = as_complex(v, b), result;
    switch (op) {
    case OP_ADD:
        result = x + y;
        break;
    case OP_SUB:
        result = x - y;
        break;
    case OP_MUL:
        result = x * y;
        break;
    case OP_DIV:
        if (creall(y) == 0 && cimagl(y) == 0)
            vmerror(v, "complex division by zero");
        result = x / y;
        break;
    default:
        vmerror(v, "unsupported complex operator");
    }
    return complex_value(v, component(creall(result)), component(cimagl(result)));
}

Val complex_call(VM *v, int id, int argc, Val *a) {
    int op = id - COMPLEX_BASE;
    if (op < 0 || (size_t)op >= sizeof names / sizeof *names)
        vmerror(v, "unknown complex operation");
    if (argc != ((op == NEW || op == POW) ? 2 : 1))
        vmerror(v, "invalid complex.%s arity", names[op]);
    if (op == NEW)
        return complex_value(v, asnum(v, a[0]), asnum(v, a[1]));
    double complex z = as_complex(v, a[0]);
    switch (op) {
    case REAL:
        return number(creal(z));
    case IMAG:
        return number(cimag(z));
    case ABS:
        return number(cabs(z));
    case ARG:
        return number(carg(z));
    case CONJ:
        z = conj(z);
        break;
    case EXP:
        z = cexp(z);
        break;
    case LOG:
        z = clog(z);
        break;
    case SQRT:
        z = csqrt(z);
        break;
    case SIN:
        z = csin(z);
        break;
    case COS:
        z = ccos(z);
        break;
    case TAN:
        z = ctan(z);
        break;
    case POW: {
        double complex exponent = as_complex(v, a[1]);
        if (creal(exponent) == 0 && cimag(exponent) == 0)
            return complex_value(v, 1, 0);
        z = cpow(z, exponent);
        break;
    }
    default:
        vmerror(v, "unknown complex operation");
    }
    return complex_value(v, creal(z), cimag(z));
}

Obj *complex_module(VM *v) {
    return named_native_module(v, "complex", names, sizeof names / sizeof *names, COMPLEX_BASE);
}
