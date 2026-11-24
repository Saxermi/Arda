#include "typed_tensor.h"
#include <complex.h>
#include <float.h>

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
                   sizeof(double) == 8 && DBL_MANT_DIG == 53,
               "typed tensors require binary32 and binary64");

const DTypeInfo dtype_info[DT_COUNT] = {
    [DT_BOOL] = {"bool", 1, SC_BOOL},        [DT_U8] = {"uint8", 1, SC_INT},
    [DT_I32] = {"int32", 4, SC_INT},         [DT_I64] = {"int64", 8, SC_INT},
    [DT_F16] = {"float16", 2, SC_REAL},      [DT_BF16] = {"bfloat16", 2, SC_REAL},
    [DT_F32] = {"float32", 4, SC_REAL},      [DT_F64] = {"float64", 8, SC_REAL},
    [DT_C64] = {"complex64", 8, SC_COMPLEX}, [DT_C128] = {"complex128", 16, SC_COMPLEX},
};

DType dtype_parse(VM *v, Val value) {
    const char *name = asstr(v, value);
    for (int i = 0; i < DT_COUNT; i++)
        if (!strcmp(name, dtype_info[i].name))
            return (DType)i;
    vmerror(v, "unknown tensor dtype '%s'", name);
}

static DType arithmetic_type(DType type) {
    if (type == DT_BOOL)
        return DT_I32;
    if (type == DT_F16 || type == DT_BF16)
        return DT_F32;
    return type;
}

DType dtype_promote(DType a, DType b, Op op) {
    a = arithmetic_type(a);
    b = arithmetic_type(b);
    int wide = a == DT_I32 || a == DT_I64 || a == DT_F64 || a == DT_C128 || b == DT_I32 ||
               b == DT_I64 || b == DT_F64 || b == DT_C128;
    if (a >= DT_C64 || b >= DT_C64)
        return wide ? DT_C128 : DT_C64;
    if (a >= DT_F32 || b >= DT_F32)
        return wide ? DT_F64 : DT_F32;
    if (op == OP_DIV)
        return DT_F64;
    return a > b ? a : b;
}

TensorScalar dtype_scalar(VM *v, Val value) {
    if (value.t == V_BOOL)
        return (TensorScalar){.kind = SC_BOOL, .i = value.n != 0};
    if (value.t == V_INT)
        return (TensorScalar){.kind = SC_INT, .i = value.i};
    if (isobj(value, O_COMPLEX))
        return (TensorScalar){
            .kind = SC_COMPLEX, .re = value.o->as.cplx.re, .im = value.o->as.cplx.im};
    return (TensorScalar){.kind = SC_REAL, .re = asnum(v, value)};
}

Val dtype_value(VM *v, TensorScalar value) {
    switch (value.kind) {
    case SC_BOOL:
        return boolean(value.i != 0);
    case SC_INT:
        return integer(value.i);
    case SC_REAL:
        return number(value.re);
    case SC_COMPLEX:
        return complex_value(v, value.re, value.im);
    }
    return nil();
}

static double real_part(TensorScalar value) {
    return value.kind <= SC_INT ? (double)value.i : value.re;
}

static uint16_t round_even(double x) {
    uint16_t low = (uint16_t)x;
    double remainder = x - low;
    return low + (remainder > 0.5 || (remainder == 0.5 && (low & 1)));
}

/* Convert from binary64 directly, avoiding double rounding through binary32. */
static uint16_t pack16(VM *v, double value, int fraction_bits, int bias) {
    unsigned sign = signbit(value) ? 0x8000 : 0;
    unsigned infinity = (unsigned)(2 * bias + 1) << fraction_bits;
    if (isnan(value))
        return (uint16_t)(sign | infinity | (1u << (fraction_bits - 1)));
    if (isinf(value))
        return (uint16_t)(sign | infinity);
    double x = fabs(value);
    int exp;
    double frac = frexp(x, &exp);
    int min_exp = 1 - bias;
    unsigned result;
    if (x == 0 || exp - 1 < min_exp)
        result = round_even(ldexp(x, fraction_bits - min_exp));
    else {
        if (exp - 1 > bias)
            vmerror(v, "floating tensor cast overflow");
        unsigned leading = 1u << fraction_bits;
        unsigned significand = round_even(ldexp(frac, fraction_bits + 1));
        result = ((unsigned)(exp - 1 + bias) << fraction_bits) + significand - leading;
        if (result >= infinity)
            vmerror(v, "floating tensor cast overflow");
    }
    return (uint16_t)(sign | result);
}

static double unpack16(uint16_t bits, int fraction_bits, int bias) {
    int exp = (bits & 0x7fff) >> fraction_bits;
    unsigned leading = 1u << fraction_bits;
    unsigned fraction = bits & (leading - 1);
    double value;
    if (exp == 2 * bias + 1)
        value = fraction ? NAN : INFINITY;
    else if (exp == 0)
        value = ldexp(fraction, 1 - bias - fraction_bits);
    else
        value = ldexp(leading + fraction, exp - bias - fraction_bits);
    return bits & 0x8000 ? -value : value;
}

TensorScalar dtype_load(DType type, const unsigned char *data) {
    TensorScalar result = {.kind = dtype_info[type].kind};
    switch (type) {
    case DT_BOOL:
        result.i = *data != 0;
        break;
    case DT_U8:
        result.i = *data;
        break;
    case DT_I32: {
        int32_t value;
        memcpy(&value, data, sizeof value);
        result.i = value;
        break;
    }
    case DT_I64:
        memcpy(&result.i, data, sizeof result.i);
        break;
    case DT_F16:
    case DT_BF16: {
        uint16_t bits;
        memcpy(&bits, data, sizeof bits);
        result.re = type == DT_F16 ? unpack16(bits, 10, 15) : unpack16(bits, 7, 127);
        break;
    }
    case DT_F32:
    case DT_C64: {
        float parts[2] = {0};
        memcpy(parts, data, dtype_info[type].size);
        result.re = parts[0];
        result.im = parts[1];
        break;
    }
    case DT_F64:
    case DT_C128: {
        double parts[2] = {0};
        memcpy(parts, data, dtype_info[type].size);
        result.re = parts[0];
        result.im = parts[1];
        break;
    }
    case DT_COUNT:
        break;
    }
    return result;
}

static int64_t exact_integer(VM *v, TensorScalar value) {
    if (value.kind <= SC_INT)
        return value.i;
    double x = value.re;
    if (value.im != 0 || !isfinite(x) || trunc(x) != x || x < -0x1p63 || x >= 0x1p63)
        vmerror(v, "tensor integer cast requires a representable integral value");
    return (int64_t)x;
}

static float narrow_float(VM *v, double x) {
    if (isfinite(x) && fabs(x) > FLT_MAX)
        vmerror(v, "floating tensor cast overflow");
    return (float)x;
}

void dtype_store(VM *v, DType type, unsigned char *data, TensorScalar value) {
    ScalarKind kind = dtype_info[type].kind;
    if (kind < SC_COMPLEX && value.im != 0)
        vmerror(v, "tensor cast would discard a nonzero imaginary part");
    if (kind <= SC_INT) {
        int64_t x = exact_integer(v, value);
        if ((type == DT_BOOL && x != 0 && x != 1) || (type == DT_U8 && (x < 0 || x > 255)) ||
            (type == DT_I32 && (x < INT32_MIN || x > INT32_MAX)))
            vmerror(v, "tensor integer cast overflow");
        if (type <= DT_U8)
            *data = (unsigned char)x;
        else if (type == DT_I32) {
            int32_t narrow = (int32_t)x;
            memcpy(data, &narrow, sizeof narrow);
        } else
            memcpy(data, &x, sizeof x);
        return;
    }
    double real = real_part(value);
    if (type == DT_F16 || type == DT_BF16) {
        uint16_t bits = type == DT_F16 ? pack16(v, real, 10, 15) : pack16(v, real, 7, 127);
        memcpy(data, &bits, sizeof bits);
    } else if (type == DT_F32 || type == DT_C64) {
        float parts[] = {narrow_float(v, real), narrow_float(v, value.im)};
        memcpy(data, parts, dtype_info[type].size);
    } else {
        double parts[] = {real, value.im};
        memcpy(data, parts, dtype_info[type].size);
    }
}

TensorScalar dtype_binary(VM *v, Op op, DType type, TensorScalar a, TensorScalar b) {
    ScalarKind kind = dtype_info[type].kind;
    if (kind <= SC_INT)
        return (TensorScalar){.kind = SC_INT, .i = checked_arith(v, op, a.i, b.i)};
    double x = real_part(a), y = real_part(b);
    if (kind == SC_COMPLEX) {
        double parts_p[] = {x, a.im}, parts_q[] = {y, b.im};
        double complex p, q;
        memcpy(&p, parts_p, sizeof p);
        memcpy(&q, parts_q, sizeof q);
        double complex z = op == OP_ADD   ? p + q
                           : op == OP_SUB ? p - q
                           : op == OP_MUL ? p * q
                                          : p / q;
        return (TensorScalar){.kind = SC_COMPLEX, .re = creal(z), .im = cimag(z)};
    }
    double z = op == OP_ADD ? x + y : op == OP_SUB ? x - y : op == OP_MUL ? x * y : x / y;
    return (TensorScalar){.kind = SC_REAL, .re = z};
}
