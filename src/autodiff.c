#include "autodiff.h"

static bool is_tensor(Val x) {
    return isobj(x, O_ARRAY) || isobj(x, O_TYPED_TENSOR);
}

Val ad_primal(Val value) {
    return isobj(value, O_AD) ? value.o->as.ad->value : value;
}

static TypedTensor layout(VM *v, Val x) {
    if (!is_tensor(x))
        return (TypedTensor){.n = 1};
    TypedTensor t = typed_tensor(v, x);
    if (t.dtype != DT_F64 || t.buffer->as.buffer.mapped)
        vmerror_kind(v, "dtype", "autodiff requires dense heap-backed Float64 tensors");
    return t;
}

static double entry(const TypedTensor *t, ptrdiff_t position) {
    double x;
    memcpy(&x, typed_element(t, position), sizeof x);
    return x;
}

static double item(const TypedTensor *t, size_t i) {
    return entry(t, typed_offset(t, i));
}

static double checked(VM *v, double x) {
    if (!isfinite(x))
        vmerror_kind(v, "nonfinite", "nonfinite autodiff value, derivative or gradient");
    return x;
}

void ad_validate(VM *v, Val value) {
    value = ad_primal(value);
    TypedTensor t = layout(v, value);
    if (t.rank) {
        for (size_t i = 0; i < t.n; i++)
            checked(v, item(&t, i));
    } else
        checked(v, asnum(v, value));
}

static unsigned saved_values(ADOperation op, bool left, bool right) {
    switch (op) {
    case AD_MUL:
    case AD_MATMUL:
        return (left ? 2u : 0u) | (right ? 1u : 0u);
    case AD_DIV:
        return (left ? 2u : 0u) | (right ? 6u : 0u);
    case AD_SIN:
    case AD_COS:
    case AD_LOG:
    case AD_POW:
        return 1;
    case AD_EXP:
    case AD_SQRT:
    case AD_TANH:
        return 4;
    default:
        return 0;
    }
}

static uint64_t version(VM *v, Val x) {
    return is_tensor(x) ? layout(v, x).buffer->as.buffer.version : 0;
}

Val ad_record(VM *v, ADOperation op, Val value, Val left, Val right, int axis, const int *axes) {
    ad_validate(v, value);
    Obj *a = isobj(left, O_AD) ? left.o : NULL, *b = isobj(right, O_AD) ? right.o : NULL;
    Obj *out = newobj(v, O_AD);
    TensorAD *node = mem(sizeof *node);
    *node = (TensorAD){.value = value,
                       .grad = nil(),
                       .left = ad_primal(left),
                       .right = ad_primal(right),
                       .parents = {a, b},
                       .saved = saved_values(op, a != NULL, b != NULL),
                       .axis = axis,
                       .operation = op};
    out->as.ad = node;
    account(v, out, 0, sizeof *node);
    if (axes)
        memcpy(node->axes, axes, sizeof node->axes);
    Val versions[] = {node->left, node->right, node->value};
    for (unsigned i = 0; i < 3; i++)
        if (node->saved & (1u << i))
            node->versions[i] = version(v, versions[i]);
    barrier(v, out, value);
    barrier(v, out, left);
    barrier(v, out, right);
    barrier(v, out, node->left);
    barrier(v, out, node->right);
    return objval(out);
}

Val ad_binary(VM *v, Op op, Val a, Val b) {
    ADOperation rule;
    switch (op) {
    case OP_ADD:
        rule = AD_ADD;
        break;
    case OP_SUB:
        rule = AD_SUB;
        break;
    case OP_MUL:
        rule = AD_MUL;
        break;
    case OP_DIV:
        rule = AD_DIV;
        break;
    default:
        vmerror_kind(v, "autodiff", "unsupported autodiff operator (use ad.value for comparisons)");
    }
    ad_validate(v, a);
    ad_validate(v, b);
    Val x = ad_primal(a), y = ad_primal(b);
    if (is_tensor(x) || is_tensor(y))
        return ad_record(v, rule, typed_tensor_binary(v, op, x, y), a, b, 0, NULL);
    double p = asnum(v, x), q = asnum(v, y);
    if (op == OP_DIV && q == 0)
        vmerror_kind(v, "nonfinite", "division by zero on differentiated value");
    double z = op == OP_ADD ? p + q : op == OP_SUB ? p - q : op == OP_MUL ? p * q : p / q;
    if (op == OP_DIV && isobj(a, O_AD))
        checked(v, 1 / q);
    if (op == OP_DIV && isobj(b, O_AD))
        checked(v, -z / q);
    return ad_record(v, rule, number(z), a, b, 0, NULL);
}

static double unary(VM *v, ADOperation op, double x, double exponent, bool tracked) {
    switch (op) {
    case AD_NEG:
        return -x;
    case AD_SIN:
        return sin(x);
    case AD_COS:
        return cos(x);
    case AD_EXP:
        return exp(x);
    case AD_LOG:
        if (tracked && x <= 0)
            vmerror_kind(v, "domain", "log domain");
        return log(x);
    case AD_SQRT:
        if (tracked && x <= 0)
            vmerror_kind(v, "domain", "sqrt derivative domain");
        return sqrt(x);
    case AD_TANH:
        return tanh(x);
    case AD_POW:
        return pow(x, exponent);
    default:
        vmerror(v, "unknown autodiff unary operation");
    }
}

static double unary_derivative(ADOperation op, double x, double z, double exponent) {
    switch (op) {
    case AD_NEG:
        return -1;
    case AD_SIN:
        return cos(x);
    case AD_COS:
        return -sin(x);
    case AD_EXP:
        return z;
    case AD_LOG:
        return 1 / x;
    case AD_SQRT:
        return 0.5 / z;
    case AD_TANH:
        return 1 - z * z;
    case AD_POW:
        return exponent == 0 ? 0 : exponent == 1 ? 1 : exponent * pow(x, exponent - 1);
    default:
        return 0;
    }
}

static double *data(Obj *x) {
    return (double *)x->as.tensor.buffer->as.buffer.data;
}

Val ad_unary(VM *v, ADOperation op, Val input, Val exponent) {
    bool tracked = isobj(input, O_AD);
    Val value = ad_primal(input);
    double power = op == AD_POW ? checked(v, asnum(v, exponent)) : 0;
    TypedTensor t = layout(v, value);
    if (tracked)
        ad_validate(v, value);
    Val result;
    if (!t.rank) {
        double x = asnum(v, value), y = unary(v, op, x, power, tracked);
        if (tracked)
            checked(v, unary_derivative(op, x, y, power));
        result = number(y);
    } else {
        Obj *out = typed_new(v, DT_F64, t.rank, t.shape);
        for (size_t i = 0; i < t.n; i++) {
            double x = item(&t, i), y = unary(v, op, x, power, tracked);
            if (tracked)
                checked(v, unary_derivative(op, x, y, power));
            data(out)[i] = y;
        }
        result = objval(out);
    }
    return tracked ? ad_record(v, op, result, input, exponent, 0, NULL) : result;
}

static Obj *topology(VM *v, Obj *root) {
    Obj *stack = list(v), *order = list(v);
    append(v, stack, objval(root));
    while (stack->as.list.n) {
        Obj *o = stack->as.list.a[stack->as.list.n - 1].o;
        TensorAD *n = o->as.ad;
        if (!n->visit) {
            n->visit = 1;
            for (int i = 1; i >= 0; i--)
                if (n->parents[i] && !n->parents[i]->as.ad->visit)
                    append(v, stack, objval(n->parents[i]));
        } else {
            stack->as.list.n--;
            if (n->visit == 1) {
                n->visit = 2;
                n->slot = order->as.list.n;
                append(v, order, objval(o));
            }
        }
    }
    for (int i = 0; i < order->as.list.n; i++)
        order->as.list.a[i].o->as.ad->visit = 0;
    return order;
}

static void coordinates(const TypedTensor *t, size_t i, int *index) {
    for (int d = t->rank - 1; d >= 0; d--) {
        index[d] = (int)(i % (size_t)t->shape[d]);
        i /= (size_t)t->shape[d];
    }
}

static size_t linear_index(const TypedTensor *t, const int *index) {
    size_t i = 0;
    for (int d = 0; d < t->rank; d++)
        i = i * (size_t)t->shape[d] + (size_t)index[d];
    return i;
}

static size_t broadcast_index(const TypedTensor *t, int rank, const int *index) {
    size_t i = 0;
    for (int d = 0; d < t->rank; d++) {
        int j = t->shape[d] == 1 ? 0 : index[d + rank - t->rank];
        i = i * (size_t)t->shape[d] + (size_t)j;
    }
    return i;
}

static void add(VM *v, Val *gradient, size_t index, double value) {
    double *target = gradient->t == V_OBJ ? &data(gradient->o)[index] : &gradient->n;
    double sum = *target + value;
    if (!isfinite(sum)) {
        // The native call still roots its arguments and previous gradients.
        // Discard unpublished gradient buffers before returning to a handler.
        gcfull(v);
        vmerror_kind(v, "nonfinite", "nonfinite autodiff gradient");
    }
    *target = sum;
}

static double read_value(VM *v, const TypedTensor *t, Val val, size_t index) {
    return t->rank ? item(t, index) : asnum(v, val);
}

static double grad_value(Val grad, size_t i) {
    return grad.t == V_OBJ ? data(grad.o)[i] : grad.n;
}

static void reverse_elementwise(VM *v, const TensorAD *n, Val g, Val *a, Val *b,
                                const TypedTensor *x, const TypedTensor *y, const TypedTensor *z) {
    for (size_t i = 0; i < z->n; i++) {
        int index[8];
        coordinates(z, i, index);
        size_t xi = broadcast_index(x, z->rank, index);
        size_t yi = y ? broadcast_index(y, z->rank, index) : 0;
        double weight = grad_value(g, i);
        switch (n->operation) {
        case AD_ADD:
        case AD_SUB:
            if (a)
                add(v, a, xi, weight);
            if (b)
                add(v, b, yi, n->operation == AD_SUB ? -weight : weight);
            break;
        case AD_MUL:
            if (a)
                add(v, a, xi, weight * read_value(v, y, n->right, yi));
            if (b)
                add(v, b, yi, weight * read_value(v, x, n->left, xi));
            break;
        case AD_DIV: {
            double denominator = read_value(v, y, n->right, yi);
            if (a)
                add(v, a, xi, weight / denominator);
            if (b)
                add(v, b, yi, -weight * (read_value(v, z, n->value, i) / denominator));
            break;
        }
        default: {
            double xv = n->saved & 1 ? read_value(v, x, n->left, xi) : 0;
            double zv = n->saved & 4 ? read_value(v, z, n->value, i) : 0;
            double power = n->operation == AD_POW ? asnum(v, n->right) : 0;
            add(v, a, xi, weight * unary_derivative(n->operation, xv, zv, power));
            break;
        }
        }
    }
}

static void matrix_index(const TypedTensor *t, int rank, const int *index, int row, int col,
                         int *out) {
    for (int d = 0; d < t->rank - 2; d++)
        out[d] = t->shape[d] == 1 ? 0 : index[d + rank - t->rank];
    out[t->rank - 2] = row;
    out[t->rank - 1] = col;
}

static void reverse_matmul(VM *v, Val g, Val *a, Val *b, const TypedTensor *x, const TypedTensor *y,
                           const TypedTensor *z) {
    for (size_t i = 0; i < z->n; i++) {
        int index[8], left[8], right[8];
        coordinates(z, i, index);
        double weight = grad_value(g, i);
        for (int k = 0; k < x->shape[x->rank - 1]; k++) {
            matrix_index(x, z->rank, index, index[z->rank - 2], k, left);
            matrix_index(y, z->rank, index, k, index[z->rank - 1], right);
            size_t xi = linear_index(x, left), yi = linear_index(y, right);
            if (a)
                add(v, a, xi, weight * item(y, yi));
            if (b)
                add(v, b, yi, weight * item(x, xi));
        }
    }
}

static void reverse_layout(VM *v, const TensorAD *n, Val g, Val *a, const TypedTensor *x,
                           const TypedTensor *z) {
    if (n->operation == AD_SUM || n->operation == AD_MEAN) {
        int axis = n->axis;
        size_t count = axis < 0 ? x->n : (size_t)x->shape[axis];
        for (size_t i = 0; i < x->n; i++) {
            size_t zi = 0;
            if (axis >= 0) {
                int index[8];
                coordinates(x, i, index);
                for (int d = 0; d < x->rank; d++)
                    if (d != axis)
                        zi = zi * (size_t)x->shape[d] + (size_t)index[d];
            }
            double w = grad_value(g, zi);
            add(v, a, i, n->operation == AD_MEAN ? w / (double)count : w);
        }
        return;
    }
    for (size_t i = 0; i < z->n; i++) {
        size_t xi = i;
        if (n->operation == AD_AT)
            xi = (size_t)n->axis;
        if (n->operation == AD_VIEW)
            xi = (size_t)(typed_offset(z, i) - (ptrdiff_t)x->offset);
        if (n->operation == AD_PERMUTE || n->operation == AD_SLICE) {
            int index[8], source[8];
            coordinates(z, i, index);
            for (int d = 0; d < x->rank; d++)
                source[n->operation == AD_PERMUTE ? n->axes[d] : d] = index[d];
            if (n->operation == AD_SLICE)
                source[n->axis] = n->axes[0] + source[n->axis] * n->axes[1];
            xi = linear_index(x, source);
        }
        add(v, a, xi, grad_value(g, i));
    }
}

static void reverse_node(VM *v, TensorAD *n, Val *grads) {
    if (n->operation == AD_LEAF)
        return;
    Val *a = n->parents[0] ? grads + n->parents[0]->as.ad->slot : NULL;
    Val *b = n->parents[1] ? grads + n->parents[1]->as.ad->slot : NULL;
    Val g = grads[n->slot];
    TypedTensor x = layout(v, n->left), z = layout(v, n->value);
    if (n->operation >= AD_SUM && n->operation < AD_MATMUL)
        reverse_layout(v, n, g, a, &x, &z);
    else {
        TypedTensor y = layout(v, n->right);
        if (n->operation == AD_MATMUL)
            reverse_matmul(v, g, a, b, &x, &y, &z);
        else
            reverse_elementwise(v, n, g, a, b, &x, &y, &z);
    }
}

static void backward(VM *v, Obj *root, Val seed) {
    TypedTensor target = layout(v, root->as.ad->value);
    TypedTensor start = layout(v, seed);
    if (seed.t == V_NIL && target.rank)
        vmerror_kind(v, "shape", "tensor backward requires an explicit seed or scalar reduction");
    if (target.rank != start.rank ||
        memcmp(target.shape, start.shape, (size_t)target.rank * sizeof *target.shape))
        vmerror_kind(v, "shape", "backward seed shape mismatch");
    if (seed.t != V_NIL)
        ad_validate(v, seed);
    Obj *order = topology(v, root), *work = list(v);
    size_t available = tensor_memory_available(v);
    for (int i = 0; i < order->as.list.n; i++) {
        TensorAD *n = order->as.list.a[i].o->as.ad;
        Val saved[] = {n->left, n->right, n->value};
        for (unsigned j = 0; j < 3; j++)
            if ((n->saved & (1u << j)) && version(v, saved[j]) != n->versions[j])
                vmerror_kind(v, "mutation",
                             "tensor saved for backward was modified through an alias");
        TypedTensor t = layout(v, n->value);
        if (t.rank) {
            if (t.n > available / sizeof(double))
                vmerror_kind(v, "memory_budget",
                             "autodiff gradients exceed the tensor memory budget");
            available -= t.n * sizeof(double);
        }
    }
    for (int i = 0; i < order->as.list.n; i++) {
        TypedTensor t = layout(v, order->as.list.a[i].o->as.ad->value);
        Val gradient = t.rank ? objval(typed_new(v, DT_F64, t.rank, t.shape)) : number(0);
        append(v, work, gradient);
    }
    Val *grads = work->as.list.a;
    if (start.rank) {
        double *dst = data(grads[root->as.ad->slot].o);
        for (size_t i = 0; i < start.n; i++)
            dst[i] = item(&start, i);
    } else
        grads[root->as.ad->slot] = number(seed.t == V_NIL ? 1 : asnum(v, seed));
    for (int i = order->as.list.n - 1; i >= 0; i--)
        reverse_node(v, order->as.list.a[i].o->as.ad, grads);
    for (int i = 0; i < order->as.list.n; i++) {
        Obj *o = order->as.list.a[i].o;
        o->as.ad->grad = grads[i];
        barrier(v, o, grads[i]);
    }
}

enum { VARIABLE, VALUE, GRAD, BACKWARD, SIN, COS, EXP, LOG, SQRT, TANH, POW };
static const char *const names[] = {"variable", "value", "grad", "backward", "sin", "cos",
                                    "exp",      "log",   "sqrt", "tanh",     "pow"};

Val ad_call(VM *v, int id, int argc, Val *args) {
    int op = id - AD_BASE;
    if (op < 0 || (size_t)op >= sizeof names / sizeof *names ||
        (argc != (op == POW ? 2 : 1) && !(op == BACKWARD && argc == 2)))
        vmerror(v, "invalid autodiff function arity");
    if (op == VARIABLE) {
        if (isobj(args[0], O_AD))
            vmerror(v, "expected ordinary input to ad.variable");
        Val input = is_tensor(args[0]) ? args[0] : number(asnum(v, args[0]));
        return ad_record(v, AD_LEAF, input, nil(), nil(), 0, NULL);
    }
    if (op == VALUE)
        return ad_primal(args[0]);
    if (op == GRAD || op == BACKWARD) {
        if (!isobj(args[0], O_AD))
            vmerror(v, "expected autodiff node");
        if (op == GRAD) {
            Val result = args[0].o->as.ad->grad;
            if (is_tensor(result)) {
                TypedTensor t = typed_tensor(v, result);
                t.readonly = true;
                return objval(typed_wrap(v, t));
            }
            return result;
        }
        backward(v, args[0].o, argc == 2 ? args[1] : nil());
        return nil();
    }
    static const ADOperation unary_ops[] = {AD_SIN,  AD_COS,  AD_EXP, AD_LOG,
                                            AD_SQRT, AD_TANH, AD_POW};
    return ad_unary(v, unary_ops[op - SIN], args[0], argc == 2 ? args[1] : nil());
}

Obj *extra_module(VM *v, const char *name) {
    if (!strcmp(name, "ad"))
        return named_native_module(v, "ad", names, sizeof names / sizeof *names, AD_BASE);
    if (!strcmp(name, "calc")) {
        int count;
        const char **calculus_names(int *n);
        const char **calculus = calculus_names(&count);
        return named_native_module(v, "calc", calculus, (size_t)count, CALC_BASE);
    }
    return NULL;
}
