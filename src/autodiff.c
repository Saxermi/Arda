#include "autodiff.h"
#include "activation_store.h"

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
    if (t.dtype != DT_F64 || (t.buffer->as.buffer.mapped && !t.buffer->as.buffer.offload))
        vmerror_kind(
            v, "dtype",
            "autodiff requires dense Float64 tensors backed by the heap or an activation offload");
    return t;
}

static double item(const TypedTensor *t, size_t i) {
    double value;
    memcpy(&value, typed_element(t, typed_offset(t, i)), sizeof value);
    return value;
}

static double saved_item(VM *v, const TypedTensor *t, size_t i) {
    const Buffer *buffer = &t->buffer->as.buffer;
    return buffer->offload
               ? offload_read_float64(v, buffer, (size_t)typed_offset(t, i) * sizeof(double))
               : item(t, i);
}

static double checked(VM *v, double x) {
    if (!isfinite(x))
        vmerror_kind(v, "nonfinite", "nonfinite autodiff value, derivative or gradient");
    return x;
}

void ad_validate(VM *v, Val value) {
    value = ad_primal(value);
    TypedTensor t = layout(v, value);
    if (!t.rank)
        checked(v, asnum(v, value));
    else if (!t.buffer->as.buffer.offload)
        for (size_t i = 0; i < t.n; i++)
            checked(v, item(&t, i));
}

static unsigned saved_values(ADOperation op, bool left, bool right) {
    switch (op) {
    case AD_MUL:
    case AD_MATMUL:
    case AD_SPARSE_MATVEC:
        return (left ? 2u : 0u) | (right ? 1u : 0u);
    case AD_CHECKPOINT:
        return 7;
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
    bool tracked = isobj(input, O_AD) && !ad_disabled(v);
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

static Obj *topology(VM *v, Obj *root, Obj *stop, Obj *other) {
    Obj *stack = list(v), *order = list(v);
    append(v, stack, objval(root));
    while (stack->as.list.n) {
        Obj *o = stack->as.list.a[stack->as.list.n - 1].o;
        TensorAD *n = o->as.ad;
        if (!n->visit) {
            n->visit = 1;
            if (o != stop && o != other)
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
    return t->rank ? saved_item(v, t, index) : asnum(v, val);
}

static double grad_value(Val grad, size_t i) {
    return grad.t == V_OBJ ? data(grad.o)[i] : grad.n;
}

typedef struct {
    VM *vm;
    TensorAD *node;
    Val *output, *parents[2];
    Val *adjoint, *tangents[2], *out_tangent;
    bool forward;
} Sweep;

static double tangent_at(const Val *tangent, size_t i) {
    return tangent ? grad_value(*tangent, i) : 0;
}

static void edge(const Sweep *s, int parent, size_t from, size_t to, double slope,
                 double slope_direction) {
    if (!s->parents[parent])
        return;
    if (s->forward) {
        add(s->vm, s->output, to, slope * grad_value(*s->parents[parent], from));
    } else {
        double value = slope * grad_value(*s->output, to);
        if (s->adjoint) {
            value += slope_direction * grad_value(*s->adjoint, to);
        }
        add(s->vm, s->parents[parent], from, value);
    }
}

static double slope_change(ADOperation op, double x, double z, double power, double dx, double dz) {
    switch (op) {
    case AD_SIN:
        return -sin(x) * dx;
    case AD_COS:
        return -cos(x) * dx;
    case AD_EXP:
        return dz;
    case AD_LOG:
        return -(dx / x) / x;
    case AD_SQRT:
        return -(0.5 / z) * (dz / z);
    case AD_TANH:
        return -2 * z * dz;
    case AD_POW:
        if (power == 0 || power == 1 || dx == 0)
            return 0;
        if (power == 2)
            return 2 * dx;
        return power * (power - 1) * pow(x, power - 2) * dx;
    default:
        return 0;
    }
}

static bool connected(const Obj *order, const Obj *node);

typedef struct {
    ADOperation op;
    int input[2];
    Val constant[2];
    uint64_t version[2];
} ReplayStep;

struct ADCheckpoint {
    int count;
    ReplayStep steps[];
};

void ad_trace_checkpoint(VM *v, const TensorAD *node) {
    const ADCheckpoint *c = node->checkpoint;
    if (c)
        for (int i = 0; i < c->count; i++)
            for (int j = 0; j < 2; j++)
                markval(v, c->steps[i].constant[j]);
}

static size_t recipe_bytes(const ADCheckpoint *c) {
    return sizeof *c + (size_t)c->count * sizeof *c->steps;
}

static void checkpoint_versions(VM *v, const ADCheckpoint *c) {
    for (int i = 0; i < c->count; i++)
        for (int j = 0; j < 2; j++)
            if (version(v, c->steps[i].constant[j]) != c->steps[i].version[j])
                vmerror_kind(v, "mutation", "a checkpoint constant changed after recording");
}

static void checkpoint_shape(VM *v, const TypedTensor *result, Val x) {
    TypedTensor t = layout(v, x);
    if (t.rank && (t.rank != result->rank ||
                   memcmp(t.shape, result->shape, (size_t)t.rank * sizeof *t.shape)))
        vmerror_kind(v, "shape", "checkpoint tensors must have the output shape or be scalars");
}

/* Copy the operation recipe. Internal nodes stay on the original tape until
 * callers drop their references, while the new node keeps only boundaries. */
static Val checkpoint(VM *v, Obj *root, Val input, Val other) {
    if (!isobj(input, O_AD) || (other.t != V_NIL && !isobj(other, O_AD)))
        vmerror_kind(v, "type", "checkpoint boundaries must be autodiff nodes");
    if (root == input.o && other.t == V_NIL)
        return input;
    Obj *second = other.t == V_NIL ? NULL : other.o;
    if (input.o == second)
        vmerror_kind(v, "autodiff", "checkpoint boundaries must be distinct");
    Obj *order = topology(v, root, input.o, second);
    if (!connected(order, input.o) || (second && !connected(order, second)))
        vmerror_kind(v, "autodiff", "checkpoint boundaries must contribute to the output");
    int count = order->as.list.n;
    if (count > INT_MAX - 2 ||
        (size_t)count > (SIZE_MAX - sizeof(ADCheckpoint)) / sizeof(ReplayStep))
        vmerror_kind(v, "memory_budget", "checkpoint recipe is too large");
    Val value = root->as.ad->value;
    TypedTensor result = layout(v, value);
    Val out = ad_record(v, AD_CHECKPOINT, value, input, other, 0, NULL);
    size_t bytes = sizeof(ADCheckpoint) + (size_t)count * sizeof(ReplayStep);
    ADCheckpoint *recipe = mem(bytes);
    recipe->count = count;
    out.o->as.ad->checkpoint = recipe;
    account(v, out.o, 0, bytes);
    for (int i = 0; i < count; i++) {
        Obj *obj = order->as.list.a[i].o;
        const TensorAD *n = obj->as.ad;
        checkpoint_shape(v, &result, n->value);
        if (obj == input.o || obj == second)
            continue;
        if (n->operation < AD_ADD || n->operation > AD_POW)
            vmerror_kind(v, "autodiff",
                         "checkpoint requires pure elementwise operations between its boundaries");
        ReplayStep *step = &recipe->steps[i];
        step->op = n->operation;
        Val args[] = {n->left, n->right, n->value};
        for (int j = 0; j < 3; j++)
            if (version(v, args[j]) != n->versions[j])
                vmerror_kind(v, "mutation", "a checkpoint operand changed after recording");
        for (int j = 0; j < 2; j++) {
            checkpoint_shape(v, &result, args[j]);
            Obj *parent = n->parents[j];
            if (parent)
                step->input[j] = parent == input.o  ? 0
                                 : parent == second ? 1
                                                    : parent->as.ad->slot + 2;
            else {
                step->input[j] = -1;
                step->constant[j] = args[j];
                step->version[j] = n->versions[j];
                barrier(v, out.o, args[j]);
            }
        }
    }
    return out;
}

static Val offload_activation(VM *v, Obj *source, Val plan_value) {
    ADOffloadPlan *plan = offload_plan(v, plan_value);
    const TensorAD *old = source->as.ad;
    if (old->operation == AD_LEAF || old->operation == AD_CONSUMED)
        vmerror_kind(v, "autodiff", "offload requires a tensor intermediate with a live tape");
    if (old->operation == AD_VIEW)
        vmerror_kind(v, "autodiff",
                     "offload a materialized result of a strided view, such as view + 0");
    TypedTensor input = layout(v, old->value);
    if (!input.rank)
        vmerror_kind(v, "dtype", "offload requires a tensor activation");
    if (input.buffer->as.buffer.version != old->versions[2])
        vmerror_kind(v, "mutation", "activation changed before it could be offloaded");
    if (!input.n)
        return objval(source);

    Val value = offload_snapshot(v, plan, &input);
    Obj *out = newobj(v, O_AD);
    TensorAD *node = mem(sizeof *node);
    *node = *old;
    node->value = value;
    node->grad = nil();
    node->batch_examples = 0;
    node->versions[2] = version(v, value);
    node->slot = 0;
    node->visit = 0;
    node->active = node->retain = false;
    out->as.ad = node;
    account(v, out, 0, sizeof *node);
    barrier(v, out, value);
    barrier(v, out, node->left);
    barrier(v, out, node->right);
    for (int i = 0; i < 2; i++)
        if (node->parents[i])
            barrier(v, out, objval(node->parents[i]));
    if (node->pattern)
        barrier(v, out, objval(node->pattern));
    if (old->checkpoint) {
        size_t bytes = recipe_bytes(old->checkpoint);
        node->checkpoint = mem(bytes);
        memcpy(node->checkpoint, old->checkpoint, bytes);
        account(v, out, 0, bytes);
        for (int i = 0; i < node->checkpoint->count; i++)
            for (int j = 0; j < 2; j++)
                barrier(v, out, node->checkpoint->steps[i].constant[j]);
    }
    return objval(out);
}

typedef struct {
    double value, tangent;
    double derivative[2], change[2];
} ReplayValue;

typedef struct {
    ReplayValue state;
    TypedTensor constants[2];
} ReplayWork;

static size_t replay_bytes(VM *v, const ADCheckpoint *c) {
    size_t steps = (size_t)c->count + 2;
    if (steps > PTRDIFF_MAX / sizeof(ReplayWork))
        vmerror_kind(v, "memory_budget", "checkpoint workspace exceeds addressable storage");
    return steps * sizeof(ReplayWork);
}

/* A zero direction contributes zero even when an unused local slope overflows. */
static double directional(double slope, double input) {
    return input == 0 ? 0 : slope * input;
}

static ReplayValue replay_step(VM *v, ADOperation op, ReplayValue a, ReplayValue b, bool second) {
    double x = a.value, y = b.value, slope[2] = {0, 0}, change[2] = {0, 0};
    ReplayValue out = {0};
    switch (op) {
    case AD_ADD:
    case AD_SUB:
        slope[0] = 1;
        slope[1] = op == AD_ADD ? 1 : -1;
        out.value = op == AD_ADD ? x + y : x - y;
        break;
    case AD_MUL:
        out.value = x * y;
        slope[0] = y;
        slope[1] = x;
        change[0] = b.tangent;
        change[1] = a.tangent;
        break;
    case AD_DIV:
        out.value = x / y;
        slope[0] = 1 / y;
        slope[1] = -out.value / y;
        if (second) {
            double tangent = directional(slope[0], a.tangent) + directional(slope[1], b.tangent);
            change[0] = -(b.tangent / y) / y;
            change[1] = (out.value * (b.tangent / y) - tangent) / y;
        }
        break;
    default:
        out.value = unary(v, op, x, y, false);
        slope[0] = unary_derivative(op, x, out.value, y);
        if (second)
            change[0] =
                slope_change(op, x, out.value, y, a.tangent, directional(slope[0], a.tangent));
        break;
    }
    if (second)
        out.tangent = directional(slope[0], a.tangent) + directional(slope[1], b.tangent);
    for (int i = 0; i < 2; i++) {
        out.derivative[i] =
            directional(slope[0], a.derivative[i]) + directional(slope[1], b.derivative[i]);
        if (second)
            out.change[i] =
                directional(slope[0], a.change[i]) + directional(slope[1], b.change[i]) +
                directional(change[0], a.derivative[i]) + directional(change[1], b.derivative[i]);
    }
    return out;
}

static void replay_edges(const Sweep *s, const TypedTensor *x, const TypedTensor *y,
                         const TypedTensor *z) {
    const ADCheckpoint *c = s->node->checkpoint;
    if (!z->n)
        return;
    size_t bytes = replay_bytes(s->vm, c);
    if (bytes > tensor_memory_available(s->vm)) {
        gcfull(s->vm);
        vmerror_kind(s->vm, "memory_budget",
                     "checkpoint workspace exceeds the tensor memory budget");
    }
    Obj *buffer = buffer_new(s->vm, bytes);
    ReplayWork *work = (ReplayWork *)buffer->as.buffer.data;
    for (int i = 0; i < c->count; i++) {
        const ReplayStep *step = &c->steps[i];
        if (step->op != AD_LEAF)
            for (int j = 0; j < 2; j++)
                work[i + 2].constants[j] = layout(s->vm, step->constant[j]);
    }
    for (size_t i = 0; i < z->n; i++) {
        size_t xi = x->rank ? i : 0, yi = y->rank ? i : 0;
        work[0].state = (ReplayValue){.value = read_value(s->vm, x, s->node->left, xi),
                                      .tangent = tangent_at(s->tangents[0], xi),
                                      .derivative = {s->parents[0] != NULL, 0}};
        work[1].state = (ReplayValue){
            .value = s->node->right.t == V_NIL ? 0 : read_value(s->vm, y, s->node->right, yi),
            .tangent = tangent_at(s->tangents[1], yi),
            .derivative = {0, s->parents[1] != NULL}};
        for (int j = 0; j < c->count; j++) {
            const ReplayStep *step = &c->steps[j];
            if (step->op == AD_LEAF)
                continue;
            ReplayValue args[2] = {0};
            for (int k = 0; k < 2; k++) {
                if (step->input[k] >= 0)
                    args[k] = work[step->input[k]].state;
                else if (step->constant[k].t != V_NIL)
                    args[k].value =
                        read_value(s->vm, &work[j + 2].constants[k], step->constant[k], i);
            }
            work[j + 2].state = replay_step(s->vm, step->op, args[0], args[1], s->adjoint != NULL);
        }
        const ReplayValue *result = &work[c->count + 1].state;
        edge(s, 0, xi, i, result->derivative[0], result->change[0]);
        edge(s, 1, yi, i, result->derivative[1], result->change[1]);
    }
    size_t boundaries = s->node->parents[1] ? 2 : 1;
    offload_record_replay(s->vm, z->n, (size_t)c->count - boundaries);
    tensor_buffer_discard(s->vm, buffer);
}

static void elementwise(const Sweep *s, const TypedTensor *x, const TypedTensor *y,
                        const TypedTensor *z) {
    TensorAD *n = s->node;
    for (size_t i = 0; i < z->n; i++) {
        int index[8];
        coordinates(z, i, index);
        size_t xi = broadcast_index(x, z->rank, index);
        size_t yi = broadcast_index(y, z->rank, index);
        double tx = tangent_at(s->tangents[0], xi);
        double ty = tangent_at(s->tangents[1], yi);
        switch (n->operation) {
        case AD_ADD:
        case AD_SUB:
            edge(s, 0, xi, i, 1, 0);
            edge(s, 1, yi, i, n->operation == AD_SUB ? -1 : 1, 0);
            break;
        case AD_MUL:
            if (s->parents[0])
                edge(s, 0, xi, i, read_value(s->vm, y, n->right, yi), ty);
            if (s->parents[1])
                edge(s, 1, yi, i, read_value(s->vm, x, n->left, xi), tx);
            break;
        case AD_DIV: {
            double denominator = read_value(s->vm, y, n->right, yi);
            if (s->parents[0])
                edge(s, 0, xi, i, 1 / denominator, -(ty / denominator) / denominator);
            if (s->parents[1]) {
                double value = read_value(s->vm, z, n->value, i);
                double change =
                    s->adjoint
                        ? (value * (ty / denominator) - tangent_at(s->out_tangent, i)) / denominator
                        : 0;
                edge(s, 1, yi, i, -value / denominator, change);
            }
            break;
        }
        default: {
            double xv = n->saved & 1 ? read_value(s->vm, x, n->left, xi) : 0;
            double zv = n->saved & 4 ? read_value(s->vm, z, n->value, i) : 0;
            double power = n->operation == AD_POW ? asnum(s->vm, n->right) : 0;
            double change = s->adjoint ? slope_change(n->operation, xv, zv, power, tx,
                                                      tangent_at(s->out_tangent, i))
                                       : 0;
            edge(s, 0, xi, i, unary_derivative(n->operation, xv, zv, power), change);
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

static void matrix_edges(const Sweep *s, const TypedTensor *x, const TypedTensor *y,
                         const TypedTensor *z) {
    for (size_t i = 0; i < z->n; i++) {
        int index[8], left[8], right[8];
        coordinates(z, i, index);
        for (int k = 0; k < x->shape[x->rank - 1]; k++) {
            matrix_index(x, z->rank, index, index[z->rank - 2], k, left);
            matrix_index(y, z->rank, index, k, index[z->rank - 1], right);
            size_t xi = linear_index(x, left), yi = linear_index(y, right);
            if (s->parents[0])
                edge(s, 0, xi, i, saved_item(s->vm, y, yi), tangent_at(s->tangents[1], yi));
            if (s->parents[1])
                edge(s, 1, yi, i, saved_item(s->vm, x, xi), tangent_at(s->tangents[0], xi));
        }
    }
}

static void layout_edges(const Sweep *s, const TypedTensor *x, const TypedTensor *z) {
    TensorAD *n = s->node;
    if (n->operation == AD_SUM || n->operation == AD_MEAN) {
        int axis = n->axis;
        size_t count = axis < 0 ? x->n : (size_t)x->shape[axis];
        double scale = n->operation == AD_MEAN && count ? 1.0 / (double)count : 1;
        for (size_t i = 0; i < x->n; i++) {
            size_t zi = 0;
            if (axis >= 0) {
                int index[8];
                coordinates(x, i, index);
                for (int d = 0; d < x->rank; d++)
                    if (d != axis)
                        zi = zi * (size_t)x->shape[d] + (size_t)index[d];
            }
            edge(s, 0, i, zi, scale, 0);
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
        edge(s, 0, xi, i, 1, 0);
    }
}

static void sparse_edges(const Sweep *s, const TypedTensor *values, const TypedTensor *input) {
    const Sparse *pattern = &s->node->pattern->as.sparse;
    for (uint32_t row = 0; row < pattern->rows; row++) {
        for (size_t i = pattern->offset[row]; i < pattern->offset[row + 1]; i++) {
            uint32_t col = pattern->col[i];
            if (s->parents[0])
                edge(s, 0, i, row, saved_item(s->vm, input, col), tangent_at(s->tangents[1], col));
            if (s->parents[1]) {
                double value = values->rank ? saved_item(s->vm, values, i) : pattern->values[i];
                edge(s, 1, col, row, value, tangent_at(s->tangents[0], i));
            }
        }
    }
}

static void add_saved_read(VM *v, ADReadSpec *spec, Val value) {
    if (is_tensor(value))
        offload_add_read(spec, &layout(v, value).buffer->as.buffer);
}

static ADReadSpec node_reads(VM *v, const TensorAD *n, unsigned mask) {
    ADReadSpec spec = {0};
    if (!mask || !v->tensor_memory.mapped)
        return spec;
    // A checkpoint reconstructs its result from the boundaries and constants.
    if (n->checkpoint) {
        mask &= 3;
        for (int i = 0; i < n->checkpoint->count; i++)
            for (int j = 0; j < 2; j++)
                add_saved_read(v, &spec, n->checkpoint->steps[i].constant[j]);
    }
    Val saved[] = {n->left, n->right, n->value};
    for (unsigned i = 0; i < 3; i++)
        if (mask & (1u << i))
            add_saved_read(v, &spec, saved[i]);
    return spec;
}

static size_t sweep_workspace(VM *v, const TensorAD *n, const TypedTensor *t, unsigned mask) {
    if (!mask || !t->n)
        return 0;
    size_t windows = offload_read_bytes(node_reads(v, n, mask));
    size_t replay = n->checkpoint ? replay_bytes(v, n->checkpoint) : 0;
    if (windows > PTRDIFF_MAX - replay)
        vmerror_kind(v, "memory_budget", "autodiff workspace exceeds addressable storage");
    return windows + replay;
}

static void visit_node(VM *v, TensorAD *n, Obj *stop, Val *work, bool forward, Val *adjoints,
                       Val *tangents) {
    if (!n->active || n->operation == AD_LEAF || (stop && n == stop->as.ad))
        return;
    Sweep s = {.vm = v,
               .node = n,
               .output = work + n->slot,
               .forward = forward,
               .adjoint = adjoints ? adjoints + n->slot : NULL,
               .out_tangent = tangents ? tangents + n->slot : NULL};
    for (int i = 0; i < 2; i++) {
        Obj *parent = n->parents[i];
        if (!parent || !parent->as.ad->active)
            continue;
        int slot = parent->as.ad->slot;
        s.parents[i] = work + slot;
        if (tangents)
            s.tangents[i] = tangents + slot;
    }

    TypedTensor x = layout(v, n->left), y = layout(v, n->right), z = layout(v, n->value);
    ADReadCache cache;
    if (z.n && v->tensor_memory.mapped) {
        unsigned mask = saved_values(n->operation, s.parents[0] != NULL, s.parents[1] != NULL);
        offload_read_start(v, &cache, node_reads(v, n, mask));
    }
    switch (n->operation) {
    case AD_SUM:
    case AD_MEAN:
    case AD_RESHAPE:
    case AD_PERMUTE:
    case AD_VIEW:
    case AD_SLICE:
    case AD_AT:
        layout_edges(&s, &x, &z);
        break;
    case AD_MATMUL:
        matrix_edges(&s, &x, &y, &z);
        break;
    case AD_SPARSE_MATVEC:
        sparse_edges(&s, &x, &y);
        break;
    case AD_CHECKPOINT:
        replay_edges(&s, &x, &y, &z);
        break;
    default:
        elementwise(&s, &x, &y, &z);
        break;
    }
    if (v->ad_read_cache)
        offload_read_finish(v);
}

static bool connected(const Obj *order, const Obj *node) {
    int slot = node->as.ad->slot;
    return slot >= 0 && slot < order->as.list.n && order->as.list.a[slot].o == node;
}

static Obj *prepare(VM *v, Obj *root, Obj *stop, size_t buffers, size_t output_bytes) {
    size_t available = tensor_memory_available(v), scratch_peak = 0;
    if (output_bytes > available)
        vmerror_kind(v, "memory_budget", "autodiff result exceeds the tensor memory budget");
    available -= output_bytes;
    Obj *order = topology(v, root, stop, NULL);
    for (int i = 0; i < order->as.list.n; i++) {
        Obj *object = order->as.list.a[i].o;
        TensorAD *n = object->as.ad;
        if (object != stop && n->operation == AD_CONSUMED)
            vmerror_kind(v, "autodiff", "backward tape was already consumed");
        bool left = object != stop && n->parents[0] && n->parents[0]->as.ad->active;
        bool right = object != stop && n->parents[1] && n->parents[1]->as.ad->active;
        n->active = !stop || object == stop || left || right;
        if (!n->active)
            continue;
        unsigned saved_mask = object == stop ? 0 : saved_values(n->operation, left, right);
        Val saved[] = {n->left, n->right, n->value};
        for (unsigned j = 0; j < 3; j++)
            if ((saved_mask & (1u << j)) && version(v, saved[j]) != n->versions[j])
                vmerror_kind(v, "mutation",
                             "tensor saved for differentiation was modified through an alias");
        TypedTensor t = layout(v, n->value);
        if (object != stop && n->checkpoint)
            checkpoint_versions(v, n->checkpoint);
        size_t scratch = sweep_workspace(v, n, &t, saved_mask);
        if (scratch > scratch_peak)
            scratch_peak = scratch;
        if (t.rank && buffers) {
            size_t per_entry = buffers * sizeof(double);
            if (t.n > available / per_entry)
                vmerror_kind(v, "memory_budget",
                             "autodiff derivatives exceed the tensor memory budget");
            available -= t.n * per_entry;
        }
    }
    if (scratch_peak > available)
        vmerror_kind(v, "memory_budget",
                     "autodiff read and replay workspace exceeds the tensor memory budget");
    return order;
}

static Val zero(VM *v, const TypedTensor *t) {
    return t->rank ? objval(typed_new(v, DT_F64, t->rank, t->shape)) : number(0);
}

static Val *allocate_work(VM *v, const Obj *order) {
    Obj *work = list(v);
    for (int i = 0; i < order->as.list.n; i++) {
        const TensorAD *n = order->as.list.a[i].o->as.ad;
        TypedTensor t = layout(v, n->value);
        append(v, work, n->active ? zero(v, &t) : nil());
    }
    return work->as.list.a;
}

static void reset_work(Val *work, int count) {
    for (int i = 0; i < count; i++) {
        if (work[i].t == V_OBJ)
            memset(data(work[i].o), 0, work[i].o->as.tensor.n * sizeof(double));
        else if (work[i].t == V_NUM)
            work[i] = number(0);
    }
}

static TypedTensor validate_seed(VM *v, const TypedTensor *target, Val seed, const char *name,
                                 bool default_one) {
    if (seed.t == V_NIL && (!default_one || target->rank))
        vmerror_kind(v, "shape", "%s requires an explicit seed for this input", name);
    TypedTensor layout_seed = layout(v, seed);
    if (target->rank != layout_seed.rank ||
        memcmp(target->shape, layout_seed.shape, (size_t)target->rank * sizeof *target->shape))
        vmerror_kind(v, "shape", "%s seed shape mismatch", name);
    if (seed.t != V_NIL)
        ad_validate(v, seed);
    return layout_seed;
}

static void assign_seed(VM *v, Val *out, Val seed, const TypedTensor *layout_seed) {
    if (layout_seed->rank) {
        double *dst = data(out->o);
        for (size_t i = 0; i < layout_seed->n; i++)
            dst[i] = item(layout_seed, i);
    } else
        *out = number(seed.t == V_NIL ? 1 : asnum(v, seed));
}

static void forward_sweep(VM *v, const Obj *order, Obj *stop, Val *work) {
    for (int i = 0; i < order->as.list.n; i++)
        visit_node(v, order->as.list.a[i].o->as.ad, stop, work, true, NULL, NULL);
}

static void reverse_sweep(VM *v, const Obj *order, Obj *stop, Val *work, Val *adjoints,
                          Val *tangents) {
    for (int i = order->as.list.n - 1; i >= 0; i--)
        visit_node(v, order->as.list.a[i].o->as.ad, stop, work, false, adjoints, tangents);
}

static Val single_sweep(VM *v, Obj *root, Obj *input, Val seed, bool forward) {
    Obj *seed_node = forward ? input : root;
    const char *name = forward ? "jvp" : input ? "vjp" : "backward";
    TypedTensor source = layout(v, seed_node->as.ad->value);
    TypedTensor seed_shape = validate_seed(v, &source, seed, name, !forward);
    Obj *order = prepare(v, root, input, 1, 0);
    if (input && !connected(order, input)) {
        TypedTensor target = layout(v, (forward ? root : input)->as.ad->value);
        return zero(v, &target);
    }
    Val *work = allocate_work(v, order);
    assign_seed(v, &work[seed_node->as.ad->slot], seed, &seed_shape);
    if (forward) {
        forward_sweep(v, order, input, work);
        return work[root->as.ad->slot];
    }
    reverse_sweep(v, order, input, work, NULL, NULL);
    if (input)
        return work[input->as.ad->slot];
    for (int i = 0; i < order->as.list.n; i++) {
        Obj *node = order->as.list.a[i].o;
        node->as.ad->grad = work[i];
        node->as.ad->batch_examples = 0;
        barrier(v, node, work[i]);
    }
    return nil();
}

static void discard_cached(VM *v, Obj **cache) {
    if (*cache) {
        tensor_buffer_discard(v, *cache);
        *cache = NULL;
    }
}

static Val live_gradient(VM *v, Val primal, Obj **cache) {
    TypedTensor source = layout(v, primal);
    if (!source.rank)
        return number(0);
    size_t bytes = source.n * sizeof(double);
    if (*cache && (*cache)->as.buffer.bytes == bytes) {
        TypedTensor out = typed_layout(v, DT_F64, source.rank, source.shape);
        out.buffer = *cache;
        *cache = NULL;
        memset(out.buffer->as.buffer.data, 0, bytes);
        return objval(typed_wrap(v, out));
    }
    if (bytes > tensor_memory_available(v))
        discard_cached(v, cache);
    if (bytes > tensor_memory_available(v)) {
        gcfull(v);
        vmerror_kind(v, "memory_budget", "live gradients exceed the tensor memory budget");
    }
    return zero(v, &source);
}

typedef struct {
    const Obj *retain;
    uint64_t samples;
    bool consume;
} BackwardOptions;

static BackwardOptions backward_options(VM *v, Val value) {
    if (!isobj(value, O_MAP))
        vmerror_kind(v, "type", "backward options must be a map");
    const Table *t = &value.o->as.map.tab;
    BackwardOptions options = {0};
    for (int i = 0; i < t->cap; i++) {
        const Entry *e = &t->a[i];
        if (!e->key)
            continue;
        if (!strcmp(e->key, "retain")) {
            if (!isobj(e->val, O_LIST) || !e->val.o->as.list.n)
                vmerror_kind(v, "type", "backward retain must be a nonempty list of tracked nodes");
            options.retain = e->val.o;
        } else if (!strcmp(e->key, "samples")) {
            if (e->val.t != V_INT || e->val.i <= 0)
                vmerror_kind(v, "shape", "backward samples must be a positive integer");
            options.samples = (uint64_t)e->val.i;
        } else if (!strcmp(e->key, "consume")) {
            if (e->val.t != V_BOOL)
                vmerror_kind(v, "type", "backward consume must be a boolean");
            options.consume = e->val.n != 0;
        } else
            vmerror_kind(v, "type", "unknown backward option '%s'", e->key);
    }
    if (!options.retain)
        vmerror_kind(v, "type", "backward options require a retain list");
    return options;
}

static void check_retained(VM *v, Obj *order, BackwardOptions options) {
    for (int i = 0; i < order->as.list.n; i++)
        order->as.list.a[i].o->as.ad->retain = false;
    for (int i = 0; i < options.retain->as.list.n; i++) {
        Val x = options.retain->as.list.a[i];
        if (!isobj(x, O_AD) || !connected(order, x.o))
            vmerror_kind(v, "autodiff", "retained node must be connected to the backward output");
        TensorAD *node = x.o->as.ad;
        if (node->retain)
            vmerror_kind(v, "autodiff", "backward retain contains a duplicate node");
        node->retain = true;
        if (options.samples && node->grad.t != V_NIL && !node->batch_examples)
            vmerror_kind(v, "autodiff",
                         "clear the unweighted gradient before accumulating batches");
        if (node->batch_examples > (uint64_t)INT64_MAX - options.samples)
            vmerror_kind(v, "overflow", "accumulated sample count exceeds Int64");
    }
}

static void blend_batch(VM *v, Val *work, const TensorAD *n, uint64_t samples) {
    if (!samples || !n->batch_examples)
        return;
    long double count = (long double)(n->batch_examples + samples);
    long double old_weight = n->batch_examples / count, new_weight = samples / count;
    if (work->t == V_OBJ) {
        double *dst = data(work->o);
        const double *old = data(n->grad.o);
        for (size_t i = 0; i < work->o->as.tensor.n; i++)
            dst[i] = checked(v, (double)(old_weight * old[i] + new_weight * dst[i]));
    } else
        work->n = checked(v, (double)(old_weight * n->grad.n + new_weight * work->n));
}

/* Release the output's ownership of history after gradients are published.
 * Other owners, including shared paths and retained nodes, stay live. */
static void consume_output(VM *v, Obj *root) {
    TensorAD *n = root->as.ad;
    if (n->operation == AD_LEAF)
        return;
    if (n->checkpoint) {
        account(v, root, recipe_bytes(n->checkpoint), 0);
        free(n->checkpoint);
        n->checkpoint = NULL;
    }
    n->operation = AD_CONSUMED;
    n->left = n->right = nil();
    n->parents[0] = n->parents[1] = n->pattern = NULL;
    gcfull(v);
}

static Val retained_backward(VM *v, Obj *root, Val seed, Val options_value) {
    BackwardOptions options = backward_options(v, options_value);
    TypedTensor target = layout(v, root->as.ad->value);
    TypedTensor seed_shape = validate_seed(v, &target, seed, "backward", true);
    Obj *order = prepare(v, root, NULL, 0, 0);
    check_retained(v, order, options);
    Obj *storage = list(v);
    for (int i = 0; i < order->as.list.n; i++)
        append(v, storage, nil());
    Val *work = storage->as.list.a;
    Obj *cache = NULL;
    int slot = root->as.ad->slot;
    work[slot] = live_gradient(v, root->as.ad->value, &cache);
    assign_seed(v, &work[slot], seed, &seed_shape);
    for (int i = order->as.list.n - 1; i >= 0; i--) {
        TensorAD *node = order->as.list.a[i].o->as.ad;
        for (int j = 0; j < 2; j++) {
            Obj *parent = node->parents[j];
            if (parent && work[parent->as.ad->slot].t == V_NIL)
                work[parent->as.ad->slot] = live_gradient(v, parent->as.ad->value, &cache);
        }
        if (cache && (node->checkpoint || v->tensor_memory.mapped)) {
            TypedTensor primal = layout(v, node->value);
            unsigned mask =
                saved_values(node->operation, node->parents[0] != NULL, node->parents[1] != NULL);
            if (sweep_workspace(v, node, &primal, mask) > tensor_memory_available(v))
                discard_cached(v, &cache);
        }
        visit_node(v, node, NULL, work, false, NULL, NULL);
        // All contributions from downstream nodes have reached this slot.
        if (node->retain)
            blend_batch(v, &work[i], node, options.samples);
        else {
            if (work[i].t == V_OBJ) {
                discard_cached(v, &cache);
                cache = work[i].o->as.tensor.buffer;
            }
            work[i] = nil();
        }
    }
    discard_cached(v, &cache);
    for (int i = 0; i < order->as.list.n; i++) {
        Obj *owner = order->as.list.a[i].o;
        TensorAD *node = owner->as.ad;
        if (node->retain) {
            node->grad = work[i];
            node->batch_examples = options.samples ? node->batch_examples + options.samples : 0;
            barrier(v, owner, work[i]);
        }
    }
    if (options.consume)
        consume_output(v, root);
    return nil();
}

static void seed_basis(Val *seed, size_t i) {
    if (seed->t == V_OBJ)
        data(seed->o)[i] = 1;
    else
        *seed = number(1);
}

static Val hessian_product(VM *v, Obj *root, Obj *input, Val direction) {
    TypedTensor x = layout(v, input->as.ad->value), y = layout(v, root->as.ad->value);
    if (y.rank)
        vmerror_kind(v, "shape", "hvp requires a scalar output");
    TypedTensor seed = validate_seed(v, &x, direction, "hvp", false);
    Obj *order = prepare(v, root, input, 3, 0);
    if (!connected(order, input))
        return zero(v, &x);
    Val *adjoints = allocate_work(v, order);
    Val *tangents = allocate_work(v, order);
    Val *product = allocate_work(v, order);
    int source = input->as.ad->slot, sink = root->as.ad->slot;
    adjoints[sink] = number(1);
    reverse_sweep(v, order, input, adjoints, NULL, NULL);
    assign_seed(v, &tangents[source], direction, &seed);
    forward_sweep(v, order, input, tangents);
    reverse_sweep(v, order, input, product, adjoints, tangents);
    return product[source];
}

static Val derivative_matrix(VM *v, Obj *root, Obj *input, bool hessian) {
    TypedTensor x = layout(v, input->as.ad->value), y = layout(v, root->as.ad->value);
    if (hessian && y.rank)
        vmerror_kind(v, "shape", "hessian requires a scalar output");
    TypedTensor rows = hessian ? x : y;
    int rank = rows.rank + x.rank, axes[8];
    if (rank > 8)
        vmerror_kind(v, "shape", "derivative result rank exceeds eight axes");
    memcpy(axes, rows.shape, (size_t)rows.rank * sizeof *axes);
    memcpy(axes + rows.rank, x.shape, (size_t)x.rank * sizeof *axes);
    TypedTensor result = rank ? typed_layout(v, DT_F64, rank, axes) : (TypedTensor){.n = 1};
    if (!result.n)
        return zero(v, &result);
    size_t bytes = rank ? result.n * sizeof(double) : 0;
    Obj *order = prepare(v, root, input, hessian ? 3 : 1, bytes);
    Val out = zero(v, &result);
    double *values = rank ? data(out.o) : &out.n;
    if (!connected(order, input))
        return out;

    bool forward = !hessian && x.n <= y.n;
    Val *work = allocate_work(v, order);
    Val *tangents = hessian ? allocate_work(v, order) : NULL;
    Val *adjoints = hessian ? allocate_work(v, order) : NULL;
    int n = order->as.list.n, source = input->as.ad->slot, sink = root->as.ad->slot;
    if (hessian) {
        seed_basis(&adjoints[sink], 0);
        reverse_sweep(v, order, input, adjoints, NULL, NULL);
    }
    size_t sweeps = hessian || forward ? x.n : y.n;
    for (size_t i = 0; i < sweeps; i++) {
        reset_work(work, n);
        if (hessian) {
            reset_work(tangents, n);
            seed_basis(&tangents[source], i);
            forward_sweep(v, order, input, tangents);
            reverse_sweep(v, order, input, work, adjoints, tangents);
            for (size_t j = 0; j < x.n; j++)
                values[j * x.n + i] = grad_value(work[source], j);
        } else if (forward) {
            seed_basis(&work[source], i);
            forward_sweep(v, order, input, work);
            for (size_t j = 0; j < y.n; j++)
                values[j * x.n + i] = grad_value(work[sink], j);
        } else {
            seed_basis(&work[sink], i);
            reverse_sweep(v, order, input, work, NULL, NULL);
            for (size_t j = 0; j < x.n; j++)
                values[i * x.n + j] = grad_value(work[source], j);
        }
    }
    return out;
}

enum {
    VARIABLE,
    VALUE,
    GRAD,
    BACKWARD,
    SIN,
    COS,
    EXP,
    LOG,
    SQRT,
    TANH,
    POW,
    ZERO_GRAD,
    DETACH,
    NO_GRAD,
    VJP,
    JVP,
    JACOBIAN,
    HESSIAN,
    CHECKPOINT,
    OFFLOAD_PLAN,
    OFFLOAD,
    OFFLOAD_STATS,
    HVP
};
static const char *const names[] = {
    "variable", "value",   "grad",       "backward",     "sin",     "cos",           "exp", "log",
    "sqrt",     "tanh",    "pow",        "zero_grad",    "detach",  "no_grad",       "vjp", "jvp",
    "jacobian", "hessian", "checkpoint", "offload_plan", "offload", "offload_stats", "hvp"};
static const int arities[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 3, 2, 2, 2, 2, 2, 1, 3};
_Static_assert(sizeof names / sizeof *names == sizeof arities / sizeof *arities, "ad arities");
_Static_assert(AD_BASE + NO_GRAD == AD_NO_GRAD_ID, "no_grad native ID");

Val ad_call(VM *v, int id, int argc, Val *args) {
    int op = id - AD_BASE;
    if (op < 0 || (size_t)op >= sizeof names / sizeof *names ||
        (argc != arities[op] && !(op == BACKWARD && (argc == 2 || argc == 3)) &&
         !((op == VJP || op == CHECKPOINT || op == OFFLOAD_PLAN) && argc == 3)))
        vmerror(v, "invalid autodiff function arity");
    if (op == VARIABLE) {
        if (ad_disabled(v))
            return ad_primal(args[0]);
        if (isobj(args[0], O_AD))
            vmerror(v, "expected ordinary input to ad.variable");
        Val input = is_tensor(args[0]) ? args[0] : number(asnum(v, args[0]));
        return ad_record(v, AD_LEAF, input, nil(), nil(), 0, NULL);
    }
    if (op == VALUE || op == DETACH)
        return ad_primal(args[0]);
    if (op >= SIN && op <= POW) {
        static const ADOperation rules[] = {AD_SIN,  AD_COS,  AD_EXP, AD_LOG,
                                            AD_SQRT, AD_TANH, AD_POW};
        return ad_unary(v, rules[op - SIN], args[0], argc == 2 ? args[1] : nil());
    }
    if (op == NO_GRAD)
        vmerror(v, "ad.no_grad must be invoked as a callable");
    if (op == OFFLOAD_PLAN)
        return offload_plan_new(v, args[0], args[1], argc == 3 ? args[2] : integer(8192));
    if (op == OFFLOAD_STATS)
        return offload_stats(v, offload_plan(v, args[0]));
    if (!isobj(args[0], O_AD) ||
        (((op >= VJP && op <= CHECKPOINT) || op == HVP) && !isobj(args[1], O_AD)))
        vmerror(v, "expected autodiff node");
    Obj *root = args[0].o;
    if (op == CHECKPOINT)
        return ad_disabled(v) ? root->as.ad->value
                              : checkpoint(v, root, args[1], argc == 3 ? args[2] : nil());
    if (op == OFFLOAD)
        return ad_disabled(v) ? root->as.ad->value : offload_activation(v, root, args[1]);
    if (op == ZERO_GRAD) {
        root->as.ad->grad = nil();
        root->as.ad->batch_examples = 0;
        return nil();
    }
    if (op == GRAD) {
        Val result = root->as.ad->grad;
        if (is_tensor(result)) {
            TypedTensor t = typed_tensor(v, result);
            t.readonly = true;
            return objval(typed_wrap(v, t));
        }
        return result;
    }
    if (op == BACKWARD) {
        if (argc == 3)
            return retained_backward(v, root, args[1], args[2]);
        return single_sweep(v, root, NULL, argc == 2 ? args[1] : nil(), false);
    }
    if (op == VJP || op == JVP)
        return single_sweep(v, root, args[1].o, argc == 3 ? args[2] : nil(), op == JVP);
    if (op == HVP)
        return hessian_product(v, root, args[1].o, args[2]);
    return derivative_matrix(v, root, args[1].o, op == HESSIAN);
}

Val ad_sparse_matvec(VM *v, Obj *pattern, Val values, Val input) {
    const Sparse *p = &pattern->as.sparse;
    TypedTensor x = layout(v, ad_primal(input)), weights = layout(v, ad_primal(values));
    if (x.rank != 1 || x.n != p->cols)
        vmerror_kind(v, "shape", "sparse.matvec input must match its column count");
    if (values.t != V_NIL && (weights.rank != 1 || weights.n != p->nnz))
        vmerror_kind(v, "shape", "sparse.matvec values must match its stored CSR entries");
    ad_validate(v, input);
    if (weights.rank)
        ad_validate(v, values);
    if (p->rows > ARDA_MAX_ELEMENTS)
        vmerror_kind(v, "shape", "sparse.matvec output exceeds the tensor element limit");
    int rows = (int)p->rows;
    Obj *out = typed_new(v, DT_F64, 1, &rows);
    double *dst = data(out);
    for (uint32_t row = 0; row < p->rows; row++) {
        long double sum = 0;
        for (size_t i = p->offset[row]; i < p->offset[row + 1]; i++) {
            double coefficient = weights.rank ? item(&weights, i) : p->values[i];
            sum += (long double)coefficient * item(&x, p->col[i]);
        }
        dst[row] = checked(v, (double)sum);
    }
    if (!isobj(values, O_AD) && !isobj(input, O_AD))
        return objval(out);
    Val result = ad_record(v, AD_SPARSE_MATVEC, objval(out), values, input, 0, NULL);
    result.o->as.ad->pattern = pattern;
    barrier(v, result.o, objval(pattern));
    return result;
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
