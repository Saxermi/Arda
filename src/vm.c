#include "compute_topology.h"
#include "autodiff.h"
#include "activation_store.h"
#include "cpu_kernels.h"
#include "numerics_source.h"
#include "ode_source.h"
#include "sim_source.h"
#include "opt_source.h"
#include "uncertainty_source.h"
static _Noreturn void jump_error(VM *v) {
    offload_read_finish(v);
    if (v->sparse_active)
        sparse_abort(v);
    solver_unwind(v);
    longjmp(v->trap, 1);
}

_Noreturn void vmerror_kind(VM *v, const char *code, const char *fmt, ...) {
    va_list a;
    va_start(a, fmt);
    vsnprintf(v->error, sizeof v->error, fmt, a);
    va_end(a);
    v->error_code = code;
    v->thrown = nil();
    jump_error(v);
}

_Noreturn void vmerror(VM *v, const char *fmt, ...) {
    va_list a;
    va_start(a, fmt);
    vsnprintf(v->error, sizeof v->error, fmt, a);
    va_end(a);
    v->error_code = "runtime_error";
    v->thrown = nil();
    jump_error(v);
}

int truth(VM *v, Val x) {
    if (isobj(x, O_AD) && ad_disabled(v))
        x = ad_primal(x);
    if (isobj(x, O_ARRAY) || isobj(x, O_TYPED_TENSOR) || isobj(x, O_LOWRANK) ||
        isobj(x, O_SPARSE) || isobj(x, O_AD) || isobj(x, O_QUANTITY) || isobj(x, O_QUANTIZED))
        vmerror_kind(v, "ambiguous_truth",
                     "tensor, quantity or differentiated value needs an explicit scalar condition");
    if (x.t == V_NIL || x.t == V_MISSING)
        return 0;
    if (x.t == V_INT)
        return x.i != 0;
    if (x.t == V_BOOL || x.t == V_NUM)
        return x.n != 0;
    if (isobj(x, O_COMPLEX))
        return x.o->as.cplx.re != 0 || x.o->as.cplx.im != 0;
    if (isobj(x, O_STRING))
        return x.o->as.str.len != 0;
    if (isobj(x, O_LIST))
        return x.o->as.list.n != 0;
    if (isobj(x, O_MAP))
        return x.o->as.map.tab.n != 0;
    if (isobj(x, O_RANGE))
        return x.o->as.range.count != 0;
    return 1;
}

static Val caught_error(VM *v) {
    if (v->thrown.t != V_NIL)
        return v->thrown;
    Obj *out = newobj(v, O_MAP);
    Table *tab = &out->as.map.tab;
    tableset(v, out, tab, "code", objval(string(v, v->error_code)));
    tableset(v, out, tab, "message", objval(string(v, v->error)));
    tableset(v, out, tab, "file", objval(string(v, v->path ? v->path : "")));
    tableset(v, out, tab, "line", integer(v->line));
    return objval(out);
}

static void reserve(VM *v, int n) {
    if (n > 1000000)
        vmerror(v, "operand stack limit");
    if (n > v->scap) {
        int cap = v->scap ? v->scap : 128;
        while (cap < n)
            cap *= 2;
        v->stack = resize(v->stack, sizeof(Val) * (size_t)cap);
        v->scap = cap;
    }
}

static void push(VM *v, Val x) {
    reserve(v, v->sp + 1);
    v->stack[v->sp++] = x;
}

static Val pop(VM *v) {
    return v->stack[--v->sp];
}

static Val top(VM *v) {
    return v->stack[v->sp - 1];
}

static Val *envslot(Obj *e, const char *s) {
    Proto *p = e->as.env.params;
    if (p)
        for (int i = 0; i < p->arity; i++)
            if (!strcmp(p->params[i], s))
                return &e->as.env.slots[i];
    return NULL;
}

static Val *lookup(Obj *env, Constant *k) {
    const char *s = k->s;
    for (Obj *e = env; e; e = e->as.env.parent) {
        Val *slot = envslot(e, s);
        if (slot)
            return slot;
        Entry *x = tablefind_cached(&e->as.env.tab, k);
        if (x)
            return &x->val;
    }
    return NULL;
}

int vm_uses_builtin(VM *v, Obj *env, const char *name) {
    Constant key = {.s = (char *)name, .hash = hashstr(name)};
    Val *current = lookup(env, &key);
    Entry *builtin = tablefind(&v->builtins->as.env.tab, name);
    return builtin && current && isobj(*current, O_NATIVE) && current->o == builtin->val.o;
}

static Val get(VM *v, Obj *e, Constant *k) {
    Val *x = lookup(e, k);
    if (!x || x->t == V_MISSING)
        vmerror(v, "undefined name '%s'", k->s);
    return *x;
}

static void set(VM *v, Obj *env, Constant *k, Val x) {
    const char *s = k->s;
    for (Obj *e = env; e && e != v->builtins; e = e->as.env.parent) {
        Val *slot = envslot(e, s);
        if (slot) {
            *slot = x;
            if (v->marking)
                barrier(v, e, x);
            return;
        }
        Entry *a = tablefind_cached(&e->as.env.tab, k);
        if (a) {
            a->val = x;
            if (v->marking)
                barrier(v, e, x);
            return;
        }
    }
    vmerror(v, "undefined name '%s' (declare with esta)", s);
}

static void define(VM *v, Obj *env, const char *s, Val x) {
    Val *slot = envslot(env, s);
    if (slot) {
        *slot = x;
        barrier(v, env, x);
        return;
    }
    tableset(v, env, &env->as.env.tab, s, x);
}

static int numeric(Val a) {
    return a.t == V_INT || a.t == V_NUM;
}

static int compare_numeric(Val a, Val b) {
    if (a.t == V_INT && b.t == V_INT)
        return (a.i > b.i) - (a.i < b.i);
    if (a.t == V_NUM && b.t == V_NUM)
        return isnan(a.n) || isnan(b.n) ? 2 : (a.n > b.n) - (a.n < b.n);
    if (a.t == V_NUM) {
        int r = compare_numeric(b, a);
        return r == 2 ? 2 : -r;
    }
    double d = b.n;
    if (isnan(d))
        return 2;
    if (d >= 0x1p63)
        return -1;
    if (d < -0x1p63)
        return 1;
    int64_t i = (int64_t)d;
    if (a.i != i)
        return a.i > i ? 1 : -1;
    double z = (double)i;
    return (z > d) - (z < d);
}

static int eq(Val a, Val b) {
    if (numeric(a) && numeric(b))
        return compare_numeric(a, b) == 0;
    if (isobj(a, O_COMPLEX)) {
        if (isobj(b, O_COMPLEX))
            return a.o->as.cplx.re == b.o->as.cplx.re && a.o->as.cplx.im == b.o->as.cplx.im;
        return numeric(b) && a.o->as.cplx.im == 0 &&
               compare_numeric(number(a.o->as.cplx.re), b) == 0;
    }
    if (isobj(b, O_COMPLEX))
        return eq(b, a);
    if (a.t != b.t)
        return 0;
    if (a.t == V_NIL)
        return 1;
    if (a.t == V_NUM || a.t == V_BOOL)
        return a.n == b.n;
    if (isobj(a, O_STRING) && isobj(b, O_STRING))
        return a.o == b.o;
    return a.o == b.o;
}

int64_t checked_arith(VM *v, Op op, int64_t x, int64_t y) {
#if defined(__GNUC__) || defined(__clang__)
    int64_t r;
    int overflow = op == OP_ADD   ? __builtin_add_overflow(x, y, &r)
                   : op == OP_SUB ? __builtin_sub_overflow(x, y, &r)
                                  : __builtin_mul_overflow(x, y, &r);
    if (overflow)
        vmerror(v, "integer overflow");
    return r;
#else
    if (op == OP_ADD) {
        if ((y > 0 && x > INT64_MAX - y) || (y < 0 && x < INT64_MIN - y))
            vmerror(v, "integer overflow");
        return x + y;
    }
    if (op == OP_SUB) {
        if ((y < 0 && x > INT64_MAX + y) || (y > 0 && x < INT64_MIN + y))
            vmerror(v, "integer overflow");
        return x - y;
    }
    if (x && y &&
        ((x > 0 && y > 0 && x > INT64_MAX / y) || (x > 0 && y < 0 && y < INT64_MIN / x) ||
         (x < 0 && y > 0 && x < INT64_MIN / y) || (x < 0 && y < 0 && x < INT64_MAX / y)))
        vmerror(v, "integer overflow");
    return x * y;
#endif
}

static Val binary(VM *v, Op op, Val a, Val b) {
    if (isobj(a, O_AD) || isobj(b, O_AD)) {
        if (!ad_disabled(v))
            return ad_binary(v, op, a, b);
        a = ad_primal(a);
        b = ad_primal(b);
    }
    if (op == OP_EQ || op == OP_NE)
        return boolean(eq(a, b) ^ (op == OP_NE));
    if (isobj(a, O_QUANTITY) || isobj(b, O_QUANTITY))
        return units_binary(v, op, a, b);
    if (isobj(a, O_TYPED_TENSOR) || isobj(b, O_TYPED_TENSOR))
        return typed_tensor_binary(v, op, a, b);
    if (isobj(a, O_ARRAY) || isobj(b, O_ARRAY))
        return array_binary(v, op, a, b);
    if (op == OP_ADD && (isobj(a, O_STRING) || isobj(b, O_STRING))) {
        char *x = value_string(a), *y = value_string(b);
        size_t n = strlen(x) + strlen(y) + 1;
        char *s = mem(n);
        snprintf(s, n, "%s%s", x, y);
        Obj *o = string(v, s);
        free(s);
        free(x);
        free(y);
        return objval(o);
    }
    if (isobj(a, O_COMPLEX) || isobj(b, O_COMPLEX))
        return complex_binary(v, op, a, b);
    if (a.t == V_INT && b.t == V_INT && op != OP_DIV) {
        int64_t x = a.i, y = b.i;
        switch (op) {
        case OP_ADD:
        case OP_SUB:
        case OP_MUL:
            return integer(checked_arith(v, op, x, y));
        case OP_MOD:
            if (!y)
                vmerror(v, "modulo by zero");
            return integer(x == INT64_MIN && y == -1 ? 0 : x % y);
        case OP_LT:
            return boolean(x < y);
        case OP_LE:
            return boolean(x <= y);
        case OP_GT:
            return boolean(x > y);
        case OP_GE:
            return boolean(x >= y);
        default:
            break;
        }
    }
    double x = asnum(v, a), y = asnum(v, b);
    switch (op) {
    case OP_ADD:
        return number(x + y);
    case OP_SUB:
        return number(x - y);
    case OP_MUL:
        return number(x * y);
    case OP_DIV:
        return number(x / y);
    case OP_MOD:
        return number(fmod(x, y));
    case OP_LT:
        return boolean(compare_numeric(a, b) == -1);
    case OP_LE:
        return boolean(compare_numeric(a, b) <= 0);
    case OP_GT:
        return boolean(compare_numeric(a, b) == 1);
    case OP_GE:
        return boolean((compare_numeric(a, b) == 1 || compare_numeric(a, b) == 0));
    default:
        vmerror(v, "invalid binary opcode");
    }
    return nil();
}

static Val indexget(VM *v, Val a, Val b) {
    if (isobj(a, O_ARRAY))
        return array_index(v, a, b);
    if (isobj(a, O_RANGE)) {
        if (b.t != V_INT)
            vmerror_kind(v, "index", "range index must be an integer");
        uint64_t count = a.o->as.range.count;
        int64_t i = b.i < 0 ? b.i + (int64_t)count : b.i;
        if (i < 0 || (uint64_t)i >= count)
            vmerror_kind(v, "index", "range index out of bounds");
        return integer(range_element(a.o, (uint64_t)i));
    }
    if (isobj(a, O_LIST))
        return a.o->as.list.a[indexnum(v, b, a.o->as.list.n)];
    if (isobj(a, O_MAP)) {
        const char *k = asstr(v, b);
        Entry *e = tablefind(&a.o->as.map.tab, k);
        if (!e)
            vmerror(v, "missing map key '%s'", k);
        return e->val;
    }
    if (isobj(a, O_STRING)) {
        if (a.o->as.str.len > INT_MAX)
            vmerror(v, "string too large");
        int i = indexnum(v, b, utf8len(a.o->as.str.s));
        const char *p = utf8at(a.o->as.str.s, i);
        char s[5] = {0};
        memcpy(s, p, (size_t)utf8width(p));
        return objval(string(v, s));
    }
    vmerror(v, "value is not indexable");
    return nil();
}

static void indexset(VM *v, Val a, Val b, Val x) {
    if (isobj(a, O_ARRAY)) {
        array_set_index(v, a, b, x);
        return;
    }
    if (isobj(a, O_LIST)) {
        a.o->as.list.a[indexnum(v, b, a.o->as.list.n)] = x;
        barrier(v, a.o, x);
        return;
    }
    if (isobj(a, O_MAP)) {
        tableset(v, a.o, &a.o->as.map.tab, asstr(v, b), x);
        return;
    }
    vmerror(v, "index assignment expects list or map");
}

static int slice_bound(VM *v, Val x, int n, int fallback) {
    if (x.t == V_NIL)
        return fallback;
    long double d = x.t == V_INT ? (long double)x.i : asnum(v, x);
    if (!isfinite(d) || truncl(d) != d)
        vmerror(v, "slice index must be integer");
    if (d < 0)
        d += n;
    return d < 0 ? 0 : d > n ? n : (int)d;
}

static Val slicing(VM *v, Val a, Val start, Val end) {
    int n;
    if (isobj(a, O_LIST))
        n = a.o->as.list.n;
    else if (isobj(a, O_STRING))
        n = utf8len(a.o->as.str.s);
    else
        vmerror(v, "slice expects list or string");
    int i = slice_bound(v, start, n, 0), j = slice_bound(v, end, n, n);
    if (j < i)
        j = i;
    if (isobj(a, O_LIST)) {
        Obj *o = list(v);
        for (; i < j; i++)
            append(v, o, a.o->as.list.a[i]);
        return objval(o);
    }
    const char *p = utf8at(a.o->as.str.s, i), *q = utf8at(p, j - i);
    char *s = slice(p, (size_t)(q - p));
    Obj *o = string(v, s);
    free(s);
    return objval(o);
}

static Entry *method(Obj *cls, const char *k) {
    for (Obj *c = cls; c; c = c->as.cls.parent) {
        Entry *e = tablefind(&c->as.cls.methods, k);
        if (e)
            return e;
    }
    return NULL;
}

static Val attr(VM *v, Val x, const char *k) {
    Entry *e = NULL;
    if (isobj(x, O_INSTANCE)) {
        e = tablefind(&x.o->as.inst.fields, k);
        if (e)
            return e->val;
        e = method(x.o->as.inst.cls, k);
        if (e) {
            Obj *b = newobj(v, O_BOUND);
            b->as.bound.fn = e->val.o;
            b->as.bound.self = x;
            return objval(b);
        }
    } else if (isobj(x, O_MODULE)) {
        if (k[0] == '_')
            vmerror(v, "private module attribute '%s'", k);
        e = tablefind(&x.o->as.mod.env->as.env.tab, k);
        if (e)
            return e->val;
    } else if (isobj(x, O_CLASS)) {
        e = method(x.o, k);
        if (e)
            return e->val;
    }
    Val documentation = mathdoc_attribute(v, x, k);
    if (documentation.t != V_NIL)
        return documentation;
    vmerror(v, "no attribute '%s'", k);
    return nil();
}

static void set_attr(VM *v, Val x, const char *k, Val y) {
    if (isobj(x, O_INSTANCE)) {
        tableset(v, x.o, &x.o->as.inst.fields, k, y);
        return;
    }
    if (isobj(x, O_MODULE)) {
        if (k[0] == '_')
            vmerror(v, "private module attribute '%s'", k);
        define(v, x.o->as.mod.env, k, y);
        return;
    }
    vmerror(v, "attribute assignment expects instance or module");
}

void remember_proto(VM *v, Proto *p) {
    if (v->np == v->pcap) {
        v->pcap = v->pcap ? 2 * v->pcap : 8;
        v->programs = resize(v->programs, sizeof(Proto *) * (size_t)v->pcap);
    }
    v->programs[v->np++] = p;
}

void vm_init(VM *v, int stress) {
    v->stress = stress;
    v->cpu_features = cpu_detect();
    v->native_threads = v->last_matmul_threads = 1;
    v->next_gc = 65536;
    v->sparse_limit = SIZE_MAX;
    v->tensor_memory.limit = tensor_memory_unlimited();
    v->builtins = environment(v, NULL);
    v->global = environment(v, v->builtins);
    install_builtins(v);
}

void vm_destroy(VM *v) {
    destroy_objects(v);
    tensor_pool_trim(v);
    free(v->compute_state);
    compute_release_affinity(v);
    for (int i = 0; i < v->np; i++)
        freeproto(v->programs[i]);
    free(v->programs);
    free(v->stack);
    free(v->handlers);
}

static void enter(VM *v, Proto *p, Obj *env, int base, Obj *module, Val constructed) {
    if (v->nf == 512)
        vmerror(v, "call depth limit (512)");
    bool no_grad = ad_disabled(v);
    v->frames[v->nf++] = (Frame){.p = p,
                                 .no_grad = no_grad,
                                 .env = env,
                                 .locals = env,
                                 .base = base,
                                 .module = module,
                                 .constructed = constructed};
}

static void enter_fn(VM *v, Obj *f, int base, int argc, Val constructed) {
    Proto *p = f->as.func.p;
    if (argc < p->required || argc > p->arity)
        vmerror(v, "%s expects %d arguments (minimum %d), got %d", p->name, p->arity, p->required,
                argc);
    Obj *env = newobj_extra(v, O_ENV, sizeof(Val) * (size_t)p->arity);
    env->as.env.parent = f->as.func.env;
    env->as.env.params = p;
    env->as.env.slots = (Val *)(env + 1);
    for (int i = 0; i < p->arity; i++)
        env->as.env.slots[i] = i < argc ? v->stack[base + 1 + i] : (Val){.t = V_MISSING};
    v->sp = base;
    enter(v, p, env, base, NULL, constructed);
}

static void insert_self(VM *v, int base, int argc, Obj *fn, Val self) {
    reserve(v, v->sp + 1);
    memmove(&v->stack[base + 2], &v->stack[base + 1], sizeof(Val) * (size_t)argc);
    v->stack[base] = objval(fn);
    v->stack[base + 1] = self;
    v->sp++;
}

static void invoke(VM *v, int argc);

static void enter_sensitivity(VM *v, int base, int argc) {
    if (argc != 2)
        vmerror(v, "getSensitivity expects a function and an argument list");
    if (!v->sensitivity_entry) {
        static const char wrapper[] = "def sensitivity(target, args):\n"
                                      "    import \"std:uncertainty\" as uncertainty\n"
                                      "    return uncertainty.sensitivity(target, args)\n";
        Proto *root = NULL;
        char error[256];
        if (prepare_source(wrapper, "<stdlib:getSensitivity>", 0, &root, error, sizeof error))
            vmerror(v, "%s", error);
        remember_proto(v, root);
        for (int i = 0; i < root->nk; i++) {
            if (root->k[i].type != K_PROTO)
                continue;
            Obj *entry = newobj(v, O_FUNC);
            entry->as.func.p = root->k[i].p;
            entry->as.func.env = v->builtins;
            v->sensitivity_entry = entry;
            break;
        }
        if (!v->sensitivity_entry)
            vmerror(v, "missing getSensitivity wrapper");
    }
    enter_fn(v, v->sensitivity_entry, base, argc, nil());
}

static void call_with_list(VM *v, int base, int argc) {
    if (argc != 2 || !isobj(v->stack[base + 2], O_LIST))
        vmerror_kind(v, "type", "uncertainty callback requires a function and an argument list");
    Val target = v->stack[base + 1];
    if (!(isobj(target, O_FUNC) || isobj(target, O_BOUND) ||
          (isobj(target, O_NATIVE) && target.o->as.native.id != UNCERTAINTY_APPLY_ID &&
           target.o->as.native.id != AD_NO_GRAD_ID)))
        vmerror_kind(v, "type", "uncertainty requires a callable model");
    Obj *arguments = v->stack[base + 2].o;
    int count = arguments->as.list.n;
    if (count > 1000000 - base - 1)
        vmerror(v, "operand stack limit");
    reserve(v, base + 1 + count);
    v->stack[base] = target;
    if (count)
        memcpy(v->stack + base + 1, arguments->as.list.a, (size_t)count * sizeof(Val));
    v->sp = base + 1 + count;
    invoke(v, count);
}

static void call_native(VM *v, Obj *function, int base, int argc) {
    int id = function->as.native.id;
    if (id == MATHDOC_SENSITIVITY_ID) {
        enter_sensitivity(v, base, argc);
        return;
    }
    if (id == UNCERTAINTY_APPLY_ID) {
        call_with_list(v, base, argc);
        return;
    }
    Val *args = &v->stack[base + 1];
    if (ad_disabled(v) && (id < AD_BASE || id >= HIGH_BASE))
        for (int i = 0; i < argc; i++)
            args[i] = ad_primal(args[i]);
    Val result = native_call(v, id, argc, args);
    v->sp = base;
    push(v, result);
}

static void invoke(VM *v, int argc) {
    int base = v->sp - argc - 1;
    Val callee = v->stack[base];
    if (isobj(callee, O_NATIVE)) {
        call_native(v, callee.o, base, argc);
    } else if (isobj(callee, O_FUNC))
        enter_fn(v, callee.o, base, argc, nil());
    else if (isobj(callee, O_BOUND)) {
        Obj *f = callee.o->as.bound.fn;
        Val self = callee.o->as.bound.self;
        insert_self(v, base, argc, f, self);
        if (f->type == O_NATIVE)
            call_native(v, f, base, argc + 1);
        else
            enter_fn(v, f, base, argc + 1, nil());
    } else if (isobj(callee, O_CLASS)) {
        Obj *o = newobj(v, O_INSTANCE);
        o->as.inst.cls = callee.o;
        Val self = objval(o);
        Entry *init = method(callee.o, "init");
        if (!init)
            init = method(callee.o, "__init__");
        if (init) {
            Obj *f = init->val.o;
            insert_self(v, base, argc, f, self);
            enter_fn(v, f, base, argc + 1, self);
        } else {
            if (argc)
                vmerror(v, "class %s has no constructor", callee.o->as.cls.name);
            v->sp = base;
            push(v, self);
        }
    } else
        vmerror(v, "attempt to call non-callable value");
}

static void call(VM *v, int argc) {
    int base = v->sp - argc - 1;
    Val callee = v->stack[base];
    if (!isobj(callee, O_NATIVE) || callee.o->as.native.id != AD_NO_GRAD_ID) {
        invoke(v, argc);
        return;
    }
    Frame *caller = &v->frames[v->nf - 1];
    bool previous = caller->no_grad;
    do {
        if (!argc)
            vmerror_kind(v, "type", "ad.no_grad expects a callable");
        argc--;
        memmove(&v->stack[base], &v->stack[base + 1], sizeof(Val) * (size_t)(argc + 1));
        v->sp--;
        callee = v->stack[base];
    } while (isobj(callee, O_NATIVE) && callee.o->as.native.id == AD_NO_GRAD_ID);
    caller->no_grad = true;
    invoke(v, argc);
    caller->no_grad = previous;
}

static void addhandler(VM *v, Frame *f, int ip) {
    if (v->nh == v->hcap) {
        v->hcap = v->hcap ? 2 * v->hcap : 16;
        v->handlers = resize(v->handlers, sizeof(Handler) * (size_t)v->hcap);
    }
    v->handlers[v->nh++] = (Handler){v->nf - 1, ip, v->sp, f->env, f->no_grad};
}

static void module_import(VM *v, Frame *f, const char *name) {
    bool standard = !strncmp(name, "std:", 4);
    if (standard)
        name += 4;
    size_t n = strlen(name);
    bool ext = n >= 5 && !strcmp(name + n - 5, ".arda");
    bool bare = !ext && !strchr(name, '/');
    const char *embedded = NULL;
    char *path = NULL;
    static const struct {
        const char *name, *path, *source;
    } stdlib[] = {
        {"ode", "<stdlib:ode>", ode_source},
        {"sim", "<stdlib:sim>", sim_source},
        {"opt", "<stdlib:opt>", opt_source},
        {"numerics", "<stdlib:numerics>", numerics_source},
        {"uncertainty", "<stdlib:uncertainty>", uncertainty_source},
    };
    if (standard || bare) {
        for (size_t i = 0; i < sizeof stdlib / sizeof *stdlib; i++)
            if (!strcmp(name, stdlib[i].name)) {
                path = copystr(stdlib[i].path);
                embedded = stdlib[i].source;
                break;
            }
        if (!path) {
            Obj *mod = native_module(v, name);
            if (mod) {
                push(v, objval(mod));
                return;
            }
            if (standard)
                vmerror_kind(v, "import", "standard module '%s' not found", name);
        }
    }
    if (!path) {
        const char *slash = strrchr(f->p->path, '/');
        size_t dir = slash && name[0] != '/' ? (size_t)(slash - f->p->path + 1) : 0;
        char *candidate = mem(dir + n + 6);
        if (dir)
            memcpy(candidate, f->p->path, dir);
        strcpy(candidate + dir, name);
        if (!ext)
            strcat(candidate, ".arda");
        path = realpath(candidate, NULL);
        free(candidate);
        if (!path)
            vmerror_kind(v, "import", "module '%s' not found", name);
    }
    Entry *e = tablefind(&v->modules, path);
    if (e) {
        Obj *o = e->val.o;
        free(path);
        if (o->as.mod.state == 0)
            vmerror(v, "circular import '%s'", name);
        if (o->as.mod.state == 2)
            vmerror(v, "module '%s' previously failed", name);
        push(v, objval(o));
        return;
    }
    char *src = embedded ? copystr(embedded) : readfile(path);
    if (!src) {
        free(path);
        vmerror(v, "cannot read module '%s'", name);
    }
    Proto *p = NULL;
    char error[1024];
    int failed = prepare_source(src, path, 0, &p, error, sizeof error);
    free(src);
    if (failed) {
        free(path);
        vmerror(v, "%s", error);
    }
    remember_proto(v, p);
    Obj *o = newobj(v, O_MODULE);
    o->as.mod.path = path;
    account(v, o, 0, strlen(path) + 1);
    o->as.mod.env = environment(v, v->builtins);
    tableset(v, NULL, &v->modules, path, objval(o));
    if (v->nf == 512) {
        o->as.mod.state = 2;
        vmerror(v, "call depth limit (512)");
    }
    enter(v, p, o->as.mod.env, v->sp, o, nil());
}

static void return_frame(VM *v, Val x) {
    Frame f = v->frames[v->nf - 1];
    while (v->nh && v->handlers[v->nh - 1].frame >= v->nf - 1)
        v->nh--;
    v->nf--;
    v->sp = f.base;
    if (f.module) {
        f.module->as.mod.state = 1;
        x = objval(f.module);
    }
    if (f.constructed.t != V_NIL)
        x = f.constructed;
    if (v->nf)
        push(v, x);
}

static void fail_modules(VM *v, int keep) {
    for (int i = v->nf - 1; i >= keep; i--)
        if (v->frames[i].module)
            v->frames[i].module->as.mod.state = 2;
}

static void dispatch(VM *v) {
    while (v->nf) {
        Frame *f = &v->frames[v->nf - 1];
        Instr in = f->p->code[f->ip++];
        v->line = in.line;
        v->path = f->p->path;
        int a = in.arg;
        switch (in.op) {
        case OP_CONST: {
            Constant k = f->p->k[a];
            push(v, k.type == K_INT   ? integer(k.i)
                    : k.type == K_NUM ? number(k.n)
                                      : objval(string(v, k.s)));
            break;
        }
        case OP_NIL:
            push(v, nil());
            break;
        case OP_TRUE:
            push(v, boolean(1));
            break;
        case OP_FALSE:
            push(v, boolean(0));
            break;
        case OP_GET_LOCAL: {
            Val x = f->locals->as.env.slots[a];
            if (x.t == V_MISSING && f->p->code[f->ip].op != OP_JPRESENT)
                vmerror(v, "parameter not initialized");
            push(v, x);
            break;
        }
        case OP_SET_LOCAL:
            f->locals->as.env.slots[a] = top(v);
            barrier(v, f->locals, top(v));
            break;
        case OP_SET_LOCAL_DROP: {
            Val x = pop(v);
            f->locals->as.env.slots[a] = x;
            if (v->marking)
                barrier(v, f->locals, x);
            break;
        }
        case OP_JPRESENT:
            if (pop(v).t != V_MISSING)
                f->ip = a;
            break;
        case OP_GET:
            push(v, get(v, f->env, &f->p->k[a]));
            break;
        case OP_DEFINE:
            define(v, f->env, f->p->k[a].s, pop(v));
            break;
        case OP_SET:
            set(v, f->env, &f->p->k[a], top(v));
            break;
        case OP_SET_DROP:
            set(v, f->env, &f->p->k[a], pop(v));
            break;
        case OP_POP:
            pop(v);
            break;
        case OP_DUP:
            push(v, top(v));
            break;
        case OP_SCOPE:
            f->env = environment(v, f->env);
            break;
        case OP_UNSCOPE:
            f->env = f->env->as.env.parent;
            break;
        case OP_LIST: {
            Obj *o = list(v);
            for (int i = v->sp - a; i < v->sp; i++)
                append(v, o, v->stack[i]);
            v->sp -= a;
            push(v, objval(o));
            break;
        }
        case OP_MAP: {
            Obj *o = newobj(v, O_MAP);
            for (int i = v->sp - a; i < v->sp; i += 2)
                tableset(v, o, &o->as.map.tab, asstr(v, v->stack[i]), v->stack[i + 1]);
            v->sp -= a;
            push(v, objval(o));
            break;
        }
        case OP_COMP_APPEND: {
            Val x = pop(v);
            append(v, v->stack[v->sp - 2].o, x);
            break;
        }
        case OP_SLICE: {
            Val end = pop(v), start = pop(v), x = pop(v);
            push(v, slicing(v, x, start, end));
            break;
        }
        case OP_INDEX: {
            Val b = pop(v), x = pop(v);
            push(v, indexget(v, x, b));
            break;
        }
        case OP_SET_INDEX: {
            Val y = pop(v), b = pop(v), x = pop(v);
            indexset(v, x, b, y);
            push(v, y);
            break;
        }
        case OP_ATTR: {
            Val x = pop(v);
            push(v, attr(v, x, f->p->k[a].s));
            break;
        }
        case OP_SET_ATTR: {
            Val y = pop(v), x = pop(v);
            set_attr(v, x, f->p->k[a].s, y);
            push(v, y);
            break;
        }
        case OP_NEG: {
            Val x = pop(v);
            if (isobj(x, O_AD) && ad_disabled(v))
                x = ad_primal(x);
            if (x.t == V_INT) {
                if (x.i == INT64_MIN)
                    vmerror(v, "integer overflow");
                push(v, integer(-x.i));
            } else if (isobj(x, O_COMPLEX))
                push(v, complex_value(v, -x.o->as.cplx.re, -x.o->as.cplx.im));
            else if (isobj(x, O_AD))
                push(v, ad_unary(v, AD_NEG, x, nil()));
            else if (isobj(x, O_ARRAY) || isobj(x, O_TYPED_TENSOR))
                push(v, binary(v, OP_MUL, x, integer(-1)));
            else
                push(v, number(-asnum(v, x)));
            break;
        }
        case OP_NOT:
            push(v, boolean(!truth(v, pop(v))));
            break;
        case OP_ADD_INT_CONST:
        case OP_SUB_INT_CONST:
        case OP_MUL_INT_CONST: {
            Op op = in.op == OP_ADD_INT_CONST   ? OP_ADD
                    : in.op == OP_SUB_INT_CONST ? OP_SUB
                                                : OP_MUL;
            Val x = top(v);
            int64_t y = f->p->k[a].i;
            v->stack[v->sp - 1] =
                x.t == V_INT ? integer(checked_arith(v, op, x.i, y)) : binary(v, op, x, integer(y));
            break;
        }
        case OP_ADD:
        case OP_SUB:
        case OP_MUL:
        case OP_DIV:
        case OP_MOD:
        case OP_EQ:
        case OP_NE:
        case OP_LT:
        case OP_LE:
        case OP_GT:
        case OP_GE: {
            Val y = pop(v);
            v->stack[v->sp - 1] = binary(v, in.op, top(v), y);
            break;
        }
        case OP_JFALSE_LT_CONST: {
            Val x = pop(v);
            int64_t y = f->p->k[in.aux].i;
            if (x.t == V_INT ? x.i >= y : !truth(v, binary(v, OP_LT, x, integer(y))))
                f->ip = a;
            break;
        }
        case OP_JUMP:
            f->ip = a;
            break;
        case OP_JFALSE:
            if (!truth(v, pop(v)))
                f->ip = a;
            break;
        case OP_JTRUE:
            if (truth(v, pop(v)))
                f->ip = a;
            break;
        case OP_CLOSURE: {
            Obj *o = newobj(v, O_FUNC);
            o->as.func.p = f->p->k[a].p;
            o->as.func.env = f->env;
            push(v, objval(o));
            break;
        }
        case OP_CALL:
            call(v, a);
            break;
        case OP_RETURN: {
            Val x = pop(v);
            return_frame(v, x);
            break;
        }
        case OP_PRINT:
            for (int i = v->sp - a; i < v->sp; i++) {
                if (i != v->sp - a)
                    putchar(' ');
                printval(v->stack[i]);
            }
            putchar('\n');
            v->sp -= a;
            break;
        case OP_ITER: {
            Val x = pop(v);
            if (isobj(x, O_MAP)) {
                Obj *l = list(v);
                Table *t = &x.o->as.map.tab;
                for (int i = 0; i < t->cap; i++)
                    if (t->a[i].key)
                        append(v, l, objval(string(v, t->a[i].key)));
                x = objval(l);
            }
            if (!isobj(x, O_LIST) && !isobj(x, O_STRING) && !isobj(x, O_ARRAY) &&
                !isobj(x, O_TENSOR_BLOCKS) && !isobj(x, O_RANGE))
                vmerror(v, "for expects list, string, map, range, array or tensor blocks");
            Obj *it = newobj(v, O_ITER);
            it->as.iter.source = x;
            push(v, objval(it));
            break;
        }
        case OP_NEXT: {
            Obj *it = top(v).o;
            Val x = it->as.iter.source;
            size_t i = it->as.iter.index;
            if (isobj(x, O_RANGE)) {
                uint64_t index = it->as.iter.range_index;
                if (index == x.o->as.range.count)
                    f->ip = a;
                else {
                    it->as.iter.range_index++;
                    push(v, integer(range_element(x.o, index)));
                }
            } else if (isobj(x, O_TENSOR_BLOCKS)) {
                Val block = tensor_block_next(v, x.o, &it->as.iter.index);
                if (block.t == V_NIL)
                    f->ip = a;
                else
                    push(v, block);
            } else if (isobj(x, O_LIST)) {
                if (i >= (size_t)x.o->as.list.n)
                    f->ip = a;
                else {
                    it->as.iter.index++;
                    push(v, x.o->as.list.a[i]);
                }
            } else if (isobj(x, O_ARRAY)) {
                if (i >= (size_t)x.o->as.array.rows)
                    f->ip = a;
                else {
                    it->as.iter.index++;
                    push(v, array_index(v, x, integer((int64_t)i)));
                }
            } else {
                if (i >= x.o->as.str.len)
                    f->ip = a;
                else {
                    const char *p = x.o->as.str.s + i;
                    int w = utf8width(p);
                    char s[5] = {0};
                    memcpy(s, p, (size_t)w);
                    it->as.iter.index += w;
                    push(v, objval(string(v, s)));
                }
            }
            break;
        }
        case OP_CLASS: {
            Val parent = pop(v);
            if (parent.t != V_NIL && !isobj(parent, O_CLASS))
                vmerror(v, "base must be a class");
            Obj *o = newobj(v, O_CLASS);
            o->as.cls.parent = parent.t == V_NIL ? NULL : parent.o;
            o->as.cls.name = copystr(f->p->k[a].s);
            account(v, o, 0, strlen(o->as.cls.name) + 1);
            push(v, objval(o));
            break;
        }
        case OP_METHOD: {
            Val fn = pop(v);
            Obj *cls = top(v).o;
            tableset(v, cls, &cls->as.cls.methods, f->p->k[a].s, fn);
            break;
        }
        case OP_IMPORT:
            module_import(v, f, f->p->k[a].s);
            break;
        case OP_TRY:
            addhandler(v, f, a);
            break;
        case OP_END_TRY:
            v->nh--;
            break;
        case OP_THROW: {
            Val x = pop(v);
            if (!isobj(x, O_MAP))
                vmerror_kind(v, "type", "throw expects an error map with code and message");
            const Entry *code = tablefind(&x.o->as.map.tab, "code");
            const Entry *message = tablefind(&x.o->as.map.tab, "message");
            if (!code || !message || !isobj(code->val, O_STRING) || !isobj(message->val, O_STRING))
                vmerror_kind(v, "type", "error map requires string code and message fields");
            v->thrown = x;
            snprintf(v->error, sizeof v->error, "%s", asstr(v, message->val));
            jump_error(v);
        }
        }
        if (v->marking || v->stress || v->bytes >= v->next_gc)
            gcstep(v, 0);
    }
}

int vm_run(VM *v, Proto *p) {
    v->sp = v->nf = v->nh = 0;
    enter(v, p, v->global, 0, NULL, nil());
    if (setjmp(v->trap)) {
        if (v->nh) {
            Handler h = v->handlers[--v->nh];
            fail_modules(v, h.frame + 1);
            v->nf = h.frame + 1;
            v->sp = h.sp;
            v->frames[h.frame].env = h.env;
            v->frames[h.frame].ip = h.ip;
            v->frames[h.frame].no_grad = h.no_grad;
            push(v, caught_error(v));
            v->thrown = nil();
        } else {
            fprintf(stderr, "%s:%d: %s\n", v->path, v->line, v->error);
            for (int i = v->nf - 1; i >= 0; i--) {
                Frame *f = &v->frames[i];
                int ip = f->ip ? f->ip - 1 : 0;
                fprintf(stderr, "  at %s (%s:%d)\n", f->p->name, f->p->path, f->p->code[ip].line);
            }
            v->thrown = nil();
            fail_modules(v, 0);
            v->nf = v->sp = v->nh = 0;
            v->path = NULL;
            gcfull(v);
            return 1;
        }
    }
    dispatch(v);
    v->path = NULL;
    gcfull(v);
    return 0;
}
