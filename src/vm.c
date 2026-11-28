#include "compute_topology.h"
#include "autodiff.h"
#include "activation_store.h"
#include "cpu_kernels.h"
#include "compute_source.h"
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
    if (isobj(x, O_LIST) || isobj(x, O_TUPLE) || isobj(x, O_SET))
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
    if (n > v->stack_limit)
        vmerror_kind(v, "stack_limit", "operand stack limit (%d)", v->stack_limit);
    if (n > v->scap) {
        int cap = v->scap ? v->scap : 128;
        while (cap < n)
            cap *= 2;
        if (cap > v->stack_limit)
            cap = v->stack_limit;
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
            if (p->param_slots[i] < 0 && !strcmp(p->params[i], s))
                return &e->as.env.slots[-1 - p->param_slots[i]];
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

static int eq(VM *v, Val a, Val b, unsigned depth);

static uint64_t mix_hash(uint64_t word) {
    word = (word ^ (word >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    word = (word ^ (word >> 27)) * UINT64_C(0x94d049bb133111eb);
    return word ^ (word >> 31);
}

static uint64_t hash_real(double value) {
    if (value >= -0x1p63 && value < 0x1p63 && trunc(value) == value)
        return mix_hash((uint64_t)(int64_t)value);
    uint64_t bits;
    memcpy(&bits, &value, sizeof bits);
    return mix_hash(bits);
}

static uint64_t value_hash(VM *v, Val x, unsigned depth) {
    if (x.t == V_INT)
        return mix_hash((uint64_t)x.i);
    if (x.t == V_NUM)
        return hash_real(x.n);
    if (x.t == V_BOOL)
        return x.n ? UINT64_C(0xe1d73235) : UINT64_C(0x1f8345ee);
    if (x.t != V_OBJ)
        return UINT64_C(0x9b3f20cf);
    if (isobj(x, O_STRING))
        return x.o->as.str.hash;
    if (isobj(x, O_COMPLEX)) {
        double re = x.o->as.cplx.re, im = x.o->as.cplx.im;
        return im == 0.0 ? hash_real(re) : mix_hash(hash_real(re) ^ (hash_real(im) << 1));
    }
    if (!isobj(x, O_TUPLE) && !isobj(x, O_SET))
        return mix_hash((uint64_t)(uintptr_t)x.o);
    if (depth == 256)
        vmerror_kind(v, "limit", "collection key nesting limit");
    uint64_t hash = mix_hash((uint64_t)x.o->as.list.n);
    if (isobj(x, O_SET)) {
        for (int i = 0; i < x.o->as.list.n; i++)
            hash += mix_hash(value_hash(v, x.o->as.list.a[i], depth + 1));
    } else {
        for (int i = 0; i < x.o->as.list.n; i++)
            hash = mix_hash(hash ^ value_hash(v, x.o->as.list.a[i], depth + 1));
    }
    return hash;
}

static bool set_probe(VM *v, Obj *set, Val item, uint64_t hash, unsigned depth, int *where) {
    unsigned mask = (unsigned)set->as.list.bucket_cap - 1;
    unsigned position = (unsigned)hash & mask;
    int value;
    while ((value = set->as.list.buckets[position])) {
        if (eq(v, set->as.list.a[value - 1], item, depth)) {
            *where = (int)position;
            return true;
        }
        position = (position + 1) & mask;
    }
    *where = (int)position;
    return false;
}

static bool set_contains(VM *v, Obj *set, Val item, unsigned depth) {
    int position;
    return set->as.list.bucket_cap &&
           set_probe(v, set, item, value_hash(v, item, depth), depth, &position);
}

void set_insert(VM *v, Obj *set, Val item) {
    uint64_t hash = value_hash(v, item, 0);
    int position, capacity = set->as.list.bucket_cap;
    if (capacity && set_probe(v, set, item, hash, 0, &position))
        return;
    if ((unsigned)set->as.list.n == ARDA_MAX_ELEMENTS)
        vmerror_kind(v, "limit", "set materialization limit exceeded");
    if (!capacity || set->as.list.n >= capacity / 2) {
        int increased = capacity ? capacity * 2 : 16;
        int *buckets = mem(sizeof *buckets * (size_t)increased);
        unsigned mask = (unsigned)increased - 1;
        for (int i = 0; i < set->as.list.n; i++) {
            unsigned slot = (unsigned)value_hash(v, set->as.list.a[i], 0) & mask;
            while (buckets[slot])
                slot = (slot + 1) & mask;
            buckets[slot] = i + 1;
        }
        free(set->as.list.buckets);
        set->as.list.buckets = buckets;
        set->as.list.bucket_cap = increased;
        account(v, set, sizeof *buckets * (size_t)capacity, sizeof *buckets * (size_t)increased);
        set_probe(v, set, item, hash, 0, &position);
    }
    append(v, set, item);
    set->as.list.buckets[position] = set->as.list.n;
}

static int eq(VM *v, Val a, Val b, unsigned depth) {
    if (numeric(a) && numeric(b))
        return compare_numeric(a, b) == 0;
    if (isobj(a, O_COMPLEX)) {
        if (isobj(b, O_COMPLEX))
            return a.o->as.cplx.re == b.o->as.cplx.re && a.o->as.cplx.im == b.o->as.cplx.im;
        return numeric(b) && a.o->as.cplx.im == 0 &&
               compare_numeric(number(a.o->as.cplx.re), b) == 0;
    }
    if (isobj(b, O_COMPLEX))
        return eq(v, b, a, depth);
    if (a.t != b.t)
        return 0;
    if (a.t == V_NIL)
        return 1;
    if (a.t == V_NUM || a.t == V_BOOL)
        return a.n == b.n;
    if (isobj(a, O_STRING) && isobj(b, O_STRING))
        return a.o == b.o;
    if (isobj(a, O_SET) && isobj(b, O_SET) && a.o != b.o) {
        if (depth == 256)
            vmerror_kind(v, "limit", "set comparison nesting limit");
        if (a.o->as.list.n != b.o->as.list.n)
            return false;
        for (int i = 0; i < a.o->as.list.n; i++)
            if (!set_contains(v, b.o, a.o->as.list.a[i], depth + 1))
                return false;
        return true;
    }
    if (isobj(a, O_TUPLE) && isobj(b, O_TUPLE) && a.o != b.o) {
        if (depth == 256)
            vmerror_kind(v, "limit", "tuple comparison nesting limit");
        if (a.o->as.list.n != b.o->as.list.n)
            return false;
        for (int i = 0; i < a.o->as.list.n; i++)
            if (!eq(v, a.o->as.list.a[i], b.o->as.list.a[i], depth + 1))
                return false;
        return true;
    }
    return a.o == b.o;
}

static bool contains_value(VM *v, Obj *collection, Val item) {
    for (int i = 0; i < collection->as.list.n; i++)
        if (eq(v, collection->as.list.a[i], item, 0))
            return true;
    return false;
}

static bool membership(VM *v, Val item, Val collection) {
    if (isobj(collection, O_SET))
        return set_contains(v, collection.o, item, 0);
    if (isobj(collection, O_LIST) || isobj(collection, O_TUPLE))
        return contains_value(v, collection.o, item);
    if (isobj(collection, O_MAP))
        return tablefind(&collection.o->as.map.tab, asstr(v, item)) != NULL;
    if (isobj(collection, O_STRING))
        return strstr(collection.o->as.str.s, asstr(v, item)) != NULL;
    if (isobj(collection, O_RANGE)) {
        Obj *r = collection.o;
        if (isobj(item, O_COMPLEX) && item.o->as.cplx.im == 0.0)
            item = number(item.o->as.cplx.re);
        if (item.t == V_NUM && item.n >= -0x1p63 && item.n < 0x1p63 && trunc(item.n) == item.n)
            item = integer((int64_t)item.n);
        if (!r->as.range.count || item.t != V_INT)
            return false;
        int64_t last = range_element(r, r->as.range.count - 1);
        int64_t first = r->as.range.start, step = r->as.range.step;
        if (step > 0)
            return item.i >= first && item.i <= last &&
                   ((uint64_t)item.i - (uint64_t)first) % (uint64_t)step == 0;
        return item.i <= first && item.i >= last &&
               ((uint64_t)first - (uint64_t)item.i) % (0 - (uint64_t)step) == 0;
    }
    vmerror_kind(v, "type", "in expects a collection");
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
        return boolean(eq(v, a, b, 0) ^ (op == OP_NE));
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
    if (isobj(a, O_LIST) || isobj(a, O_TUPLE))
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
    if (isobj(a, O_LIST) || isobj(a, O_TUPLE))
        n = a.o->as.list.n;
    else if (isobj(a, O_STRING))
        n = utf8len(a.o->as.str.s);
    else
        vmerror(v, "slice expects list or string");
    int i = slice_bound(v, start, n, 0), j = slice_bound(v, end, n, n);
    if (j < i)
        j = i;
    if (isobj(a, O_LIST) || isobj(a, O_TUPLE)) {
        Obj *o = newobj(v, a.o->type);
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

static Entry *cached_method(Obj *cls, Constant *key) {
    for (Obj *c = cls; c; c = c->as.cls.parent) {
        Entry *e = tablefind_cached(&c->as.cls.methods, key);
        if (e)
            return e;
    }
    return NULL;
}

static Val attr(VM *v, Val x, Constant *key, Val *receiver) {
    const char *name = key->s;
    Entry *e = NULL;
    if (isobj(x, O_INSTANCE)) {
        e = tablefind_cached(&x.o->as.inst.fields, key);
        if (e)
            return e->val;
        e = cached_method(x.o->as.inst.cls, key);
        if (e) {
            if (receiver) {
                *receiver = x;
                return e->val;
            }
            Obj *b = newobj(v, O_BOUND);
            b->as.bound.fn = e->val.o;
            b->as.bound.self = x;
            return objval(b);
        }
    } else if (isobj(x, O_MODULE)) {
        if (name[0] == '_')
            vmerror(v, "private module attribute '%s'", name);
        e = tablefind_cached(&x.o->as.mod.env->as.env.tab, key);
        if (e)
            return e->val;
    } else if (isobj(x, O_CLASS)) {
        e = cached_method(x.o, key);
        if (e)
            return e->val;
    }
    Val documentation = mathdoc_attribute(v, x, name);
    if (documentation.t != V_NIL)
        return documentation;
    vmerror(v, "no attribute '%s'", name);
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
    v->stack_limit = ARDA_MAX_STACK;
    v->frame_limit = ARDA_MAX_FRAMES;
    v->vm_byte_limit = SIZE_MAX;
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

static void reserve_handlers(VM *v, int count) {
    if (count <= v->hcap)
        return;
    int capacity = v->hcap ? v->hcap : 16;
    while (capacity < count)
        capacity *= 2;
    v->handlers = resize(v->handlers, sizeof(Handler) * (size_t)capacity);
    v->hcap = capacity;
}

static void suspend_generator(VM *v, Obj *object) {
    Generator *g = object->as.generator;
    Frame *f = &v->frames[v->nf - 1];
    g->frame = *f;
    g->frame.base = 0;
    g->nstack = v->sp - f->base;
    g->nhandlers = v->nh - f->handler_base;
    if (g->nstack > g->stack_cap) {
        size_t old = sizeof(Val) * (size_t)g->stack_cap;
        size_t bytes = sizeof(Val) * (size_t)g->nstack;
        g->stack = resize(g->stack, bytes);
        account(v, object, old, bytes);
        g->stack_cap = g->nstack;
    }
    if (g->nstack)
        memcpy(g->stack, v->stack + f->base, sizeof(Val) * (size_t)g->nstack);
    if (g->nhandlers > g->handler_cap) {
        size_t old = sizeof(Handler) * (size_t)g->handler_cap;
        size_t bytes = sizeof(Handler) * (size_t)g->nhandlers;
        g->handlers = resize(g->handlers, bytes);
        account(v, object, old, bytes);
        g->handler_cap = g->nhandlers;
    }
    for (int i = 0; i < g->nhandlers; i++) {
        g->handlers[i] = v->handlers[f->handler_base + i];
        g->handlers[i].sp -= f->base;
    }
    /* The suspended values were on the VM stack, which may change before the
       next GC step. Preserve references when the iterator is already marked. */
    barrier(v, object, objval(f->env));
    barrier(v, object, objval(f->locals));
    if (v->marking && object->mark) {
        for (int i = 0; i < g->nstack; i++)
            markval(v, g->stack[i]);
        for (int i = 0; i < g->nhandlers; i++)
            markobj(v, g->handlers[i].env);
    }
    g->fallback = nil();
    g->state = GENERATOR_READY;
    v->sp = f->base;
    v->nh = f->handler_base;
    v->nf--;
}

static void finish_generator(VM *v, Obj *object) {
    Generator *g = object->as.generator;
    size_t old = sizeof(Val) * (size_t)g->stack_cap + sizeof(Handler) * (size_t)g->handler_cap;
    free(g->stack);
    free(g->handlers);
    account(v, object, old, 0);
    *g = (Generator){.state = GENERATOR_DONE};
}

void iterator_close(VM *v, Val cursor) {
    Obj *generator = isobj(cursor, O_GENERATOR) ? cursor.o : NULL;
    if (generator) {
        if (generator->as.generator->state == GENERATOR_RUNNING)
            vmerror_kind(v, "state", "cannot close a running generator");
        finish_generator(v, generator);
    }
    if (isobj(cursor, O_ITER))
        cursor.o->as.iter.source = nil();
    else if (!generator)
        vmerror_kind(v, "type", "close_iterator expects an iterator or generator");
}

static void end_iteration(VM *v, int stop_ip, Val fallback) {
    if (stop_ip >= 0) {
        v->frames[v->nf - 1].ip = stop_ip;
    } else if (fallback.t != V_MISSING) {
        push(v, fallback);
    } else {
        Frame *caller = &v->frames[v->nf - 1];
        v->path = caller->p->path;
        v->line = caller->p->code[caller->ip - 1].line;
        vmerror_kind(v, "stop_iteration", "iterator is exhausted");
    }
}

static void resume_generator(VM *v, Obj *object, int base, int stop_ip, Val fallback) {
    Generator *g = object->as.generator;
    if (g->state == GENERATOR_RUNNING)
        vmerror_kind(v, "state", "cannot advance a running generator");
    if (g->state == GENERATOR_DONE) {
        v->sp = base;
        end_iteration(v, stop_ip, fallback);
        return;
    }
    if (v->nf >= v->frame_limit)
        vmerror_kind(v, "stack_limit", "call depth limit (%d)", v->frame_limit);
    reserve(v, base + g->nstack);
    reserve_handlers(v, v->nh + g->nhandlers);
    if (g->nstack)
        memcpy(v->stack + base, g->stack, sizeof(Val) * (size_t)g->nstack);
    Frame frame = g->frame;
    frame.base = base;
    frame.generator = object;
    frame.handler_base = v->nh;
    for (int i = 0; i < g->nhandlers; i++) {
        Handler handler = g->handlers[i];
        handler.frame = v->nf;
        handler.sp += base;
        v->handlers[v->nh++] = handler;
    }
    v->frames[v->nf++] = frame;
    v->sp = base + g->nstack;
    g->nstack = g->nhandlers = 0;
    g->stop_ip = stop_ip;
    g->fallback = fallback;
    g->state = GENERATOR_RUNNING;
    barrier(v, object, fallback);
}

static void enter(VM *v, Proto *p, Obj *env, int base, Obj *module, Val constructed) {
    if (v->nf >= v->frame_limit)
        vmerror_kind(v, "stack_limit", "call depth limit (%d)", v->frame_limit);
    bool no_grad = ad_disabled(v);
    v->frames[v->nf++] = (Frame){.p = p,
                                 .no_grad = no_grad,
                                 .env = env,
                                 .locals = env,
                                 .base = base,
                                 .handler_base = v->nh,
                                 .module = module,
                                 .constructed = constructed};
}

static void enter_fn(VM *v, Obj *f, int base, int argc, Val constructed) {
    Proto *p = f->as.func.p;
    if (argc < p->required || argc > p->arity)
        vmerror(v, "%s expects %d arguments (minimum %d), got %d", p->name, p->arity, p->required,
                argc);
    if (p->generator && constructed.t != V_NIL)
        vmerror_kind(v, "type", "a constructor cannot be a generator");
    Obj *env = f->as.func.env;
    reserve(v, base + p->slots);
    if (p->heap_scope) {
        env = newobj_extra(v, O_ENV, sizeof(Val) * (size_t)p->heap_params);
        env->as.env.parent = f->as.func.env;
    }
    if (p->heap_params) {
        env->as.env.params = p;
        env->as.env.slots = (Val *)(env + 1);
        for (int i = 0; i < p->arity; i++) {
            Val x = i < argc ? v->stack[base + 1 + i] : (Val){.t = V_MISSING};
            int slot = p->param_slots[i];
            if (slot < 0)
                env->as.env.slots[-1 - slot] = x;
            else
                v->stack[base + slot] = x;
        }
    } else {
        if (argc)
            memmove(v->stack + base, v->stack + base + 1, sizeof(Val) * (size_t)argc);
        for (int i = argc; i < p->arity; i++)
            v->stack[base + i] = (Val){.t = V_MISSING};
    }
    for (int i = p->arity - p->heap_params; i < p->slots; i++)
        v->stack[base + i] = nil();
    v->sp = base + p->slots;
    enter(v, p, env, base, NULL, constructed);
    if (p->generator) {
        Obj *object = newobj_extra(v, O_GENERATOR, sizeof(Generator));
        object->as.generator = (Generator *)(object + 1);
        suspend_generator(v, object);
        push(v, objval(object));
    }
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
    if (count > v->stack_limit - base - 1)
        vmerror_kind(v, "stack_limit", "operand stack limit (%d)", v->stack_limit);
    reserve(v, base + 1 + count);
    v->stack[base] = target;
    if (count)
        memcpy(v->stack + base + 1, arguments->as.list.a, (size_t)count * sizeof(Val));
    v->sp = base + 1 + count;
    invoke(v, count);
}

static void call_native(VM *v, Obj *function, int base, int argc) {
    int id = function->as.native.id;
    if (id == BUILTIN_NEXT && (argc == 1 || argc == 2)) {
        Val gen = v->stack[base + 1];
        if (isobj(gen, O_GENERATOR)) {
            Val fallback = argc == 2 ? v->stack[base + 2] : (Val){.t = V_MISSING};
            resume_generator(v, gen.o, base, -1, fallback);
            return;
        }
    }
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

static void call_named(VM *v, int positional, int named) {
    int base = v->sp - positional - 2 * named - 1;
    Val callee = v->stack[base], self = nil(), constructed = nil();
    Obj *function = isobj(callee, O_FUNC) ? callee.o : NULL;
    if (isobj(callee, O_BOUND)) {
        function = callee.o->as.bound.fn;
        self = callee.o->as.bound.self;
    } else if (isobj(callee, O_CLASS)) {
        Entry *init = method(callee.o, "init");
        if (!init)
            init = method(callee.o, "__init__");
        if (init) {
            function = init->val.o;
            Obj *instance = newobj(v, O_INSTANCE);
            instance->as.inst.cls = callee.o;
            self = constructed = objval(instance);
        }
    }
    if (!function || function->type != O_FUNC)
        vmerror_kind(v, "arity", "named arguments require an Arda function, method or constructor");
    Proto *p = function->as.func.p;
    int implicit = self.t != V_NIL;
    if (positional > p->arity - implicit)
        vmerror_kind(v, "arity", "%s received too many positional arguments", p->name);
    reserve(v, v->sp + p->arity);
    Val *args = v->stack + v->sp;
    for (int i = 0; i < p->arity; i++)
        args[i] = (Val){.t = V_MISSING};
    if (implicit)
        args[0] = self;
    for (int i = 0; i < positional; i++)
        args[i + implicit] = v->stack[base + i + 1];
    for (int i = base + positional + 1; i < v->sp; i += 2) {
        const char *name = asstr(v, v->stack[i]);
        int k = 0;
        while (k < p->arity && strcmp(p->params[k], name))
            k++;
        if (k == p->arity)
            vmerror_kind(v, "arity", "%s has no parameter '%s'", p->name, name);
        if (args[k].t != V_MISSING)
            vmerror_kind(v, "arity", "%s received '%s' twice", p->name, name);
        args[k] = v->stack[i + 1];
    }
    for (int i = 0; i < p->required; i++)
        if (args[i].t == V_MISSING)
            vmerror_kind(v, "arity", "%s requires '%s'", p->name, p->params[i]);
    if (p->arity)
        memmove(v->stack + base + 1, args, sizeof(Val) * (size_t)p->arity);
    v->sp = base + p->arity + 1;
    enter_fn(v, function, base, p->arity, constructed);
}

static void addhandler(VM *v, Frame *f, int ip, int slots) {
    reserve_handlers(v, v->nh + 1);
    v->handlers[v->nh++] = (Handler){v->nf - 1, ip, v->sp, slots, f->env, f->no_grad};
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
        {"compute", "<stdlib:compute>", compute_source},
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
        vm_require_file_access(v);
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
    if (v->nf >= v->frame_limit) {
        o->as.mod.state = 2;
        vmerror_kind(v, "stack_limit", "call depth limit (%d)", v->frame_limit);
    }
    enter(v, p, o->as.mod.env, v->sp, o, nil());
}

static void return_frame(VM *v, Val x) {
    Frame f = v->frames[v->nf - 1];
    while (v->nh && v->handlers[v->nh - 1].frame >= v->nf - 1)
        v->nh--;
    v->nf--;
    v->sp = f.base;
    if (f.generator) {
        Generator *g = f.generator->as.generator;
        int stop_ip = g->stop_ip;
        Val fallback = g->fallback;
        finish_generator(v, f.generator);
        end_iteration(v, stop_ip, fallback);
        return;
    }
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
    for (int i = v->nf - 1; i >= keep; i--) {
        Frame *f = &v->frames[i];
        if (f->module)
            f->module->as.mod.state = 2;
        if (f->generator)
            finish_generator(v, f->generator);
    }
}

Obj *iterator(VM *v, Val source) {
    if (isobj(source, O_ITER) || isobj(source, O_GENERATOR))
        return source.o;
    if (isobj(source, O_MAP)) {
        Obj *keys = list(v);
        Table *table = &source.o->as.map.tab;
        for (int i = 0; i < table->cap; i++)
            if (table->a[i].key)
                append(v, keys, objval(string(v, table->a[i].key)));
        source = objval(keys);
    }
    if (!isobj(source, O_LIST) && !isobj(source, O_STRING) && !isobj(source, O_ARRAY) &&
        !isobj(source, O_TENSOR_BLOCKS) && !isobj(source, O_RANGE) && !isobj(source, O_TUPLE) &&
        !isobj(source, O_SET))
        vmerror_kind(v, "type", "expected an iterable value");
    Obj *it = newobj(v, O_ITER);
    it->as.iter.source = source;
    return it;
}

static bool advance_iterator(VM *v, Obj *it, Val *value) {
    Obj *source = it->as.iter.source.o;
    size_t i = it->as.iter.index;
    switch (source->type) {
    case O_RANGE: {
        uint64_t index = it->as.iter.range_index;
        if (index == source->as.range.count)
            return false;
        *value = integer(range_element(source, index));
        it->as.iter.range_index++;
        return true;
    }
    case O_TENSOR_BLOCKS:
        *value = tensor_block_next(v, source, &it->as.iter.index);
        return value->t != V_NIL;
    case O_LIST:
    case O_TUPLE:
    case O_SET:
        if (i >= (size_t)source->as.list.n)
            return false;
        *value = source->as.list.a[i];
        break;
    case O_ARRAY:
        if (i >= (size_t)source->as.array.rows)
            return false;
        *value = array_index(v, objval(source), integer((int64_t)i));
        break;
    case O_STRING: {
        if (i >= source->as.str.len)
            return false;
        const char *p = source->as.str.s + i;
        int width = utf8width(p);
        char text[5] = {0};
        memcpy(text, p, (size_t)width);
        it->as.iter.index += (size_t)width;
        *value = objval(string(v, text));
        return true;
    }
    default:
        vmerror_kind(v, "type", "invalid iterator source");
    }
    it->as.iter.index++;
    return true;
}

bool iterator_next(VM *v, Obj *it, Val *value) {
    if (it->as.iter.source.t == V_NIL)
        return false;
    if (advance_iterator(v, it, value))
        return true;
    it->as.iter.source = nil();
    return false;
}

static _Noreturn void terminate(VM *v, const char *code, const char *message) {
    v->nh = 0;
    vmerror_kind(v, code, "%s", message);
}

static void check_managed_bytes(VM *v) {
    if (v->bytes <= v->vm_byte_limit)
        return;
    gcfull(v);
    if (v->bytes > v->vm_byte_limit)
        terminate(v, "memory_limit", "managed VM memory limit exceeded");
}

static void dispatch(VM *v) {
    /* Remaining instructions live on the VM across caught longjmps. */
    while (v->nf) {
        Frame *f = &v->frames[v->nf - 1];
        Instr in = f->p->code[f->ip++];
        v->line = in.line;
        v->path = f->p->path;
        if (v->instructions_active && !v->instructions_left--)
            terminate(v, "instruction_limit", "instruction limit exceeded");
        if (v->interrupted && *v->interrupted)
            terminate(v, "interrupted", "execution interrupted");
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
            Val x = in.aux ? f->locals->as.env.slots[a] : v->stack[f->base + a];
            if (x.t == V_MISSING && f->p->code[f->ip].op != OP_JPRESENT)
                vmerror(v, "parameter not initialized");
            push(v, x);
            break;
        }
        case OP_SET_LOCAL:
        case OP_SET_LOCAL_DROP: {
            Val x = in.op == OP_SET_LOCAL_DROP ? pop(v) : top(v);
            if (!in.aux)
                v->stack[f->base + a] = x;
            else {
                f->locals->as.env.slots[a] = x;
                if (v->marking)
                    barrier(v, f->locals, x);
            }
            break;
        }
        case OP_UNSCOPE:
            f->env = f->env->as.env.parent;
            /* fall through */
        case OP_CLEAR_LOCALS:
            for (int slot = a; slot < in.aux; slot++)
                v->stack[f->base + slot] = nil();
            break;
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
        case OP_LIST:
        case OP_TUPLE:
        case OP_SET_LITERAL: {
            OType type = in.op == OP_LIST ? O_LIST : in.op == OP_TUPLE ? O_TUPLE : O_SET;
            Obj *o = newobj(v, type);
            for (int i = v->sp - a; i < v->sp; i++)
                if (type == O_SET)
                    set_insert(v, o, v->stack[i]);
                else
                    append(v, o, v->stack[i]);
            v->sp -= a;
            push(v, objval(o));
            break;
        }
        case OP_FORMAT: {
            char *text = NULL;
            size_t used = 0, cap = 0;
            for (int i = v->sp - a; i < v->sp; i++) {
                char *part = value_string(v->stack[i]);
                size_t length = strlen(part);
                if (length >= SIZE_MAX - used) {
                    free(part);
                    free(text);
                    vmerror_kind(v, "limit", "formatted string is too large");
                }
                size_t needed = used + length + 1;
                if (needed > cap) {
                    cap = cap > SIZE_MAX / 2 ? needed : (needed > cap * 2 ? needed : cap * 2);
                    text = resize(text, cap);
                }
                memcpy(text + used, part, length + 1);
                used += length;
                free(part);
            }
            Obj *result = string(v, text ? text : "");
            free(text);
            v->sp -= a;
            push(v, objval(result));
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
        case OP_IN: {
            Val collection = pop(v), item = pop(v);
            push(v, boolean(membership(v, item, collection)));
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
        case OP_ATTR:
        case OP_ATTR_CALL: {
            Val x = pop(v), receiver = (Val){.t = V_MISSING};
            bool direct = in.op == OP_ATTR_CALL;
            push(v, attr(v, x, &f->p->k[a], direct ? &receiver : NULL));
            if (direct)
                push(v, receiver);
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
        case OP_JFALSE_LT: {
            Val right = pop(v), left = pop(v);
            bool less = left.t == V_INT && right.t == V_INT
                            ? left.i < right.i
                            : truth(v, binary(v, OP_LT, left, right));
            if (!less)
                f->ip = a;
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
        case OP_CALL_NAMED:
            call_named(v, a, in.aux);
            break;
        case OP_CALL_ATTR: {
            int count = a + 2 * in.aux;
            int base = v->sp - count - 2;
            if (v->stack[base + 1].t == V_MISSING) {
                memmove(v->stack + base + 1, v->stack + base + 2, sizeof(Val) * (size_t)count);
                v->sp--;
            } else {
                a++;
            }
            if (in.aux)
                call_named(v, a, in.aux);
            else
                call(v, a);
            break;
        }
        case OP_YIELD: {
            Val value = pop(v);
            suspend_generator(v, f->generator);
            push(v, value);
            break;
        }
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
            push(v, objval(iterator(v, x)));
            break;
        }
        case OP_NEXT: {
            Obj *it = top(v).o;
            if (it->type == O_GENERATOR) {
                resume_generator(v, it, v->sp, a, nil());
                break;
            }
            Val value;
            if (iterator_next(v, it, &value))
                push(v, value);
            else
                f->ip = a;
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
            addhandler(v, f, a, in.aux);
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
        if (v->gc_profile && v->bytes > v->gc_profile->peak_bytes)
            v->gc_profile->peak_bytes = v->bytes;
        if (v->bytes > v->vm_byte_limit)
            check_managed_bytes(v);
        else if (v->marking || v->stress || v->bytes >= v->next_gc)
            gcstep(v, 0);
    }
}

int vm_run(VM *v, Proto *p) {
    v->sp = v->nf = v->nh = 0;
    v->instructions_left = v->instruction_limit;
    v->instructions_active = v->instruction_limit != 0;
    enter(v, p, v->global, 0, NULL, nil());
    if (setjmp(v->trap)) {
        if (v->nh) {
            Handler h = v->handlers[--v->nh];
            fail_modules(v, h.frame + 1);
            v->nf = h.frame + 1;
            v->sp = h.sp;
            Frame *f = &v->frames[h.frame];
            for (int i = h.slots; i < f->p->slots; i++)
                v->stack[f->base + i] = nil();
            f->env = h.env;
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
    v->path = v->frames[v->nf - 1].p->path;
    check_managed_bytes(v);
    dispatch(v);
    v->path = NULL;
    gcfull(v);
    return 0;
}
