#include "autodiff.h"
#include "simulation_checkpoint.h"
#include <time.h>
static const char *names[] = {"len",        "push",      "pop",        "str",        "num",
                              "sqrt",       "heap",      "clock_gc",   "heap_bytes", "collect",
                              "range",      "read_text", "write_text", "keys",       "type",
                              "int",        "float",     "abs",        "sin",        "cos",
                              "floor",      "ceil",      "monotonic",  "split",      "join",
                              "startswith", "endswith",  "isinstance", "list"};

const char *native_builtin_name(int id) {
    return id >= 0 && (size_t)id < sizeof names / sizeof *names ? names[id] : NULL;
}
void install_builtins(VM *v) {
    for (size_t i = 0; i < sizeof names / sizeof *names; i++) {
        Obj *o = newobj(v, O_NATIVE);
        o->as.native.id = (int)i;
        tableset(v, v->builtins, &v->builtins->as.env.tab, names[i], objval(o));
    }
}

static const char *typename(Val v) {
    switch (v.t) {
    case V_NIL:
        return "nil";
    case V_NUM:
        return "float";
    case V_INT:
        return "int";
    case V_MISSING:
        return "missing";
    case V_BOOL:
        return "bool";
    case V_OBJ: {
        static const char *n[] = {"string",
                                  "list",
                                  "map",
                                  "scope",
                                  "function",
                                  "native",
                                  "class",
                                  "instance",
                                  "method",
                                  "module",
                                  "iterator",
                                  "array",
                                  "ad",
                                  "sparse",
                                  "sparse_builder",
                                  "factor",
                                  "generator",
                                  "moments",
                                  "complex",
                                  "linear_operator",
                                  "buffer",
                                  "tensor",
                                  "tensor_blocks",
                                  "lowrank",
                                  "quantity",
                                  "range",
                                  "offload_plan",
                                  "quantized_matrix",
                                  "compute_plan",
                                  "fem_mesh",
                                  "tensor_output"};
        _Static_assert(sizeof n / sizeof *n == O_TENSOR_OUTPUT + 1, "object type names");
        return n[v.o->type];
    }
    }
    return "?";
}

char *value_string(Val v) {
    char b[128];
    switch (v.t) {
    case V_NIL:
        return copystr("mun");
    case V_BOOL:
        return copystr(v.n ? "nae" : "law");
    case V_INT:
        snprintf(b, sizeof b, "%" PRId64, v.i);
        return copystr(b);
    case V_MISSING:
        return copystr("<missing>");
    case V_NUM:
        snprintf(b, sizeof b, "%.15g", v.n);
        return copystr(b);
    case V_OBJ:
        if (isobj(v, O_AD))
            return value_string(ad_primal(v));
        if (isobj(v, O_STRING))
            return copystr(v.o->as.str.s);
        if (isobj(v, O_LIST))
            snprintf(b, sizeof b, "<list length=%d>", v.o->as.list.n);
        else if (isobj(v, O_ARRAY)) {
            size_t pos = (size_t)snprintf(b, sizeof b, "<array shape=");
            for (int i = 0; i < v.o->as.array.rank; i++)
                pos += (size_t)snprintf(b + pos, sizeof b - pos, "%s%d", i ? "x" : "",
                                        v.o->as.array.shape[i]);
            snprintf(b + pos, sizeof b - pos, ">");
        } else if (isobj(v, O_FACTOR)) {
            snprintf(b, sizeof b, "<linear algebra factor>");
        } else if (isobj(v, O_COMPLEX)) {
            snprintf(b, sizeof b, "(%.15g%+.15gi)", v.o->as.cplx.re, v.o->as.cplx.im);
        } else if (isobj(v, O_LINOP)) {
            snprintf(b, sizeof b, "<linear operator shape=%dx%d>", v.o->as.linop.rows,
                     v.o->as.linop.cols);
        } else if (isobj(v, O_SPARSE_BUILDER)) {
            snprintf(b, sizeof b, "<sparse builder>");
        } else if (isobj(v, O_SPARSE)) {
            Sparse *s = &v.o->as.sparse;
            snprintf(b, sizeof b, "<%s shape=%" PRIu32 "x%" PRIu32 " nnz=%zu>",
                     s->csr ? "csr" : "coo", s->rows, s->cols, s->nnz);
        } else if (isobj(v, O_MAP))
            snprintf(b, sizeof b, "<map length=%d>", v.o->as.map.tab.n);
        else if (isobj(v, O_CLASS))
            snprintf(b, sizeof b, "<class %s>", v.o->as.cls.name);
        else if (isobj(v, O_INSTANCE))
            snprintf(b, sizeof b, "<%s instance>", v.o->as.inst.cls->as.cls.name);
        else
            snprintf(b, sizeof b, "<%s>", typename(v));
        return copystr(b);
    }
    return copystr("?");
}

static void print_rec(Val v, int depth) {
    if (isobj(v, O_ARRAY)) {
        array_print(v.o);
        return;
    }
    if (depth >= 8 && (isobj(v, O_LIST) || isobj(v, O_MAP))) {
        fputs("...", stdout);
        return;
    }
    if (isobj(v, O_LIST)) {
        putchar('[');
        for (int i = 0; i < v.o->as.list.n; i++) {
            if (i)
                fputs(", ", stdout);
            print_rec(v.o->as.list.a[i], depth + 1);
        }
        putchar(']');
    } else if (isobj(v, O_MAP)) {
        putchar('{');
        int first = 1;
        Table *t = &v.o->as.map.tab;
        for (int i = 0; i < t->cap; i++)
            if (t->a[i].key) {
                if (!first)
                    fputs(", ", stdout);
                first = 0;
                printf("\"%s\": ", t->a[i].key);
                print_rec(t->a[i].val, depth + 1);
            }
        putchar('}');
    } else {
        char *s = value_string(v);
        fputs(s, stdout);
        free(s);
    }
}

void printval(Val v) {
    print_rec(v, 0);
}

double asnum(VM *v, Val x) {
    if (x.t == V_INT)
        return (double)x.i;
    if (x.t != V_NUM)
        vmerror(v, "expected number, got %s", typename(x));
    return x.n;
}

const char *asstr(VM *v, Val x) {
    if (!isobj(x, O_STRING))
        vmerror(v, "expected string, got %s", typename(x));
    return x.o->as.str.s;
}

int indexnum(VM *v, Val x, int n) {
    long double d = x.t == V_INT ? (long double)x.i : asnum(v, x);
    if (!isfinite(d) || truncl(d) != d || d < -(double)n || d >= n)
        vmerror(v, "index out of bounds or not an integer");
    int i = (int)d;
    return i < 0 ? i + n : i;
}

static void arity(VM *v, int id, int argc, int n) {
    if (argc != n)
        vmerror(v, "%s expects %d arguments, got %d", names[id], n, argc);
}

static int64_t range_integer(VM *v, Val value) {
    if (value.t == V_INT)
        return value.i;
    double x = asnum(v, value);
    if (!isfinite(x) || trunc(x) != x || x < -0x1p63 || x >= 0x1p63)
        vmerror_kind(v, "range", "range requires integral Int64 endpoints and step");
    return (int64_t)x;
}

int64_t range_element(const Obj *range, uint64_t index) {
    uint64_t bits = (uint64_t)range->as.range.start + index * (uint64_t)range->as.range.step;
    return bits <= INT64_MAX ? (int64_t)bits : INT64_MIN + (int64_t)(bits - (UINT64_C(1) << 63));
}

Val native_call(VM *v, int id, int argc, Val *a) {
    if (id >= PLOT_BASE)
        return plot_call(v, id, argc, a);
    if (id >= CHECKPOINT_BASE)
        return checkpoint_call(v, id, argc, a);
    if (id >= FEM_BASE)
        return fem_call(v, id, argc, a);
    if (id >= PDE_BASE)
        return pde_call(v, id, argc, a);
    if (id >= SOLVER_BASE)
        return solver_call(v, id, argc, a);
    if (id >= QUANT_BASE)
        return quant_call(v, id, argc, a);
    if (id >= UNCERTAINTY_BASE)
        return uncertainty_call(v, id, argc, a);
    if (id >= OPTIMIZER_BASE)
        return optimizer_call(v, id, argc, a);
    if (id >= UNITS_BASE)
        return units_call(v, id, argc, a);
    if (id >= LOWRANK_BASE)
        return lowrank_call(v, id, argc, a);
    if (id >= COMPUTE_BASE)
        return compute_call(v, id, argc, a);
    if (id >= TYPED_TENSOR_BASE)
        return typed_tensor_call(v, id, argc, a);
    if (id >= SPECIAL_BASE)
        return special_call(v, id, argc, a);
    if (id >= MATHDOC_BASE)
        return mathdoc_call(v, id, argc, a);
    if (id >= LINOP_BASE)
        return linop_call(v, id, argc, a);
    if (id >= COMPLEX_BASE)
        return complex_call(v, id, argc, a);
    if (id >= STATS_BASE)
        return stats_call(v, id, argc, a);
    if (id >= RANDOM_BASE)
        return random_call(v, id, argc, a);
    if (id >= LINALG_BASE)
        return linalg_call(v, id, argc, a);
    if (id >= SPARSE_BASE)
        return sparse_call(v, id, argc, a);
    if (id >= HIGH_BASE)
        return high_call(v, id, argc, a);
    if (id >= AD_BASE)
        return ad_call(v, id, argc, a);
    if (id >= CALC_BASE)
        return calculus_call(v, id, argc, a);
    if (id >= TENSOR_BASE)
        return tensor_call(v, id, argc, a);
    if (id >= SCI_BASE)
        return scientific_call(v, id, argc, a);
    switch (id) {
    case 0:
        arity(v, id, argc, 1);
        if (isobj(a[0], O_ARRAY))
            return integer(a[0].o->as.array.rows);
        if (isobj(a[0], O_RANGE))
            return integer((int64_t)a[0].o->as.range.count);
        if (isobj(a[0], O_LIST))
            return integer(a[0].o->as.list.n);
        if (isobj(a[0], O_STRING))
            return integer(utf8len(a[0].o->as.str.s));
        if (isobj(a[0], O_MAP))
            return integer(a[0].o->as.map.tab.n);
        vmerror(v, "len expects array, list, string, map or range");
        break;
    case 1:
        arity(v, id, argc, 2);
        if (!isobj(a[0], O_LIST))
            vmerror(v, "push expects list");
        append(v, a[0].o, a[1]);
        return nil();
    case 2:
        arity(v, id, argc, 1);
        if (!isobj(a[0], O_LIST) || !a[0].o->as.list.n)
            vmerror(v, "pop expects nonempty list");
        return a[0].o->as.list.a[--a[0].o->as.list.n];
    case 3: {
        arity(v, id, argc, 1);
        char *s = value_string(a[0]);
        Obj *o = string(v, s);
        free(s);
        return objval(o);
    }
    case 4: {
        arity(v, id, argc, 1);
        if (a[0].t == V_NUM || a[0].t == V_INT)
            return a[0];
        const char *s = asstr(v, a[0]);
        char *end;
        errno = 0;
        intmax_t i = strtoimax(s, &end, 10);
        if (end != s && !*end) {
            if (errno == ERANGE)
                vmerror(v, "integer overflow");
            return integer((int64_t)i);
        }
        double d = strtod(s, &end);
        if (end == s || *end)
            vmerror(v, "invalid number");
        return number(d);
    }
    case 5: {
        arity(v, id, argc, 1);
        if (isobj(a[0], O_AD) && ad_primal(a[0]).t == V_NUM)
            return ad_unary(v, AD_SQRT, a[0], nil());
        double x = asnum(v, a[0]);
        return number(sqrt(x));
    }
    case 6:
        arity(v, id, argc, 0);
        return integer((int64_t)v->live);
    case 7:
        arity(v, id, argc, 0);
        return integer((int64_t)v->cycles);
    case 8:
        arity(v, id, argc, 0);
        return integer((int64_t)v->bytes);
    case 9:
        arity(v, id, argc, 0);
        gcfull(v);
        return nil();
    case 10: {
        if (argc < 1 || argc > 3)
            vmerror(v, "range expects 1 to 3 arguments");
        int64_t start = argc == 1 ? 0 : range_integer(v, a[0]);
        int64_t end = range_integer(v, a[argc == 1 ? 0 : 1]);
        int64_t step = argc == 3 ? range_integer(v, a[2]) : 1;
        if (!step)
            vmerror_kind(v, "range", "range step must be nonzero");
        uint64_t distance = 0;
        if (step > 0 && end > start)
            distance = (uint64_t)end - (uint64_t)start;
        if (step < 0 && end < start)
            distance = (uint64_t)start - (uint64_t)end;
        uint64_t width = step < 0 ? 0 - (uint64_t)step : (uint64_t)step;
        uint64_t count = distance ? (distance - 1) / width + 1 : 0;
        if (count > INT64_MAX)
            vmerror_kind(v, "range", "range length exceeds Int64");
        Obj *o = newobj(v, O_RANGE);
        o->as.range.start = start;
        o->as.range.step = step;
        o->as.range.count = count;
        return objval(o);
    }
    case 11: {
        arity(v, id, argc, 1);
        const char *path = asstr(v, a[0]);
        char *s = readfile(path);
        if (!s)
            vmerror(v, "cannot read text file '%s'", path);
        if (!utf8valid(s)) {
            free(s);
            vmerror(v, "invalid UTF-8 file");
        }
        Obj *o = string(v, s);
        free(s);
        return objval(o);
    }
    case 12: {
        arity(v, id, argc, 2);
        const char *path = asstr(v, a[0]), *s = asstr(v, a[1]);
        FILE *f = fopen(path, "wb");
        if (!f)
            vmerror(v, "cannot write '%s'", path);
        size_t n = strlen(s);
        size_t w = fwrite(s, 1, n, f);
        int rc = fclose(f);
        if (w != n || rc)
            vmerror(v, "write failed '%s'", path);
        return integer((int64_t)n);
    }
    case 13: {
        arity(v, id, argc, 1);
        if (!isobj(a[0], O_MAP))
            vmerror(v, "keys expects map");
        Obj *o = list(v);
        Table *t = &a[0].o->as.map.tab;
        for (int i = 0; i < t->cap; i++)
            if (t->a[i].key)
                append(v, o, objval(string(v, t->a[i].key)));
        return objval(o);
    }
    case 14:
        arity(v, id, argc, 1);
        return objval(string(v, typename(a[0])));
    case 15: {
        arity(v, id, argc, 1);
        Val x = a[0];
        if (x.t == V_INT)
            return x;
        if (isobj(x, O_STRING)) {
            const char *s = asstr(v, x);
            char *e;
            errno = 0;
            intmax_t n = strtoimax(s, &e, 10);
            if (errno || e == s || *e)
                vmerror(v, "invalid integer");
            return integer((int64_t)n);
        }
        double d = asnum(v, x);
        if (!isfinite(d) || d < -0x1p63 || d >= 0x1p63)
            vmerror(v, "integer overflow");
        return integer((int64_t)d);
    }
    case 16: {
        arity(v, id, argc, 1);
        if (isobj(a[0], O_STRING)) {
            const char *s = asstr(v, a[0]);
            char *e;
            double d = strtod(s, &e);
            if (e == s || *e)
                vmerror(v, "invalid float");
            return number(d);
        }
        return number(asnum(v, a[0]));
    }
    case 17:
        arity(v, id, argc, 1);
        if (isobj(a[0], O_COMPLEX))
            return number(hypot(a[0].o->as.cplx.re, a[0].o->as.cplx.im));
        if (a[0].t == V_INT) {
            if (a[0].i == INT64_MIN)
                vmerror(v, "integer overflow");
            return integer(a[0].i < 0 ? -a[0].i : a[0].i);
        }
        return number(fabs(asnum(v, a[0])));
    case 18:
    case 19: {
        arity(v, id, argc, 1);
        if (isobj(a[0], O_AD) && ad_primal(a[0]).t == V_NUM)
            return ad_unary(v, id == 18 ? AD_SIN : AD_COS, a[0], nil());
        double x = asnum(v, a[0]);
        return number(id == 18 ? sin(x) : cos(x));
    }
    case 20:
    case 21: {
        arity(v, id, argc, 1);
        if (a[0].t == V_INT)
            return a[0];
        double d = id == 20 ? floor(asnum(v, a[0])) : ceil(asnum(v, a[0]));
        if (!isfinite(d) || d < -0x1p63 || d >= 0x1p63)
            vmerror(v, "integer overflow");
        return integer((int64_t)d);
    }
    case 22: {
        arity(v, id, argc, 0);
        struct timespec ts;
        if (clock_gettime(CLOCK_MONOTONIC, &ts))
            vmerror(v, "clock failed");
        return number((double)ts.tv_sec + ts.tv_nsec / 1e9);
    }
    case 23: {
        arity(v, id, argc, 2);
        const char *s = asstr(v, a[0]), *sep = asstr(v, a[1]);
        size_t n = strlen(sep);
        if (!n)
            vmerror(v, "empty separator");
        Obj *o = list(v);
        const char *p;
        while ((p = strstr(s, sep))) {
            char *part = slice(s, (size_t)(p - s));
            append(v, o, objval(string(v, part)));
            free(part);
            s = p + n;
        }
        append(v, o, objval(string(v, s)));
        return objval(o);
    }
    case 24: {
        arity(v, id, argc, 2);
        const char *sep = asstr(v, a[0]);
        if (!isobj(a[1], O_LIST))
            vmerror(v, "join expects list");
        Obj *l = a[1].o;
        size_t n = 1, k = strlen(sep);
        for (int i = 0; i < l->as.list.n; i++)
            n += strlen(asstr(v, l->as.list.a[i])) + (i ? k : 0);
        char *s = mem(n), *p = s;
        for (int i = 0; i < l->as.list.n; i++) {
            if (i) {
                memcpy(p, sep, k);
                p += k;
            }
            const char *t = asstr(v, l->as.list.a[i]);
            size_t m = strlen(t);
            memcpy(p, t, m);
            p += m;
        }
        Obj *o = string(v, s);
        free(s);
        return objval(o);
    }
    case 25:
    case 26: {
        arity(v, id, argc, 2);
        const char *s = asstr(v, a[0]), *p = asstr(v, a[1]);
        size_t n = strlen(s), m = strlen(p);
        return boolean(m <= n && !memcmp(id == 25 ? s : s + n - m, p, m));
    }
    case 27: {
        arity(v, id, argc, 2);
        if (!isobj(a[1], O_CLASS))
            vmerror(v, "isinstance expects class");
        if (isobj(a[0], O_INSTANCE))
            for (Obj *c = a[0].o->as.inst.cls; c; c = c->as.cls.parent)
                if (c == a[1].o)
                    return boolean(1);
        return boolean(0);
    }
    case 28: {
        arity(v, id, argc, 1);
        if (!isobj(a[0], O_RANGE))
            vmerror_kind(v, "type", "list expects a range");
        uint64_t count = a[0].o->as.range.count;
        if (count > 1000000)
            vmerror_kind(v, "range", "list materialization limit: 1000000 elements");
        Obj *out = list(v);
        for (uint64_t i = 0; i < count; i++)
            append(v, out, integer(range_element(a[0].o, i)));
        return objval(out);
    }
    }
    return nil();
}

Obj *named_native_module(VM *v, const char *name, const char *const *names, size_t count,
                         int base) {
    char key[64];
    snprintf(key, sizeof key, "<native:%s>", name);
    Entry *entry = tablefind(&v->modules, key);
    if (entry)
        return entry->val.o;
    Obj *module = newobj(v, O_MODULE);
    module->as.mod.path = copystr(key);
    account(v, module, 0, strlen(key) + 1);
    module->as.mod.state = 1;
    module->as.mod.env = environment(v, NULL);
    for (size_t i = 0; i < count; i++) {
        Obj *function = newobj(v, O_NATIVE);
        function->as.native.id = base + (int)i;
        tableset(v, module->as.mod.env, &module->as.mod.env->as.env.tab, names[i],
                 objval(function));
    }
    tableset(v, NULL, &v->modules, key, objval(module));
    return module;
}

Obj *native_module(VM *v, const char *name) {
    if (!strcmp(name, "plot"))
        return plot_module(v);
    if (!strcmp(name, "_checkpoint"))
        return checkpoint_module(v);
    if (!strcmp(name, "fem"))
        return fem_module(v);
    if (!strcmp(name, "pde"))
        return pde_module(v);
    if (!strcmp(name, "solver"))
        return solver_module(v);
    if (!strcmp(name, "quant"))
        return quant_module(v);
    if (!strcmp(name, "_uncertainty"))
        return uncertainty_module(v);
    if (!strcmp(name, "_optimizer"))
        return optimizer_module(v);
    if (!strcmp(name, "units"))
        return units_module(v);
    if (!strcmp(name, "lowrank"))
        return lowrank_module(v);
    if (!strcmp(name, "_compute"))
        return compute_module(v);
    if (!strcmp(name, "tensor"))
        return typed_tensor_module(v);
    if (!strcmp(name, "special"))
        return special_module(v);
    if (!strcmp(name, "mathdoc"))
        return mathdoc_module(v);
    if (!strcmp(name, "complex"))
        return complex_module(v);
    if (!strcmp(name, "linop"))
        return linop_module(v);
    if (!strcmp(name, "random"))
        return random_module(v);
    if (!strcmp(name, "stats"))
        return stats_module(v);
    if (!strcmp(name, "linalg"))
        return linalg_module(v);
    if (!strcmp(name, "sparse"))
        return sparse_module(v);
    if (!strcmp(name, "high"))
        return high_module(v);
    if (!strcmp(name, "numc"))
        return scientific_module(v);
    Obj *extra = extra_module(v, name);
    if (extra)
        return extra;
    const int *ids = NULL;
    static const int math_ids[] = {5, 17, 18, 19, 20, 21, -1}, time_ids[] = {22, -1},
                     text_ids[] = {23, 24, 25, 26, -1}, io_ids[] = {11, 12, -1},
                     collections_ids[] = {0, 1, 2, 10, 13, -1};
    if (!strcmp(name, "math"))
        ids = math_ids;
    else if (!strcmp(name, "time"))
        ids = time_ids;
    else if (!strcmp(name, "text"))
        ids = text_ids;
    else if (!strcmp(name, "io"))
        ids = io_ids;
    else if (!strcmp(name, "collections"))
        ids = collections_ids;
    else
        return NULL;
    char key[64];
    snprintf(key, sizeof key, "<native:%s>", name);
    Entry *cached = tablefind(&v->modules, key);
    if (cached)
        return cached->val.o;
    Obj *o = newobj(v, O_MODULE);
    o->as.mod.path = copystr(key);
    account(v, o, 0, strlen(key) + 1);
    o->as.mod.state = 1;
    o->as.mod.env = environment(v, NULL);
    for (; *ids >= 0; ids++) {
        Entry *e = tablefind(&v->builtins->as.env.tab, names[*ids]);
        tableset(v, o->as.mod.env, &o->as.mod.env->as.env.tab, names[*ids], e->val);
    }
    if (!strcmp(name, "math"))
        tableset(v, o->as.mod.env, &o->as.mod.env->as.env.tab, "pi",
                 number(3.14159265358979323846));
    tableset(v, NULL, &v->modules, key, objval(o));
    return o;
}
