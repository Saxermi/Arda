#include "autodiff.h"
#include "activation_store.h"
void account(VM *v, Obj *o, size_t old, size_t now) {
    if (!o)
        return;
    if (sparse_owns_payload(o->type))
        v->sparse_bytes = v->sparse_bytes - old + now;
    o->bytes = o->bytes - old + now;
    v->bytes = v->bytes - old + now;
}

void markobj(VM *v, Obj *o) {
    if (!o || o->mark)
        return;
    o->mark = 1;
    if (v->ng == v->gcap) {
        v->gcap = v->gcap ? 2 * v->gcap : 128;
        v->gray = resize(v->gray, sizeof(Obj *) * v->gcap);
    }
    v->gray[v->ng++] = o;
}

void markval(VM *v, Val x) {
    if (x.t == V_OBJ)
        markobj(v, x.o);
}

void barrier(VM *v, Obj *owner, Val x) {
    if (v->marking && (!owner || owner->mark))
        markval(v, x);
}

Obj *newobj_extra(VM *v, OType t, size_t extra) {
    Obj *o = mem(sizeof *o + extra);
    o->type = t;
    o->next = v->objects;
    v->objects = o;
    o->bytes = sizeof *o + extra;
    v->bytes += o->bytes;
    v->live++;
    if (v->marking)
        markobj(v, o);
    return o;
}

Obj *newobj(VM *v, OType t) {
    return newobj_extra(v, t, 0);
}

uint64_t hashstr(const char *s) {
    uint64_t h = 14695981039346656037ULL;
    for (; *s; s++) {
        h ^= (unsigned char)*s;
        h *= 1099511628211ULL;
    }
    return h;
}

static Entry *slot(Entry *a, int cap, const char *key, uint64_t h) {
    size_t i = h & (unsigned)(cap - 1);
    while (a[i].key && (a[i].hash != h || strcmp(a[i].key, key)))
        i = (i + 1) & (unsigned)(cap - 1);
    return &a[i];
}
Entry *tablefind(Table *t, const char *key) {
    if (!t->cap)
        return NULL;
    Entry *e = slot(t->a, t->cap, key, hashstr(key));
    return e->key ? e : NULL;
}
// Layout stamps are unique within the owning VM, and change on each resize.
// A compiled name caches an index, not a pointer; GC cannot invalidate a hit.
// Tables never delete keys. Existing values may change without changing layout.
Entry *tablefind_cached(Table *t, Constant *key) {
    if (!t->cap)
        return NULL;
    if (key->stamp == t->stamp)
        return &t->a[key->hint];
    Entry *e = slot(t->a, t->cap, key->s, key->hash);
    if (!e->key)
        return NULL;
    key->hint = (int)(e - t->a);
    key->stamp = t->stamp;
    return e;
}

void tableset(VM *v, Obj *owner, Table *t, const char *key, Val x) {
    Entry *e = tablefind(t, key);
    if (!e) {
        if (t->n + 1 > t->cap * 3 / 4) {
            int nc = t->cap ? 2 * t->cap : 8;
            Entry *a = mem(sizeof(Entry) * (size_t)nc);
            for (int i = 0; i < t->cap; i++)
                if (t->a[i].key)
                    *slot(a, nc, t->a[i].key, t->a[i].hash) = t->a[i];
            account(v, owner, sizeof(Entry) * (size_t)t->cap, sizeof(Entry) * (size_t)nc);
            free(t->a);
            t->a = a;
            t->cap = nc;
            if (++v->next_table_stamp == 0)
                die("table stamp overflow");
            t->stamp = v->next_table_stamp;
        }
        uint64_t h = hashstr(key);
        e = slot(t->a, t->cap, key, h);
        e->key = copystr(key);
        e->hash = h;
        t->n++;
        account(v, owner, 0, strlen(key) + 1);
    }
    e->val = x;
    barrier(v, owner, x);
}

void tablefree(Table *t) {
    for (int i = 0; i < t->cap; i++)
        free(t->a[i].key);
    free(t->a);
}

Obj *environment(VM *v, Obj *parent) {
    Obj *o = newobj(v, O_ENV);
    o->as.env.parent = parent;
    return o;
}

static void intern_resize(VM *v, size_t cap) {
    Obj **b = mem(cap * sizeof *b);
    for (size_t i = 0; i < v->intern_cap; i++) {
        Obj *o = v->intern[i];
        while (o) {
            Obj *next = o->as.str.intern_next;
            size_t k = o->as.str.hash & (cap - 1);
            o->as.str.intern_next = b[k];
            b[k] = o;
            o = next;
        }
    }
    free(v->intern);
    v->intern = b;
    v->intern_cap = cap;
}

Obj *string(VM *v, const char *s) {
    uint64_t h = hashstr(s);
    if (!v->intern_cap)
        intern_resize(v, 64);
    size_t k = h & (v->intern_cap - 1);
    for (Obj *o = v->intern[k]; o; o = o->as.str.intern_next)
        if (o->as.str.hash == h && !strcmp(o->as.str.s, s)) {
            if (v->marking)
                markobj(v, o);
            return o;
        }
    if (v->intern_count >= v->intern_cap * 3 / 4) {
        intern_resize(v, v->intern_cap * 2);
        k = h & (v->intern_cap - 1);
    }
    Obj *o = newobj(v, O_STRING);
    o->as.str.s = copystr(s);
    o->as.str.len = strlen(s);
    o->as.str.hash = h;
    o->as.str.intern_next = v->intern[k];
    v->intern[k] = o;
    v->intern_count++;
    account(v, o, 0, o->as.str.len + 1);
    return o;
}

static void intern_prune(VM *v) {
    for (size_t i = 0; i < v->intern_cap; i++) {
        Obj **p = &v->intern[i];
        while (*p) {
            Obj *o = *p;
            if (!o->mark) {
                *p = o->as.str.intern_next;
                v->intern_count--;
            } else
                p = &o->as.str.intern_next;
        }
    }
    if (v->intern_cap > 64 && v->intern_count < v->intern_cap / 8)
        intern_resize(v, v->intern_cap / 2);
}

Obj *list(VM *v) {
    return newobj(v, O_LIST);
}

void append(VM *v, Obj *o, Val x) {
    if (o->as.list.n == o->as.list.cap) {
        int nc = o->as.list.cap ? 2 * o->as.list.cap : 8;
        o->as.list.a = resize(o->as.list.a, sizeof(Val) * (size_t)nc);
        account(v, o, sizeof(Val) * (size_t)o->as.list.cap, sizeof(Val) * (size_t)nc);
        o->as.list.cap = nc;
    }
    o->as.list.a[o->as.list.n++] = x;
    barrier(v, o, x);
}

static void marktable(VM *v, Table *t) {
    for (int i = 0; i < t->cap; i++)
        if (t->a[i].key)
            markval(v, t->a[i].val);
}

static void markproto(Proto *p) {
    if (p)
        p->root->marked = 1;
}

static void trace(VM *v, Obj *o) {
    switch (o->type) {
    case O_STRING:
    case O_RANGE:
    case O_OFFLOAD_PLAN:
    case O_COMPUTE_PLAN:
    case O_NATIVE:
    case O_RNG:
    case O_MOMENTS:
    case O_COMPLEX:
    case O_BUFFER:
    case O_TENSOR_BLOCKS:
    case O_TENSOR_OUTPUT:
    case O_SPARSE:
    case O_SPARSE_BUILDER:
        break;
    case O_FEM_MESH:
        markobj(v, o->as.fem.nodes);
        markobj(v, o->as.fem.elements);
        markobj(v, o->as.fem.hierarchy);
        break;
    case O_FACTOR:
        markobj(v, o->as.factor.packed);
        break;
    case O_LINOP:
        markobj(v, o->as.linop.first);
        markobj(v, o->as.linop.second);
        break;
    case O_ARRAY:
        markobj(v, o->as.array.buffer);
        break;
    case O_QUANTIZED:
        markobj(v, o->as.quantized.buffer);
        break;
    case O_TYPED_TENSOR:
        markobj(v, o->as.tensor.buffer);
        break;
    case O_LOWRANK:
        markobj(v, o->as.lowrank.coefficients);
        for (int i = 0; i < o->as.lowrank.rank; i++)
            markobj(v, o->as.lowrank.factor[i]);
        break;
    case O_QUANTITY:
        markval(v, o->as.quantity.value);
        markobj(v, o->as.quantity.symbol);
        break;
    case O_AD:
        markval(v, o->as.ad->value);
        markval(v, o->as.ad->grad);
        markval(v, o->as.ad->left);
        markval(v, o->as.ad->right);
        markobj(v, o->as.ad->parents[0]);
        markobj(v, o->as.ad->parents[1]);
        markobj(v, o->as.ad->pattern);
        ad_trace_checkpoint(v, o->as.ad);
        break;
    case O_LIST:
        for (int i = 0; i < o->as.list.n; i++)
            markval(v, o->as.list.a[i]);
        break;
    case O_MAP:
        marktable(v, &o->as.map.tab);
        break;
    case O_ENV:
        markproto(o->as.env.params);
        markobj(v, o->as.env.parent);
        marktable(v, &o->as.env.tab);
        if (o->as.env.params)
            for (int i = 0; i < o->as.env.params->arity; i++)
                markval(v, o->as.env.slots[i]);
        break;
    case O_FUNC:
        markproto(o->as.func.p);
        markobj(v, o->as.func.env);
        break;
    case O_CLASS:
        markobj(v, o->as.cls.parent);
        marktable(v, &o->as.cls.methods);
        break;
    case O_INSTANCE:
        markobj(v, o->as.inst.cls);
        marktable(v, &o->as.inst.fields);
        break;
    case O_BOUND:
        markobj(v, o->as.bound.fn);
        markval(v, o->as.bound.self);
        break;
    case O_MODULE:
        markobj(v, o->as.mod.env);
        break;
    case O_ITER:
        markval(v, o->as.iter.source);
        break;
    }
}

static void roots(VM *v) {
    markobj(v, v->global);
    markobj(v, v->builtins);
    markobj(v, v->sensitivity_entry);
    markval(v, v->thrown);
    marktable(v, &v->modules);
    for (int i = 0; i < v->sp; i++)
        markval(v, v->stack[i]);
    for (int i = 0; i < v->nf; i++) {
        Frame *f = &v->frames[i];
        markproto(f->p);
        markobj(v, f->env);
        markobj(v, f->locals);
        markobj(v, f->module);
        markval(v, f->constructed);
    }
    for (int i = 0; i < v->nh; i++)
        markobj(v, v->handlers[i].env);
}

static void release(VM *v, Obj *o) {
    switch (o->type) {
    case O_STRING:
        free(o->as.str.s);
        break;
    case O_SPARSE:
    case O_SPARSE_BUILDER:
        sparse_release_payload(o);
        break;
    case O_AD:
        free(o->as.ad->checkpoint);
        free(o->as.ad);
        break;
    case O_OFFLOAD_PLAN:
        offload_plan_release(o->as.offload_plan);
        break;
    case O_BUFFER:
        tensor_buffer_release(v, &o->as.buffer);
        break;
    case O_TENSOR_BLOCKS:
        tensor_blocks_close(o);
        break;
    case O_TENSOR_OUTPUT:
        tensor_output_abort(o);
        break;
    case O_LIST:
        free(o->as.list.a);
        break;
    case O_MAP:
        tablefree(&o->as.map.tab);
        break;
    case O_ENV:
        tablefree(&o->as.env.tab);
        break;
    case O_CLASS:
        free(o->as.cls.name);
        tablefree(&o->as.cls.methods);
        break;
    case O_INSTANCE:
        tablefree(&o->as.inst.fields);
        break;
    case O_MODULE:
        free(o->as.mod.path);
        break;
    default:
        break;
    }
    free(o);
}
// Incremental tracing; sweep is atomic. GC runs only between complete instructions.
// All mutating operations use an insertion barrier and new objects enter gray.
void gcstep(VM *v, int force) {
    if (!v->marking) {
        if (!force && !v->stress && v->bytes < v->next_gc)
            return;
        v->marking = 1;
    }
    roots(v);
    size_t budget = force ? SIZE_MAX : 32;
    while (v->ng && budget--) {
        Obj *o = v->gray[--v->ng];
        trace(v, o);
    }
    if (v->ng)
        return;
    intern_prune(v);
    Obj **p = &v->objects;
    while (*p) {
        Obj *o = *p;
        if (o->mark) {
            o->mark = 0;
            p = &o->next;
        } else {
            *p = o->next;
            if (sparse_owns_payload(o->type))
                v->sparse_bytes -= o->bytes - sizeof *o;
            v->bytes -= o->bytes;
            v->live--;
            release(v, o);
        }
    }
    /* Sweep code after objects: live closures, parameter environments and
     * active frames have marked their compilation roots during tracing. */
    int keep = 0;
    for (int i = 0; i < v->np; i++) {
        Proto *code = v->programs[i];
        if (code->marked) {
            code->marked = 0;
            v->programs[keep++] = code;
        } else
            freeproto(code);
    }
    v->np = keep;
    v->marking = 0;
    v->cycles++;
    v->next_gc = v->bytes > 32768 ? v->bytes * 2 : 65536;
}

void gcfull(VM *v) {
    gcstep(v, 1);
}

void destroy_objects(VM *v) {
    Obj *o = v->objects;
    while (o) {
        Obj *next = o->next;
        release(v, o);
        o = next;
    }
    free(v->intern);
    free(v->gray);
    tablefree(&v->modules);
}
