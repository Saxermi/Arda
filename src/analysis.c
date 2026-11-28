#include "analysis.h"
#include "frontend.h"

enum { ANALYSIS_NODES = 192, ANALYSIS_NAMES = 128, ANALYSIS_DEPTH = 48, ANALYSIS_VISITS = 1024 };
typedef enum { SCALAR, SEQUENCE, BRANCH, RANGE } CostKind;

typedef struct {
    int parameter;
    int64_t literal;
} Endpoint;

typedef struct {
    CostKind kind;
    int a, b;
    Endpoint first, stop;
    int64_t step;
} CostNode;

struct FunctionAnalysis {
    int root, rank, loops, branches;
    bool recursive, uncertain;
    CostNode nodes[];
};

typedef struct {
    const Node *function;
    CostNode nodes[ANALYSIS_NODES];
    const char *names[ANALYSIS_NAMES];
    int count, names_used, loop_depth, rank, loops, branches, visits;
    bool recursive, uncertain;
} Analysis;

typedef struct {
    uint64_t lo, hi, calls;
} Bounds;

static int record(Analysis *a, CostNode n) {
    if (a->count == ANALYSIS_NODES)
        return -1;
    a->nodes[a->count] = n;
    return a->count++;
}

static int pair(Analysis *a, CostKind kind, int left, int right) {
    if (left < 0 || right < 0)
        return -1;
    if (kind == SEQUENCE && (!left || !right))
        return left + right;
    return record(a, (CostNode){.kind = kind, .a = left, .b = right});
}

static int named(const Analysis *a, const char *name) {
    for (int i = a->names_used - 1; i >= 0; i--)
        if (!strcmp(a->names[i], name))
            return i;
    return -1;
}

static bool bind(Analysis *a, const char *name) {
    if (a->names_used == ANALYSIS_NAMES || !strcmp(name, "range") ||
        !strcmp(name, a->function->str))
        return false;
    int i = named(a, name);
    if (i >= 0 && i < a->function->n_names)
        return false;
    a->names[a->names_used++] = name;
    return true;
}

static bool visit(Analysis *a, int depth) {
    return depth <= ANALYSIS_DEPTH && ++a->visits <= ANALYSIS_VISITS;
}

static bool expression(Analysis *a, const Node *n, int depth, int *ops, int *calls) {
    if (!n || !visit(a, depth))
        return false;
    switch (n->t) {
    case N_NUM:
    case N_NIL:
    case N_BOOL:
        return true;
    case N_VAR:
        return named(a, n->str) >= 0;
    case N_UNARY:
        (*ops)++;
        return expression(a, n->a, depth + 1, ops, calls);
    case N_BINARY:
        if (n->op == T_IN)
            return false;
        a->uncertain |= n->op == T_AND || n->op == T_OR;
        (*ops)++;
        return expression(a, n->a, depth + 1, ops, calls) &&
               expression(a, n->b, depth + 1, ops, calls);
    case N_ASSIGN:
        return n->a->t == N_VAR && named(a, n->a->str) >= a->function->n_names &&
               expression(a, n->b, depth + 1, ops, calls);
    case N_CALL:
        if (n->a->t != N_VAR || strcmp(n->a->str, a->function->str) || named(a, n->a->str) >= 0 ||
            n->count != a->function->n_names)
            return false;
        a->recursive = true;
        (*calls)++;
        for (int i = 0; i < n->count; i++)
            if (!expression(a, n->items[i], depth + 1, ops, calls))
                return false;
        return true;
    default:
        return false;
    }
}

static int expression_cost(Analysis *a, const Node *n, int depth) {
    int ops = 0, calls = 0;
    if (!expression(a, n, depth, &ops, &calls))
        return -1;
    if (!ops && !calls)
        return 0;
    return record(a, (CostNode){.kind = SCALAR, .a = ops, .b = calls});
}

static bool literal(const Node *n, int64_t *result) {
    if (n->t == N_NUM && n->isint == 1) {
        *result = n->integer;
        return true;
    }
    if (n->t == N_UNARY && n->op == T_MINUS && n->a->t == N_NUM && n->a->isint == 1 &&
        n->a->integer != INT64_MIN) {
        *result = -n->a->integer;
        return true;
    }
    return false;
}

static bool endpoint(const Analysis *a, const Node *n, Endpoint *end) {
    if (n->t == N_VAR) {
        int i = named(a, n->str);
        if (i < 0 || i >= a->function->n_names)
            return false;
        *end = (Endpoint){.parameter = i};
        return true;
    }
    end->parameter = -1;
    return literal(n, &end->literal);
}

static int statement(Analysis *a, const Node *n, int depth);

static int range_cost(Analysis *a, const Node *loop, int depth) {
    const Node *r = loop->a;
    if (r->t != N_CALL || r->a->t != N_VAR || strcmp(r->a->str, "range") ||
        named(a, "range") >= 0 || r->count < 1 || r->count > 3)
        return -1;
    CostNode n = {.kind = RANGE, .first = {.parameter = -1}, .step = 1};
    if ((r->count > 1 && !endpoint(a, r->items[0], &n.first)) ||
        !endpoint(a, r->items[r->count > 1], &n.stop) ||
        (r->count == 3 && (!literal(r->items[2], &n.step) || !n.step)))
        return -1;
    int names = a->names_used;
    if (!bind(a, loop->str))
        return -1;
    a->loops++;
    if (++a->loop_depth > a->rank)
        a->rank = a->loop_depth;
    n.a = statement(a, loop->b, depth + 1);
    a->loop_depth--;
    a->names_used = names;
    return n.a < 0 ? -1 : record(a, n);
}

static int statement(Analysis *a, const Node *n, int depth) {
    if (!visit(a, depth))
        return -1;
    switch (n->t) {
    case N_BLOCK: {
        int sum = 0, names = a->names_used;
        for (int i = 0; i < n->count && sum >= 0; i++) {
            if (n->items[i]->t == N_RETURN && (depth || i != n->count - 1))
                a->uncertain = true;
            sum = pair(a, SEQUENCE, sum, statement(a, n->items[i], depth + 1));
        }
        a->names_used = names;
        return sum;
    }
    case N_DECLARE: {
        int cost = expression_cost(a, n->a, depth + 1);
        return cost >= 0 && bind(a, n->str) ? cost : -1;
    }
    case N_RETURN:
    case N_EXPR:
        return n->a ? expression_cost(a, n->a, depth + 1) : 0;
    case N_FOR:
        return range_cost(a, n, depth);
    case N_IF: {
        a->branches++;
        int condition = expression_cost(a, n->a, depth + 1);
        int yes = statement(a, n->b, depth + 1);
        int no = n->c ? statement(a, n->c, depth + 1) : 0;
        return pair(a, SEQUENCE, condition, pair(a, BRANCH, yes, no));
    }
    case N_PASS:
        return 0;
    default:
        return -1;
    }
}

struct FunctionAnalysis *function_analyze(const Node *fn) {
    if (fn->n_names > ANALYSIS_NAMES)
        return NULL;
    Analysis a = {.function = fn, .count = 1, .names_used = fn->n_names};
    for (int i = 0; i < fn->n_names; i++) {
        if (fn->defaults && fn->defaults[i])
            return NULL;
        a.names[i] = fn->names[i];
    }
    int root = fn->t == N_LAMBDA ? expression_cost(&a, fn->a, 0) : statement(&a, fn->a, 0);
    if (root < 0 || (a.recursive && a.loops))
        return NULL;
    struct FunctionAnalysis *result =
        front_mem(sizeof *result + sizeof(CostNode) * (size_t)a.count);
    *result =
        (struct FunctionAnalysis){root, a.rank, a.loops, a.branches, a.recursive, a.uncertain};
    memcpy(result->nodes, a.nodes, sizeof(CostNode) * (size_t)a.count);
    return result;
}

static Val lookup(Val map, const char *name) {
    if (!isobj(map, O_MAP))
        return nil();
    Entry *e = tablefind(&map.o->as.map.tab, name);
    return e ? e->val : nil();
}

static bool interval(const Proto *p, Val parameters, Endpoint e, int64_t *lo, int64_t *hi) {
    if (e.parameter < 0) {
        *lo = *hi = e.literal;
        return true;
    }
    Val value = lookup(parameters, p->params[e.parameter]);
    if (value.t == V_INT) {
        *lo = *hi = value.i;
        return true;
    }
    if (!isobj(value, O_LIST) || value.o->as.list.n != 2)
        return false;
    const Val *bound = value.o->as.list.a;
    if (bound[0].t != V_INT || bound[1].t != V_INT || bound[0].i > bound[1].i)
        return false;
    *lo = bound[0].i;
    *hi = bound[1].i;
    return true;
}

static uint64_t length(int64_t first, int64_t stop, int64_t step) {
    if (step > 0)
        return first >= stop ? 0 : 1 + ((uint64_t)stop - (uint64_t)first - 1) / (uint64_t)step;
    return first <= stop ? 0 : 1 + ((uint64_t)first - (uint64_t)stop - 1) / (0 - (uint64_t)step);
}

static bool combine(uint64_t a, uint64_t b, bool multiply, uint64_t *result) {
    if (a > INT64_MAX || b > INT64_MAX)
        return false;
    if (multiply ? a && b > (uint64_t)INT64_MAX / a : b > (uint64_t)INT64_MAX - a)
        return false;
    *result = multiply ? a * b : a + b;
    return true;
}

static bool evaluate(const Proto *p, int index, Val parameters, Bounds *out) {
    const CostNode *n = &p->analysis->nodes[index];
    if (n->kind == SCALAR) {
        *out = (Bounds){(uint64_t)n->a, (uint64_t)n->a, (uint64_t)n->b};
        return true;
    }
    Bounds a, b;
    if (!evaluate(p, n->a, parameters, &a))
        return false;
    if (n->kind == RANGE) {
        int64_t flo, fhi, slo, shi;
        if (!interval(p, parameters, n->first, &flo, &fhi) ||
            !interval(p, parameters, n->stop, &slo, &shi))
            return false;
        uint64_t lo = length(n->step > 0 ? fhi : flo, n->step > 0 ? slo : shi, n->step);
        uint64_t hi = length(n->step > 0 ? flo : fhi, n->step > 0 ? shi : slo, n->step);
        return combine(lo, a.lo, true, &out->lo) && combine(hi, a.hi, true, &out->hi) &&
               combine(hi, a.calls, true, &out->calls);
    }
    if (!evaluate(p, n->b, parameters, &b))
        return false;
    if (n->kind == BRANCH) {
        *out = (Bounds){a.lo < b.lo ? a.lo : b.lo, a.hi > b.hi ? a.hi : b.hi,
                        a.calls > b.calls ? a.calls : b.calls};
        return true;
    }
    return combine(a.lo, b.lo, false, &out->lo) && combine(a.hi, b.hi, false, &out->hi) &&
           combine(a.calls, b.calls, false, &out->calls);
}

static void put(VM *v, Obj *map, const char *key, Val value) {
    tableset(v, map, &map->as.map.tab, key, value);
}

static const struct FunctionAnalysis *resolved(VM *v, Val fn) {
    if (!isobj(fn, O_FUNC))
        return NULL;
    Proto *p = fn.o->as.func.p;
    const struct FunctionAnalysis *a = p->analysis;
    if (!a || (a->loops && !vm_uses_builtin(v, fn.o->as.func.env, "range")) ||
        (a->recursive && !vm_resolves_function(fn.o)))
        return NULL;
    return a;
}

Val function_cost(VM *v, Val fn, Val spec) {
    const struct FunctionAnalysis *a = resolved(v, fn);
    if (!a)
        return nil();
    if (spec.t != V_NIL && !isobj(spec, O_MAP))
        vmerror_kind(v, "type", "static cost specification must be a map");
    Val parameters = lookup(spec, "parameters");
    if (parameters.t != V_NIL && !isobj(parameters, O_MAP))
        vmerror_kind(v, "type", "static cost parameters must be a map");
    Bounds b;
    if (!evaluate(fn.o->as.func.p, a->root, parameters, &b))
        return nil();
    uint64_t invocations = 1;
    Val depth = lookup(spec, "recursion_depth");
    if (a->recursive) {
        if (depth.t != V_INT || depth.i < 1 || depth.i > 512)
            return nil();
        /* Upper bound on a tree with at most b.calls children per node. */
        for (int i = 1; i < depth.i; i++)
            if (!combine(invocations, b.calls, true, &invocations) ||
                !combine(invocations, 1, false, &invocations))
                return nil();
        if (!combine(b.hi, invocations, true, &b.hi))
            return nil();
    }
    if (a->uncertain || a->recursive)
        b.lo = 0;
    Obj *out = newobj(v, O_MAP);
    put(v, out, "estimated", boolean(true));
    put(v, out, "min_scalar_operations", integer((int64_t)b.lo));
    put(v, out, "max_scalar_operations", integer((int64_t)b.hi));
    put(v, out, "max_invocations", integer((int64_t)invocations));
    put(v, out, "recursion_depth_assumption", a->recursive ? depth : nil());
    return objval(out);
}

const char *function_algorithm(VM *v, Val fn) {
    const struct FunctionAnalysis *a = resolved(v, fn);
    if (!a)
        return NULL;
    if (a->recursive)
        return "Source-level direct recursive calls with a supplied depth bound";
    if (a->loops)
        return "Source-level ordered scalar updates over native half-open ranges";
    return "Source-level scalar expressions and conditional branches";
}

Val function_plan(VM *v, Val fn, Val spec) {
    const struct FunctionAnalysis *a = resolved(v, fn);
    if (!a)
        return nil();
    Val cost = function_cost(v, fn, spec);
    Obj *out = newobj(v, O_MAP);
    put(v, out, "kind", objval(string(v, "static_function")));
    put(v, out, "algorithm", objval(string(v, function_algorithm(v, fn))));
    put(v, out, "loops", integer(a->loops));
    put(v, out, "loop_depth", integer(a->rank));
    put(v, out, "branches", integer(a->branches));
    put(v, out, "recursive", boolean(a->recursive));
    put(v, out, "cost", cost);
    put(v, out, "assumptions",
        objval(string(
            v, "Numeric scalar operands and supplied Int64 intervals. Direct recursive calls are "
               "conditional on their supplied depth bound. Source-level counts exclude VM "
               "overhead.")));
    return objval(out);
}

Obj *contract_module(VM *v) {
    const char *const names[] = {"require", "ensure", "assume"};
    return named_native_module(v, "contract", names, sizeof names / sizeof *names, CONTRACT_BASE);
}

Val contract_call(VM *v, int id, int argc, Val *args) {
    const char *const code[] = {"precondition", "postcondition", "assumption"};
    int method = id - CONTRACT_BASE;
    if ((argc != 1 && argc != 2) || method < 0 || method > 2)
        vmerror_kind(v, "arity", "contract expects a Boolean and an optional message");
    if (args[0].t != V_BOOL)
        vmerror_kind(v, "type", "contract predicate must be Boolean");
    const char *message = argc == 2 ? asstr(v, args[1]) : "contract predicate was false";
    if (!args[0].n)
        vmerror_kind(v, code[method], "%s", message);
    return args[0];
}
