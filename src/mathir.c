#include "frontend.h"

#include <float.h>

/* A bounded, source-level mathematical IR, owned by a compilation prototype.
 * All names are copied into the same allocation as the immutable nodes.
 * No VM values, AST pointers or runtime call evaluation are retained. */
enum { MAX_NODES = 128, MAX_DEPTH = 32, MAX_NAME = 128, MAX_PARAMS = 32 };
typedef enum {
    MI_INT,
    MI_REAL,
    MI_PARAM,
    MI_EXTERNAL,
    MI_LOCAL,
    MI_INDEX,
    MI_SUBSCRIPT,
    MI_STATE,
    MI_NEG,
    MI_ADD,
    MI_SUB,
    MI_MUL,
    MI_DIV,
    MI_CALL,
    MI_BIND
} Kind;
typedef struct {
    Kind kind;
    int a, b, next;
    int64_t integer; // Literal value, or eligibility for a bare builtin call.
    double real;
    const char *name;
} MathNode;
typedef enum { NO_LOOP, RECURRENCE, SUM, PRODUCT } LoopKind;
typedef struct {
    LoopKind kind;
    int index, accumulator, start, stop, step, update;
} MathLoop;
struct MathIR {
    int count, result, has_calls, has_external, has_subscript;
    const MathLoop *loop;
    MathNode nodes[];
};
typedef struct {
    MathNode nodes[MAX_NODES];
    int count, has_calls, has_external, has_subscript;
    const Node *fn;
    MathLoop loop;
} Extract;

/* Accepted 0.15.4 policy: compact metadata is on by default. */
static int enabled = 1;

void mathir_set_enabled(int value) {
    enabled = value != 0;
}

static int valid_name(const char *name) {
    return name && strlen(name) <= MAX_NAME;
}

static int add(Extract *e, Kind kind, const char *name, int a, int b) {
    if (e->count == MAX_NODES || (name && !valid_name(name)))
        return -1;
    int i = e->count++;
    e->nodes[i] = (MathNode){.kind = kind, .a = a, .b = b, .next = -1, .name = name};
    return i;
}

static int binding(const Extract *e, const char *name) {
    for (int i = e->count - 1; i >= 0; i--)
        if (e->nodes[i].kind == MI_BIND && !strcmp(e->nodes[i].name, name))
            return i;
    return -1;
}

static int parameter(const Extract *e, const char *name) {
    for (int i = 0; i < e->fn->n_names; i++)
        if (!strcmp(e->fn->names[i], name))
            return i;
    return -1;
}

static int expression(Extract *e, const Node *n, int depth) {
    if (!n || depth >= MAX_DEPTH)
        return -1;
    int a, b, i;
    Kind kind;
    switch (n->t) {
    case N_NUM:
        if (!n->isint && !isfinite(n->num))
            return -1;
        i = add(e, n->isint ? MI_INT : MI_REAL, NULL, -1, -1);
        if (i >= 0) {
            e->nodes[i].integer = n->integer;
            e->nodes[i].real = n->num;
        }
        return i;
    case N_VAR:
        if (e->loop.kind != NO_LOOP && !strcmp(e->nodes[e->loop.index].name, n->str))
            return add(e, MI_INDEX, n->str, -1, -1);
        a = binding(e, n->str);
        if (a >= 0)
            return add(e, e->loop.kind != NO_LOOP && a == e->loop.accumulator ? MI_STATE : MI_LOCAL,
                       n->str, a, -1);
        if (parameter(e, n->str) >= 0)
            return add(e, MI_PARAM, n->str, -1, -1);
        e->has_external = 1;
        return add(e, MI_EXTERNAL, n->str, -1, -1);
    case N_UNARY:
        if (n->op != T_MINUS)
            return -1;
        a = expression(e, n->a, depth + 1);
        return a < 0 ? -1 : add(e, MI_NEG, NULL, a, -1);
    case N_INDEX:
        if (e->loop.kind != NO_LOOP)
            return -1;
        a = expression(e, n->a, depth + 1);
        if (a < 0)
            return -1;
        b = expression(e, n->b, depth + 1);
        e->has_subscript = 1;
        return b < 0 ? -1 : add(e, MI_SUBSCRIPT, NULL, a, b);
    case N_BINARY:
        switch (n->op) {
        case T_PLUS:
            kind = MI_ADD;
            break;
        case T_MINUS:
            kind = MI_SUB;
            break;
        case T_STAR:
            kind = MI_MUL;
            break;
        case T_SLASH:
            kind = MI_DIV;
            break;
        default:
            return -1;
        }
        a = expression(e, n->a, depth + 1);
        if (a < 0)
            return -1;
        b = expression(e, n->b, depth + 1);
        return b < 0 ? -1 : add(e, kind, NULL, a, b);
    case N_CALL: {
        /* Source names are not identities: mutable lexical environments can
         * rebind even a builtin. Document the spelling, not a guessed callee. */
        const Node *callee = n->a;
        if (e->loop.kind != NO_LOOP || !callee ||
            (callee->t != N_VAR && (callee->t != N_ATTR || !callee->a || callee->a->t != N_VAR)))
            return -1;
        int qualifier = -1;
        if (callee->t == N_ATTR) {
            qualifier = add(e, MI_EXTERNAL, callee->a->str, -1, -1);
            if (qualifier < 0)
                return -1;
        }
        e->has_calls = 1;
        int first = -1, last = -1;
        for (int j = 0; j < n->count; j++) {
            int arg = expression(e, n->items[j], depth + 1);
            if (arg < 0)
                return -1;
            if (last >= 0)
                e->nodes[last].next = arg;
            else
                first = arg;
            last = arg;
        }
        i = add(e, MI_CALL, callee->str, first, qualifier);
        if (i >= 0 && qualifier < 0 && n->count == 1 && binding(e, callee->str) < 0 &&
            parameter(e, callee->str) < 0)
            e->nodes[i].integer = 1;
        return i;
    }
    default:
        return -1;
    }
}

// A literal can be valid (1), invalid (-1), or unavailable for a symbolic bound (0).
static int range_literal(const Node *n, int64_t *value) {
    int negative = n->t == N_UNARY && n->op == T_MINUS;
    if (negative)
        n = n->a;
    if (!n || n->t != N_NUM)
        return 0;
    long double x = n->isint ? (long double)n->integer : n->num;
    if (!isfinite(x) || fabsl(x) > 9e15L || truncl(x) != x)
        return -1;
    *value = negative ? -(int64_t)x : (int64_t)x;
    return 1;
}

static int range_bounds_valid(const Node *first, const Node *last, const Node *stride) {
    int64_t start, stop, step;
    int known_start = range_literal(first, &start);
    int known_stop = range_literal(last, &stop);
    if (known_start < 0 || known_stop < 0 || range_literal(stride, &step) != 1 || !step)
        return 0;
    if (known_start && known_stop) {
        int64_t distance = step > 0 ? stop - start : start - stop;
        int64_t width = step > 0 ? step : -step;
        return distance <= 0 || (distance - 1) / width < INT64_C(1000000);
    }
    return 1;
}

static int loop_expression(Extract *e, const Node *loop, const Node *ret) {
    const Node *range = loop->a;
    const Node *body = loop->b;
    if (range->t != N_CALL || range->a->t != N_VAR || strcmp(range->a->str, "range") ||
        range->count < 1 || range->count > 3 || parameter(e, "range") >= 0 ||
        binding(e, "range") >= 0 || !valid_name(loop->str) || parameter(e, loop->str) >= 0 ||
        binding(e, loop->str) >= 0 || body->count != 1 || body->items[0]->t != N_EXPR || !ret ||
        ret->t != N_VAR)
        return 0;
    const Node *assignment = body->items[0]->a;
    if (assignment->t != N_ASSIGN || assignment->a->t != N_VAR ||
        strcmp(assignment->a->str, ret->str))
        return 0;
    int accumulator = binding(e, ret->str);
    if (accumulator < 0)
        return 0;
    // Bounds are evaluated before the loop variable and iteration state exist.
    const Node zero = {.t = N_NUM, .isint = 1};
    const Node one = {.t = N_NUM, .isint = 1, .integer = 1};
    const Node *first = range->count == 1 ? &zero : range->items[0];
    const Node *last = range->items[range->count == 1 ? 0 : 1];
    const Node *stride = range->count == 3 ? range->items[2] : &one;
    if (!range_bounds_valid(first, last, stride))
        return 0;
    int start = expression(e, first, 0);
    int stop = expression(e, last, 0);
    int step = expression(e, stride, 0);
    int index = add(e, MI_INDEX, loop->str, -1, -1);
    if (start < 0 || stop < 0 || step < 0 || index < 0 || e->has_calls || e->has_external ||
        e->has_subscript)
        return 0;
    e->loop = (MathLoop){.kind = RECURRENCE,
                         .index = index,
                         .accumulator = accumulator,
                         .start = start,
                         .stop = stop,
                         .step = step};
    int update = expression(e, assignment->b, 0);
    if (update < 0 || e->has_external)
        return 0;
    e->loop.update = update;
    const MathNode *value = &e->nodes[update];
    if ((value->kind == MI_ADD || value->kind == MI_MUL) && e->nodes[value->a].kind == MI_STATE) {
        int occurrences = 0;
        for (int i = index + 1; i <= update; i++)
            occurrences += e->nodes[i].kind == MI_STATE;
        if (occurrences == 1)
            e->loop.kind = value->kind == MI_ADD ? SUM : PRODUCT;
    }
    return 1;
}

static MathIR *extract(const Node *fn) {
    if (!fn || !valid_name(fn->str) || fn->n_names > MAX_PARAMS)
        return NULL;
    for (int i = 0; i < fn->n_names; i++)
        if (!valid_name(fn->names[i]) || (fn->defaults && fn->defaults[i]))
            return NULL;
    Extract e = {.fn = fn};
    int result = -1;
    if (fn->t == N_LAMBDA)
        result = expression(&e, fn->a, 0);
    else {
        const Node *body = fn->a;
        if (!body || body->count < 1 || body->count > MAX_NODES)
            return NULL;
        for (int i = 0; i < body->count; i++) {
            const Node *s = body->items[i];
            if (s->t == N_DECLARE && i < body->count - 1) {
                /* Rebinding and shadowing are deliberately outside v1. */
                if (!valid_name(s->str) || parameter(&e, s->str) >= 0 || binding(&e, s->str) >= 0)
                    return NULL;
                int value = expression(&e, s->a, 0);
                if (value < 0 || add(&e, MI_BIND, s->str, value, -1) < 0)
                    return NULL;
            } else if (s->t == N_FOR && i == body->count - 2 && body->items[i + 1]->t == N_RETURN) {
                if (e.has_subscript || !loop_expression(&e, s, body->items[i + 1]->a))
                    return NULL;
            } else if (s->t == N_RETURN && i == body->count - 1) {
                // The return in a supported loop is the final accumulator.
                result = expression(&e, s->a, 0);
            } else
                return NULL;
        }
    }
    if (result < 0)
        return NULL;
    /* A reference to an external name that is later declared locally could
     * become ambiguous in a symbolic formula. Do not document that pattern. */
    for (int i = 0; i < e.count; i++)
        if (e.nodes[i].kind == MI_EXTERNAL && binding(&e, e.nodes[i].name) >= 0)
            return NULL;
    size_t bytes = sizeof(MathIR) + (size_t)e.count * sizeof(MathNode);
    size_t loop_bytes = e.loop.kind == NO_LOOP ? 0 : sizeof(MathLoop);
    size_t names = 0;
    for (int i = 0; i < e.count; i++)
        if (e.nodes[i].name)
            names += strlen(e.nodes[i].name) + 1;
    /* Both counts are bounded by the constants above. */
    MathIR *ir = front_mem(bytes + loop_bytes + names);
    ir->count = e.count;
    ir->result = result;
    ir->has_calls = e.has_calls;
    ir->has_external = e.has_external;
    ir->has_subscript = e.has_subscript;
    memcpy(ir->nodes, e.nodes, (size_t)e.count * sizeof(MathNode));
    char *dst = (char *)ir + bytes;
    ir->loop = NULL;
    if (loop_bytes) {
        memcpy(dst, &e.loop, loop_bytes);
        ir->loop = (const MathLoop *)dst;
        dst += loop_bytes;
    }
    for (int i = 0; i < ir->count; i++)
        if (ir->nodes[i].name) {
            size_t len = strlen(ir->nodes[i].name) + 1;
            memcpy(dst, ir->nodes[i].name, len);
            ir->nodes[i].name = dst;
            dst += len;
        }
    return ir;
}

MathIR *mathir_extract(const Node *fn) {
    return enabled ? extract(fn) : NULL;
}

typedef struct {
    char *out;
    size_t size, used;
    int failed;
} Writer;

static void put(Writer *w, const char *s) {
    if (w->failed)
        return;
    size_t n = strlen(s);
    if (n >= w->size - w->used) {
        w->failed = 1;
        return;
    }
    memcpy(w->out + w->used, s, n + 1);
    w->used += n;
}

/* Identifiers are validated UTF-8 by the lexer. Use only safe ASCII LaTeX
 * syntax: underscores are escaped; non-ASCII code points are explicit U+hex
 * tokens so the output does not depend on a particular TeX font or engine. */
static void identifier(Writer *w, const char *name) {
    const unsigned char *s = (const unsigned char *)name;
    if (s[0] && !s[1] && ((s[0] >= 'a' && s[0] <= 'z') || (s[0] >= 'A' && s[0] <= 'Z'))) {
        char one[] = {(char)s[0], 0};
        put(w, one);
        return;
    }
    put(w, "\\mathrm{");
    while (*s && !w->failed) {
        unsigned int cp = *s++;
        if (cp < 128) {
            if (cp == '_')
                put(w, "\\_");
            else if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') ||
                     (cp >= '0' && cp <= '9')) {
                char one[] = {(char)cp, 0};
                put(w, one);
            } else {
                w->failed = 1;
                return;
            }
        } else {
            int tails = cp < 0xe0 ? 1 : cp < 0xf0 ? 2 : 3;
            cp &= (1u << (6 - tails)) - 1;
            while (tails--) {
                if (!*s || (*s & 0xc0) != 0x80) {
                    w->failed = 1;
                    return;
                }
                cp = (cp << 6) | (*s++ & 0x3f);
            }
            char token[32];
            snprintf(token, sizeof token, "[U+%04X]", cp);
            put(w, token);
        }
    }
    put(w, "}");
}

static void render_number(Writer *w, const MathNode *n) {
    char text[64];
    if (n->kind == MI_INT) {
        snprintf(text, sizeof text, "%" PRId64, n->integer);
    } else {
        // Use only as many significant digits as the stored double needs.
        for (int digits = 1; digits <= DBL_DECIMAL_DIG; digits++) {
            snprintf(text, sizeof text, "%.*g", digits, n->real);
            double parsed = strtod(text, NULL);
            if (parsed == n->real && !!signbit(parsed) == !!signbit(n->real))
                break;
        }
    }
    char *exp = strchr(text, 'e');
    if (!exp) {
        put(w, text);
        return;
    }
    *exp++ = 0;
    put(w, text);
    put(w, "\\times 10^{");
    char power[24];
    snprintf(power, sizeof power, "%ld", strtol(exp, NULL, 10));
    put(w, power);
    put(w, "}");
}

static int precedence(const MathNode *n) {
    switch (n->kind) {
    case MI_ADD:
    case MI_SUB:
        return 1;
    case MI_MUL:
        return 2;
    case MI_NEG:
        return 3;
    case MI_INT:
        return n->integer < 0 ? 3 : 4;
    case MI_REAL:
        return signbit(n->real) ? 3 : 4;
    default:
        return 4;
    }
}

static int integer_literal(const MathNode *n, int64_t value) {
    return n->kind == MI_INT && n->integer == value;
}

static int same_parameter(const MathNode *a, const MathNode *b) {
    return a->kind == MI_PARAM && b->kind == MI_PARAM && !strcmp(a->name, b->name);
}

static void render(Writer *w, const MathIR *ir, int i, int expand, int minimum, int depth) {
    if (w->failed)
        return;
    if (depth >= MAX_DEPTH * 2 || i < 0 || i >= ir->count) {
        w->failed = 1;
        return;
    }
    const MathNode *n = &ir->nodes[i];
    if (n->kind == MI_LOCAL && expand > 0) {
        render(w, ir, ir->nodes[n->a].a, expand, minimum, depth + 1);
        return;
    }
    /* Only the compact, call-free, external-free mathematical view may
     * simplify. These are formal scalar identities, not VM rewrites. Never
     * erase a nonliteral operand, cancel a denominator or fold arithmetic.
     * Levels 1 and 3 preserve the source operations. */
    if (expand > 0) {
        int keep = -1;
        if (n->kind == MI_NEG && ir->nodes[n->a].kind == MI_NEG)
            keep = ir->nodes[n->a].a;
        if (n->kind == MI_ADD || n->kind == MI_SUB || n->kind == MI_MUL || n->kind == MI_DIV) {
            const MathNode *a = &ir->nodes[n->a], *b = &ir->nodes[n->b];
            if ((n->kind == MI_ADD || n->kind == MI_SUB) && integer_literal(b, 0))
                keep = n->a;
            else if (n->kind == MI_ADD && integer_literal(a, 0))
                keep = n->b;
            else if ((n->kind == MI_MUL || n->kind == MI_DIV) && integer_literal(b, 1))
                keep = n->a;
            else if (n->kind == MI_MUL && integer_literal(a, 1))
                keep = n->b;
            if (n->kind == MI_MUL && same_parameter(a, b)) {
                identifier(w, a->name);
                put(w, "^{2}");
                return;
            }
            if (n->kind == MI_MUL && a->kind == MI_MUL && same_parameter(&ir->nodes[a->b], b)) {
                int parens = minimum > 2;
                if (parens)
                    put(w, "\\left(");
                if (!integer_literal(&ir->nodes[a->a], 1)) {
                    render(w, ir, a->a, expand, 2, depth + 1);
                    put(w, "\\cdot ");
                }
                identifier(w, b->name);
                put(w, "^{2}");
                if (parens)
                    put(w, "\\right)");
                return;
            }
        }
        if (keep >= 0) {
            render(w, ir, keep, expand, minimum, depth + 1);
            return;
        }
    }
    int prec = precedence(n);
    int parens = prec < minimum;
    if (parens)
        put(w, "\\left(");
    switch (n->kind) {
    case MI_INT:
    case MI_REAL:
        render_number(w, n);
        break;
    case MI_PARAM:
    case MI_EXTERNAL:
    case MI_LOCAL:
        identifier(w, n->name);
        break;
    case MI_INDEX:
    case MI_STATE:
        identifier(w, n->name);
        if (expand < 0)
            put(w, "_{\\mathsf{k}}");
        break;
    case MI_SUBSCRIPT:
        put(w, "{");
        render(w, ir, n->a, expand, 4, depth + 1);
        put(w, "}_{");
        render(w, ir, n->b, expand, 0, depth + 1);
        put(w, "}");
        break;
    case MI_NEG:
        put(w, "-");
        render(w, ir, n->a, expand, 4, depth + 1);
        break;
    case MI_ADD:
    case MI_SUB:
    case MI_MUL:
        render(w, ir, n->a, expand, prec, depth + 1);
        put(w, n->kind == MI_ADD ? "+" : n->kind == MI_SUB ? "-" : "\\cdot ");
        /* Preserve the AST association of right operands. */
        render(w, ir, n->b, expand, prec + 1, depth + 1);
        break;
    case MI_DIV:
        put(w, "\\frac{");
        render(w, ir, n->a, expand, 0, depth + 1);
        put(w, "}{");
        render(w, ir, n->b, expand, 0, depth + 1);
        put(w, "}");
        break;
    case MI_CALL:
        if (n->b >= 0) {
            identifier(w, ir->nodes[n->b].name);
            put(w, ".");
        }
        identifier(w, n->name);
        put(w, "\\left(");
        for (int arg = n->a; arg >= 0 && !w->failed; arg = ir->nodes[arg].next) {
            if (arg != n->a)
                put(w, ",");
            render(w, ir, arg, expand, 0, depth + 1);
        }
        put(w, "\\right)");
        break;
    default:
        w->failed = 1;
        return;
    }
    if (parens)
        put(w, "\\right)");
}

static void signature(Writer *w, const Proto *p) {
    if (!strcmp(p->name, "<lambda>"))
        put(w, "\\lambda");
    else
        identifier(w, p->name);
    put(w, "\\left(");
    for (int i = 0; i < p->arity; i++) {
        if (i)
            put(w, ",");
        identifier(w, p->params[i]);
    }
    put(w, "\\right)");
}

static void render_bindings(Writer *w, const MathIR *ir, int level) {
    for (int i = 0; i < ir->count; i++)
        if (ir->nodes[i].kind == MI_BIND) {
            identifier(w, ir->nodes[i].name);
            put(w, level == 3 ? "&\\gets " : "&=");
            render(w, ir, ir->nodes[i].a, 0, 0, 0);
            put(w, "\\\\");
        }
}

static void render_loop(Writer *w, const Proto *p, int level) {
    const MathIR *ir = p->mathir;
    const MathLoop *loop = ir->loop;
    const char *index = ir->nodes[loop->index].name;
    const char *state = ir->nodes[loop->accumulator].name;
    put(w, "\\begin{aligned}");
    render_bindings(w, ir, level);
    if (level == 3) {
        put(w, "\\text{for }");
        identifier(w, index);
        put(w, "&\\in\\operatorname{range}\\left(");
        render(w, ir, loop->start, 0, 0, 0);
        put(w, ",");
        render(w, ir, loop->stop, 0, 0, 0);
        put(w, ",");
        render(w, ir, loop->step, 0, 0, 0);
        put(w, "\\right)\\\\");
        identifier(w, state);
        put(w, "&\\gets ");
        render(w, ir, loop->update, 0, 0, 0);
        put(w, "\\\\\\text{return}&\\quad ");
        identifier(w, state);
    } else {
        put(w, "\\mathsf{N}&=\\max\\left(0,\\left\\lceil\\frac{");
        render(w, ir, loop->stop, 0, 1, 0);
        put(w, "-");
        render(w, ir, loop->start, 0, 2, 0);
        put(w, "}{");
        render(w, ir, loop->step, 0, 0, 0);
        put(w, "}\\right\\rceil\\right)\\\\");
        identifier(w, index);
        put(w, "_{\\mathsf{k}}&=");
        render(w, ir, loop->start, 0, 1, 0);
        put(w, "+\\mathsf{k}\\cdot ");
        render(w, ir, loop->step, 0, 4, 0);
        put(w, ",\\quad 0\\le\\mathsf{k}<\\mathsf{N}\\\\");
        identifier(w, state);
        put(w, "_0&=");
        identifier(w, state);
        put(w, "\\\\");
        if (level == 0 && loop->kind != RECURRENCE) {
            signature(w, p);
            put(w, "&=");
            identifier(w, state);
            put(w, loop->kind == SUM ? "_0+\\sum_{\\mathsf{k}=0}^{\\mathsf{N}-1}"
                                     : "_0\\cdot\\prod_{\\mathsf{k}=0}^{\\mathsf{N}-1}");
            render(w, ir, ir->nodes[loop->update].b, -1, 3, 0);
        } else {
            identifier(w, state);
            put(w, "_{\\mathsf{k}+1}&=");
            render(w, ir, loop->update, -1, 0, 0);
            put(w, "\\\\");
            signature(w, p);
            put(w, "&=");
            identifier(w, state);
            put(w, "_{\\mathsf{N}}");
        }
    }
    put(w, "\\end{aligned}");
}

int mathir_has_loop(const Proto *p) {
    return p->mathir && p->mathir->loop;
}

const char *mathir_assumptions(const Proto *p) {
    if (!p->mathir)
        return NULL;
    if (mathir_has_loop(p))
        return "Native range at query time, bounded integral endpoints, a nonzero integer step "
               "and at most 1000000 iterations. Numeric scalar accumulator operations assumed. "
               "The half-open range can be empty. A mathematical sum or product may round "
               "differently from the ordered source updates.";
    if (p->mathir->has_subscript)
        return "Subscripts show source indices. Operand types, shapes, bounds and dynamic "
               "call targets are unresolved. The function is never evaluated by this query.";
    return "Source-level mathematical notation conditional on numeric operands and compatible "
           "operations. "
           "Level 0 may use formal identities for ordinary finite scalar real arithmetic; "
           "levels 1 and 3 retain source operations. "
           "Types, shapes, domains, overflow, external values and dynamic call targets are not "
           "inferred. "
           "Not a claim of floating-point or runtime equivalence for arbitrary values.";
}

const char *mathir_algorithm(const Proto *p) {
    if (mathir_has_loop(p))
        return "Single half-open native range and one ordered scalar accumulator update per "
               "iteration, followed by return of the final accumulator.";
    return p->mathir ? "Straight-line source expressions and local declarations; actual dynamic "
                       "dispatch and native call implementations are not inferred."
                     : NULL;
}

int mathir_tex(const Proto *p, int level, char *out, size_t size) {
    const MathIR *ir = p->mathir;
    if (!ir || !size || level < 0 || level > 3)
        return 0;
    Writer w = {.out = out, .size = size};
    out[0] = 0;
    if (ir->loop) {
        if (level == 2) {
            put(&w,
                "\\begin{gathered}\\text{Half-open native range with integral bounds and "
                "nonzero step.}"
                "\\\\\\text{An empty range returns the initial state.}"
                "\\\\\\text{Real-valued formulas can round differently from sequential updates.}"
                "\\end{gathered}");
        } else
            render_loop(&w, p, level);
        return !w.failed;
    }
    if (level == 2) {
        put(&w, "\\text{Numeric operands assumed; types, shapes and domains unknown.}");
        if (!ir->has_calls && !ir->has_external && !ir->has_subscript)
            put(&w, "\\quad\\text{Level 0 may use formal finite scalar real identities.}");
        if (ir->has_subscript)
            put(&w, "\\quad\\text{Subscripts show source indices; bounds unknown.}");
        if (ir->has_external)
            put(&w, "\\quad\\text{External values unresolved.}");
        if (ir->has_calls)
            put(&w, "\\quad\\text{Dynamic call targets unresolved.}");
        return !w.failed;
    }
    /* For calls and external dependencies keep declarations visible rather
     * than expanding a possibly effectful expression multiple times. */
    if (level == 0 && !ir->has_calls && !ir->has_external && !ir->has_subscript) {
        signature(&w, p);
        put(&w, "=");
        render(&w, ir, ir->result, 1, 0, 0);
        if (!w.failed)
            return 1;
        /* Exponential expansion or depth limit: fall back to named steps. */
        w = (Writer){.out = out, .size = size};
        out[0] = 0;
    }
    put(&w, "\\begin{aligned}");
    render_bindings(&w, ir, level);
    if (level == 3)
        put(&w, "\\text{return}&\\quad ");
    else {
        signature(&w, p);
        put(&w, "&=");
    }
    render(&w, ir, ir->result, 0, 0, 0);
    if (ir->has_calls)
        put(&w, "\\\\\\text{Dynamic call targets unresolved.}&");
    if (ir->has_external)
        put(&w, "\\\\\\text{External values unresolved.}&");
    put(&w, "\\end{aligned}");
    return !w.failed;
}

typedef struct {
    Extract derivative;
    int memo[MAX_NODES];
    int zero, one;
    const char *parameter;
    VM *vm;
    Obj *environment;
} Symbolic;

static int combine(Symbolic *s, Kind kind, int a, int b) {
    if (a < 0 || (kind != MI_NEG && b < 0))
        return -1;
    if (kind == MI_MUL && (a == s->zero || b == s->zero))
        return s->zero;
    if (kind == MI_MUL && a == s->one)
        return b;
    if (kind == MI_MUL && b == s->one)
        return a;
    if ((kind == MI_ADD || kind == MI_SUB) && b == s->zero)
        return a;
    if (kind == MI_ADD && a == s->zero)
        return b;
    if (kind == MI_NEG && a == s->zero)
        return a;
    return add(&s->derivative, kind, NULL, a, b);
}

static int differentiate(Symbolic *s, int i) {
    if (s->memo[i] != -2)
        return s->memo[i];
    Extract *e = &s->derivative;
    const MathNode *n = &e->nodes[i];
    int d = -1;
    switch (n->kind) {
    case MI_INT:
    case MI_REAL:
        d = s->zero;
        break;
    case MI_PARAM:
        d = !strcmp(n->name, s->parameter) ? s->one : s->zero;
        break;
    case MI_LOCAL:
    case MI_BIND:
        d = differentiate(s, n->a);
        break;
    case MI_NEG:
        d = combine(s, MI_NEG, differentiate(s, n->a), -1);
        break;
    case MI_ADD:
    case MI_SUB:
    case MI_MUL:
    case MI_DIV: {
        int a = differentiate(s, n->a), b = differentiate(s, n->b);
        if (a < 0 || b < 0)
            break;
        if (n->kind == MI_ADD || n->kind == MI_SUB) {
            d = combine(s, n->kind, a, b);
            break;
        }
        int left = combine(s, MI_MUL, a, n->b);
        int right = combine(s, MI_MUL, n->a, b);
        if (n->kind == MI_MUL)
            d = combine(s, MI_ADD, left, right);
        else {
            int top = combine(s, MI_SUB, left, right);
            int bottom = combine(s, MI_MUL, n->b, n->b);
            d = combine(s, MI_DIV, top, bottom);
        }
        break;
    }
    case MI_CALL: {
        if (!n->integer ||
            (strcmp(n->name, "sin") && strcmp(n->name, "cos") && strcmp(n->name, "sqrt")))
            break;
        if (!vm_uses_builtin(s->vm, s->environment, n->name))
            break;
        int inner = differentiate(s, n->a);
        if (inner < 0)
            break;
        if (!strcmp(n->name, "sqrt")) {
            int two = add(e, MI_INT, NULL, -1, -1);
            if (two < 0)
                break;
            e->nodes[two].integer = 2;
            d = combine(s, MI_DIV, inner, combine(s, MI_MUL, two, i));
        } else {
            bool cosine = !strcmp(n->name, "cos");
            int outer = add(e, MI_CALL, cosine ? "sin" : "cos", n->a, -1);
            if (cosine)
                outer = combine(s, MI_NEG, outer, -1);
            d = combine(s, MI_MUL, outer, inner);
        }
        break;
    }
    default:
        break;
    }
    s->memo[i] = d;
    return d;
}

int mathir_derivative_tex(VM *v, Obj *function, const char *parameter, char *out, size_t size) {
    const Proto *p = function->as.func.p;
    const MathIR *source = p->mathir;
    if (!source || source->loop || source->has_external || source->has_subscript || !size ||
        source->count > MAX_NODES - 2)
        return 0;
    bool known = false;
    for (int i = 0; i < p->arity; i++)
        if (!strcmp(p->params[i], parameter))
            known = true;
    if (!known)
        vmerror_kind(v, "domain", "unknown symbolic derivative parameter");
    Symbolic s = {.parameter = parameter, .vm = v, .environment = function->as.func.env};
    Extract *e = &s.derivative;
    e->count = source->count;
    memcpy(e->nodes, source->nodes, (size_t)e->count * sizeof *e->nodes);
    for (int i = 0; i < e->count; i++)
        s.memo[i] = -2;
    s.zero = add(e, MI_INT, NULL, -1, -1);
    s.one = add(e, MI_INT, NULL, -1, -1);
    e->nodes[s.one].integer = 1;
    int result = differentiate(&s, source->result);
    if (result < 0)
        return 0;

    size_t bytes = (size_t)e->count * sizeof *e->nodes;
    MathIR *ir = mem(sizeof *ir + bytes);
    ir->count = e->count;
    memcpy(ir->nodes, e->nodes, bytes);
    Writer w = {.out = out, .size = size};
    out[0] = '\0';
    put(&w, "\\frac{\\partial ");
    identifier(&w, p->name);
    put(&w, "}{\\partial ");
    identifier(&w, parameter);
    put(&w, "}=");
    render(&w, ir, result, 1, 0, 0);
    free(ir);
    return !w.failed;
}
