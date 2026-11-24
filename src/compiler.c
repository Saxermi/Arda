#include "frontend.h"
typedef struct Loop {
    struct Loop *parent;
    int start, depth, tries;
    int *breaks, n, cap;
} Loop;
typedef struct {
    Proto *p;
    int depth, tries, isfn;
    Loop *loop;
} C;
static Proto *proto(const char *name, const char *path) {
    Proto *p = front_mem(sizeof * p);
    p->root = p;
    p->name = front_copystr(name);
    p->path = front_copystr(path);
    return p;
}

static void err(C *c, Node *n, const char *s) {
    front_error("%s:%d: %s", c->p->path, n->line, s);
}

static int emit(C *c, Op op, int arg, int line) {
    Proto *p = c->p;
    if (p->count == p->cap) {
        p->cap = p->cap ? 2 * p->cap : 64;
        p->code = front_resize(p->code, sizeof(Instr) * (size_t) p->cap);
    }
    int i = p->count++;
    p->code[i] = (Instr) {
        .op = op, .arg = arg, .line = line
    };
    return i;
}

static void patch(C *c, int i) {
    c->p->code[i].arg = c->p->count;
}

static int constant(C *c, Constant k) {
    Proto *p = c->p;
    for (int i = 0; i < p->nk; i++) {
        Constant a = p->k[i];
        if (k.type == a.type && ((k.type == K_NUM && k.n == a.n) || (k.type == K_INT && k.i == a.i) || (k.type == K_STR && !strcmp(k.s,
                            a.s))))
            return i;
    }
    if (p->nk == p->kcap) {
        p->kcap = p->kcap ? 2 * p->kcap : 16;
        p->k = front_resize(p->k, sizeof(Constant) * (size_t) p->kcap);
    }
    if (k.type == K_STR) {
        k.s = front_copystr(k.s);
        k.hash = hashstr(k.s);
        k.hint = -1;
    }
    p->k[p->nk] = k;
    return p->nk++;
}

static int key(C *c, const char *s) {
    return constant(c, (Constant) {
            .type = K_STR, .s = (char * ) s
            });
}

static void expression(C *c, Node *n);
static void statement(C *c, Node *n);
static void scope(C *c, int line) {
    emit(c, OP_SCOPE, 0, line);
    c->depth++;
}

static void unscope(C *c, int line) {
    emit(c, OP_UNSCOPE, 0, line);
    c->depth--;
}

static void body(C *c, Node *n) {
    for (int i = 0; i < n->count; i++)
        statement(c, n->items[i]);
}

static void block(C *c, Node *n) {
    int needed = 0;
    for (int i = 0; i < n->count; i++) {
        NodeType t = n->items[i]->t;
        if (t == N_DECLARE || t == N_FUNCTION || t == N_CLASS || t == N_IMPORT)
            needed = 1;
    }
    if (needed)
        scope(c, n->line);
    body(c, n);
    if (needed)
        unscope(c, n->line);
}

static int local(C *c, const char *s) {
    if (c->depth)
        return -1;
    for (int i = 0; i < c->p->arity; i++)
        if (!strcmp(c->p->params[i], s))
            return i;
    return -1;
}

static void function(C *c, Node *n) {
    Proto *p = proto(n->str, c->p->path);
    p->root = c->p->root;
    p->arity = n->n_names;
    p->required = p->arity;
    p->params = front_mem(sizeof(char * ) * (size_t) p->arity);
    for (int i = 0; i < p->arity; i++)
        p->params[i] = front_copystr(n->names[i]);
    C child = {
        .p = p, .isfn = 1
    };
    for (int i = 0; i < p->arity; i++)
        if (n->defaults && n->defaults[i]) {
            if (p->required == p->arity)
                p->required = i;
            emit(&child, OP_GET_LOCAL, i, n->line);
            int j = emit(&child, OP_JPRESENT, 0, n->line);
            expression(&child, n->defaults[i]);
            emit(&child, OP_SET_LOCAL, i, n->line);
            emit(&child, OP_POP, 0, n->line);
            patch(&child, j);
        }
    if (n->t == N_LAMBDA) {
        expression(&child, n->a);
        emit(&child, OP_RETURN, 0, n->line);
    } else
        body(&child, n->a);
    emit(&child, OP_NIL, 0, n->line);
    emit(&child, OP_RETURN, 0, n->line);
    emit(c, OP_CLOSURE, constant(c, (Constant) {
                .type = K_PROTO, .p = p
                }), n->line);
}

static void expression(C *c, Node *n) {
    int l = n->line;
    switch (n->t) {
    case N_NUM:
        emit(c, OP_CONST, constant(c, (Constant) {
                    .type = n->isint ? K_INT : K_NUM, .n = n->num, .i = n->integer
                    }), l);
        break;
    case N_STR:
        emit(c, OP_CONST, key(c, n->str), l);
        break;
    case N_NIL:
        emit(c, OP_NIL, 0, l);
        break;
    case N_BOOL:
        emit(c, n->num ? OP_TRUE : OP_FALSE, 0, l);
        break;
    case N_VAR:
        {
            int i = local(c, n->str);
            emit(c, i >= 0 ? OP_GET_LOCAL : OP_GET, i >= 0 ? i : key(c, n->str), l);
            break;
        }
    case N_LAMBDA:
        function(c, n);
        break;
    case N_LISTCOMP:
        {
            emit(c, OP_LIST, 0, l);
            expression(c, n->b);
            emit(c, OP_ITER, 0, l);
            int start = c->p->count, end = emit(c, OP_NEXT, 0, l);
            scope(c, l);
            emit(c, OP_DEFINE, key(c, n->str), l);
            int skip = -1;
            if (n->c) {
                expression(c, n->c);
                skip = emit(c, OP_JFALSE, 0, l);
            }
            expression(c, n->a);
            emit(c, OP_COMP_APPEND, 0, l);
            if (skip >= 0)
                patch(c, skip);
            unscope(c, l);
            emit(c, OP_JUMP, start, l);
            patch(c, end);
            emit(c, OP_POP, 0, l);
            break;
        }
    case N_LIST:
    case N_MAP:
        for (int i = 0; i < n->count; i++)
            expression(c, n->items[i]);
        emit(c, n->t == N_LIST ? OP_LIST : OP_MAP, n->count, l);
        break;
    case N_INDEX:
        expression(c, n->a);
        expression(c, n->b);
        emit(c, OP_INDEX, 0, l);
        break;
    case N_SLICE:
        expression(c, n->a);
        if (n->b)
            expression(c, n->b);
        else
            emit(c, OP_NIL, 0, l);
        if (n->c)
            expression(c, n->c);
        else
            emit(c, OP_NIL, 0, l);
        emit(c, OP_SLICE, 0, l);
        break;
    case N_ATTR:
        expression(c, n->a);
        emit(c, OP_ATTR, key(c, n->str), l);
        break;
    case N_ASSIGN:
        if (n->a->t == N_VAR) {
            expression(c, n->b);
            int i = local(c, n->a->str);
            emit(c, i >= 0 ? OP_SET_LOCAL : OP_SET, i >= 0 ? i : key(c, n->a->str), l);
        } else {
            expression(c, n->a->a);
            if (n->a->t == N_INDEX)
                expression(c, n->a->b);
            expression(c, n->b);
            emit(c, n->a->t == N_INDEX ? OP_SET_INDEX : OP_SET_ATTR, n->a->t == N_ATTR ? key(c, n->a->str) : 0,
                    l);
        } break;
    case N_UNARY:
        expression(c, n->a);
        emit(c, n->op == T_MINUS ? OP_NEG : OP_NOT, 0, l);
        break;
    case N_BINARY:
        {
            expression(c, n->a);
            if (n->op == T_AND || n->op == T_OR) {
                emit(c, OP_DUP, 0, l);
                int j = emit(c, n->op == T_AND ? OP_JFALSE : OP_JTRUE, 0, l);
                emit(c, OP_POP, 0, l);
                expression(c, n->b);
                patch(c, j);
                break;
            }
            if (n->b->t == N_NUM && n->b->isint && (n->op == T_PLUS || n->op == T_MINUS || n->op == T_STAR)) {
                emit(c, n->op == T_PLUS ? OP_ADD_INT_CONST : n->op == T_MINUS ? OP_SUB_INT_CONST : OP_MUL_INT_CONST,
                        constant(c, (Constant) {
                            .type = K_INT, .i = n->b->integer
                            }), l);
                break;
            }
            expression(c, n->b);
            Op op;
            switch (n->op) {
            case T_PLUS:
                op = OP_ADD;
                break;
            case T_MINUS:
                op = OP_SUB;
                break;
            case T_STAR:
                op = OP_MUL;
                break;
            case T_SLASH:
                op = OP_DIV;
                break;
            case T_PERCENT:
                op = OP_MOD;
                break;
            case T_EQEQ:
                op = OP_EQ;
                break;
            case T_NE:
                op = OP_NE;
                break;
            case T_LT:
                op = OP_LT;
                break;
            case T_LE:
                op = OP_LE;
                break;
            case T_GT:
                op = OP_GT;
                break;
            case T_GE:
                op = OP_GE;
                break;
            default:
                err(c, n, "unknown operator");
                return;
            }
            emit(c, op, 0, l);
            break;
        }
    case N_CALL:
        expression(c, n->a);
        for (int i = 0; i < n->count; i++)
            expression(c, n->items[i]);
        emit(c, OP_CALL, n->count, l);
        break;
    default:
        err(c, n, "invalid expression");
    }
}
// Fuse a comparison with its consuming branch; preserve generic comparison
// behavior for floats, invalid operands and all other dynamic cases.
static int false_jump(C *c, Node *n, int line) {
    if (n->t == N_BINARY && n->op == T_LT && n->b->t == N_NUM && n->b->isint) {
        expression(c, n->a);
        int k = constant(c, (Constant) {
                .type = K_INT, .i = n->b->integer
                });
        int j = emit(c, OP_JFALSE_LT_CONST, 0, line);
        c->p->code[j].aux = k;
        return j;
    }
    expression(c, n);
    return emit(c, OP_JFALSE, 0, line);
}

static void finishloop(C *c, Loop *loop) {
    for (int i = 0; i < loop->n; i++)
        patch(c, loop->breaks[i]);
    front_free(loop->breaks);
    c->loop = loop->parent;
}

static void statement(C *c, Node *n) {
    int l = n->line;
    switch (n->t) {
    case N_DECLARE:
        expression(c, n->a);
        emit(c, OP_DEFINE, key(c, n->str), l);
        break;
    case N_FUNCTION:
        function(c, n);
        emit(c, OP_DEFINE, key(c, n->str), l);
        break;
    case N_EXPR:
        expression(c, n->a);
        if (n->a->t == N_ASSIGN && n->a->a->t == N_VAR) {
            Instr *last = &c->p->code[c->p->count - 1];
            last->op = last->op == OP_SET_LOCAL ? OP_SET_LOCAL_DROP : OP_SET_DROP;
        } else
            emit(c, OP_POP, 0, l);
        break;
    case N_PRINT:
        for (int i = 0; i < n->count; i++)
            expression(c, n->items[i]);
        emit(c, OP_PRINT, n->count, l);
        break;
    case N_BLOCK:
        block(c, n);
        break;
    case N_IF:
        {
            int j = false_jump(c, n->a, l);
            block(c, n->b);
            if (n->c) {
                int end = emit(c, OP_JUMP, 0, l);
                patch(c, j);
                statement(c, n->c);
                patch(c, end);
            } else
                patch(c, j);
            break;
        }
    case N_WHILE:
        {
            Loop loop = {
                .parent = c->loop, .start = c->p->count, .depth = c->depth, .tries = c->tries
            };
            c->loop = &loop;
            int end = false_jump(c, n->a, l);
            block(c, n->b);
            emit(c, OP_JUMP, loop.start, l);
            patch(c, end);
            finishloop(c, &loop);
            break;
        }
    case N_FOR:
        {
            scope(c, l);
            expression(c, n->a);
            emit(c, OP_ITER, 0, l);
            Loop loop = {
                .parent = c->loop, .start = c->p->count, .depth = c->depth, .tries = c->tries
            };
            c->loop = &loop;
            int end = emit(c, OP_NEXT, 0, l);
            scope(c, l);
            emit(c, OP_DEFINE, key(c, n->str), l);
            body(c, n->b);
            unscope(c, l);
            emit(c, OP_JUMP, loop.start, l);
            patch(c, end);
            finishloop(c, &loop);
            emit(c, OP_POP, 0, l);
            unscope(c, l);
            break;
        }
    case N_BREAK:
    case N_CONTINUE:
        {
            Loop *loop = c->loop;
            if (!loop) {
                err(c, n, "loop control outside loop");
                break;
            }
            for (int i = c->tries; i > loop->tries; i--)
                emit(c, OP_END_TRY, 0, l);
            for (int i = c->depth; i > loop->depth; i--)
                emit(c, OP_UNSCOPE, 0, l);
            if (n->t == N_CONTINUE)
                emit(c, OP_JUMP, loop->start, l);
            else {
                if (loop->n == loop->cap) {
                    loop->cap = loop->cap ? 2 * loop->cap : 4;
                    loop->breaks = front_resize(loop->breaks, sizeof(int) * (size_t) loop->cap);
                }
                loop->breaks[loop->n++] = emit(c, OP_JUMP, 0, l);
            } break;
        }
    case N_RETURN:
        if (!c->isfn)
            err(c, n, "return outside function");
        if (n->a)
            expression(c, n->a);
        else
            emit(c, OP_NIL, 0, l);
        emit(c, OP_RETURN, 0, l);
        break;
    case N_CLASS:
        if (n->b)
            expression(c, n->b);
        else
            emit(c, OP_NIL, 0, l);
        emit(c, OP_CLASS, key(c, n->str), l);
        emit(c, OP_DUP, 0, l);
        emit(c, OP_DEFINE, key(c, n->str), l);
        for (int i = 0; i < n->a->count; i++) {
            Node *m = n->a->items[i];
            if (m->t == N_PASS)
                continue;
            if (m->t != N_FUNCTION || !m->n_names || strcmp(m->names[0], "self"))
                err(c, m, "class body expects methods with first parameter self");
            function(c, m);
            emit(c, OP_METHOD, key(c, m->str), m->line);
        }
        emit(c, OP_POP, 0, l);
        break;
    case N_IMPORT:
        emit(c, OP_IMPORT, key(c, n->str), l);
        emit(c, OP_DEFINE, key(c, n->n_names ? n->names[0] : n->str), l);
        break;
    case N_TRY:
        {
            int catchip = emit(c, OP_TRY, 0, l);
            c->tries++;
            block(c, n->a);
            c->tries--;
            emit(c, OP_END_TRY, 0, l);
            int end = emit(c, OP_JUMP, 0, l);
            patch(c, catchip);
            scope(c, l);
            emit(c, OP_DEFINE, key(c, n->str), l);
            body(c, n->b);
            unscope(c, l);
            patch(c, end);
            break;
        }
    case N_THROW:
        expression(c, n->a);
        emit(c, OP_THROW, 0, l);
        break;
    case N_PASS:
        break;
    default:
        err(c, n, "invalid statement");
    }
}

Proto *compile(Node *root, const char *path) {
    Proto * p = proto("<main>", path);
    C c = {
        .p = p
    };
    body(&c, root);
    emit(&c, OP_NIL, 0, root->line);
    emit(&c, OP_RETURN, 0, root->line);
    return p;
}

void freeproto(Proto *p) {
    for (int i = 0; i < p->nk; i++) {
        if (p->k[i].type == K_STR)
            front_free(p->k[i].s);
        if (p->k[i].type == K_PROTO)
            freeproto(p->k[i].p);
    }
    for (int i = 0; i < p->arity; i++)
        front_free(p->params[i]);
    front_free(p->params);
    front_free(p->k);
    front_free(p->code);
    front_free(p->name);
    front_free(p->path);
    front_free(p);
}

void disassemble(Proto *p, int depth) {
    static const char *names[] = {
        "CONST", "NIL", "TRUE", "FALSE", "GET", "DEFINE", "SET", "POP", "DUP", "SCOPE", "UNSCOPE", "LIST",
        "MAP", "INDEX", "SET_INDEX", "ATTR", "SET_ATTR", "NEG", "NOT", "ADD", "SUB", "MUL", "DIV", "MOD",
        "EQ", "NE", "LT", "LE", "GT", "GE", "JUMP", "JFALSE", "JTRUE", "CLOSURE", "CALL", "RETURN", "PRINT",
        "ITER", "NEXT", "CLASS", "METHOD", "IMPORT", "TRY", "END_TRY", "THROW", "GET_LOCAL", "SET_LOCAL",
        "JPRESENT", "SLICE", "COMP_APPEND", "SET_DROP", "SET_LOCAL_DROP", "ADD_INT_CONST", "SUB_INT_CONST",
        "MUL_INT_CONST", "JFALSE_LT_CONST"
    };
    printf("%*s== %s (%s) ==\n", depth * 2, "", p->name, p->path);
    for (int i = 0; i < p->count; i++) {
        Instr x = p->code[i];
        printf("%*s%04d L%-4d %-12s %d", depth * 2, "", i, x.line, names[x.op], x.arg);
        if (x.op == OP_CONST || x.op == OP_GET || x.op == OP_SET || x.op == OP_SET_DROP || x.op == OP_ADD_INT_CONST || x.op == OP_SUB_INT_CONST || x.op == OP_MUL_INT_CONST || x.op == OP_DEFINE || x.op == OP_ATTR || x.op == OP_SET_ATTR || x.op == OP_CLASS || x.op == OP_METHOD || x.op == OP_IMPORT || x.op == OP_CLOSURE) {
            Constant k = p->k[x.arg];
            if (k.type == K_INT)
                printf("  %" PRId64, k.i);
            else if (k.type == K_NUM)
                printf("  %.15g", k.n);
            else if (k.type == K_STR)
                printf("  %s", k.s);
            else
                printf("  %s", k.p->name);
        }
        if (x.op == OP_JFALSE_LT_CONST)
            printf("  if not < %" PRId64, p->k[x.aux].i);
        puts("");
    }
    for (int i = 0; i < p->nk; i++)
        if (p->k[i].type == K_PROTO)
            disassemble(p->k[i].p, depth + 1);
}
