#include "frontend.h"
typedef struct {
    Tokens *ts;
    int i, depth;
    const char *path;
} Parser;
static Token *peek(Parser *p) {
    return &p->ts->a[p->i];
}

static Token *prev(Parser *p) {
    return &p->ts->a[p->i - 1];
}

static int match(Parser *p, TokenType t) {
    if (peek(p)->t == t) {
        p->i++;
        return 1;
    }
    return 0;
}

static void err(Parser *p, const char *msg) {
    front_error("%s:%d: %s (got %s '%s')", p->path, peek(p)->line, msg, tokname(peek(p)->t), peek(p)->lex);
}

static Token *expect(Parser *p, TokenType t) {
    if (!match(p, t))
        err(p, tokname(t));
    return prev(p);
}

static Node *node(NodeType t, int line) {
    Node *n = front_mem(sizeof * n);
    n->t = t;
    n->line = line;
    return n;
}

static void add(Node *n, Node *x) {
    if (n->count == n->cap) {
        n->cap = n->cap ? 2 * n->cap : 8;
        n->items = front_resize(n->items, sizeof(Node * ) * (size_t) n->cap);
    }
    n->items[n->count++] = x;
}

static Node *expr(Parser *p);
static Node *stmt(Parser *p);
static Node *primary(Parser *p) {
    Token *t = peek(p);
    Node *n;
    if (match(p, T_NUM)) {
        n = node(N_NUM, t->line);
        if (t->isint == 2)
            err(p, "integer literal out of range");
        n->num = t->num;
        n->isint = t->isint;
        n->integer = t->integer;
        return n;
    }
    if (match(p, T_STR)) {
        n = node(N_STR, t->line);
        n->str = t->lex;
        return n;
    }
    if (match(p, T_TRUE) || match(p, T_FALSE)) {
        n = node(N_BOOL, t->line);
        n->num = t->t == T_TRUE;
        return n;
    }
    if (match(p, T_NIL))
        return node(N_NIL, t->line);
    if (match(p, T_LAMBDA)) {
        n = node(N_LAMBDA, t->line);
        n->str = "<lambda>";
        while (peek(p)->t != T_COLON) {
            char *s = expect(p, T_ID)->lex;
            for (int i = 0; i < n->n_names; i++)
                if (!strcmp(s, n->names[i]))
                    err(p, "duplicate parameter");
            n->names = front_resize(n->names, sizeof(char * ) * (size_t)(n->n_names + 1));
            n->names[n->n_names++] = s;
            if (!match(p, T_COMMA))
                break;
        }
        expect(p, T_COLON);
        n->a = expr(p);
        return n;
    }
    if (match(p, T_ID)) {
        n = node(N_VAR, t->line);
        n->str = t->lex;
        return n;
    }
    if (match(p, T_LP)) {
        n = expr(p);
        expect(p, T_RP);
        return n;
    }
    if (match(p, T_LS)) {
        n = node(N_LIST, t->line);
        if (peek(p)->t != T_RS) {
            Node *first = expr(p);
            if (match(p, T_FOR)) {
                n->t = N_LISTCOMP;
                n->a = first;
                n->str = expect(p, T_ID)->lex;
                expect(p, T_IN);
                n->b = expr(p);
                if (match(p, T_QUI))
                    n->c = expr(p);
            } else {
                add(n, first);
                while (match(p, T_COMMA) && peek(p)->t != T_RS)
                    add(n, expr(p));
            }
        }
        expect(p, T_RS);
        return n;
    }
    if (match(p, T_LB)) {
        n = node(N_MAP, t->line);
        while (peek(p)->t != T_RB) {
            add(n, expr(p));
            expect(p, T_COLON);
            add(n, expr(p));
            if (!match(p, T_COMMA))
                break;
        }
        expect(p, T_RB);
        return n;
    }
    err(p, "expected expression");
    return NULL;
}

static Node *postfix(Parser *p) {
    Node *n = primary(p);
    while (1) {
        if (match(p, T_LP)) {
            Node *c = node(N_CALL, n->line);
            c->a = n;
            while (peek(p)->t != T_RP) {
                add(c, expr(p));
                if (!match(p, T_COMMA))
                    break;
            }
            expect(p, T_RP);
            n = c;
        } else if (match(p, T_LS)) {
            Node *c = node(N_INDEX, n->line);
            c->a = n;
            if (peek(p)->t != T_COLON)
                c->b = expr(p);
            if (match(p, T_COLON)) {
                c->t = N_SLICE;
                if (peek(p)->t != T_RS)
                    c->c = expr(p);
            } else if (!c->b)
                err(p, "expected index");
            expect(p, T_RS);
            n = c;
        } else if (match(p, T_DOT)) {
            Node *c = node(N_ATTR, n->line);
            c->a = n;
            c->str = expect(p, T_ID)->lex;
            n = c;
        } else
            break;
    }
    return n;
}

static Node *unary(Parser *p) {
    if (match(p, T_MINUS) || match(p, T_BANG)) {
        Token *t = prev(p);
        if (t->t == T_MINUS && peek(p)->t == T_NUM && peek(p)->isint == 2) {
            p->i++;
            Node *m = node(N_NUM, t->line);
            m->isint = 1;
            m->integer = INT64_MIN;
            return m;
        }
        Node *n = node(N_UNARY, t->line);
        n->op = t->t;
        if (++p->depth > 256)
            err(p, "expression too deep");
        n->a = unary(p);
        p->depth--;
        return n;
    }
    return postfix(p);
}

static Node *binary(Parser *p, int level) {
    if (!level)
        return unary(p);
    Node *left = binary(p, level - 1);
    while (1) {
        TokenType t = peek(p)->t;
        int ok = (level == 1 && (t == T_STAR || t == T_SLASH || t == T_PERCENT)) || (level == 2 && (t == T_PLUS || t == T_MINUS)) || (level == 3 && (t == T_LT || t == T_LE || t == T_GT || t == T_GE)) || (level == 4 && (t == T_EQEQ || t == T_NE)) || (level == 5 && t == T_AND) || (level == 6 && t == T_OR);
        if (!ok)
            break;
        p->i++;
        Node *n = node(N_BINARY, prev(p)->line);
        n->op = t;
        n->a = left;
        n->b = binary(p, level - 1);
        left = n;
    }
    return left;
}

static Node *expr(Parser *p) {
    if (++p->depth > 256)
        err(p, "expression too deep");
    Node *n = binary(p, 6);
    if (match(p, T_EQ)) {
        if (n->t != N_VAR && n->t != N_INDEX && n->t != N_ATTR)
            err(p, "invalid assignment target");
        Node *a = node(N_ASSIGN, n->line);
        a->a = n;
        a->b = expr(p);
        n = a;
    }
    p->depth--;
    return n;
}

static Node *block(Parser *p) {
    int line = expect(p, T_COLON)->line;
    expect(p, T_NL);
    expect(p, T_INDENT);
    Node *n = node(N_BLOCK, line);
    while (peek(p)->t != T_DEDENT && peek(p)->t != T_EOF) {
        if (match(p, T_NL))
            continue;
        add(n, stmt(p));
    }
    expect(p, T_DEDENT);
    return n;
}

static Node *ifstmt(Parser *p, int line) {
    Node *n = node(N_IF, line);
    n->a = expr(p);
    n->b = block(p);
    if (match(p, T_ELIF))
        n->c = ifstmt(p, prev(p)->line);
    else if (match(p, T_AI))
        n->c = block(p);
    return n;
}

static Node *stmt(Parser *p) {
    Token *t = peek(p);
    Node *n;
    if (match(p, T_ESTA)) {
        n = node(N_DECLARE, t->line);
        n->str = expect(p, T_ID)->lex;
        expect(p, T_EQ);
        n->a = expr(p);
    } else if (match(p, T_CARO)) {
        n = node(N_FUNCTION, t->line);
        n->str = expect(p, T_ID)->lex;
        expect(p, T_LP);
        while (peek(p)->t != T_RP) {
            char *s = expect(p, T_ID)->lex;
            for (int i = 0; i < n->n_names; i++)
                if (!strcmp(s, n->names[i]))
                    err(p, "duplicate parameter");
            n->names = front_resize(n->names, sizeof(char * ) * (size_t)(n->n_names + 1));
            n->names[n->n_names] = s;
            n->defaults = front_resize(n->defaults, sizeof(Node * ) * (size_t)(n->n_names + 1));
            n->defaults[n->n_names] = match(p, T_EQ) ? expr(p) : NULL;
            if (n->n_names && n->defaults[n->n_names - 1] && !n->defaults[n->n_names])
                err(p, "required parameter after default");
            n->n_names++;
            if (!match(p, T_COMMA))
                break;
        }
        expect(p, T_RP);
        n->a = block(p);
        return n;
    } else if (match(p, T_QUI))
        return ifstmt(p, t->line);
    else if (match(p, T_TENNA)) {
        n = node(N_WHILE, t->line);
        n->a = expr(p);
        n->b = block(p);
        return n;
    } else if (match(p, T_FOR)) {
        n = node(N_FOR, t->line);
        n->str = expect(p, T_ID)->lex;
        expect(p, T_IN);
        n->a = expr(p);
        n->b = block(p);
        return n;
    } else if (match(p, T_CLASS)) {
        n = node(N_CLASS, t->line);
        n->str = expect(p, T_ID)->lex;
        if (match(p, T_LP)) {
            n->b = expr(p);
            expect(p, T_RP);
        }
        n->a = block(p);
        return n;
    } else if (match(p, T_TRY)) {
        n = node(N_TRY, t->line);
        n->a = block(p);
        expect(p, T_CATCH);
        n->str = expect(p, T_ID)->lex;
        n->b = block(p);
        return n;
    } else if (match(p, T_IMPORT)) {
        n = node(N_IMPORT, t->line);
        Token *m = peek(p);
        if (!match(p, T_ID) && !match(p, T_STR))
            err(p, "module name or quoted relative path");
        n->str = m->lex;
        if (match(p, T_AS)) {
            n->names = front_mem(sizeof(char * ));
            n->names[0] = expect(p, T_ID)->lex;
            n->n_names = 1;
        } else if (m->t == T_STR)
            err(p, "quoted module path requires 'as name'");
    } else if (match(p, T_NAN)) {
        n = node(N_RETURN, t->line);
        if (peek(p)->t != T_NL)
            n->a = expr(p);
    } else if (match(p, T_THROW)) {
        n = node(N_THROW, t->line);
        n->a = expr(p);
    } else if (match(p, T_PEDO)) {
        n = node(N_PRINT, t->line);
        expect(p, T_LP);
        while (peek(p)->t != T_RP) {
            add(n, expr(p));
            if (!match(p, T_COMMA))
                break;
        }
        expect(p, T_RP);
    } else if (match(p, T_BREAK))
        n = node(N_BREAK, t->line);
    else if (match(p, T_CONTINUE))
        n = node(N_CONTINUE, t->line);
    else if (match(p, T_PASS))
        n = node(N_PASS, t->line);
    else {
        n = node(N_EXPR, t->line);
        n->a = expr(p);
    }
    expect(p, T_NL);
    return n;
}

Node *parse(Tokens *ts, const char *path) {
    Parser p = {
        .ts = ts, .path = path
    };
    Node *n = node(N_BLOCK, 1);
    while (peek(&p)->t != T_EOF) {
        if (match(&p, T_NL))
            continue;
        add(n, stmt(&p));
    }
    return n;
}

void free_ast(Node *n) {
    if (!n)
        return;
    free_ast(n->a);
    free_ast(n->b);
    free_ast(n->c);
    for (int i = 0; i < n->count; i++)
        free_ast(n->items[i]);
    front_free(n->items);
    if (n->defaults) {
        for (int i = 0; i < n->n_names; i++)
            free_ast(n->defaults[i]);
        front_free(n->defaults);
    }
    front_free(n->names);
    front_free(n);
}
