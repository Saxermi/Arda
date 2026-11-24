#include "frontend.h"
static void add(Tokens *v, Token t) {
    if (v->n == v->cap) {
        v->cap = v->cap ? 2 * v->cap : 128;
        v->a = front_resize(v->a, sizeof(Token) * (size_t)v->cap);
    }
    v->a[v->n++] = t;
}

static void simple(Tokens *v, TokenType t, const char *s, int line) {
    add(v, (Token){.t = t, .lex = front_copystr(s), .line = line});
}

static TokenType keyword(const char *s) {
    struct {
        const char *s;
        TokenType t;
    } k[] = {
        {"esta", T_ESTA},    {"let", T_ESTA},      {"caro", T_CARO},   {"fn", T_CARO},
        {"def", T_CARO},     {"nan", T_NAN},       {"return", T_NAN},  {"qui", T_QUI},
        {"if", T_QUI},       {"ai", T_AI},         {"else", T_AI},     {"elif", T_ELIF},
        {"tenna", T_TENNA},  {"while", T_TENNA},   {"pedo", T_PEDO},   {"print", T_PEDO},
        {"nae", T_TRUE},     {"true", T_TRUE},     {"law", T_FALSE},   {"false", T_FALSE},
        {"mun", T_NIL},      {"nil", T_NIL},       {"break", T_BREAK}, {"continue", T_CONTINUE},
        {"and", T_AND},      {"or", T_OR},         {"not", T_BANG},    {"an", T_FOR},
        {"for", T_FOR},      {"in", T_IN},         {"nost", T_CLASS},  {"class", T_CLASS},
        {"tol", T_IMPORT},   {"import", T_IMPORT}, {"as", T_AS},       {"try", T_TRY},
        {"catch", T_CATCH},  {"throw", T_THROW},   {"raise", T_THROW}, {"pass", T_PASS},
        {"lambda", T_LAMBDA}};
    for (size_t i = 0; i < sizeof k / sizeof *k; i++)
        if (!strcmp(s, k[i].s))
            return k[i].t;
    return T_ID;
}

const char *tokname(TokenType t) {
    static const char *n[] = {"EOF",      "ID",    "NUM",  "STR",   "(",       ")",      "{",
                              "}",        "[",     "]",    ",",     "NEWLINE", "INDENT", "DEDENT",
                              ":",        ".",     "+",    "-",     "*",       "/",      "%",
                              "!",        "=",     "==",   "!=",    "<",       "<=",     ">",
                              ">=",       "and",   "or",   "esta",  "caro",    "nan",    "qui",
                              "ai",       "tenna", "pedo", "true",  "false",   "nil",    "break",
                              "continue", "for",   "in",   "class", "import",  "as",     "try",
                              "catch",    "throw", "pass", "elif",  "lambda"};
    return n[t];
}

Tokens tokenize(const char *src, const char *path) {
    if (!utf8valid(src))
        front_error("%s: invalid UTF-8", path);
    Tokens out = {0};
    int line = 1, bol = 1, depth = 0, ind[256] = {0}, ni = 0;
    char delimiters[512];
    const char *p = src;
#define ERR(...) front_lex_error(path, line, __VA_ARGS__)
    while (*p) {
        if (bol) {
            int spaces = 0;
            while (*p == ' ' || *p == '\t') {
                if (*p == '\t' && !depth)
                    ERR("use spaces, not tabs, for indentation");
                spaces++;
                p++;
            }
            bol = 0;
            if (!*p)
                break;
            if (*p == '\r' || *p == '\n' || *p == '#' || (p[0] == '/' && p[1] == '/')) {
            } else if (!depth) {
                if (spaces > ind[ni]) {
                    if (ni == 255)
                        ERR("indentation too deep");
                    ind[++ni] = spaces;
                    simple(&out, T_INDENT, "", line);
                } else {
                    while (spaces < ind[ni]) {
                        ni--;
                        simple(&out, T_DEDENT, "", line);
                    }
                    if (spaces != ind[ni])
                        ERR("inconsistent indentation");
                }
            }
        }
        if (*p == '\n' || *p == '\r') {
            if (!depth && out.n && out.a[out.n - 1].t != T_NL && out.a[out.n - 1].t != T_DEDENT)
                simple(&out, T_NL, "", line);
            if (*p == '\r' && p[1] == '\n')
                p++;
            p++;
            line++;
            bol = 1;
            continue;
        }
        if (isspace((unsigned char)*p)) {
            p++;
            continue;
        }
        if (*p == '#' || (p[0] == '/' && p[1] == '/')) {
            while (*p && *p != '\n' && *p != '\r')
                p++;
            continue;
        }
        const char *st = p;
        Token t = {.line = line};
        if (isalpha((unsigned char)*p) || *p == '_' || (unsigned char)*p >= 128) {
            p++;
            while (isalnum((unsigned char)*p) || *p == '_' || (unsigned char)*p >= 128)
                p++;
            t.lex = front_slice(st, (size_t)(p - st));
            t.t = keyword(t.lex);
        } else if (isdigit((unsigned char)*p)) {
            char *end;
            t.num = strtod(p, &end);
            p = end;
            t.lex = front_slice(st, (size_t)(p - st));
            t.t = T_NUM;
            if (!strpbrk(t.lex, ".eE")) {
                errno = 0;
                char *ie;
                intmax_t q = strtoimax(t.lex, &ie, 10);
                if (!errno && !*ie) {
                    t.isint = 1;
                    t.integer = (int64_t)q;
                } else if (errno == ERANGE && !strcmp(t.lex, "9223372036854775808")) {
                    t.isint = 2;
                    t.integer = INT64_MIN;
                } else if (errno == ERANGE)
                    ERR("integer literal out of range");
            }
        } else if (*p == '"' || *p == '\'') {
            char quote = *p++;
            char *buf = front_mem(strlen(p) + 1);
            size_t n = 0;
            while (*p && *p != quote) {
                if (*p == '\n' || *p == '\r')
                    ERR("unterminated string");
                if (*p == '\\') {
                    p++;
                    if (!*p)
                        ERR("unterminated escape");
                    switch (*p) {
                    case 'n':
                        buf[n++] = '\n';
                        break;
                    case 't':
                        buf[n++] = '\t';
                        break;
                    case 'r':
                        buf[n++] = '\r';
                        break;
                    default:
                        buf[n++] = *p;
                        break;
                    }
                    p++;
                } else
                    buf[n++] = *p++;
            }
            if (!*p)
                ERR("unterminated string");
            p++;
            buf[n] = 0;
            t.lex = buf;
            t.t = T_STR;
        } else {
            char c = *p++;
            switch (c) {
            case '(':
            case '[':
            case '{':
                if (depth == 512)
                    ERR("nesting too deep");
                delimiters[depth++] = c;
                t.t = c == '(' ? T_LP : c == '[' ? T_LS : T_LB;
                break;
            case ')':
            case ']':
            case '}':
                if (!depth || delimiters[depth - 1] != (c == ')' ? '(' : c == ']' ? '[' : '{'))
                    ERR("mismatched delimiter");
                depth--;
                t.t = c == ')' ? T_RP : c == ']' ? T_RS : T_RB;
                break;
            case ',':
                t.t = T_COMMA;
                break;
            case ':':
                t.t = T_COLON;
                break;
            case '.':
                t.t = T_DOT;
                break;
            case '+':
                t.t = T_PLUS;
                break;
            case '-':
                t.t = T_MINUS;
                break;
            case '*':
                t.t = T_STAR;
                break;
            case '/':
                t.t = T_SLASH;
                break;
            case '%':
                t.t = T_PERCENT;
                break;
            case '!':
                t.t = T_BANG;
                if (*p == '=') {
                    p++;
                    t.t = T_NE;
                }
                break;
            case '=':
                t.t = T_EQ;
                if (*p == '=') {
                    p++;
                    t.t = T_EQEQ;
                }
                break;
            case '<':
                t.t = T_LT;
                if (*p == '=') {
                    p++;
                    t.t = T_LE;
                }
                break;
            case '>':
                t.t = T_GT;
                if (*p == '=') {
                    p++;
                    t.t = T_GE;
                }
                break;
            case ';':
                ERR("semicolons are not used; start a new line");
            default:
                ERR("unexpected character '%c'", c);
            }
            t.lex = front_slice(st, (size_t)(p - st));
        }
        add(&out, t);
    }
    if (depth)
        ERR("unclosed delimiter");
    if (out.n && out.a[out.n - 1].t != T_NL)
        simple(&out, T_NL, "", line);
    while (ni--)
        simple(&out, T_DEDENT, "", line);
    simple(&out, T_EOF, "", line);
    return out;
#undef ERR
}

void freetokens(Tokens *t) {
    for (int i = 0; i < t->n; i++)
        front_free(t->a[i].lex);
    front_free(t->a);
}
