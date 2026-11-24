#include "frontend.h"
#include <stddef.h>

/* The interpreter is single-threaded; a thread-local scope also prevents
 * unrelated threads from sharing the temporary allocation chain. */
typedef union Allocation Allocation;
union Allocation {
    max_align_t alignment;
    struct {
        Allocation *prev, *next;
        int tracked;
    } link;
};
typedef struct {
    Allocation *head;
    jmp_buf trap;
    char *error;
    size_t error_size;
} FrontScope;
static _Thread_local FrontScope *active;

static void attach(Allocation *a) {
    a->link.tracked = active != NULL;
    a->link.prev = NULL;
    a->link.next = active ? active->head : NULL;
    if (active) {
        if (active->head)
            active->head->link.prev = a;
        active->head = a;
    }
}
static void detach(Allocation *a) {
    if (!a->link.tracked)
        return;
    if (a->link.prev)
        a->link.prev->link.next = a->link.next;
    else
        active->head = a->link.next;
    if (a->link.next)
        a->link.next->link.prev = a->link.prev;
    a->link.tracked = 0;
}
static size_t allocation_size(size_t n) {
    if (n > SIZE_MAX - sizeof(Allocation))
        die("frontend allocation overflow");
    return sizeof(Allocation) + (n ? n : 1);
}
void *front_mem(size_t n) {
    Allocation *a = mem(allocation_size(n));
    attach(a);
    return a + 1;
}
void *front_resize(void *p, size_t n) {
    if (!p)
        return front_mem(n);
    Allocation *a = (Allocation *)p - 1;
    detach(a);
    a = resize(a, allocation_size(n));
    attach(a);
    return a + 1;
}
void front_free(void *p) {
    if (!p)
        return;
    Allocation *a = (Allocation *)p - 1;
    detach(a);
    free(a);
}
char *front_slice(const char *s, size_t n) {
    char *p = front_mem(n + 1);
    memcpy(p, s, n);
    return p;
}
char *front_copystr(const char *s) {
    return front_slice(s, strlen(s));
}

static void report(const char *prefix, const char *fmt, va_list args) {
    if (!active) {
        fputs(prefix, stderr);
        vfprintf(stderr, fmt, args);
        fputc('\n', stderr);
        return;
    }
    size_t offset = 0;
    if (active->error_size) {
        snprintf(active->error, active->error_size, "%s", prefix);
        offset = strlen(active->error);
        vsnprintf(active->error + offset, active->error_size - offset, fmt, args);
    }
}
static _Noreturn void abort_frontend(void) {
    if (active)
        longjmp(active->trap, 1);
    exit(1);
}
_Noreturn void front_error(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    report("", fmt, args);
    va_end(args);
    abort_frontend();
}
_Noreturn void front_lex_error(const char *path, int line, const char *fmt, ...) {
    char prefix[1024];
    snprintf(prefix, sizeof prefix, "%s:%d: ", path, line);
    va_list args;
    va_start(args, fmt);
    report(prefix, fmt, args);
    va_end(args);
    abort_frontend();
}

/* Return 0 on success, 1 on a recoverable frontend error. On error, every
 * partial token, AST node and bytecode allocation is released. The scope
 * lives on the heap so its state remains valid across longjmp. */
int prepare_source(const char *src, const char *path, int tokens_only, Proto **out, char *error,
                   size_t error_size) {
    FrontScope *scope = mem(sizeof *scope);
    scope->error = error;
    scope->error_size = error_size;
    *out = NULL;
    active = scope;
    if (setjmp(scope->trap)) {
        while (scope->head)
            front_free(scope->head + 1);
        active = NULL;
        free(scope);
        return 1;
    }
    Tokens t = tokenize(src, path);
    if (tokens_only) {
        for (int i = 0; i < t.n; i++)
            printf("%4d %-10s %s\n", t.a[i].line, tokname(t.a[i].t), t.a[i].lex);
    } else {
        Node *ast = parse(&t, path);
        *out = compile(ast, path);
        free_ast(ast);
    }
    freetokens(&t);
    /* Only bytecode allocations should remain; detach without freeing. */
    while (scope->head)
        detach(scope->head);
    active = NULL;
    free(scope);
    return 0;
}
