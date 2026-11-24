#ifndef ARDA_FRONTEND_H
#define ARDA_FRONTEND_H
#include "arda.h"

/* All allocations made by the lexer/parser/compiler use this allocator.
 * Successful compilation transfers ownership to the returned Proto. */
void *front_mem(size_t n);
void *front_resize(void *p, size_t n);
void front_free(void *p);
char *front_slice(const char *s, size_t n);
char *front_copystr(const char *s);
_Noreturn void front_error(const char *fmt, ...);
_Noreturn void front_lex_error(const char *path, int line, const char *fmt, ...);
#endif
