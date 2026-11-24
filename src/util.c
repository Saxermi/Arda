#include "arda.h"
_Noreturn void die(const char *fmt, ...) {
    va_list a;
    va_start(a, fmt);
    vfprintf(stderr, fmt, a);
    va_end(a);
    fputc('\n', stderr);
    exit(1);
}

void *mem(size_t n) {
    void *p = calloc(1, n ? n : 1);
    if (!p)
        die("out of memory");
    return p;
}

void *resize(void *p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p)
        die("out of memory");
    return p;
}

char *slice(const char *s, size_t n) {
    char *p = mem(n + 1);
    memcpy(p, s, n);
    return p;
}

char *copystr(const char *s) {
    return slice(s, strlen(s));
}

char *readfile(const char *path) {
    FILE * f = fopen(path, "rb");
    if (!f)
        return NULL;
    if (fseek(f, 0, SEEK_END)) {
        fclose(f);
        return NULL;
    }
    long n = ftell(f);
    if (n < 0) {
        fclose(f);
        return NULL;
    }
    rewind(f);
    char *s = mem((size_t) n + 1);
    size_t got = fread(s, 1, (size_t) n, f);
    fclose(f);
    if (got != (size_t) n || memchr(s, 0, got)) {
        free(s);
        return NULL;
    }
    return s;
}
// Strict UTF-8, Unicode scalar values (not grapheme clusters).
int utf8width(const char *s) {
    const unsigned char *p = (const unsigned char * ) s;
    unsigned c = p[0];
    if (c < 0x80)
        return 1;
    int w = c >= 0xc2 && c <= 0xdf ? 2 : c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 0;
    if (!w)
        return 0;
    for (int i = 1; i < w; i++)
        if ((p[i] & 0xc0) != 0x80)
            return 0;
    if ((c == 0xe0 && p[1] < 0xa0) || (c == 0xed && p[1] >= 0xa0) || (c == 0xf0 && p[1] < 0x90) || (c == 0xf4 && p[1] >= 0x90))
        return 0;
    return w;
}

int utf8valid(const char *s) {
    while (*s) {
        int w = utf8width(s);
        if (!w)
            return 0;
        s += w;
    }
    return 1;
}

int utf8len(const char *s) {
    int n = 0;
    while (*s) {
        int w = utf8width(s);
        if (!w)
            return -1;
        s += w;
        n++;
    }
    return n;
}

const char *utf8at(const char *s, int i) {
    while (i-- > 0 && *s)
        s += utf8width(s);
    return s;
}
