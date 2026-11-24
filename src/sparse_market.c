#include "sparse_internal.h"
#include <strings.h>
#include <unistd.h>

#define S(o) ((o)->as.sparse)

enum { MARKET_LINE = 1024 };

static int market_line(VM *v, FILE *f, char *line) {
    size_t n = 0;
    int ch, truncated = 0;
    while ((ch = fgetc(f)) != EOF && ch != '\n') {
        if (ch == '\0')
            vmerror(v, "NUL byte in Matrix Market file");
        if (n + 1 < MARKET_LINE)
            line[n++] = (char)ch;
        else
            truncated = 1;
    }
    if (ch == EOF && ferror(f))
        vmerror(v, "could not read Matrix Market file");
    line[n] = '\0';
    if (truncated) {
        const char *p = line;
        while (isspace((unsigned char)*p))
            p++;
        if (*p != '%' || p[1] == '%')
            vmerror(v, "Matrix Market line exceeds 1023 bytes");
    }
    return n != 0 || ch != EOF;
}

static int market_tokens(char *line, char **tokens, int max) {
    int count = 0;
    while (*line) {
        while (isspace((unsigned char)*line))
            line++;
        if (!*line)
            break;
        if (count == max)
            return -1;
        tokens[count++] = line;
        while (*line && !isspace((unsigned char)*line))
            line++;
        if (*line)
            *line++ = '\0';
    }
    return count;
}

static int market_record(VM *v, FILE *f, char *line) {
    while (market_line(v, f, line)) {
        const char *p = line;
        while (isspace((unsigned char)*p))
            p++;
        if (*p && *p != '%')
            return 1;
    }
    return 0;
}

static uintmax_t market_uint(VM *v, const char *token, uintmax_t limit) {
    for (const char *p = token; *p; p++)
        if (*p < '0' || *p > '9')
            vmerror(v, "expected a Matrix Market unsigned integer");
    errno = 0;
    uintmax_t value = strtoumax(token, NULL, 10);
    if (errno == ERANGE || value > limit)
        vmerror(v, "Matrix Market dimension, index or entry count out of range");
    return value;
}

static double market_real(VM *v, const char *token) {
    if (token[strspn(token, "+-0123456789.eE")])
        vmerror(v, "invalid Matrix Market real value");
    char *end;
    errno = 0;
    double value = strtod(token, &end);
    if (end == token || *end || !isfinite(value) || (errno == ERANGE && value == 0.0))
        vmerror(v, "invalid or out-of-range Matrix Market real value");
    return value;
}

Obj *sparse_read_mm(VM *v, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f)
        vmerror(v, "could not open Matrix Market input");
    v->sparse_file = f;
    char line[MARKET_LINE], *tokens[5];
    if (!market_line(v, f, line) || market_tokens(line, tokens, 5) != 5 ||
        strcasecmp(tokens[0], "%%MatrixMarket") || strcasecmp(tokens[1], "matrix") ||
        strcasecmp(tokens[2], "coordinate") || strcasecmp(tokens[3], "real"))
        vmerror(v, "expected Matrix Market coordinate real header");
    int symmetric = !strcasecmp(tokens[4], "symmetric");
    if (!symmetric && strcasecmp(tokens[4], "general"))
        vmerror(v, "Matrix Market symmetry must be general or symmetric");
    if (!market_record(v, f, line) || market_tokens(line, tokens, 3) != 3)
        vmerror(v, "expected Matrix Market rows, columns and entry count");
    uint32_t rows = (uint32_t)market_uint(v, tokens[0], UINT32_MAX);
    uint32_t cols = (uint32_t)market_uint(v, tokens[1], UINT32_MAX);
    size_t count = (size_t)market_uint(v, tokens[2], SIZE_MAX);
    if (symmetric && rows != cols)
        vmerror(v, "Matrix Market symmetric input must be square");
    if (count && (!rows || !cols))
        vmerror(v, "Matrix Market empty shape cannot store entries");
    /* Grow as records are read. A truncated file cannot force its declared
     * count to be allocated before any entries have been validated. */
    Obj *builder = sparse_builder_new(v, rows, cols, 0, SIZE_MAX);
    for (size_t i = 0; i < count; i++) {
        if (!market_record(v, f, line) || market_tokens(line, tokens, 3) != 3)
            vmerror(v, "invalid or missing Matrix Market entry");
        uint32_t row = (uint32_t)market_uint(v, tokens[0], rows);
        uint32_t col = (uint32_t)market_uint(v, tokens[1], cols);
        if (!row || !col || (symmetric && row < col))
            vmerror(
                v,
                "Matrix Market requires one-based indices and lower triangular symmetric entries");
        double real = market_real(v, tokens[2]);
        int mirror = symmetric && row != col;
        sparse_builder_reserve(v, builder, 1u + (unsigned)mirror);
        sparse_builder_write(builder, row - 1, col - 1, real);
        if (mirror)
            sparse_builder_write(builder, col - 1, row - 1, real);
    }
    if (market_record(v, f, line))
        vmerror(v, "Matrix Market has more entries than declared");
    v->sparse_file = NULL;
    if (fclose(f) != 0)
        vmerror(v, "could not close Matrix Market input");
    return sparse_builder_finish(v, builder);
}

static int market_write(FILE *f, const Sparse *s, int symmetric) {
    size_t count = s->nnz;
    if (symmetric) {
        count = 0;
        for (size_t row = 0; row < s->rows; row++)
            for (size_t j = s->offset[row]; j < s->offset[row + 1]; j++)
                count += s->col[j] <= row;
    }
    if (fprintf(f, "%%%%MatrixMarket matrix coordinate real %s\n%" PRIu32 " %" PRIu32 " %zu\n",
                symmetric ? "symmetric" : "general", s->rows, s->cols, count) < 0)
        return 0;
    for (size_t row = 0; row < s->rows; row++) {
        for (size_t j = s->offset[row]; j < s->offset[row + 1]; j++) {
            if (symmetric && s->col[j] > row)
                break;
            if (fprintf(f, "%zu %" PRIu32 " %.17g\n", row + 1, s->col[j] + (uint32_t)1,
                        s->values[j]) < 0)
                return 0;
        }
    }
    return 1;
}

void sparse_write_mm(VM *v, const char *path, Obj *matrix, int symmetric) {
    Obj *csr = sparse_as_csr(v, matrix);
    const Sparse *s = &S(csr);
    if (symmetric && !sparse_symmetric(s))
        vmerror(v, "Matrix Market symmetric export requires a symmetric matrix");
    size_t length = strlen(path);
    const char suffix[] = ".arda-XXXXXX";
    if (!length || length > SIZE_MAX - sizeof suffix)
        vmerror(v, "invalid Matrix Market output path");
    char *temp = malloc(length + sizeof suffix);
    if (!temp)
        vmerror(v, "could not allocate Matrix Market output path");
    memcpy(temp, path, length);
    memcpy(temp + length, suffix, sizeof suffix);
    int fd = mkstemp(temp);
    if (fd < 0) {
        free(temp);
        vmerror(v, "could not create Matrix Market output");
    }
    FILE *f = fdopen(fd, "w");
    int ok = 0;
    if (f) {
        ok = market_write(f, s, symmetric);
        if (fclose(f) != 0)
            ok = 0;
    } else
        close(fd);
    if (ok)
        ok = rename(temp, path) == 0;
    if (!ok)
        unlink(temp);
    free(temp);
    if (!ok)
        vmerror(v, "could not write Matrix Market output");
    sparse_drop_temporary(v, csr, matrix);
}
