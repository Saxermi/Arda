#include "tensor_file.h"

#include <ctype.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>

enum { CHUNK_BYTES = 65536, NPY_HEADER_MAX = 4096, FIELD_MAX = 256 };

static FILE *open_input(VM *v, const char *path, off_t *length) {
    vm_require_file_access(v);
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0)
        vmerror_kind(v, "io", "cannot open tensor import: %s", strerror(errno));
    struct stat info;
    if (fstat(fd, &info) || !S_ISREG(info.st_mode) || info.st_size < 0) {
        close(fd);
        vmerror_kind(v, "io", "tensor import requires a regular file");
    }
    FILE *file = fdopen(fd, "rb");
    if (!file) {
        int cause = errno;
        close(fd);
        vmerror_kind(v, "io", "cannot read tensor import: %s", strerror(cause));
    }
    *length = info.st_size;
    return file;
}

static void check_import_budget(VM *v, FILE *file, const TypedTensor *layout, Val budget) {
    size_t bytes;
    if (!tensor_file_size(layout, &bytes)) {
        fclose(file);
        vmerror_kind(v, "shape", "import exceeds the addressable tensor file size");
    }
    if (!tensor_file_fits_budget(budget, bytes)) {
        fclose(file);
        vmerror_kind(v, "disk_budget", "import exceeds the disk budget or budget is invalid");
    }
}

static void skip_space(const char **p) {
    while (isspace((unsigned char)**p))
        (*p)++;
}

static bool symbol(const char **p, char expected) {
    skip_space(p);
    if (**p != expected)
        return false;
    (*p)++;
    return true;
}

static bool quoted_word(const char **p, char *word, size_t capacity) {
    skip_space(p);
    char quote = **p;
    if (quote != '\'' && quote != '"')
        return false;
    (*p)++;
    size_t n = 0;
    while (**p && **p != quote) {
        if (n + 1 == capacity || **p == '\\')
            return false;
        word[n++] = *(*p)++;
    }
    if (**p != quote)
        return false;
    (*p)++;
    word[n] = '\0';
    return true;
}

static bool npy_dtype(const char **p, DType *dtype) {
    char description[8];
    if (!quoted_word(p, description, sizeof description))
        return false;
    const char *const names[DT_COUNT] = {
        [DT_BOOL] = "|b1", [DT_U8] = "|u1",  [DT_I32] = "<i4", [DT_I64] = "<i8",   [DT_F16] = "<f2",
        [DT_F32] = "<f4",  [DT_F64] = "<f8", [DT_C64] = "<c8", [DT_C128] = "<c16",
    };
    for (int i = 0; i < DT_COUNT; i++)
        if (names[i] && !strcmp(names[i], description)) {
            *dtype = (DType)i;
            return true;
        }
    return false;
}

static bool npy_shape(const char **p, int *rank, int shape[8]) {
    if (!symbol(p, '('))
        return false;
    for (;;) {
        skip_space(p);
        if (*rank == 8 || !isdigit((unsigned char)**p))
            return false;
        int dim = 0;
        do {
            int digit = *(*p)++ - '0';
            if (dim > (INT_MAX - digit) / 10)
                return false;
            dim = 10 * dim + digit;
        } while (isdigit((unsigned char)**p));
        shape[(*rank)++] = dim;
        if (!symbol(p, ','))
            return *rank > 1 && symbol(p, ')');
        if (symbol(p, ')'))
            return true;
    }
}

static bool npy_dictionary(const char *p, TypedTensor *layout) {
    if (!symbol(&p, '{'))
        return false;
    enum { DESCR = 1, ORDER = 2, SHAPE = 4 };
    unsigned fields = 0;
    int rank = 0, shape[8];
    DType dtype = DT_COUNT;
    while (!symbol(&p, '}')) {
        char key[16];
        if (!quoted_word(&p, key, sizeof key) || !symbol(&p, ':'))
            return false;
        unsigned field;
        if (!strcmp(key, "descr")) {
            field = DESCR;
            if (!npy_dtype(&p, &dtype))
                return false;
        } else if (!strcmp(key, "fortran_order")) {
            field = ORDER;
            skip_space(&p);
            if (strncmp(p, "False", 5))
                return false;
            p += 5;
        } else if (!strcmp(key, "shape")) {
            field = SHAPE;
            if (!npy_shape(&p, &rank, shape))
                return false;
        } else {
            return false;
        }
        if (fields & field)
            return false;
        fields |= field;
        if (!symbol(&p, ',')) {
            if (!symbol(&p, '}'))
                return false;
            break;
        }
    }
    skip_space(&p);
    return !*p && fields == (DESCR | ORDER | SHAPE) &&
           typed_layout_checked(dtype, rank, shape, layout);
}

static bool npy_header(FILE *file, off_t file_size, TypedTensor *layout) {
    unsigned char prefix[8], width[4];
    if (fread(prefix, 1, sizeof prefix, file) != sizeof prefix || memcmp(prefix, "\x93NUMPY", 6) ||
        prefix[6] < 1 || prefix[6] > 3 || prefix[7] != 0)
        return false;
    size_t length_width = prefix[6] == 1 ? 2 : 4;
    if (fread(width, 1, length_width, file) != length_width)
        return false;
    uint32_t header_length = 0;
    for (size_t i = 0; i < length_width; i++)
        header_length |= (uint32_t)width[i] << (8 * i);
    size_t prefix_size = sizeof prefix + length_width;
    char header[NPY_HEADER_MAX + 1];
    if (!header_length || header_length > NPY_HEADER_MAX ||
        (uintmax_t)file_size < prefix_size + header_length ||
        fread(header, 1, header_length, file) != header_length ||
        memchr(header, '\0', header_length) || header[header_length - 1] != '\n')
        return false;
    header[header_length] = '\0';
    if (!npy_dictionary(header, layout))
        return false;
    return (uintmax_t)file_size - prefix_size - header_length ==
           layout->n * dtype_info[layout->dtype].size;
}

typedef struct {
    FILE *file;
    size_t bytes;
} NpyCopy;

static bool expect_end(FILE *input) {
    if (fgetc(input) == EOF && !ferror(input))
        return true;
    errno = EIO;
    return false;
}

static bool copy_npy(int fd, const void *context) {
    const NpyCopy *copy = context;
    unsigned char chunk[CHUNK_BYTES];
    size_t remaining = copy->bytes;
    while (remaining) {
        size_t count = remaining < sizeof chunk ? remaining : sizeof chunk;
        if (fread(chunk, 1, count, copy->file) != count) {
            errno = EIO;
            return false;
        }
        if (!tensor_file_write(fd, chunk, count))
            return false;
        remaining -= count;
    }
    return expect_end(copy->file);
}

Val tensor_import_npy(VM *v, Val input, Val output, Val budget) {
    tensor_file_require_little_endian(v);
    const char *source = asstr(v, input), *destination = asstr(v, output);
    off_t size;
    FILE *file = open_input(v, source, &size);
    TypedTensor layout;
    if (!npy_header(file, size, &layout)) {
        fclose(file);
        vmerror_kind(v, "file_format", "unsupported NPY dtype, layout, header or payload length");
    }
    check_import_budget(v, file, &layout, budget);
    NpyCopy copy = {.file = file, .bytes = layout.n * dtype_info[layout.dtype].size};
    int cause = tensor_publish_payload(destination, &layout, copy_npy, &copy);
    fclose(file);
    if (cause)
        vmerror_kind(v, "io", "cannot publish NPY import: %s", strerror(cause));
    return nil();
}

typedef struct {
    FILE *file;
    int rows, columns, delimiter;
    bool header;
    bool *invalid;
} DelimitedCopy;

enum { FIELD_ERROR = -1, FIELD_END, FIELD_NEXT, FIELD_ROW };

static bool append_field(char field[FIELD_MAX + 1], size_t *length, int c) {
    if (c == 0 || c == EOF || *length == FIELD_MAX)
        return false;
    field[(*length)++] = (char)c;
    return true;
}

static int read_field(const DelimitedCopy *copy, char field[FIELD_MAX + 1]) {
    int c = fgetc(copy->file);
    size_t length = 0;
    if (c == '"') {
        for (;;) {
            c = fgetc(copy->file);
            if (c == '"') {
                c = fgetc(copy->file);
                if (c != '"')
                    break;
            }
            if (!append_field(field, &length, c))
                return FIELD_ERROR;
        }
    } else {
        while (c != EOF && c != copy->delimiter && c != '\n' && c != '\r') {
            if (c == '"' || !append_field(field, &length, c))
                return FIELD_ERROR;
            c = fgetc(copy->file);
        }
    }
    field[length] = '\0';
    if (c == copy->delimiter)
        return FIELD_NEXT;
    if (c == '\r')
        return fgetc(copy->file) == '\n' ? FIELD_ROW : FIELD_ERROR;
    if (c == '\n')
        return FIELD_ROW;
    return c == EOF && !ferror(copy->file) ? FIELD_END : FIELD_ERROR;
}

static bool parse_real(const char *field, double *result) {
    skip_space(&field);
    if (!*field)
        return false;
    size_t n = strspn(field, "0123456789.+-eE");
    const char *tail = field + n;
    skip_space(&tail);
    if (*tail)
        return false;
    char *end;
    errno = 0;
    *result = strtod(field, &end);
    return end != field && end == field + n && isfinite(*result) &&
           (errno != ERANGE || *result != 0);
}

static bool bad_delimited(const DelimitedCopy *copy) {
    *copy->invalid = !ferror(copy->file);
    errno = *copy->invalid ? EINVAL : EIO;
    return false;
}

static bool copy_delimited(int fd, const void *context) {
    const DelimitedCopy *copy = context;
    if (copy->header) {
        int first = fgetc(copy->file);
        if (first == EOF || ungetc(first, copy->file) == EOF)
            return bad_delimited(copy);
    }
    double chunk[CHUNK_BYTES / sizeof(double)];
    size_t count = 0;
    for (int row = copy->header ? -1 : 0; row < copy->rows; row++) {
        for (int col = 0; col < copy->columns; col++) {
            char field[FIELD_MAX + 1];
            int end = read_field(copy, field);
            if (col + 1 < copy->columns
                    ? end != FIELD_NEXT
                    : end != FIELD_ROW && !(end == FIELD_END && row + 1 == copy->rows))
                return bad_delimited(copy);
            if (row < 0)
                continue;
            if (!parse_real(field, chunk + count))
                return bad_delimited(copy);
            if (++count == sizeof chunk / sizeof *chunk) {
                if (!tensor_file_write(fd, chunk, sizeof chunk))
                    return false;
                count = 0;
            }
        }
    }
    if (!expect_end(copy->file))
        return bad_delimited(copy);
    return tensor_file_write(fd, chunk, count * sizeof *chunk);
}

Val tensor_import_delimited(VM *v, Val input, Val output, Val shape, Val options) {
    tensor_file_require_little_endian(v);
    const char *source = asstr(v, input), *destination = asstr(v, output);
    int dims[8];
    int rank = typed_dimensions(v, shape, dims);
    if (rank != 2 || dims[1] == 0)
        vmerror_kind(v, "shape", "delimited import needs a [rows, columns] shape with columns > 0");
    TypedTensor layout = typed_layout(v, DT_F64, rank, dims);
    int delimiter = ',';
    bool header = false;
    Val limit = nil();
    if (options.t != V_NIL) {
        if (!isobj(options, O_MAP))
            vmerror_kind(v, "type", "delimited import options must be a map");
        Table *table = &options.o->as.map.tab;
        Entry *sep = tablefind(table, "delimiter"), *skip = tablefind(table, "header"),
              *budget = tablefind(table, "disk_budget");
        if (table->n != (sep != NULL) + (skip != NULL) + (budget != NULL))
            vmerror_kind(v, "domain", "unknown delimited import option");
        if (sep) {
            const char *text = asstr(v, sep->val);
            if (strcmp(text, ",") && strcmp(text, "\t"))
                vmerror_kind(v, "domain", "delimited import supports comma or tab");
            delimiter = text[0];
        }
        if (skip) {
            if (skip->val.t != V_BOOL)
                vmerror_kind(v, "type", "delimited header option must be Boolean");
            header = skip->val.n != 0;
        }
        if (budget)
            limit = budget->val;
    }
    off_t size;
    FILE *file = open_input(v, source, &size);
    check_import_budget(v, file, &layout, limit);
    bool invalid = false;
    DelimitedCopy copy = {.file = file,
                          .rows = dims[0],
                          .columns = dims[1],
                          .delimiter = delimiter,
                          .header = header,
                          .invalid = &invalid};
    int cause = tensor_publish_payload(destination, &layout, copy_delimited, &copy);
    fclose(file);
    if (invalid)
        vmerror_kind(v, "file_format",
                     "delimited import does not match the shape or numeric format");
    if (cause)
        vmerror_kind(v, "io", "cannot publish delimited import: %s", strerror(cause));
    return nil();
}
