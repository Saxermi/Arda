#include "tensor_file.h"

#include <limits.h>
#include <stdarg.h>
#include <unistd.h>

enum { SCAN_ELEMENTS = 8192, VIEW_WIDTH = 640, VIEW_HEIGHT = 400, MARGIN = 24 };
typedef enum { LINE, SCATTER, HISTOGRAM, HEATMAP, CONTOUR, VECTOR } PlotKind;
static const char *const plot_names[] = {"line",    "scatter", "histogram",
                                         "heatmap", "contour", "vector"};

typedef struct {
    FILE *file;
    uint64_t remaining;
} ExportStream;

static bool emit(ExportStream *out, const void *bytes, size_t size) {
    if (size > out->remaining) {
        errno = EFBIG;
        return false;
    }
    if (fwrite(bytes, 1, size, out->file) != size)
        return false;
    out->remaining -= size;
    return true;
}

static bool format(ExportStream *out, const char *pattern, ...) {
    char line[512];
    va_list args;
    va_start(args, pattern);
    int size = vsnprintf(line, sizeof line, pattern, args);
    va_end(args);
    if (size < 0 || (size_t)size >= sizeof line) {
        errno = EIO;
        return false;
    }
    return emit(out, line, (size_t)size);
}

typedef struct {
    TensorFile input;
    uint64_t disk_budget;
    size_t samples, stride, scan_count;
    int rows, columns, bins;
    PlotKind kind;
    long double level;
    unsigned char *scratch;
} PlotJob;

typedef struct {
    const TensorFile *input;
    size_t count, first, valid;
    unsigned char *data;
} SampleReader;

static bool sample(SampleReader *reader, size_t index, long double *value) {
    if (index < reader->first || index - reader->first >= reader->valid) {
        reader->first = index - index % reader->count;
        reader->valid = reader->input->layout.n - reader->first;
        if (reader->valid > reader->count)
            reader->valid = reader->count;
        if (!tensor_file_read(reader->input, reader->first, reader->valid, reader->data))
            return false;
    }
    DType dtype = reader->input->layout.dtype;
    const unsigned char *data = reader->data + (index - reader->first) * dtype_info[dtype].size;
    TensorScalar scalar = dtype_load(dtype, data);
    *value = scalar.kind <= SC_INT ? (long double)scalar.i : (long double)scalar.re;
    if (!isfinite(*value)) {
        errno = EDOM;
        return false;
    }
    return true;
}

typedef struct {
    long double low, high;
} Range;
typedef struct {
    double x, y;
} Point;

static void include(Range *range, long double value) {
    range->low = fminl(range->low, value);
    range->high = fmaxl(range->high, value);
}

static double fraction(long double value, long double low, long double high) {
    if (high == low)
        return 0.5;
    long double result;
    if (!isfinite(high - low))
        result = (value * 0.5L - low * 0.5L) / (high * 0.5L - low * 0.5L);
    else
        result = (value - low) / (high - low);
    return (double)fminl(1, fmaxl(0, result));
}

static bool series_point(const PlotJob *job, SampleReader *reader, size_t i, long double *x,
                         long double *y) {
    size_t row = i * job->stride;
    if (job->input.layout.rank == 1) {
        *x = (long double)row;
        return sample(reader, row, y);
    }
    return sample(reader, 2 * row, x) && sample(reader, 2 * row + 1, y);
}

static bool draw_series(const PlotJob *job, SampleReader *reader, ExportStream *out) {
    Range xs = {INFINITY, -INFINITY}, ys = xs;
    for (size_t i = 0; i < job->samples; i++) {
        long double x, y;
        if (!series_point(job, reader, i, &x, &y))
            return false;
        include(&xs, x);
        include(&ys, y);
    }
    if (!format(out,
                "<desc>x %.8Lg to %.8Lg, y %.8Lg to %.8Lg</desc>\n"
                "<path d=\"M24 24V376H616\" fill=\"none\" stroke=\"#64748b\"/>\n",
                xs.low, xs.high, ys.low, ys.high))
        return false;
    for (size_t i = 0; i < job->samples; i++) {
        long double x, y;
        if (!series_point(job, reader, i, &x, &y))
            return false;
        double px = MARGIN + fraction(x, xs.low, xs.high) * (VIEW_WIDTH - 2 * MARGIN);
        double py =
            VIEW_HEIGHT - MARGIN - fraction(y, ys.low, ys.high) * (VIEW_HEIGHT - 2 * MARGIN);
        if (job->kind == SCATTER || job->samples == 1) {
            if (!format(out, "<circle cx=\"%.4f\" cy=\"%.4f\" r=\"2.5\" fill=\"#1855a0\"/>\n", px,
                        py))
                return false;
        } else if (!format(out, "%s%c%.4f %.4f\n",
                           i == 0 ? "<path fill=\"none\" stroke=\"#1855a0\" d=\"" : "",
                           i == 0 ? 'M' : 'L', px, py))
            return false;
    }
    return job->kind != LINE || job->samples == 1 || format(out, "\"/>\n");
}

static bool draw_histogram(const PlotJob *job, SampleReader *reader, ExportStream *out) {
    Range range = {INFINITY, -INFINITY};
    for (size_t i = 0; i < job->samples; i++) {
        long double value;
        if (!sample(reader, i, &value))
            return false;
        include(&range, value);
    }
    size_t *counts = (size_t *)job->scratch;
    memset(counts, 0, (size_t)job->bins * sizeof *counts);
    size_t max_count = 1;
    for (size_t i = 0; i < job->samples; i++) {
        long double value;
        if (!sample(reader, i, &value))
            return false;
        int bin =
            range.high == range.low ? 0 : (int)(job->bins * fraction(value, range.low, range.high));
        if (bin == job->bins)
            bin--;
        if (++counts[bin] > max_count)
            max_count = counts[bin];
    }
    if (!format(out, "<desc>Range %.8Lg to %.8Lg, %zu values, %d bins</desc>\n", range.low,
                range.high, job->samples, job->bins))
        return false;
    double width = (double)(VIEW_WIDTH - 2 * MARGIN) / job->bins;
    for (int i = 0; i < job->bins; i++) {
        double height = (double)counts[i] / (double)max_count * (VIEW_HEIGHT - 2 * MARGIN);
        if (!format(out,
                    "<rect x=\"%.4f\" y=\"%.4f\" width=\"%.4f\" height=\"%.4f\" "
                    "fill=\"#1855a0\" data-count=\"%zu\"/>\n",
                    MARGIN + i * width, VIEW_HEIGHT - MARGIN - height, width, height, counts[i]))
            return false;
    }
    return true;
}

static size_t grid_offset(const PlotJob *job, int row, int col) {
    return ((size_t)row * job->input.layout.shape[1] + (size_t)col) * job->stride;
}

static bool draw_grid(const PlotJob *job, SampleReader *reader, ExportStream *out) {
    int components = job->kind == VECTOR ? 2 : 1;
    Range range = {INFINITY, -INFINITY};
    for (int row = 0; row < job->rows; row++)
        for (int col = 0; col < job->columns; col++)
            for (int c = 0; c < components; c++) {
                long double value;
                if (!sample(reader, grid_offset(job, row, col) * components + c, &value))
                    return false;
                include(&range, components == 2 ? fabsl(value) : value);
            }
    if (components == 2 &&
        !format(out, "<defs><marker id=\"arrow\" markerWidth=\"6\" markerHeight=\"4\" "
                     "refX=\"6\" refY=\"2\" orient=\"auto\"><path d=\"M0 0L6 2L0 4Z\" "
                     "fill=\"#1855a0\"/></marker></defs>\n"))
        return false;
    if (!format(out, "<desc>Range %.8Lg to %.8Lg, sample stride %zu</desc>\n", range.low,
                range.high, job->stride))
        return false;
    double dx = (double)VIEW_WIDTH / job->columns, dy = (double)VIEW_HEIGHT / job->rows;
    for (int row = 0; row < job->rows; row++)
        for (int col = 0; col < job->columns; col++) {
            long double value;
            size_t pos = grid_offset(job, row, col) * components;
            if (!sample(reader, pos, &value))
                return false;
            if (components == 1) {
                double mix = fraction(value, range.low, range.high);
                if (!format(out,
                            "<rect x=\"%.4f\" y=\"%.4f\" width=\"%.4f\" height=\"%.4f\" "
                            "fill=\"#%02x%02x%02x\"/>\n",
                            col * dx, row * dy, dx, dy, (unsigned)(29 + 221 * mix),
                            (unsigned)(69 + 111 * mix), (unsigned)(149 - 94 * mix)))
                    return false;
            } else {
                long double vertical;
                if (!sample(reader, pos + 1, &vertical))
                    return false;
                if (value == 0 && vertical == 0)
                    continue;
                double x = (col + 0.5) * dx, y = (row + 0.5) * dy;
                double scale = 0.35 * fmin(dx, dy);
                long double peak = fmaxl(range.high, fmaxl(fabsl(value), fabsl(vertical)));
                if (!format(out,
                            "<line x1=\"%.4f\" y1=\"%.4f\" x2=\"%.4f\" y2=\"%.4f\" "
                            "stroke=\"#1855a0\" marker-end=\"url(#arrow)\"/>\n",
                            x, y, x + scale * (double)(value / peak),
                            y - scale * (double)(vertical / peak)))
                    return false;
            }
        }
    return true;
}

static bool contour_segment(ExportStream *out, Point a, Point b) {
    return format(out, "<path d=\"M%.4f %.4fL%.4f %.4f\" fill=\"none\" stroke=\"#1855a0\"/>\n", a.x,
                  a.y, b.x, b.y);
}

static bool draw_contour(const PlotJob *job, SampleReader *reader, ExportStream *out) {
    double dx = (double)VIEW_WIDTH / (job->columns - 1);
    double dy = (double)VIEW_HEIGHT / (job->rows - 1);
    const int ox[] = {0, 1, 1, 0}, oy[] = {0, 0, 1, 1};
    SampleReader bottom = *reader;
    bottom.data += reader->count * dtype_info[job->input.layout.dtype].size;
    for (int row = 0; row + 1 < job->rows; row++)
        for (int col = 0; col + 1 < job->columns; col++) {
            long double val[4];
            for (int i = 0; i < 4; i++)
                if (!sample(i < 2 ? reader : &bottom, grid_offset(job, row + oy[i], col + ox[i]),
                            val + i))
                    return false;
            Point crossing[4];
            int count = 0;
            for (int i = 0; i < 4; i++) {
                int j = (i + 1) % 4;
                if ((val[i] >= job->level) == (val[j] >= job->level))
                    continue;
                double t = fraction(job->level, val[i], val[j]);
                crossing[count++] = (Point){(col + ox[i] + (ox[j] - ox[i]) * t) * dx,
                                            (row + oy[i] + (oy[j] - oy[i]) * t) * dy};
            }
            if (count == 2 && !contour_segment(out, crossing[0], crossing[1]))
                return false;
            if (count == 4) {
                long double center =
                    val[0] * 0.25L + val[1] * 0.25L + val[2] * 0.25L + val[3] * 0.25L;
                int first = (center >= job->level) == (val[0] >= job->level) ? 0 : 1;
                if (!contour_segment(out, crossing[first], crossing[first + 1]) ||
                    !contour_segment(out, crossing[(first + 2) % 4], crossing[(first + 3) % 4]))
                    return false;
            }
        }
    return true;
}

static bool svg_content(const PlotJob *job, ExportStream *out) {
    size_t histogram_bytes = (size_t)job->bins * sizeof(size_t);
    SampleReader reader = {
        .input = &job->input, .count = job->scan_count, .data = job->scratch + histogram_bytes};
    if (!format(out,
                "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 %d %d\">\n"
                "<title>%s</title><rect width=\"100%%\" height=\"100%%\" fill=\"white\"/>\n",
                VIEW_WIDTH, VIEW_HEIGHT, plot_names[job->kind]))
        return false;
    bool ok;
    switch (job->kind) {
    case LINE:
    case SCATTER:
        ok = draw_series(job, &reader, out);
        break;
    case HISTOGRAM:
        ok = draw_histogram(job, &reader, out);
        break;
    case HEATMAP:
    case VECTOR:
        ok = draw_grid(job, &reader, out);
        break;
    case CONTOUR:
        ok = draw_contour(job, &reader, out);
        break;
    default:
        errno = EINVAL;
        return false;
    }
    return ok && format(out, "</svg>\n");
}

static bool vtk_content(const PlotJob *job, ExportStream *out) {
    const TypedTensor *layout = &job->input.layout;
    int rank = layout->rank, x = layout->shape[rank - 1], y = layout->shape[rank - 2];
    int z = rank == 3 ? layout->shape[0] : 1;
    if (!format(out,
                "# vtk DataFile Version 3.0\nArda tensor\nBINARY\n"
                "DATASET STRUCTURED_POINTS\nDIMENSIONS %d %d %d\nORIGIN 0 0 0\n"
                "SPACING 1 1 1\nPOINT_DATA %zu\nSCALARS values double 1\n"
                "LOOKUP_TABLE default\n",
                x, y, z, layout->n))
        return false;
    unsigned char *chunk = job->scratch;
    for (size_t pos = 0; pos < layout->n;) {
        size_t count = layout->n - pos;
        if (count > job->scan_count)
            count = job->scan_count;
        if (!tensor_file_read(&job->input, pos, count, chunk))
            return false;
        for (size_t i = 0; i < count; i++)
            for (size_t j = 0; j < 4; j++) {
                unsigned char *word = chunk + 8 * i, first = word[j];
                word[j] = word[7 - j];
                word[7 - j] = first;
            }
        if (!emit(out, chunk, count * 8))
            return false;
        pos += count;
    }
    return emit(out, "\n", 1);
}

static bool write_export(int fd, const PlotJob *job,
                         bool (*content)(const PlotJob *, ExportStream *)) {
    int stream_fd = dup(fd);
    if (stream_fd < 0)
        return false;
    FILE *stream = fdopen(stream_fd, "wb");
    if (!stream) {
        int cause = errno;
        close(stream_fd);
        errno = cause;
        return false;
    }
    ExportStream out = {.file = stream, .remaining = job->disk_budget};
    bool success = content(job, &out);
    int cause = errno;
    if (fclose(stream) && success) {
        success = false;
        cause = errno;
    }
    errno = cause;
    return success;
}

static bool write_svg(int fd, const void *context) {
    return write_export(fd, context, svg_content);
}

static bool write_vtk(int fd, const void *context) {
    return write_export(fd, context, vtk_content);
}

static uint64_t output_budget(VM *v, Val value) {
    if (value.t == V_NIL)
        return UINT64_MAX;
    if (value.t != V_INT || value.i < 0)
        vmerror_kind(v, "disk_budget", "plot disk budget must be a nonnegative integer");
    return (uint64_t)value.i;
}

static void svg_options(VM *v, PlotJob *job, Val value) {
    if (value.t != V_NIL && !isobj(value, O_MAP))
        vmerror_kind(v, "type", "plot options must be a map");
    Table *table = value.t == V_NIL ? NULL : &value.o->as.map.tab;
    Entry *stride = table ? tablefind(table, "stride") : NULL;
    Entry *bins = table ? tablefind(table, "bins") : NULL;
    Entry *level = table ? tablefind(table, "level") : NULL;
    Entry *budget = table ? tablefind(table, "disk_budget") : NULL;
    if (table && table->n != !!stride + !!bins + !!level + !!budget)
        vmerror_kind(v, "domain", "unknown SVG plot option");
    if (stride && (stride->val.t != V_INT || stride->val.i < 1 || stride->val.i > INT_MAX ||
                   job->kind == HISTOGRAM))
        vmerror_kind(v, "domain",
                     "plot stride must be a positive integer for series or grid plots");
    if (bins &&
        (job->kind != HISTOGRAM || bins->val.t != V_INT || bins->val.i < 1 || bins->val.i > 4096))
        vmerror_kind(v, "domain", "histogram bins must be an integer from 1 to 4096");
    if ((job->kind == CONTOUR) != !!level)
        vmerror_kind(v, "domain", "a contour requires a level option");
    if (level) {
        job->level = level->val.t == V_INT ? (long double)level->val.i : asnum(v, level->val);
        if (!isfinite(job->level))
            vmerror_kind(v, "domain", "contour level must be finite");
    }
    job->stride = stride ? (size_t)stride->val.i : 1;
    job->bins = job->kind == HISTOGRAM ? (bins ? (int)bins->val.i : 32) : 0;
    job->disk_budget = output_budget(v, budget ? budget->val : nil());
}

static bool svg_shape(PlotJob *job) {
    const TypedTensor *t = &job->input.layout;
    if (!t->n || dtype_info[t->dtype].kind == SC_COMPLEX)
        return false;
    switch (job->kind) {
    case LINE:
    case SCATTER:
        if (!(t->rank == 2 && t->shape[1] == 2) && !(job->kind == LINE && t->rank == 1))
            return false;
        job->samples = 1 + ((size_t)t->shape[0] - 1) / job->stride;
        return true;
    case HISTOGRAM:
        job->samples = t->n;
        return true;
    case HEATMAP:
    case CONTOUR:
    case VECTOR:
        if (job->kind == VECTOR ? t->rank != 3 || t->shape[2] != 2 : t->rank != 2)
            return false;
        job->rows = 1 + (int)(((size_t)t->shape[0] - 1) / job->stride);
        job->columns = 1 + (int)(((size_t)t->shape[1] - 1) / job->stride);
        return job->kind != CONTOUR || (job->rows >= 2 && job->columns >= 2);
    }
    return false;
}

static Val run_export(VM *v, const char *source, const char *destination, PlotJob *job, bool svg) {
    tensor_file_require_little_endian(v);
    int cause = tensor_file_open(source, &job->input);
    if (cause)
        vmerror_kind(v, cause == EINVAL ? "file_format" : "io", "cannot open plot source: %s",
                     strerror(cause));
    const TypedTensor *layout = &job->input.layout;
    bool shape_ok = svg ? svg_shape(job)
                        : (layout->dtype == DT_F64 && layout->n && layout->n <= INT_MAX &&
                           (layout->rank == 2 || layout->rank == 3));
    if (!shape_ok) {
        close(job->input.fd);
        vmerror_kind(v, "shape", "tensor shape or dtype is incompatible with the export");
    }
    size_t width = dtype_info[layout->dtype].size * (svg && job->kind == CONTOUR ? 2 : 1);
    size_t counters = (size_t)job->bins * sizeof(size_t), available = tensor_memory_available(v);
    size_t capacity = available > counters ? (available - counters) / width : 0;
    job->scan_count = layout->n < SCAN_ELEMENTS ? layout->n : SCAN_ELEMENTS;
    if (job->scan_count > capacity)
        job->scan_count = capacity;
    job->scratch =
        job->scan_count ? tensor_workspace_acquire(v, job->scan_count * width + counters) : NULL;
    if (!job->scratch) {
        close(job->input.fd);
        vmerror_kind(v, "memory_budget", "export exceeds the tensor workspace budget");
    }
    cause = tensor_publish_file(destination, svg ? write_svg : write_vtk, job);
    tensor_workspace_release(v, job->scratch);
    close(job->input.fd);
    if (cause)
        vmerror_kind(v,
                     cause == EDOM    ? "domain"
                     : cause == EFBIG ? "disk_budget"
                                      : "io",
                     "export failed: %s", strerror(cause));
    return nil();
}

Val plot_call(VM *v, int id, int argc, Val *args) {
    PlotJob job = {0};
    if (id == PLOT_BASE && (argc == 3 || argc == 4)) {
        const char *kind = asstr(v, args[0]), *source = asstr(v, args[1]);
        const char *destination = asstr(v, args[2]);
        for (size_t i = 0; i < sizeof plot_names / sizeof *plot_names; i++)
            if (!strcmp(kind, plot_names[i])) {
                job.kind = (PlotKind)i;
                svg_options(v, &job, argc == 4 ? args[3] : nil());
                return run_export(v, source, destination, &job, true);
            }
        vmerror_kind(v, "domain", "unknown SVG plot kind '%s'", kind);
    }
    if (id == PLOT_BASE + 1 && (argc == 2 || argc == 3)) {
        const char *source = asstr(v, args[0]), *destination = asstr(v, args[1]);
        job.disk_budget = output_budget(v, argc == 3 ? args[2] : nil());
        return run_export(v, source, destination, &job, false);
    }
    vmerror_kind(v, "arity", "invalid plot API arity");
}

Obj *plot_module(VM *v) {
    static const char *const names[] = {"svg", "vtk"};
    return named_native_module(v, "plot", names, sizeof names / sizeof *names, PLOT_BASE);
}
