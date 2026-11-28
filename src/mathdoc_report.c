#include "mathdoc_report.h"
#include "metadata.h"

typedef enum { MARKDOWN, JSON, HTML, LATEX } Format;
typedef struct {
    Val record;
    Format format;
    size_t limit;
} Report;

static Val get(Val map, const char *key) {
    Entry *entry = tablefind(&map.o->as.map.tab, key);
    return entry ? entry->val : nil();
}

static void field(VM *v, Obj *map, const char *name, Val value) {
    tableset(v, map, &map->as.map.tab, name, value);
}

static bool escaped(RecordOutput *out, const char *text, Format format) {
    if (!utf8valid(text)) {
        errno = EINVAL;
        return false;
    }
    const char *begin = text;
    for (const char *p = text; *p; p++) {
        const char *replacement = NULL;
        if (format == MARKDOWN) {
            if (*p == '\n')
                replacement = "\n    ";
            else if (*p == '\r')
                replacement = " ";
        } else if (format == HTML) {
            switch (*p) {
            case '&':
                replacement = "&amp;";
                break;
            case '<':
                replacement = "&lt;";
                break;
            case '>':
                replacement = "&gt;";
                break;
            case '"':
                replacement = "&quot;";
                break;
            }
        } else {
            switch (*p) {
            case '\\':
                replacement = "\\textbackslash{}";
                break;
            case '{':
                replacement = "\\{";
                break;
            case '}':
                replacement = "\\}";
                break;
            case '#':
                replacement = "\\#";
                break;
            case '$':
                replacement = "\\$";
                break;
            case '%':
                replacement = "\\%";
                break;
            case '&':
                replacement = "\\&";
                break;
            case '_':
                replacement = "\\_";
                break;
            case '~':
                replacement = "\\textasciitilde{}";
                break;
            case '^':
                replacement = "\\textasciicircum{}";
                break;
            case '\n':
            case '\r':
                replacement = " ";
                break;
            }
        }
        if (!replacement)
            continue;
        if (!record_emit(out, begin, (size_t)(p - begin)) || !record_text(out, replacement))
            return false;
        begin = p + 1;
    }
    return record_text(out, begin);
}

static bool section(RecordOutput *out, Format format, const char *name, Val content, bool formula) {
    bool text = isobj(content, O_STRING), absent = content.t == V_NIL;
    const char *value = absent ? "Unavailable" : text ? content.o->as.str.s : NULL;
    switch (format) {
    case MARKDOWN:
        return record_text(out, "\n## ") && record_text(out, name) &&
               record_text(out, "\n\n    ") &&
               (value ? escaped(out, value, format) : metadata_json(out, content)) &&
               record_text(out, "\n");
    case HTML:
        return record_text(out, "<h2>") && record_text(out, name) &&
               record_text(out, "</h2>\n<pre>") &&
               (value ? escaped(out, value, format) : metadata_json(out, content)) &&
               record_text(out, "</pre>\n");
    case LATEX:
        if (!record_text(out, "\\subsection*{") || !record_text(out, name) ||
            !record_text(out, "}\n"))
            return false;
        // Formula text only comes from Arda's static registry or MathIR.
        if (formula && text)
            return record_text(out, "\\[\n") && record_text(out, value) &&
                   record_text(out, "\n\\]\n");
        if (value)
            return escaped(out, value, format) && record_text(out, "\\par\n");
        return record_text(out, "\\begin{verbatim}\n") && metadata_json(out, content) &&
               record_text(out, "\n\\end{verbatim}\n");
    default:
        return false;
    }
}

static bool render(RecordOutput *out, const Report *report) {
    Format format = report->format;
    Val record = report->record;
    if (format == JSON)
        return metadata_json(out, record) && record_text(out, "\n");
    const char *title = get(record, "title").o->as.str.s;
    if (format == MARKDOWN) {
        if (!record_text(out, "# Arda scientific report\n\n    ") || !escaped(out, title, format) ||
            !record_text(out, "\n"))
            return false;
    } else if (format == HTML) {
        if (!record_text(out, "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
                              "<title>Arda scientific report</title></head><body><main><h1>") ||
            !escaped(out, title, format) || !record_text(out, "</h1>\n"))
            return false;
    } else {
        if (!record_text(out, "\\documentclass{article}\n\\usepackage{amsmath}\n"
                              "\\begin{document}\n\\section*{") ||
            !escaped(out, title, format) || !record_text(out, "}\n"))
            return false;
    }
    Val description = get(record, "static"), observed = get(record, "observed");
    if (!(section(out, format, "Formula (static)", get(description, "formula_tex"), true) &&
          section(out, format, "Algorithm (static)", get(description, "algorithm"), false) &&
          section(out, format, "Assumptions (static)", get(description, "assumptions"), false) &&
          section(out, format, "Complexity (static)", get(description, "complexity"), false) &&
          section(out, format, "Units (declared)", get(record, "units_declared"), false) &&
          section(out, format, "Verification (observed)", get(observed, "verification"), false) &&
          section(out, format, "Trace (observed)", get(observed, "trace"), false)))
        return false;
    if (format == HTML)
        return record_text(out, "</main></body></html>\n");
    if (format == LATEX)
        return record_text(out, "\\end{document}\n");
    return true;
}

static bool write_report(int fd, const void *context) {
    const Report *report = context;
    RecordOutput out = {.fd = fd, .limit = report->limit};
    return render(&out, report) && record_flush(&out);
}

Val mathdoc_export(VM *v, int argc, Val *args) {
    vm_require_file_access(v);
    if (argc != 3 && argc != 4)
        vmerror_kind(v, "arity",
                     "mathdoc.export expects target, path, format and optional options");
    const char *file = asstr(v, args[1]), *kind = asstr(v, args[2]);
    const char *const formats[] = {"markdown", "json", "html", "latex"};
    _Static_assert(sizeof formats / sizeof *formats == LATEX + 1, "report formats");
    int format = 0;
    while (format <= LATEX && strcmp(kind, formats[format]))
        format++;
    if (format > LATEX)
        vmerror_kind(v, "domain", "unknown AMDE report format");
    Val options = argc == 4 ? args[3] : nil();
    if (options.t != V_NIL && !isobj(options, O_MAP))
        vmerror_kind(v, "type", "AMDE export options must be a map");
    Val title = objval(string(v, "Arda scientific report")), level = integer(0);
    Val units = nil(), verification = nil(), trace = nil();
    size_t limit = METADATA_BYTES;
    if (isobj(options, O_MAP)) {
        Table *table = &options.o->as.map.tab;
        for (int i = 0; i < table->cap; i++) {
            Entry entry = table->a[i];
            if (!entry.key)
                continue;
            if (!strcmp(entry.key, "title") && isobj(entry.val, O_STRING))
                title = entry.val;
            else if (!strcmp(entry.key, "level"))
                level = entry.val;
            else if (!strcmp(entry.key, "units"))
                units = entry.val;
            else if (!strcmp(entry.key, "verification"))
                verification = entry.val;
            else if (!strcmp(entry.key, "trace"))
                trace = entry.val;
            else if (!strcmp(entry.key, "max_bytes"))
                limit = record_budget(v, entry.val, METADATA_BYTES);
            else
                vmerror_kind(v, "domain", "unknown or invalid AMDE export option: %s", entry.key);
        }
    }
    Obj *record = newobj(v, O_MAP), *observed = newobj(v, O_MAP);
    field(v, record, "version", integer(1));
    field(v, record, "title", title);
    field(v, record, "static", mathdoc_describe(v, args[0], level));
    field(v, record, "units_declared", units);
    field(v, observed, "verification", verification);
    field(v, observed, "trace", trace);
    field(v, record, "observed", objval(observed));
    Report report = {.record = objval(record), .format = (Format)format, .limit = limit};
    RecordOutput count = {.fd = -1, .limit = limit};
    if (!render(&count, &report))
        record_error(v, "AMDE export", errno);
    int cause = tensor_publish_file(file, write_report, &report);
    if (cause)
        record_error(v, "AMDE publication", cause);
    return integer((int64_t)count.size);
}
