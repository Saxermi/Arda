#include "metadata.h"

enum { MAX_DEPTH = 16, MAX_VALUES = 4096 };
enum { META_NIL, META_FALSE, META_TRUE, META_INT, META_REAL, META_TEXT, META_LIST, META_MAP };

static bool invalid(void) {
    errno = EINVAL;
    return false;
}

static bool write_word(RecordOutput *out, uint64_t word) {
    unsigned char bytes[8];
    record_put_u64(bytes, word);
    return record_emit(out, bytes, sizeof bytes);
}

static bool write_tag(RecordOutput *out, unsigned char tag) {
    return record_emit(out, &tag, 1);
}

static bool json_text(RecordOutput *out, const char *text) {
    if (!record_text(out, "\""))
        return false;
    const unsigned char *begin = (const unsigned char *)text, *p = begin;
    for (; *p; p++) {
        // Unicode escapes for backslashes and HTML-sensitive characters also
        // allow JSON to be embedded as literal text in HTML or LaTeX reports.
        if (*p >= 32 && *p != '"' && *p != '\\' && *p != '<' && *p != '>' && *p != '&')
            continue;
        char escaped[7];
        snprintf(escaped, sizeof escaped, "\\u%04x", (unsigned)*p);
        if (!record_emit(out, begin, (size_t)(p - begin)) || !record_emit(out, escaped, 6))
            return false;
        begin = p + 1;
    }
    return record_emit(out, begin, (size_t)(p - begin)) && record_text(out, "\"");
}

static bool write_text(RecordOutput *out, const char *text, bool json) {
    if (!utf8valid(text))
        return invalid();
    if (json)
        return json_text(out, text);
    size_t length = strlen(text);
    return write_word(out, length) && record_emit(out, text, length);
}

static bool write_scalar(RecordOutput *out, Val value, bool json) {
    unsigned char tag;
    uint64_t word;
    char repr[48];
    switch (value.t) {
    case V_NIL:
        return json ? record_text(out, "null") : write_tag(out, META_NIL);
    case V_BOOL:
        return json ? record_text(out, value.n ? "true" : "false")
                    : write_tag(out, value.n ? META_TRUE : META_FALSE);
    case V_INT:
        if (json) {
            snprintf(repr, sizeof repr, "%" PRId64, value.i);
            return record_text(out, repr);
        }
        tag = META_INT;
        word = (uint64_t)value.i;
        break;
    case V_NUM:
        if (!isfinite(value.n))
            return invalid();
        if (json) {
            snprintf(repr, sizeof repr, "%.17g", value.n);
            return record_text(out, repr);
        }
        tag = META_REAL;
        _Static_assert(sizeof word == sizeof value.n, "metadata uses binary64");
        memcpy(&word, &value.n, sizeof word);
        break;
    default:
        return invalid();
    }
    return write_tag(out, tag) && write_word(out, word);
}

static bool write_value(RecordOutput *out, Val value, bool json, unsigned depth, unsigned *left) {
    if (!*left || depth > MAX_DEPTH)
        return invalid();
    --*left;
    if (value.t != V_OBJ)
        return write_scalar(out, value, json);
    if (isobj(value, O_STRING))
        return (json || write_tag(out, META_TEXT)) && write_text(out, value.o->as.str.s, json);
    if (isobj(value, O_LIST)) {
        Obj *list = value.o;
        if (json ? !record_text(out, "[")
                 : !(write_tag(out, META_LIST) && write_word(out, (uint64_t)list->as.list.n)))
            return false;
        for (int i = 0; i < list->as.list.n; i++)
            if ((json && i && !record_text(out, ",")) ||
                !write_value(out, list->as.list.a[i], json, depth + 1, left))
                return false;
        return !json || record_text(out, "]");
    }
    if (isobj(value, O_MAP)) {
        const Table *table = &value.o->as.map.tab;
        if (json ? !record_text(out, "{")
                 : !(write_tag(out, META_MAP) && write_word(out, (uint64_t)table->n)))
            return false;
        bool first = true;
        for (int i = 0; i < table->cap; i++) {
            const Entry *entry = &table->a[i];
            if (!entry->key)
                continue;
            if ((json && !first && !record_text(out, ",")) || !write_text(out, entry->key, json) ||
                (json && !record_text(out, ":")) ||
                !write_value(out, entry->val, json, depth + 1, left))
                return false;
            first = false;
        }
        return !json || record_text(out, "}");
    }
    return invalid();
}

bool metadata_binary(RecordOutput *out, Val value) {
    unsigned left = MAX_VALUES;
    return write_value(out, value, false, 0, &left);
}

bool metadata_json(RecordOutput *out, Val value) {
    unsigned left = MAX_VALUES;
    return write_value(out, value, true, 0, &left);
}

typedef struct {
    const unsigned char *data;
    size_t remaining;
    unsigned left;
} Decoder;

static bool read_word(Decoder *in, uint64_t *word) {
    if (in->remaining < 8)
        return false;
    *word = record_u64(in->data);
    in->data += 8;
    in->remaining -= 8;
    return true;
}

static Obj *read_text(VM *v, Decoder *in) {
    uint64_t length;
    if (!read_word(in, &length) || length > in->remaining || memchr(in->data, '\0', (size_t)length))
        return NULL;
    char *text = slice((const char *)in->data, (size_t)length);
    Obj *out = utf8valid(text) ? string(v, text) : NULL;
    free(text);
    in->data += (size_t)length;
    in->remaining -= (size_t)length;
    return out;
}

static bool read_value(VM *v, Decoder *in, unsigned depth, Val *value) {
    if (!in->remaining || !in->left || depth > MAX_DEPTH)
        return false;
    --in->left;
    unsigned tag = *in->data++;
    --in->remaining;
    uint64_t bits;
    switch (tag) {
    case META_NIL:
        *value = nil();
        return true;
    case META_FALSE:
    case META_TRUE:
        *value = boolean(tag == META_TRUE);
        return true;
    case META_INT:
        if (!read_word(in, &bits))
            return false;
        *value = integer(bits <= INT64_MAX ? (int64_t)bits
                                           : INT64_MIN + (int64_t)(bits - (UINT64_C(1) << 63)));
        return true;
    case META_REAL: {
        if (!read_word(in, &bits))
            return false;
        double real;
        memcpy(&real, &bits, sizeof real);
        if (!isfinite(real))
            return false;
        *value = number(real);
        return true;
    }
    case META_TEXT: {
        Obj *text = read_text(v, in);
        if (!text)
            return false;
        *value = objval(text);
        return true;
    }
    case META_LIST:
    case META_MAP: {
        uint64_t count;
        if (!read_word(in, &count) || count > in->left || count > in->remaining)
            return false;
        Obj *out = tag == META_LIST ? list(v) : newobj(v, O_MAP);
        for (uint64_t i = 0; i < count; i++) {
            Obj *key = NULL;
            if (tag == META_MAP) {
                key = read_text(v, in);
                if (!key || tablefind(&out->as.map.tab, key->as.str.s))
                    return false;
            }
            Val entry;
            if (!read_value(v, in, depth + 1, &entry))
                return false;
            if (key)
                tableset(v, out, &out->as.map.tab, key->as.str.s, entry);
            else
                append(v, out, entry);
        }
        *value = objval(out);
        return true;
    }
    default:
        return false;
    }
}

bool metadata_decode(VM *v, const unsigned char *data, size_t size, Val *value) {
    Decoder in = {data, size, MAX_VALUES};
    return read_value(v, &in, 0, value) && !in.remaining;
}
