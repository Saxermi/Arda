#include "arda.h"

#define ARDA_VERSION "0.15.4"

static int source(VM *v, const char *s, const char *path, int mode) {
    Proto *p = NULL;
    char error[1024];
    if (prepare_source(s, path, mode == 1, &p, error, sizeof error)) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    if (mode == 1)
        return 0;
    if (mode == 2) {
        disassemble(p, 0);
        freeproto(p);
        return 0;
    }
    remember_proto(v, p);
    return vm_run(v, p);
}

static void help(void) {
    puts("Arda " ARDA_VERSION " | C11 bytecode VM\nusage: arda [--tokens|--bytecode] [--gc-stress] "
         "[--gc-stats] [--mathdoc=none|compact] file.arda\n       arda "
         "[--gc-stress] [--gc-stats] [--mathdoc=none|compact]  # REPL; blank "
         "line executes a block; :quit exits\n       arda --version");
}

int main(int argc, char **argv) {
    int mode = 0, stress = 0, stats = 0;
    const char *path = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help")) {
            help();
            return 0;
        }
        if (!strcmp(argv[i], "--version")) {
            puts("Arda " ARDA_VERSION);
            return 0;
        }
        if (!strcmp(argv[i], "--tokens"))
            mode = 1;
        else if (!strcmp(argv[i], "--bytecode"))
            mode = 2;
        else if (!strcmp(argv[i], "--gc-stress"))
            stress = 1;
        else if (!strcmp(argv[i], "--gc-stats"))
            stats = 1;
        else if (!strcmp(argv[i], "--mathdoc=none"))
            mathir_set_enabled(0);
        else if (!strcmp(argv[i], "--mathdoc=compact"))
            mathir_set_enabled(1);
        else if (!path && argv[i][0] != '-')
            path = argv[i];
        else {
            help();
            return 1;
        }
    }
    VM *v = mem(sizeof *v);
    vm_init(v, stress);
    int status = 0;
    if (path) {
        char *s = readfile(path);
        if (!s) {
            fprintf(stderr, "cannot read '%s'\n", path);
            vm_destroy(v);
            free(v);
            return 1;
        }
        char *canonical = realpath(path, NULL);
        status = source(v, s, canonical ? canonical : path, mode);
        free(canonical);
        free(s);
    } else if (mode) {
        help();
        status = 1;
    } else {
        puts("Arda " ARDA_VERSION " | blank line runs a block | :quit exits");
        char *line = NULL, *buf = NULL;
        size_t cap = 0, n = 0;
        int block = 0;
        while (1) {
            fputs(n ? "... " : "arda> ", stdout);
            fflush(stdout);
            ssize_t got = getline(&line, &cap, stdin);
            if (got < 0) {
                if (n)
                    source(v, buf, "<repl>", 0);
                break;
            }
            if (!n && (!strcmp(line, ":quit\n") || !strcmp(line, ":q\n")))
                break;
            if (n && (!strcmp(line, "\n") || !strcmp(line, "\r\n"))) {
                source(v, buf, "<repl>", 0);
                n = 0;
                block = 0;
                continue;
            }
            if (!n && line[0] == '\n')
                continue;
            buf = resize(buf, n + (size_t)got + 1);
            memcpy(buf + n, line, (size_t)got + 1);
            n += (size_t)got;
            size_t k = (size_t)got;
            while (k && isspace((unsigned char)line[k - 1]))
                k--;
            if (k && line[k - 1] == ':')
                block = 1;
            if (!block) {
                source(v, buf, "<repl>", 0);
                n = 0;
            }
        }
        free(buf);
        free(line);
    }
    if (stats) {
        gcfull(v);
        fprintf(stderr, "gc: objects=%zu bytes=%zu program_roots=%d cycles=%zu\n", v->live,
                v->bytes, v->np, v->cycles);
    }
    vm_destroy(v);
    free(v);
    return status;
}
