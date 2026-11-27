#include "arda.h"

#define ARDA_VERSION "0.22.1"

static volatile sig_atomic_t interrupted;

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
    int status = vm_run(v, p);
    if (v->interrupted)
        interrupted = 0;
    return status;
}

static void interrupt_handler(int signal_number) {
    (void)signal_number;
    interrupted = 1;
}

static bool option_number(const char *arg, uint64_t max, uint64_t *value) {
    const char *text = strchr(arg, '=') + 1;
    char *end;
    errno = 0;
    uintmax_t parsed = strtoumax(text, &end, 10);
    if (*text < '0' || *text > '9' || *end || errno || !parsed || parsed > max) {
        fprintf(stderr, "invalid positive limit: %s\n", arg);
        return false;
    }
    *value = (uint64_t)parsed;
    return true;
}

static void help(void) {
    puts("Arda " ARDA_VERSION " | C11 bytecode VM\nusage: arda [--tokens|--bytecode] [--gc-stress] "
         "[--gc-stats] [--mathdoc=none|compact] file.arda\n       arda "
         "[--gc-stress] [--gc-stats] [--mathdoc=none|compact]  # REPL; blank "
         "line executes a block; :quit exits\n       arda --version\n"
         "runtime options: --gc-profile --max-instructions=N --max-stack=N "
         "--max-frames=N --max-vm-bytes=N --interruptible --no-file-io");
}

int main(int argc, char **argv) {
    int mode = 0, stress = 0, stats = 0;
    bool profile_gc = false, interruptible = false, no_file_io = false;
    uint64_t instructions = 0, stack = ARDA_MAX_STACK, frames = ARDA_MAX_FRAMES;
    uint64_t bytes = SIZE_MAX;
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
        else if (!strcmp(argv[i], "--gc-profile"))
            profile_gc = true;
        else if (!strcmp(argv[i], "--no-file-io"))
            no_file_io = true;
        else if (!strcmp(argv[i], "--interruptible"))
            interruptible = true;
        else if (!strncmp(argv[i], "--max-instructions=", 19)) {
            if (!option_number(argv[i], UINT64_MAX, &instructions))
                return 1;
        } else if (!strncmp(argv[i], "--max-stack=", 12)) {
            if (!option_number(argv[i], ARDA_MAX_STACK, &stack))
                return 1;
        } else if (!strncmp(argv[i], "--max-frames=", 13)) {
            if (!option_number(argv[i], ARDA_MAX_FRAMES, &frames))
                return 1;
        } else if (!strncmp(argv[i], "--max-vm-bytes=", 15)) {
            if (!option_number(argv[i], SIZE_MAX, &bytes))
                return 1;
        } else if (!strcmp(argv[i], "--mathdoc=none"))
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
    v->instruction_limit = instructions;
    v->no_file_io = no_file_io;
    v->stack_limit = (int)stack;
    v->frame_limit = (int)frames;
    v->vm_byte_limit = (size_t)bytes;
    GCProfile profile = {.peak_bytes = v->bytes};
    if (profile_gc)
        v->gc_profile = &profile;
    if (interruptible) {
        struct sigaction action = {.sa_handler = interrupt_handler};
        sigemptyset(&action.sa_mask);
        if (sigaction(SIGINT, &action, NULL)) {
            perror("install interrupt handler");
            vm_destroy(v);
            free(v);
            return 1;
        }
        v->interrupted = &interrupted;
    }
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
            if (got < 0 && v->interrupted && interrupted && errno == EINTR) {
                clearerr(stdin);
                interrupted = 0;
                n = 0;
                block = 0;
                putchar('\n');
                continue;
            }
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
    if (profile_gc)
        fprintf(stderr,
                "gc-profile: sweeps=%zu total_sweep_ns=%" PRIu64 " max_sweep_ns=%" PRIu64
                " max_objects=%zu peak_managed_bytes=%zu\n",
                profile.sweeps, profile.total_ns, profile.max_ns, profile.max_objects,
                profile.peak_bytes);
    vm_destroy(v);
    free(v);
    return status;
}
