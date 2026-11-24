#ifndef ARDA_H
#define ARDA_H
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <setjmp.h>
#include <limits.h>
#include <inttypes.h>
#include <errno.h>
#include "sparse.h"
_Noreturn void die(const char *fmt, ...);
void *mem(size_t n);
void *resize(void *p, size_t n);
char *copystr(const char *s);
char *slice(const char *s, size_t n);
char *readfile(const char *path);
typedef enum {
    T_EOF,
    T_ID,
    T_NUM,
    T_STR,
    T_LP,
    T_RP,
    T_LB,
    T_RB,
    T_LS,
    T_RS,
    T_COMMA,
    T_NL,
    T_INDENT,
    T_DEDENT,
    T_COLON,
    T_DOT,
    T_PLUS,
    T_MINUS,
    T_STAR,
    T_SLASH,
    T_PERCENT,
    T_BANG,
    T_EQ,
    T_EQEQ,
    T_NE,
    T_LT,
    T_LE,
    T_GT,
    T_GE,
    T_AND,
    T_OR,
    T_ESTA,
    T_CARO,
    T_NAN,
    T_QUI,
    T_AI,
    T_TENNA,
    T_PEDO,
    T_TRUE,
    T_FALSE,
    T_NIL,
    T_BREAK,
    T_CONTINUE,
    T_FOR,
    T_IN,
    T_CLASS,
    T_IMPORT,
    T_AS,
    T_TRY,
    T_CATCH,
    T_THROW,
    T_PASS,
    T_ELIF,
    T_LAMBDA
} TokenType;
typedef struct {
    TokenType t;
    char *lex;
    double num;
    int line;
    int64_t integer;
    int isint;
} Token;
typedef struct {
    Token *a;
    int n, cap;
} Tokens;
Tokens tokenize(const char *src, const char *path);
const char *tokname(TokenType t);
void freetokens(Tokens *t);
typedef enum {
    N_NUM,
    N_STR,
    N_BOOL,
    N_NIL,
    N_VAR,
    N_LIST,
    N_MAP,
    N_INDEX,
    N_ATTR,
    N_ASSIGN,
    N_BINARY,
    N_UNARY,
    N_CALL,
    N_BLOCK,
    N_DECLARE,
    N_FUNCTION,
    N_IF,
    N_WHILE,
    N_RETURN,
    N_PRINT,
    N_EXPR,
    N_BREAK,
    N_CONTINUE,
    N_FOR,
    N_CLASS,
    N_IMPORT,
    N_TRY,
    N_THROW,
    N_PASS,
    N_SLICE,
    N_LAMBDA,
    N_LISTCOMP
} NodeType;
typedef struct Node Node;
struct Node {
    NodeType t;
    int line, op;
    double num;
    int64_t integer;
    int isint;
    char *str;
    Node *a, *b, *c;
    Node **items;
    int count, cap;
    char **names;
    Node **defaults;
    int n_names;
};
Node *parse(Tokens *ts, const char *path);
void free_ast(Node *n);
// Bytecode uses fixed-width instructions and per-function constant pools.
typedef enum {
    OP_CONST,
    OP_NIL,
    OP_TRUE,
    OP_FALSE,
    OP_GET,
    OP_DEFINE,
    OP_SET,
    OP_POP,
    OP_DUP,
    OP_SCOPE,
    OP_UNSCOPE,
    OP_LIST,
    OP_MAP,
    OP_INDEX,
    OP_SET_INDEX,
    OP_ATTR,
    OP_SET_ATTR,
    OP_NEG,
    OP_NOT,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_MOD,
    OP_EQ,
    OP_NE,
    OP_LT,
    OP_LE,
    OP_GT,
    OP_GE,
    OP_JUMP,
    OP_JFALSE,
    OP_JTRUE,
    OP_CLOSURE,
    OP_CALL,
    OP_RETURN,
    OP_PRINT,
    OP_ITER,
    OP_NEXT,
    OP_CLASS,
    OP_METHOD,
    OP_IMPORT,
    OP_TRY,
    OP_END_TRY,
    OP_THROW,
    OP_GET_LOCAL,
    OP_SET_LOCAL,
    OP_JPRESENT,
    OP_SLICE,
    OP_COMP_APPEND,
    OP_SET_DROP,
    OP_SET_LOCAL_DROP,
    OP_ADD_INT_CONST,
    OP_SUB_INT_CONST,
    OP_MUL_INT_CONST,
    OP_JFALSE_LT_CONST
} Op;
typedef struct {
    Op op;
    int arg, line, aux;
} Instr;
typedef struct Proto Proto;
typedef struct MathIR MathIR;
typedef enum { K_NUM, K_STR, K_PROTO, K_INT } KType;
typedef struct {
    KType type;
    double n;
    char *s;
    Proto *p;
    int64_t i;
    uint64_t hash, stamp;
    int hint;
} Constant;
/* Child prototypes share the lifetime of their compilation root. */
struct Proto {
    Proto *root;
    unsigned char marked;
    char *name, *path;
    char **params;
    MathIR *mathir;
    int arity, required;
    Instr *code;
    int count, cap;
    Constant *k;
    int nk, kcap;
};
Proto *compile(Node *root, const char *path);
void freeproto(Proto *p);
void mathir_set_enabled(int enabled);
MathIR *mathir_extract(const Node *function);
int mathir_has_loop(const Proto *p);
int mathir_tex(const Proto *p, int level, char *out, size_t size);
const char *mathir_assumptions(const Proto *p);
const char *mathir_algorithm(const Proto *p);
#define MATHIR_TEX_LIMIT 8192
int prepare_source(const char *src, const char *path, int tokens_only, Proto **out, char *error,
                   size_t error_size);
void disassemble(Proto *p, int depth);
// Runtime objects and VM.
typedef struct Obj Obj;
typedef enum { V_NIL, V_NUM, V_BOOL, V_OBJ, V_INT, V_MISSING } VType;
typedef struct {
    VType t;
    union {
        double n;
        Obj *o;
        int64_t i;
    };
} Val;
typedef enum {
    O_STRING,
    O_LIST,
    O_MAP,
    O_ENV,
    O_FUNC,
    O_NATIVE,
    O_CLASS,
    O_INSTANCE,
    O_BOUND,
    O_MODULE,
    O_ITER,
    O_ARRAY,
    O_AD,
    O_SPARSE,
    O_SPARSE_BUILDER,
    O_SPARSE_WORKSPACE,
    O_FACTOR,
    O_RNG,
    O_MOMENTS,
    O_COMPLEX,
    O_LINOP
} OType;

static inline int sparse_owns_payload(OType type) {
    return type == O_SPARSE || type == O_SPARSE_BUILDER || type == O_SPARSE_WORKSPACE;
}
typedef struct {
    char *key;
    Val val;
    uint64_t hash;
} Entry;
typedef struct {
    Entry *a;
    int n, cap;
    uint64_t stamp;
} Table;
typedef struct {
    uint64_t state, increment;
} RandomState;
struct Obj {
    OType type;
    unsigned char mark;
    Obj *next;
    size_t bytes;
    union {
        struct {
            char *s;
            size_t len;
            uint64_t hash;
            Obj *intern_next;
        } str;
        struct {
            Val *a;
            int n, cap;
        } list;
        struct {
            Table tab;
        } map;
        struct {
            Table tab;
            Obj *parent;
            Proto *params;
            Val *slots;
        } env;
        struct {
            Proto *p;
            Obj *env;
        } func;
        struct {
            int id;
        } native;
        struct {
            char *name;
            Table methods;
            Obj *parent;
        } cls;
        struct {
            Obj *cls;
            Table fields;
        } inst;
        struct {
            Obj *fn;
            Val self;
        } bound;
        struct {
            char *path;
            Obj *env;
            int state;
        } mod;
        struct {
            Val source;
            int index;
        } iter;
        struct {
            double *data;
            int rank, rows, cols;
            size_t n;
            int shape[8];
            size_t stride[8];
            Obj *owner;
        } array;
        struct {
            double value, grad, da, db;
            Obj *a, *b;
            unsigned char visited;
        } ad;
        struct {
            Obj *packed;
            int kind;
        } factor;
        RandomState rng;
        struct {
            double re, im;
        } cplx;
        struct {
            Obj *first, *second;
            int kind, rows, cols, lower, upper;
        } linop;
        Sparse sparse;
        SparseBuilder sparse_builder;
        struct {
            size_t length;
            double *data;
        } sparse_workspace;
    } as;
};
typedef struct {
    Proto *p;
    int ip, base;
    Obj *env, *module, *locals;
    Val constructed;
} Frame;
typedef struct {
    int frame, ip, sp;
    Obj *env;
} Handler;
typedef struct VM {
    Obj *objects, *global, *builtins;
    Obj **intern;
    size_t intern_cap, intern_count;
    Table modules;
    uint64_t next_table_stamp;
    Val *stack;
    int sp, scap;
    Frame frames[512];
    int nf;
    Handler *handlers;
    int nh, hcap;
    Obj **gray;
    size_t ng, gcap, live, bytes, next_gc, cycles;
    int marking, stress;
    size_t sparse_bytes, sparse_limit;
    Obj *sparse_checkpoint;
    FILE *sparse_file;
    int sparse_active;
    Proto **programs;
    int np, pcap;
    jmp_buf trap;
    char error[1024];
    int line;
    const char *path;
    int failed;
} VM;
static inline Val nil(void) {
    return (Val){.t = V_NIL};
}

static inline Val number(double n) {
    return (Val){.t = V_NUM, .n = n};
}

static inline Val integer(int64_t n) {
    return (Val){.t = V_INT, .i = n};
}

static inline Val boolean(int n) {
    return (Val){.t = V_BOOL, .n = !!n};
}

static inline Val objval(Obj *o) {
    return (Val){.t = V_OBJ, .o = o};
}

static inline int isobj(Val v, OType t) {
    return v.t == V_OBJ && v.o->type == t;
}

static inline int truth(Val v) {
    if (v.t == V_NIL || v.t == V_MISSING)
        return 0;
    if (v.t == V_INT)
        return v.i != 0;
    if (v.t == V_BOOL || v.t == V_NUM)
        return v.n != 0;
    if (isobj(v, O_COMPLEX))
        return v.o->as.cplx.re != 0 || v.o->as.cplx.im != 0;
    if (isobj(v, O_STRING))
        return v.o->as.str.len != 0;
    if (isobj(v, O_LIST))
        return v.o->as.list.n != 0;
    if (isobj(v, O_MAP))
        return v.o->as.map.tab.n != 0;
    return 1;
}

Obj *newobj(VM *v, OType t);
Obj *newobj_extra(VM *v, OType t, size_t extra);
void account(VM *v, Obj *o, size_t old, size_t now);
void markval(VM *v, Val x);
void markobj(VM *v, Obj *o);
void barrier(VM *v, Obj *owner, Val x);
void gcstep(VM *v, int force);
void gcfull(VM *v);
void destroy_objects(VM *v);
uint64_t hashstr(const char *s);
Entry *tablefind(Table *t, const char *key);
Entry *tablefind_cached(Table *t, Constant *key);
void tableset(VM *v, Obj *owner, Table *t, const char *key, Val x);
void tablefree(Table *t);
Obj *environment(VM *v, Obj *parent);
Obj *string(VM *v, const char *s);
Obj *list(VM *v);
void append(VM *v, Obj *o, Val x);
_Noreturn void vmerror(VM *v, const char *fmt, ...);
void vm_init(VM *v, int stress);
void vm_destroy(VM *v);
int vm_run(VM *v, Proto *p);
void remember_proto(VM *v, Proto *p);
void install_builtins(VM *v);
Val native_call(VM *v, int id, int argc, Val *a);
const char *native_builtin_name(int id);
char *value_string(Val v);
void printval(Val v);
double asnum(VM *v, Val x);
const char *asstr(VM *v, Val x);
int indexnum(VM *v, Val x, int n);
int utf8width(const char *s);
int utf8len(const char *s);
int utf8valid(const char *s);
const char *utf8at(const char *s, int i);
Obj *native_module(VM *v, const char *name);
#define SCI_BASE 100
Obj *scientific_module(VM *v);
Val scientific_call(VM *v, int id, int argc, Val *a);
const char *scientific_name(int id);
Val array_binary(VM *v, Op op, Val a, Val b);
Val array_index(VM *v, Val a, Val b);
void array_set_index(VM *v, Val a, Val b, Val x);
void array_print(Obj *o);
Obj *array_new(VM *v, int rank, int rows, int cols);
#define ARDA_MAX_ELEMENTS 16000000u
Obj *tensor_new(VM *v, int rank, const int *shape);
Obj *tensor_copy(VM *v, Obj *x);
size_t tensor_offset(Obj *x, size_t i);
int tensor_contiguous(Obj *x);
Val tensor_binary(VM *v, Op op, Val a, Val b);
Val tensor_index(VM *v, Obj *x, int i);
void tensor_print(Obj *x);
Val tensor_tolist(VM *v, Obj *x);
Obj *tensor_from_list(VM *v, Val a);
Val tensor_call(VM *v, int id, int argc, Val *a);
Val calculus_call(VM *v, int id, int argc, Val *a);
Val ad_call(VM *v, int id, int argc, Val *a);
Val ad_binary(VM *v, Op op, Val a, Val b);
Obj *extra_module(VM *v, const char *name);
void install_tensor(VM *v, Obj *module);
#define TENSOR_BASE 200
#define CALC_BASE 300
#define AD_BASE 400
#define HIGH_BASE 500
#define SPARSE_BASE 600
#define LINALG_BASE 700
#define RANDOM_BASE 800
#define STATS_BASE 900
#define COMPLEX_BASE 1000
#define LINOP_BASE 1100
#define MATHDOC_BASE 1200
#define SPECIAL_BASE 1300
Val linalg_call(VM *v, int id, int argc, Val *a);
const char *linalg_name(int id);
Val linalg_eigh(VM *v, Val a);
Obj *linalg_module(VM *v);
Obj *named_native_module(VM *v, const char *name, const char *const *names, size_t count, int base);
Obj *random_module(VM *v);
Val random_call(VM *v, int id, int argc, Val *a);
void random_seed(RandomState *rng, int64_t seed, uint64_t stream);
long double random_normal_unit(RandomState *rng);
Obj *stats_module(VM *v);
Val stats_call(VM *v, int id, int argc, Val *a);
Obj *complex_module(VM *v);
Val complex_call(VM *v, int id, int argc, Val *a);
Val complex_value(VM *v, double re, double im);
Val complex_binary(VM *v, Op op, Val a, Val b);
Obj *linop_module(VM *v);
Val linop_call(VM *v, int id, int argc, Val *a);
Obj *mathdoc_module(VM *v);
Val mathdoc_call(VM *v, int id, int argc, Val *a);
Val special_call(VM *v, int id, int argc, Val *a);
Obj *special_module(VM *v);
int vm_uses_builtin(VM *v, Obj *env, const char *name);
Val mathdoc_attribute(VM *v, Val target, const char *name);
Obj *dense_lu_solve(VM *v, Obj *a, Obj *b);
Val sparse_call(VM *v, int id, int argc, Val *a);
const char *sparse_name(int id);
Obj *sparse_module(VM *v);
void sparse_abort(VM *v);
void sparse_release_payload(Obj *o);
Val high_call(VM *v, int id, int argc, Val *a);
Obj *high_module(VM *v);
#endif
