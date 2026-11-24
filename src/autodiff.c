#include "arda.h"
#define D(x) ((x)->as.ad)
#define A(x) ((x)->as.array)
static Obj *node(VM *v, Val a) {
    if (!isobj(a, O_AD))
        vmerror(v, "expected autodiff node");
    return a.o;
}

static double value(VM *v, Val a) {
    return isobj(a, O_AD) ? D(a.o).value : asnum(v, a);
}

static Obj *make(VM *v, double x, Obj *a, Obj *b, double da, double db) {
    if (!isfinite(x) || (a && !isfinite(da)) || (b && !isfinite(db)))
        vmerror(v, "nonfinite autodiff value or derivative");
    Obj *o = newobj(v, O_AD);
    D(o).value = x;
    D(o).a = a;
    D(o).b = b;
    D(o).da = da;
    D(o).db = db;
    return o;
}

Val ad_binary(VM *v, Op op, Val a, Val b) {
    double x = value(v, a), y = value(v, b), z = 0, da = 0, db = 0;
    Obj *p = isobj(a, O_AD) ? a.o : NULL, *q = isobj(b, O_AD) ? b.o : NULL;
    switch (op) {
    case OP_ADD:
        z = x + y;
        da = db = 1;
        break;
    case OP_SUB:
        z = x - y;
        da = 1;
        db = -1;
        break;
    case OP_MUL:
        z = x * y;
        da = y;
        db = x;
        break;
    case OP_DIV:
        if (y == 0)
            vmerror(v, "division by zero");
        z = x / y;
        da = 1 / y;
        db = -z / y;
        break;
    default:
        vmerror(v, "unsupported autodiff operator (use ad.value for comparisons)");
    }
    return objval(make(v, z, p, q, da, db));
}
/* Iterative postorder DFS, then reverse topological accumulation. Each node
 * has at most two incoming operand links; shared subexpressions visit once. */
static void backward(VM *v, Obj *root) {
    Obj *stack = list(v), *order = list(v);
    append(v, stack, objval(root));
    while (stack->as.list.n) {
        Obj *o = stack->as.list.a[stack->as.list.n - 1].o;
        if (D(o).visited == 0) {
            D(o).visited = 1;
            D(o).grad = 0;
            if (D(o).b && !D(D(o).b).visited)
                append(v, stack, objval(D(o).b));
            if (D(o).a && !D(D(o).a).visited)
                append(v, stack, objval(D(o).a));
        } else {
            stack->as.list.n--;
            if (D(o).visited == 1) {
                D(o).visited = 2;
                append(v, order, objval(o));
            }
        }
    }
    D(root).grad = 1;
    int finite = 1;
    for (int i = order->as.list.n - 1; i >= 0; i--) {
        Obj *o = order->as.list.a[i].o;
        double g = D(o).grad;
        if (!isfinite(g))
            finite = 0;
        if (D(o).a)
            D(D(o).a).grad += g * D(o).da;
        if (D(o).b)
            D(D(o).b).grad += g * D(o).db;
        D(o).visited = 0;
    }
    if (!finite)
        vmerror(v, "nonfinite autodiff gradient");
}

static const char *names[] = {"variable", "value", "grad", "backward", "sin", "cos", "exp",
                              "log",      "sqrt",  "tanh", "pow",      "sgd", "adam"};
static Obj *vec(VM *v, Val x) {
    if (!isobj(x, O_ARRAY) || A(x.o).rank != 1)
        vmerror(v, "optimizer expects vectors");
    return tensor_contiguous(x.o) ? x.o : tensor_copy(v, x.o);
}

static double rate(VM *v, Val a) {
    double x = asnum(v, a);
    if (!isfinite(x) || x <= 0)
        vmerror(v, "learning rate must be positive and finite");
    return x;
}

static Val optimizer(VM *v, int k, Val *a) {
    Obj *p = vec(v, a[0]), *g = vec(v, a[1]);
    int n = A(p).rows;
    if (A(g).rows != n)
        vmerror(v, "optimizer shape mismatch");
    double lr = rate(v, a[k == 11 ? 2 : 5]);
    Obj *out = array_new(v, 1, n, 1);
    if (k == 11) {
        for (int i = 0; i < n; i++) {
            double x = A(p).data[i], d = A(g).data[i], z = x - lr * d;
            if (!isfinite(x) || !isfinite(d) || !isfinite(z))
                vmerror(v, "nonfinite SGD step");
            A(out).data[i] = z;
        }
        return objval(out);
    }
    Obj *m = vec(v, a[2]), *s = vec(v, a[3]);
    if (A(m).rows != n || A(s).rows != n)
        vmerror(v, "optimizer state shape mismatch");
    if (a[4].t != V_INT || a[4].i < 1)
        vmerror(v, "Adam step must start at 1");
    double t = (double)a[4].i, b1 = .9, b2 = .999, c1 = -expm1(t * log(b1)),
           c2 = -expm1(t * log(b2));
    Obj *nm = array_new(v, 1, n, 1), *ns = array_new(v, 1, n, 1);
    for (int i = 0; i < n; i++) {
        double x = A(p).data[i], d = A(g).data[i], mi = A(m).data[i], si = A(s).data[i];
        if (!isfinite(x) || !isfinite(d) || !isfinite(mi) || !isfinite(si) || si < 0)
            vmerror(v, "invalid Adam input");
        double u = b1 * mi + (1 - b1) * d, w = b2 * si + (1 - b2) * d * d,
               z = x - lr * (u / c1) / (sqrt(w / c2) + 1e-8);
        if (!isfinite(u) || !isfinite(w) || !isfinite(z))
            vmerror(v, "nonfinite Adam step");
        A(nm).data[i] = u;
        A(ns).data[i] = w;
        A(out).data[i] = z;
    }
    Obj *l = list(v);
    append(v, l, objval(out));
    append(v, l, objval(nm));
    append(v, l, objval(ns));
    return objval(l);
}

Val ad_call(VM *v, int id, int argc, Val *a) {
    int k = id - AD_BASE;
    int ar[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 6};
    if (k < 0 || k >= 13 || argc != ar[k])
        vmerror(v, "invalid autodiff function arity");
    if (k >= 11)
        return optimizer(v, k, a);
    if (k == 0)
        return objval(make(v, asnum(v, a[0]), NULL, NULL, 0, 0));
    if (k == 1)
        return number(value(v, a[0]));
    if (k == 2)
        return number(D(node(v, a[0])).grad);
    if (k == 3) {
        backward(v, node(v, a[0]));
        return nil();
    }
    double x = value(v, a[0]), z = 0, d = 0;
    Obj *p = isobj(a[0], O_AD) ? a[0].o : NULL;
    switch (k) {
    case 4:
        z = sin(x);
        d = cos(x);
        break;
    case 5:
        z = cos(x);
        d = -sin(x);
        break;
    case 6:
        z = exp(x);
        d = z;
        break;
    case 7:
        if (x <= 0)
            vmerror(v, "log domain");
        z = log(x);
        d = 1 / x;
        break;
    case 8:
        if (x < 0 || (x == 0 && p))
            vmerror(v, "sqrt derivative domain");
        z = sqrt(x);
        d = .5 / z;
        break;
    case 9:
        z = tanh(x);
        d = 1 - z * z;
        break;
    case 10: {
        double y = asnum(v, a[1]);
        if (!isfinite(y))
            vmerror(v, "invalid power exponent");
        z = pow(x, y);
        d = y == 0 ? 0 : y == 1 ? 1 : y * pow(x, y - 1);
        break;
    }
    }
    if (!p) {
        if (!isfinite(z))
            vmerror(v, "nonfinite result");
        return number(z);
    }
    return objval(make(v, z, p, NULL, d, 0));
}

const char **calculus_names(int *n);
Obj *extra_module(VM *v, const char *name) {
    int n = 0, base;
    const char **ns;
    if (!strcmp(name, "calc")) {
        ns = calculus_names(&n);
        base = CALC_BASE;
    } else if (!strcmp(name, "ad")) {
        ns = names;
        n = 13;
        base = AD_BASE;
    } else
        return NULL;
    char key[64];
    snprintf(key, sizeof key, "<native:%s>", name);
    Entry *e = tablefind(&v->modules, key);
    if (e)
        return e->val.o;
    Obj *o = newobj(v, O_MODULE);
    o->as.mod.path = copystr(key);
    account(v, o, 0, strlen(key) + 1);
    o->as.mod.state = 1;
    o->as.mod.env = environment(v, NULL);
    for (int i = 0; i < n; i++) {
        Obj *f = newobj(v, O_NATIVE);
        f->as.native.id = base + i;
        tableset(v, o->as.mod.env, &o->as.mod.env->as.env.tab, ns[i], objval(f));
    }
    tableset(v, NULL, &v->modules, key, objval(o));
    return o;
}
