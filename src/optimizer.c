#include "arda.h"

#define A(x) ((x)->as.array)

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

Val optimizer_call(VM *v, int id, int argc, Val *a) {
    int k = id - OPTIMIZER_BASE;
    if ((k != 0 && k != 1) || argc != (k == 0 ? 3 : 6))
        vmerror(v, "invalid optimizer function arity");
    Obj *p = vec(v, a[0]), *g = vec(v, a[1]);
    int n = A(p).rows;
    if (A(g).rows != n)
        vmerror(v, "optimizer shape mismatch");
    double lr = rate(v, a[k == 0 ? 2 : 5]);
    Obj *out = array_new(v, 1, n, 1);
    if (k == 0) {
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
    Obj *report = newobj(v, O_MAP);
    Table *tab = &report->as.map.tab;
    tableset(v, report, tab, "parameters", objval(out));
    tableset(v, report, tab, "first_moment", objval(nm));
    tableset(v, report, tab, "second_moment", objval(ns));
    return objval(report);
}

Obj *optimizer_module(VM *v) {
    static const char *const names[] = {"sgd", "adam"};
    return named_native_module(v, "_optimizer", names, 2, OPTIMIZER_BASE);
}
