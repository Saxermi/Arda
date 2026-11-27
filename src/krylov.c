#include "solver.h"
#include <float.h>

size_t krylov_elements(SolverMethod method, size_t m, size_t n, int restart, bool preconditioned) {
    size_t k = (size_t)restart;
    switch (method) {
    case SOLVER_CG:
        return (preconditioned ? 4 : 3) * n;
    case SOLVER_MINRES:
        return 7 * n;
    case SOLVER_GMRES:
        return (k + 3) * n + (k + 1) * k + 3 * k + 1;
    case SOLVER_BICGSTAB:
        return 8 * n;
    case SOLVER_LSQR:
        return 2 * m + 3 * n;
    case SOLVER_LSMR:
        return 2 * m + 4 * n;
    }
    return 0;
}

static void apply(const SolverProblem *p, const double *x, double *out) {
    p->op.apply(p->op.context, x, 1, out, 1);
}

static void transpose(const SolverProblem *p, const double *x, double *out) {
    p->op.apply_transpose(p->op.context, x, 1, out, 1);
}

static void scale_copy(size_t n, const double *x, long double divisor, double *out) {
    for (size_t i = 0; i < n; i++)
        out[i] = (double)(x[i] / divisor);
}

static void plus(size_t n, double *x, double factor, const double *y) {
    for (size_t i = 0; i < n; i++)
        x[i] += factor * y[i];
}

static bool cg_iteration(SolverProblem *p, double *r, double *z, double *d, double *q,
                         long double *old, bool restart) {
    size_t n = p->op.cols;
    if (z != r)
        preconditioner_apply(&p->preconditioner, r, z);
    long double rho = linear_dot(n, r, z);
    if (!(rho > 0) || !isfinite(rho))
        return false;
    if (restart)
        memcpy(d, z, n * sizeof(double));
    else {
        double beta = (double)(rho / *old);
        for (size_t i = 0; i < n; i++)
            d[i] = z[i] + beta * d[i];
    }
    apply(p, d, q);
    long double curvature = linear_dot(n, d, q);
    if (!(curvature > 0) || !isfinite(curvature)) {
        if (isfinite(curvature))
            p->stop = STOP_CURVATURE;
        return false;
    }
    double alpha = (double)(rho / curvature);
    if (!alpha || !solver_update(n, p->x, alpha, d))
        return false;
    plus(n, r, -alpha, q);
    *old = rho;
    return true;
}

static void cg(SolverProblem *p, double *r) {
    size_t n = p->op.cols;
    double *d = r + n, *q = p->residual;
    double *z = p->options.preconditioner == PC_NONE ? r : d + n;
    memcpy(r, p->b, n * sizeof(double));
    long double old = 0;
    bool restart = true;
    while (p->iterations < p->options.max_iterations) {
        if (!cg_iteration(p, r, z, d, q, &old, restart)) {
            if (p->stop != STOP_CURVATURE)
                p->stop = STOP_BREAKDOWN;
            return;
        }
        p->iterations++;
        long double estimate = linear_norm(n, r, 1) / p->rhs_norm;
        solver_trace(p, estimate);
        if (!isfinite(estimate)) {
            p->stop = STOP_BREAKDOWN;
            return;
        }
        restart = estimate <= p->options.tolerance;
        if (restart) {
            if (solver_residual(p) <= p->options.tolerance)
                return;
            memcpy(r, p->residual, n * sizeof(double));
        }
    }
}

static void minres(SolverProblem *p, double *r1) {
    size_t n = p->op.cols;
    double *r2 = r1 + n, *y = r2 + n, *v = y + n;
    double *w = v + n, *w1 = w + n, *w2 = w1 + n;
    memcpy(r2, p->b, n * sizeof(double));
    long double old_beta = 0, beta = p->rhs_norm;
    long double cs = -1, sn = 0, dbar = 0, eps = 0, phi_bar = beta;
    while (p->iterations < p->options.max_iterations && beta > 0) {
        scale_copy(n, r2, beta, v);
        apply(p, v, y);
        if (p->iterations)
            plus(n, y, (double)(-beta / old_beta), r1);
        long double alpha = linear_dot(n, v, y);
        plus(n, y, (double)(-alpha / beta), r2);
        double *old = r1;
        r1 = r2;
        r2 = y;
        y = old;
        old_beta = beta;
        beta = linear_norm(n, r2, 1);
        long double old_eps = eps, delta = cs * dbar + sn * alpha;
        long double gbar = sn * dbar - cs * alpha;
        eps = sn * beta;
        dbar = -cs * beta;
        long double gamma = hypotl(gbar, beta);
        if (!(gamma > 0) || !isfinite(gamma))
            break;
        cs = gbar / gamma;
        sn = beta / gamma;
        long double phi = cs * phi_bar;
        phi_bar *= sn;
        for (size_t i = 0; i < n; i++)
            w[i] = (double)((v[i] - old_eps * w1[i] - delta * w2[i]) / gamma);
        if (!solver_update(n, p->x, (double)phi, w))
            break;
        old = w1;
        w1 = w2;
        w2 = w;
        w = old;
        p->iterations++;
        long double estimate = fabsl(phi_bar) / p->rhs_norm;
        solver_trace(p, estimate);
        if (estimate <= p->options.tolerance && solver_residual(p) <= p->options.tolerance)
            return;
    }
    if (p->iterations < p->options.max_iterations)
        p->stop = STOP_BREAKDOWN;
}

/* Modified Gram-Schmidt twice for a column of the Arnoldi projection. */
long double krylov_orthogonalize(size_t n, const double *basis, int columns, double *vector,
                                 double *coefficients) {
    for (int pass = 0; pass < 2; pass++)
        for (int i = 0; i < columns; i++) {
            const double *v = basis + (size_t)i * n;
            double dot = (double)linear_dot(n, v, vector);
            if (coefficients)
                coefficients[i] += dot;
            plus(n, vector, -dot, v);
        }
    return linear_norm(n, vector, 1);
}

static double arnoldi(SolverProblem *p, double *basis, double *q, double *tmp, double *column,
                      int j) {
    size_t n = p->op.cols;
    apply(p, basis + (size_t)j * n, tmp);
    preconditioner_apply(&p->preconditioner, tmp, q);
    return (double)krylov_orthogonalize(n, basis, j + 1, q, column);
}

static bool gmres_update(SolverProblem *p, double *basis, double *h, double *g, double *delta,
                         int taken, int columns) {
    size_t n = p->op.cols, pitch = (size_t)columns + 1;
    for (int i = taken; i-- > 0;) {
        long double value = g[i];
        for (int j = i + 1; j < taken; j++)
            value -= (long double)h[(size_t)j * pitch + i] * g[j];
        g[i] = (double)(value / h[(size_t)i * pitch + i]);
        if (!isfinite(g[i]))
            return false;
    }
    memset(delta, 0, n * sizeof(double));
    for (int i = 0; i < taken; i++)
        plus(n, delta, g[i], basis + (size_t)i * n);
    return solver_update(n, p->x, 1, delta);
}

static void gmres(SolverProblem *p, double *basis) {
    size_t n = p->op.cols;
    int restart = p->options.restart;
    size_t pitch = (size_t)restart + 1;
    double *q = basis + pitch * n, *tmp = q + n, *h = tmp + n;
    double *cs = h + pitch * restart, *sn = cs + restart, *g = sn + restart;
    preconditioner_apply(&p->preconditioner, p->b, tmp);
    long double scale = linear_norm(n, tmp, 1);
    if (!(scale > 0) || !isfinite(scale)) {
        p->stop = STOP_BREAKDOWN;
        return;
    }
    long double actual = solver_residual(p);
    while (p->iterations < p->options.max_iterations && isfinite(actual)) {
        if (actual <= p->options.tolerance)
            return;
        preconditioner_apply(&p->preconditioner, p->residual, q);
        long double beta = linear_norm(n, q, 1);
        if (!(beta > 0) || !isfinite(beta) || beta > DBL_MAX)
            break;
        scale_copy(n, q, beta, basis);
        memset(g, 0, pitch * sizeof(double));
        memset(h, 0, pitch * restart * sizeof(double));
        g[0] = (double)beta;
        int taken = 0;
        bool invariant = false, breakdown = false;
        for (int j = 0; j < restart && p->iterations < p->options.max_iterations; j++) {
            double *c = h + (size_t)j * pitch;
            double length = arnoldi(p, basis, q, tmp, c, j);
            if (!isfinite(length)) {
                breakdown = true;
                break;
            }
            for (int i = 0; i < j; i++) {
                double next = cs[i] * c[i] + sn[i] * c[i + 1];
                c[i + 1] = -sn[i] * c[i] + cs[i] * c[i + 1];
                c[i] = next;
            }
            double rho = hypot(c[j], length);
            if (!(rho > 0) || !isfinite(rho)) {
                breakdown = true;
                break;
            }
            cs[j] = c[j] / rho;
            sn[j] = length / rho;
            c[j] = rho;
            g[j + 1] = -sn[j] * g[j];
            g[j] *= cs[j];
            taken++;
            p->iterations++;
            long double estimate = fabs(g[j + 1]) / scale;
            solver_trace(p, estimate);
            invariant = length <= DBL_EPSILON * rho;
            if (invariant || estimate <= p->options.tolerance)
                break;
            scale_copy(n, q, length, basis + ((size_t)j + 1) * n);
        }
        if (!taken || !gmres_update(p, basis, h, g, tmp, taken, restart))
            break;
        if (breakdown)
            break;
        if (p->iterations == p->options.max_iterations)
            return;
        // In finite precision the updated residual can point outside the
        // nearly invariant basis. Restart only when the update made progress.
        long double next = solver_residual(p);
        if (invariant && isfinite(next) && next > p->options.tolerance &&
            next >= actual * (1 - 8 * DBL_EPSILON)) {
            p->stop = STOP_STAGNATION;
            return;
        }
        actual = next;
    }
    p->stop = STOP_BREAKDOWN;
}

static void bicgstab(SolverProblem *p, double *r) {
    size_t n = p->op.cols;
    double *r0 = r + n, *d = r0 + n, *v = d + n, *s = v + n;
    double *t = s + n, *phat = t + n, *shat = phat + n;
    memcpy(r, p->b, n * sizeof(double));
    memcpy(r0, r, n * sizeof(double));
    long double prev = 1, alpha = 1, omega = 1;
    while (p->iterations < p->options.max_iterations) {
        long double rho = linear_dot(n, r0, r);
        if (!rho || !isfinite(rho))
            break;
        double beta = (double)((rho / prev) * (alpha / omega));
        for (size_t i = 0; i < n; i++)
            d[i] = r[i] + beta * (d[i] - (double)omega * v[i]);
        preconditioner_apply(&p->preconditioner, d, phat);
        apply(p, phat, v);
        long double denom = linear_dot(n, r0, v);
        if (!denom || !isfinite(denom))
            break;
        alpha = rho / denom;
        for (size_t i = 0; i < n; i++)
            s[i] = r[i] - (double)alpha * v[i];
        long double estimate = linear_norm(n, s, 1) / p->rhs_norm;
        if (estimate <= p->options.tolerance) {
            if (!solver_update(n, p->x, (double)alpha, phat))
                break;
            p->iterations++;
            solver_trace(p, estimate);
            if (solver_residual(p) <= p->options.tolerance)
                return;
            // Restart from the recomputed residual when the recursive estimate drifts.
            memcpy(r, p->residual, n * sizeof(double));
            memcpy(r0, r, n * sizeof(double));
            memset(d, 0, 2 * n * sizeof(double));
            prev = alpha = omega = 1;
            continue;
        }
        preconditioner_apply(&p->preconditioner, s, shat);
        apply(p, shat, t);
        long double tt = linear_dot(n, t, t);
        if (!(tt > 0) || !isfinite(tt))
            break;
        omega = linear_dot(n, t, s) / tt;
        if (!omega || !isfinite(omega))
            break;
        for (size_t i = 0; i < n; i++) {
            phat[i] = (double)alpha * phat[i] + (double)omega * shat[i];
            r[i] = s[i] - (double)omega * t[i];
        }
        if (!solver_update(n, p->x, 1, phat))
            break;
        p->iterations++;
        estimate = linear_norm(n, r, 1) / p->rhs_norm;
        solver_trace(p, estimate);
        if (estimate <= p->options.tolerance && solver_residual(p) <= p->options.tolerance)
            return;
        prev = rho;
    }
    if (p->iterations < p->options.max_iterations)
        p->stop = STOP_BREAKDOWN;
}

typedef struct {
    long double alpha, beta;
    double *u, *v, *work_m, *work_n;
} Bidiagonal;

static void bidiagonal_step(const SolverProblem *p, Bidiagonal *s) {
    size_t m = p->op.rows, n = p->op.cols;
    apply(p, s->v, s->work_m);
    plus(m, s->work_m, (double)-s->alpha, s->u);
    s->beta = linear_norm(m, s->work_m, 1);
    if (s->beta)
        scale_copy(m, s->work_m, s->beta, s->u);
    else
        memset(s->u, 0, m * sizeof(double));
    transpose(p, s->u, s->work_n);
    plus(n, s->work_n, (double)-s->beta, s->v);
    s->alpha = linear_norm(n, s->work_n, 1);
    if (s->alpha)
        scale_copy(n, s->work_n, s->alpha, s->v);
    else
        memset(s->v, 0, n * sizeof(double));
}

typedef struct {
    long double rho, c, s;
} Rotation;

static Rotation rotate(long double a, long double b) {
    long double rho = hypotl(a, b);
    return rho ? (Rotation){rho, a / rho, b / rho} : (Rotation){0, 1, 0};
}

static bool least_stop(SolverProblem *p) {
    long double residual = solver_residual(p);
    long double normal = solver_normal_residual(p);
    solver_trace(p, residual);
    return residual <= p->options.tolerance || normal <= p->options.tolerance;
}

static void lsqr(SolverProblem *p, Bidiagonal *b, double *w) {
    size_t n = p->op.cols;
    memcpy(w, b->v, n * sizeof(double));
    long double rho_bar = b->alpha, phi_bar = b->beta;
    while (p->iterations < p->options.max_iterations) {
        bidiagonal_step(p, b);
        Rotation q = rotate(rho_bar, b->beta);
        if (!(q.rho > 0) || !isfinite(q.rho) || !isfinite(b->alpha))
            break;
        long double theta = q.s * b->alpha, phi = q.c * phi_bar;
        rho_bar = -q.c * b->alpha;
        phi_bar *= q.s;
        if (!solver_update(n, p->x, (double)(phi / q.rho), w))
            break;
        for (size_t i = 0; i < n; i++)
            w[i] = b->v[i] - (double)(theta / q.rho) * w[i];
        p->iterations++;
        if (least_stop(p))
            return;
    }
    if (p->iterations < p->options.max_iterations)
        p->stop = STOP_BREAKDOWN;
}

static void lsmr(SolverProblem *p, Bidiagonal *b, double *h) {
    size_t n = p->op.cols;
    double *hbar = h + n;
    memcpy(h, b->v, n * sizeof(double));
    long double zeta_bar = b->alpha * b->beta, alpha_bar = b->alpha;
    Rotation q = {1, 1, 0}, qbar = {1, 1, 0};
    while (p->iterations < p->options.max_iterations) {
        bidiagonal_step(p, b);
        long double old_rho = q.rho, old_bar = qbar.rho;
        q = rotate(alpha_bar, b->beta);
        long double theta = q.s * b->alpha, theta_bar = qbar.s * q.rho;
        alpha_bar = q.c * b->alpha;
        qbar = rotate(qbar.c * q.rho, theta);
        if (!(q.rho > 0 && qbar.rho > 0) || !isfinite(q.rho) || !isfinite(qbar.rho))
            break;
        long double zeta = qbar.c * zeta_bar;
        zeta_bar *= -qbar.s;
        double hscale = (double)((theta_bar / old_bar) * (q.rho / old_rho));
        for (size_t i = 0; i < n; i++)
            hbar[i] = h[i] - hscale * hbar[i];
        if (!solver_update(n, p->x, (double)((zeta / qbar.rho) / q.rho), hbar))
            break;
        for (size_t i = 0; i < n; i++)
            h[i] = b->v[i] - (double)(theta / q.rho) * h[i];
        p->iterations++;
        if (least_stop(p))
            return;
    }
    if (p->iterations < p->options.max_iterations)
        p->stop = STOP_BREAKDOWN;
}

void krylov_run(SolverProblem *p, SolverMethod method, double *w) {
    switch (method) {
    case SOLVER_CG:
        cg(p, w);
        return;
    case SOLVER_MINRES:
        minres(p, w);
        return;
    case SOLVER_GMRES:
        gmres(p, w);
        return;
    case SOLVER_BICGSTAB:
        bicgstab(p, w);
        return;
    case SOLVER_LSQR:
    case SOLVER_LSMR: {
        size_t m = p->op.rows, n = p->op.cols;
        Bidiagonal b = {.u = w, .v = w + m, .work_m = w + m + n, .work_n = w + 2 * m + n};
        b.beta = p->rhs_norm;
        scale_copy(m, p->b, b.beta, b.u);
        transpose(p, b.u, b.v);
        b.alpha = linear_norm(n, b.v, 1);
        if (!isfinite(b.alpha)) {
            p->stop = STOP_BREAKDOWN;
            return;
        }
        if (!b.alpha)
            return;
        scale_copy(n, b.v, b.alpha, b.v);
        if (method == SOLVER_LSQR)
            lsqr(p, &b, b.work_n + n);
        else
            lsmr(p, &b, b.work_n + n);
        return;
    }
    }
}
