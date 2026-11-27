#include "mathdoc_report.h"

enum { TEX, ALGORITHM, ASSUMPTIONS, COMPLEXITY, COST, DERIVATIVE_TEX, SENSITIVITY, EXPORT };
static const char *const methods[] = {"getTex",         "getAlgorithm", "getAssumptions",
                                      "getComplexity",  "getCost",      "getDerivativeTex",
                                      "getSensitivity", "export"};
_Static_assert(MATHDOC_BASE + SENSITIVITY == MATHDOC_SENSITIVITY_ID, "sensitivity method ID");

typedef enum { NO_COST, DENSE_SOLVE, DENSE_MATMUL, DENSE_DOT, CSR_MATVEC } CostModel;

typedef struct {
    int family;
    const char *name;
    const char *tex[4];
    const char *algorithm, *assumptions, *complexity, *backend;
    CostModel model;
} MathDescriptor;

#ifdef ARDA_BLAS
#define MATMUL_ALGORITHM                                                                           \
    "Calls the linked CBLAS dgemm provider. Its internal kernel is provider-dependent."
#define MATMUL_TEX "C=AB\\quad\\text{via CBLAS dgemm}"
#define MATMUL_BACKEND "cblas"
#else
#define MATMUL_ALGORITHM                                                                           \
    "Blocked row-major matrix multiply with runtime-selected register tiles. The kernel "          \
    "depends on CPU features and the selected compute mode."
#define MATMUL_TEX "C_{ij}\\gets C_{ij}+A_{it}B_{tj}\\quad\\text{over row-major blocks}"
#define MATMUL_BACKEND "portable-c"
#endif

#ifdef ARDA_LAPACK
#define RSVD_ALGORITHM                                                                             \
    "Gaussian range finding and reorthogonalized power iteration, followed by LAPACK DGESVD on "   \
    "the projection. A separate Gaussian stream estimates the residual."
#define RSVD_TEX                                                                                   \
    "Q=\\operatorname{orth}(A\\Omega),\\quad "                                                     \
    "Q\\gets\\operatorname{orth}(A\\operatorname{orth}(A^TQ)),\\quad "                             \
    "A^TQ=\\widetilde U\\Sigma W^T,\\quad U=QW"
#else
#define RSVD_ALGORITHM "LAPACK backend unavailable in this build."
#define RSVD_TEX NULL
#endif

enum { ODE_DESCRIPTORS = -1 };

static const MathDescriptor registry[] = {
    {
        .family = 0,
        .name = "cos",
        .tex = {"\\cos(x)", NULL, "x\\in\\mathbb{R}", NULL},
        .algorithm =
            "Calls the platform C libm cos. Its internal approximation is provider-dependent.",
        .assumptions = "A real scalar in radians. Special values follow the platform library.",
        .complexity = "One scalar libm call. Provider-internal cost is unavailable.",
    },
    {
        .family = SCI_BASE,
        .name = "cos",
        .tex = {"Y_i=\\cos(X_i)", NULL, "\\operatorname{shape}(Y)=\\operatorname{shape}(X)", NULL},
        .algorithm = "Applies the platform C libm cos elementwise. Its internal approximation is "
                     "provider-dependent.",
        .assumptions =
            "A real Float64 tensor in radians. Special values follow the platform library.",
        .complexity = "One libm call per element and a same-shape output tensor.",
    },
    {
        .family = SCI_BASE,
        .name = "solve",
        .tex =
            {
                "Ax=b",
                "PA=LU,\\quad Ly=Pb,\\quad Ux=y",
                "A\\in\\mathbb{R}^{n\\times n},\\quad b,x\\in\\mathbb{R}^{n},\\quad \\det(A)\\ne0",
                "\\begin{aligned}p_k&=\\underset{i\\ge k}{\\operatorname{arg\\,max}}\\,|a_{ik}|\\\\"
                "\\ell_{ik}&=a_{ik}/a_{kk}\\quad\\text{after swapping rows }k,p_k\\\\"
                "a_{ij}&\\gets a_{ij}-\\ell_{ik}a_{kj}\\\\Ly&=Pb,\\quad Ux=y\\end{aligned}",
            },
        .algorithm = "LU factorization with partial row pivoting, followed by forward and backward "
                     "substitution.",
        .assumptions = "Finite square Float64 matrix and matching vector. A zero pivot raises a "
                       "singularity error.",
        .complexity = "O(n^3) arithmetic and O(n^2) workspace for the copied factors.",
        .backend = "portable-c",
        .model = DENSE_SOLVE,
    },
    {
        .family = SCI_BASE,
        .name = "matmul",
        .tex =
            {
                "C=AB",
                "C_{ij}=\\sum_{t=1}^{k}A_{it}B_{tj}",
                "A\\in\\mathbb{R}^{m\\times k},\\quad B\\in\\mathbb{R}^{k\\times n},\\quad "
                "C\\in\\mathbb{R}^{m\\times n}",
                MATMUL_TEX,
            },
        .algorithm = MATMUL_ALGORITHM,
        .assumptions = "Two rank-two Float64 arrays with matching inner dimensions.",
        .complexity = "O(mkn) arithmetic and O(mn) output. Strided inputs can require packing.",
        .backend = MATMUL_BACKEND,
        .model = DENSE_MATMUL,
    },
    {
        .family = SCI_BASE,
        .name = "dot",
        .tex = {"x^Ty", "\\sum_{i=1}^{n}x_i y_i", "x,y\\in\\mathbb{R}^n", NULL},
        .algorithm = "Accumulates products in four independent long double sums and a scalar tail.",
        .assumptions = "Two Float64 vectors with the same length.",
        .complexity =
            "O(n) arithmetic and constant scalar workspace. Strided inputs can require packing.",
        .backend = "portable-c",
        .model = DENSE_DOT,
    },
    {
        .family = LINALG_BASE,
        .name = "solve",
        .tex = {"AX=B\\quad\\text{or}\\quad\\min_X\\|AX-B\\|_F", NULL, NULL, NULL},
        .algorithm = "Uses triangular solves with the supplied LU or Cholesky factor, or "
                     "Householder QR least squares with a full-column-rank factor.",
        .assumptions = "A compatible reusable factor with finite right-hand sides in a vector or "
                       "in matrix columns. QR requires full column rank.",
        .complexity = "O(n^2 r) arithmetic for an n-by-n LU or Cholesky factor and r right-hand "
                      "sides. QR with an m-by-n factor uses O(mnr) arithmetic.",
    },
    {
        .family = LINALG_BASE,
        .name = "rsvd",
        .tex =
            {
                "A\\approx U_k\\operatorname{diag}(s_k)V_k^T",
                "Y=A\\Omega,\\quad Q=\\operatorname{orth}(Y),\\quad B=Q^TA",
                "1\\le k\\le\\min(m,n)\\quad(k<\\min(m,n)\\text{ for operators})",
                RSVD_TEX,
            },
        .algorithm = RSVD_ALGORITHM,
        .assumptions = "Optional LAPACK build and a finite dense, sparse, or structured real "
                       "matrix. Operator rank must be strictly below full width.",
        .complexity = "Cost depends on operator products, capped sketch width, and the "
                      "power-iteration count. The projected SVD uses skinny buffers.",
    },
    {
        .family = SPARSE_BASE,
        .name = "matvec",
        .tex =
            {
                "y=Ax",
                "y_i=\\sum_{j:(i,j)\\in\\operatorname{supp}(A)}A_{ij}x_j",
                "A\\in\\mathbb{R}^{m\\times n},\\quad x\\in\\mathbb{R}^n,\\quad y\\in\\mathbb{R}^m",
                "y_i=\\sum_{p=\\mathrm{offset}_i}^{\\mathrm{offset}_{i+1}-1}"
                "\\mathrm{values}_p x_{\\mathrm{column}_p}\\quad\\text{in CSR}",
            },
        .algorithm = "Reduces each CSR row in long double. COO input is converted to canonical CSR "
                     "before the product.",
        .assumptions = "Native real sparse matrix with a matching finite Float64 vector.",
        .complexity =
            "O(m+nnz) work for CSR and O(m) output. COO conversion adds time and workspace.",
        .backend = "portable-c",
        .model = CSR_MATVEC,
    },
    {
        .family = PDE_BASE,
        .name = "grid",
        .tex = {"L_h u=-\\Delta_h u", "(L_hu)_i=\\sum_a(2u_i-u_{i-e_a}-u_{i+e_a})/h_a^2",
                "\\text{Dirichlet: }u_g=0;\\quad\\text{Neumann: }u_g=u_i;"
                "\\quad\\text{periodic: wrap}"},
        .algorithm = "Matrix-free centered differences on a row-major Cartesian grid.",
        .assumptions = "One to three positive extents, positive spacing per axis and homogeneous "
                       "Dirichlet or Neumann boundaries, or periodic wrapping per axis.",
        .complexity = "O(dN) work per apply on N cells in d dimensions.",
    },
    {
        .family = PDE_BASE,
        .name = "poisson",
        .tex = {"L_h u=b", "\\langle u,1\\rangle=0\\quad\\text{for a constant nullspace}"},
        .algorithm =
            "Conjugate gradients on the centered negative Laplacian. With periodic and "
            "Neumann faces only, the returned solution is centered and its residual checked.",
        .assumptions = "Homogeneous boundary conditions from the grid. Without a Dirichlet axis "
                       "the right-hand side must have zero mean to within roundoff.",
        .complexity = "O(dN) work per CG iteration and O(N) workspace.",
    },
    {
        .family = PDE_BASE,
        .name = "heat",
        .tex = {"u_t=\\kappa\\Delta u", "u^{n+1}=u^n-\\Delta t\\,\\kappa L_hu^n",
                "2\\kappa\\Delta t\\sum_{a\\,\\mathrm{active}} h_a^{-2}\\le 1",
                "\\sum_i u_i^{n+1}=\\sum_i u_i^n\\quad\\text{with closed or periodic faces}"},
        .algorithm = "Centered differences in space and forward Euler in time, clipping the last "
                     "step. Periodic and zero-flux Neumann faces conserve the sum to roundoff.",
        .assumptions = "Constant nonnegative diffusivity, finite Float64 state, Cartesian grid "
                       "and homogeneous per-axis boundaries. The explicit CFL bound is checked.",
        .complexity = "O(dN) work per step and O(N) numeric workspace.",
    },
    {
        .family = PDE_BASE,
        .name = "wave",
        .tex = {"u_{tt}=c^2\\Delta u",
                "v^{n+1/2}=v^n-\\tfrac12\\Delta t\\,c^2L_hu^n,\\quad"
                "u^{n+1}=u^n+\\Delta t\\,v^{n+1/2}",
                "c\\Delta t\\sqrt{\\sum_{a\\,\\mathrm{active}} h_a^{-2}}\\le 1",
                "v^{n+1}=v^{n+1/2}-\\tfrac12\\Delta t\\,c^2L_hu^{n+1}"},
        .algorithm = "Centered differences with kick-drift-kick velocity Verlet, clipping the "
                     "last step and checking the explicit stability limit.",
        .assumptions = "Constant wave speed, finite displacement and velocity, and homogeneous "
                       "Cartesian grid boundary conditions.",
        .complexity = "O(dN) work per step and O(N) numeric workspace.",
    },
    {
        .family = PDE_BASE,
        .name = "advection_diffusion",
        .tex = {"u_t+\\nabla\\cdot(vu)=\\kappa\\Delta u",
                "F_{i+1/2}=v^+u_i+v^-u_{i+1}-\\kappa(u_{i+1}-u_i)/h",
                "\\Delta t\\sum_a\\left(|v_a|/h_a+2\\kappa/h_a^2\\right)\\le 1",
                "\\sum_i u_i^{n+1}=\\sum_i u_i^n\\quad\\text{with closed or periodic faces}"},
        .algorithm = "Conservative donor-cell upwind advection, centered diffusive fluxes and "
                     "forward Euler. Interior and periodic face contributions telescope.",
        .assumptions = "Constant velocity per axis and nonnegative diffusivity. Closed Neumann "
                       "faces carry zero flux. A sufficient CFL bound is checked.",
        .complexity = "O(dN) work per step and O(N) numeric workspace.",
    },
    {
        .family = PDE_BASE,
        .name = "apply_subdomain",
        .tex = {"(L_hu)|_S=L_{h,S}(u|_S,u_{\\mathrm{ghost}})",
                "(L_hu)_i=\\sum_a(2u_i-u_{i-e_a}-u_{i+e_a})/h_a^2"},
        .algorithm = "Applies the centered negative Laplacian on an axis slab. Supplied adjacent "
                     "faces are indexed in transverse row-major order.",
        .assumptions = "Ghost data matches neighboring slabs at the same state time. Physical "
                       "faces inherit the grid boundary, and periodic joins require wrap faces.",
        .complexity = "O(dN_S) work and O(N_S + F) workspace for slab size N_S and face size F.",
    },
    {
        .family = FEM_BASE,
        .name = "assemble",
        .tex = {"-\\nabla\\cdot(k\\nabla u)+cu=f",
                "A_{ij}=\\sum_T\\int_T (k_T\\nabla\\phi_i\\cdot\\nabla\\phi_j"
                "+c_T\\phi_i\\phi_j)\\,dx",
                "k_T>0,\\quad c_T\\ge0,\\quad u|_{\\Gamma_D}=g,"
                "\\quad\\partial_n u|_{\\Gamma_N}=0"},
        .algorithm = "Assembles P1 segment or triangle stiffness and exact simplex mass terms "
                     "into CSR. Symmetric Dirichlet elimination preserves symmetry.",
        .assumptions = "Conforming nondegenerate mesh, cellwise coefficients and finite nodal "
                       "source. A prescribed node or positive reaction grounds each component.",
        .complexity =
            "O(E+N) assembly storage before CSR canonicalization for E elements and N nodes.",
    },
    {
        .family = FEM_BASE,
        .name = "integral",
        .tex = {"\\int_\\Omega u_h\\,dx", "\\sum_T |T|\\frac{\\sum_{i\\in T} u_i}{\\#T}"},
        .algorithm = "Integrates the P1 interpolant exactly per simplex up to Float64 rounding.",
        .assumptions = "Finite nodal values on a conforming segment or triangle mesh.",
        .complexity = "O(E+N) work including input validation.",
    },
    {
        .family = FEM_BASE,
        .name = "refine",
        .tex = {"u_h|_T=u_{h'}|_T\\quad\\text{under P1 prolongation}",
                "u((x_i+x_j)/2)=(u_i+u_j)/2"},
        .algorithm = "Marks cells by the indicators and refines with conforming midpoint splits "
                     "or newest-vertex bisection. P1 prolongation preserves the integral.",
        .assumptions = "Conforming mesh, finite P1 field and nonnegative element indicators.",
        .complexity = "O(E log E) marking and face ordering with O(N+E) working storage.",
    },
    {
        .family = ODE_DESCRIPTORS,
        .name = "rk45",
        .tex = {"y'=f(t,y)", "\\mathrm{err}=\\operatorname{WRMS}(y_5-y_4)"},
        .algorithm = "Dormand-Prince 5(4) with adaptive rejection, Shampine dense output and "
                     "optional bracketed event localization.",
        .assumptions = "Finite scalar or Float64 state and positive error tolerances. Events "
                       "use accepted-step endpoint brackets.",
        .complexity = "Seven right-hand-side evaluations per attempted step. Retained dense "
                      "output stores data per accepted step.",
    },
};

static const MathDescriptor *descriptor_for(Val target) {
    int family;
    const char *name;
    if (isobj(target, O_FUNC)) {
        const Proto *proto = target.o->as.func.p;
        if (strcmp(proto->path, "<stdlib:ode>"))
            return NULL;
        family = ODE_DESCRIPTORS;
        name = proto->name;
    } else if (isobj(target, O_NATIVE)) {
        int id = target.o->as.native.id;
        family = id / 100 * 100;
        switch (family) {
        case 0:
            name = native_builtin_name(id);
            break;
        case SCI_BASE:
            name = scientific_name(id);
            break;
        case SPARSE_BASE:
            name = sparse_name(id);
            break;
        case LINALG_BASE:
            name = linalg_name(id);
            break;
        case PDE_BASE:
            name = pde_name(id);
            break;
        case FEM_BASE:
            name = fem_name(id);
            break;
        default:
            return NULL;
        }
    } else {
        return NULL;
    }
    if (name)
        for (size_t i = 0; i < sizeof registry / sizeof *registry; i++)
            if (family == registry[i].family && !strcmp(name, registry[i].name))
                return &registry[i];
    return NULL;
}

static int documented_target(Val target) {
    return isobj(target, O_NATIVE) || isobj(target, O_FUNC) || isobj(target, O_BOUND);
}

static Val text_or_nil(VM *v, const char *text) {
    return text ? objval(string(v, text)) : nil();
}

static int64_t dimension(VM *v, Val value) {
    if (value.t != V_INT || value.i < 0 || value.i > ARDA_MAX_ELEMENTS)
        vmerror(v, "mathdoc cost dimensions must be nonnegative integers within the tensor limit");
    return value.i;
}

static void matrix_limit(VM *v, int64_t rows, int64_t cols) {
    if ((rows ? rows : 1) * (cols ? cols : 1) > ARDA_MAX_ELEMENTS)
        vmerror(v, "mathdoc cost shape exceeds the tensor element limit");
}

static Val cost(VM *v, const MathDescriptor *descriptor, Val spec) {
    CostModel model = descriptor->model;
    if (model == NO_COST)
        return nil();
    if (!isobj(spec, O_MAP))
        vmerror(v, "mathdoc getCost expects a shape specification map");
    Table *table = &spec.o->as.map.tab;
    Entry *dtype = tablefind(table, "dtype");
    if (dtype && (!isobj(dtype->val, O_STRING) || strcmp(asstr(v, dtype->val), "float64")))
        vmerror(v, "mathdoc cost models require dtype float64");
    Entry *shape = tablefind(table, "shape");
    int rank = model == DENSE_DOT ? 1 : model == DENSE_MATMUL ? 3 : 2;
    if (!shape || !isobj(shape->val, O_LIST) || shape->val.o->as.list.n != rank)
        vmerror(v, "mathdoc cost model expects a shape with %d dimensions", rank);
    const Val *axes = shape->val.o->as.list.a;
    int64_t m = dimension(v, axes[0]);
    int64_t k = rank > 1 ? dimension(v, axes[1]) : 0;
    int64_t n = rank > 2 ? dimension(v, axes[2]) : 0;
    int64_t flops, output, workspace = 0;
    const char *label;
    switch (model) {
    case DENSE_SOLVE:
        if (m != k)
            vmerror(v, "mathdoc solve cost requires a square matrix shape");
        matrix_limit(v, m, m);
        flops = m * (m - 1) * (2 * m - 1) / 3 + m * (m - 1) / 2 + 2 * m * (m - 1) + m;
        output = m * sizeof(double);
        workspace = m * m * sizeof(double) + m * sizeof(int);
        label = "Successful LU solve with packed inputs and one right-hand side";
        break;
    case DENSE_MATMUL:
        matrix_limit(v, m, k);
        matrix_limit(v, k, n);
        matrix_limit(v, m, n);
        flops = 2 * m * k * n;
        output = m * n * sizeof(double);
        label = "Nominal multiply-add work with packed inputs";
        break;
    case DENSE_DOT:
        flops = 2 * m;
        output = sizeof(double);
        label = "Nominal multiply-add work with packed vectors";
        break;
    default: {
        Entry *format = tablefind(table, "format");
        if (format) {
            const char *name = asstr(v, format->val);
            if (!strcmp(name, "coo"))
                return nil();
            if (strcmp(name, "csr"))
                vmerror(v, "mathdoc sparse cost format must be csr or coo");
        }
        Entry *nnz = tablefind(table, "nnz");
        if (!nnz || nnz->val.t != V_INT || nnz->val.i < 0 || nnz->val.i > m * k)
            vmerror(v, "mathdoc CSR cost requires nnz within the supplied shape");
        flops = 2 * nnz->val.i;
        output = m * sizeof(double);
        label = "Nominal multiply-add work for canonical CSR";
        break;
    }
    }
    Obj *result = newobj(v, O_MAP);
    Table *out = &result->as.map.tab;
    tableset(v, result, out, "estimated", boolean(1));
    tableset(v, result, out, "flops", integer(flops));
    tableset(v, result, out, "output_bytes", integer(output));
    tableset(v, result, out, "workspace_bytes", integer(workspace));
    tableset(v, result, out, "backend", objval(string(v, descriptor->backend)));
    tableset(v, result, out, "model", objval(string(v, label)));
    return objval(result);
}

Val mathdoc_call(VM *v, int id, int argc, Val *args) {
    int method = id - MATHDOC_BASE;
    if (method < 0 || (size_t)method >= sizeof methods / sizeof *methods)
        vmerror(v, "unknown mathdoc query");
    if (method == EXPORT)
        return mathdoc_export(v, argc, args);
    if (argc != (method >= COST ? 2 : 1) && !(method == TEX && argc == 2))
        vmerror(v, "invalid mathdoc.%s arity", methods[method]);
    if (!documented_target(args[0]))
        vmerror(v, "mathdoc expects a function or bound method");
    if (method == DERIVATIVE_TEX) {
        const char *parameter = asstr(v, args[1]);
        if (!isobj(args[0], O_FUNC))
            return nil();
        char tex[MATHIR_TEX_LIMIT];
        return mathir_derivative_tex(v, args[0].o, parameter, tex, sizeof tex) ? text_or_nil(v, tex)
                                                                               : nil();
    }
    int level = 0;
    if (method == TEX && argc == 2) {
        if (args[1].t != V_INT || args[1].i < 0 || args[1].i > 3)
            vmerror(v, "static getTex level must be an integer from 0 to 3");
        level = (int)args[1].i;
    }
    const MathDescriptor *descriptor = descriptor_for(args[0]);
    if (!descriptor && isobj(args[0], O_FUNC)) {
        Obj *function = args[0].o;
        const Proto *p = function->as.func.p;
        if (mathir_has_loop(p) && !vm_uses_builtin(v, function->as.func.env, "range"))
            return nil();
        if (method == TEX) {
            char tex[MATHIR_TEX_LIMIT];
            return mathir_tex(p, level, tex, sizeof tex) ? text_or_nil(v, tex) : nil();
        }
        if (method == ALGORITHM)
            return text_or_nil(v, mathir_algorithm(p));
        if (method == ASSUMPTIONS)
            return text_or_nil(v, mathir_assumptions(p));
        return nil();
    }
    if (!descriptor)
        return nil();
    switch (method) {
    case TEX:
        return text_or_nil(v, descriptor->tex[level]);
    case ALGORITHM:
        return text_or_nil(v, descriptor->algorithm);
    case ASSUMPTIONS:
        return text_or_nil(v, descriptor->assumptions);
    case COMPLEXITY:
        return text_or_nil(v, descriptor->complexity);
    case COST:
        return cost(v, descriptor, args[1]);
    default:
        return nil();
    }
}

Val mathdoc_describe(VM *v, Val target, Val level) {
    Obj *record = newobj(v, O_MAP);
    const char *const keys[] = {"formula_tex", "algorithm", "assumptions", "complexity"};
    Val args[] = {target, level};
    for (int i = TEX; i <= COMPLEXITY; i++) {
        Val value = mathdoc_call(v, MATHDOC_BASE + i, i == TEX ? 2 : 1, args);
        tableset(v, record, &record->as.map.tab, keys[i], value);
    }
    return objval(record);
}

Val mathdoc_attribute(VM *v, Val target, const char *name) {
    if (documented_target(target))
        for (size_t i = 0; i < EXPORT; i++) {
            if (strcmp(methods[i], name))
                continue;
            Obj *method = newobj(v, O_NATIVE);
            method->as.native.id = MATHDOC_BASE + (int)i;
            Obj *bound = newobj(v, O_BOUND);
            bound->as.bound.fn = method;
            bound->as.bound.self = target;
            barrier(v, bound, objval(method));
            barrier(v, bound, target);
            return objval(bound);
        }
    return nil();
}

Obj *mathdoc_module(VM *v) {
    return named_native_module(v, "mathdoc", methods, sizeof methods / sizeof *methods,
                               MATHDOC_BASE);
}
