#include "fem.h"
#include <float.h>

double fem_checked(VM *v, double value) {
    if (!isfinite(value))
        vmerror_kind(v, "domain", "nonfinite FEM value");
    return value;
}

static int positive_count(VM *v, Val x) {
    if (x.t != V_INT || x.i < 1 || x.i > ARDA_MAX_ELEMENTS)
        vmerror_kind(v, "shape", "invalid FEM resource limit");
    return (int)x.i;
}

FemOptions fem_options(VM *v, Val value, bool refinement) {
    FemOptions out = {.diffusion = number(1),
                      .reaction = number(0),
                      .workspace = SIZE_MAX,
                      .max_nodes = ARDA_MAX_ELEMENTS,
                      .max_elements = ARDA_MAX_ELEMENTS,
                      .fraction = 0.5};
    if (value.t == V_NIL)
        return out;
    if (!isobj(value, O_MAP))
        vmerror_kind(v, "type", "FEM options must be a map");
    Table *table = &value.o->as.map.tab;
    for (int i = 0; i < table->cap; i++) {
        Entry e = table->a[i];
        if (!e.key)
            continue;
        if (!strcmp(e.key, "workspace_bytes")) {
            if (e.val.t != V_INT || e.val.i < 0 || (uint64_t)e.val.i > SIZE_MAX)
                vmerror_kind(v, "shape", "invalid FEM workspace_bytes");
            out.workspace = (size_t)e.val.i;
        } else if (refinement && !strcmp(e.key, "max_nodes")) {
            out.max_nodes = positive_count(v, e.val);
        } else if (refinement && !strcmp(e.key, "max_elements")) {
            out.max_elements = positive_count(v, e.val);
        } else if (refinement && !strcmp(e.key, "method")) {
            const char *method = asstr(v, e.val);
            if (strcmp(method, "midpoint") && strcmp(method, "nvb"))
                vmerror_kind(v, "domain", "unknown FEM refinement method");
            out.nvb = !strcmp(method, "nvb");
        } else if (refinement && !strcmp(e.key, "fraction")) {
            out.fraction = sparse_finite(v, e.val);
            if (out.fraction <= 0 || out.fraction > 1)
                vmerror_kind(v, "domain", "FEM marking fraction must be in (0, 1]");
        } else if (!refinement && !strcmp(e.key, "diffusivity")) {
            out.diffusion = e.val;
        } else if (!refinement && !strcmp(e.key, "reaction")) {
            out.reaction = e.val;
        } else {
            vmerror_kind(v, "domain", "unknown FEM option '%s'", e.key);
        }
    }
    return out;
}

Obj *fem_work(VM *v, size_t bytes, size_t output, size_t limit) {
    SolverOptions options = {.workspace_limit = limit};
    return solver_workspace(v, bytes, 0, output, &options);
}

Obj *fem_mesh_wrap(VM *v, Obj *nodes, Obj *elements, int count) {
    Obj *out = newobj(v, O_FEM_MESH);
    out->as.fem.nodes = nodes;
    out->as.fem.elements = elements;
    out->as.fem.dimension = nodes->as.array.rank;
    out->as.fem.count = count;
    barrier(v, out, objval(nodes));
    barrier(v, out, objval(elements));
    return out;
}

static FemMesh mesh_view(VM *v, Val x) {
    if (!isobj(x, O_FEM_MESH))
        vmerror_kind(v, "type", "expected a FEM mesh");
    Obj *mesh = x.o, *nodes = mesh->as.fem.nodes;
    Obj *hierarchy = mesh->as.fem.hierarchy;
    const FemBranch *tree = hierarchy ? (const FemBranch *)hierarchy->as.buffer.data : NULL;
    return (FemMesh){.x = nodes->as.array.data,
                     .cells = (const uint32_t *)mesh->as.fem.elements->as.buffer.data,
                     .nodes = nodes->as.array.rows,
                     .count = mesh->as.fem.count,
                     .dimension = mesh->as.fem.dimension,
                     .arity = mesh->as.fem.dimension + 1,
                     .tree = tree,
                     .leaves = tree ? (const uint32_t *)(tree + mesh->as.fem.tree_count) : NULL,
                     .tree_count = hierarchy ? mesh->as.fem.tree_count : mesh->as.fem.count};
}

FemElement fem_geometry(VM *v, FemMesh mesh, const uint32_t *cell) {
    FemElement out = {0};
    const double *a = mesh.x + (size_t)mesh.dimension * cell[0];
    const double *b = mesh.x + (size_t)mesh.dimension * cell[1];
    double dx = b[0] - a[0];
    if (mesh.dimension == 1) {
        out.measure = out.diameter = fabs(dx);
        out.gradient[0][0] = -1 / dx;
        out.gradient[1][0] = 1 / dx;
    } else {
        const double *c = mesh.x + 2 * (size_t)cell[2];
        double dy = b[1] - a[1], ex = c[0] - a[0], ey = c[1] - a[1];
        out.diameter = fmax(hypot(dx, dy), fmax(hypot(ex, ey), hypot(c[0] - b[0], c[1] - b[1])));
        double det =
            (dx / out.diameter) * (ey / out.diameter) - (dy / out.diameter) * (ex / out.diameter);
        if (fabs(det) <= 16 * DBL_EPSILON)
            vmerror_kind(v, "domain", "degenerate FEM triangle");
        double scale = det * out.diameter;
        out.measure = (fabs(scale) * 0.5) * out.diameter;
        out.gradient[1][0] = (ey / out.diameter) / scale;
        out.gradient[1][1] = -(ex / out.diameter) / scale;
        out.gradient[2][0] = -(dy / out.diameter) / scale;
        out.gradient[2][1] = (dx / out.diameter) / scale;
        for (int d = 0; d < 2; d++)
            out.gradient[0][d] = -out.gradient[1][d] - out.gradient[2][d];
    }
    if (out.measure <= 0 || !isfinite(out.measure))
        vmerror_kind(v, "domain", "invalid FEM element measure");
    for (int i = 0; i < mesh.arity; i++)
        for (int d = 0; d < mesh.dimension; d++)
            fem_checked(v, out.gradient[i][d]);
    return out;
}

static uint32_t node_index(VM *v, Val value, int n) {
    if (value.t != V_INT || value.i < 0 || value.i >= n)
        vmerror_kind(v, "shape", "invalid FEM node index");
    return (uint32_t)value.i;
}

static Val mesh_new(VM *v, Val coordinates, Val elements) {
    if (!isobj(coordinates, O_ARRAY) || !isobj(elements, O_LIST))
        vmerror_kind(v, "type", "FEM mesh needs NumC coordinates and a list of cells");
    Obj *input = coordinates.o;
    int dim = input->as.array.rank, n = input->as.array.rows, count = elements.o->as.list.n;
    if (dim < 1 || dim > 2 || n < dim + 1 || (dim == 2 && input->as.array.cols != 2) || count < 1 ||
        (size_t)count > ARDA_MAX_ELEMENTS)
        vmerror_kind(v, "shape", "FEM needs nonempty segments or triangles");
    Obj *nodes = tensor_copy(v, input);
    Obj *cells = buffer_new(v, (size_t)count * (dim + 1) * sizeof(uint32_t));
    Obj *used = buffer_new(v, (size_t)n);
    memset(used->as.buffer.data, 0, (size_t)n);
    uint32_t *indices = (uint32_t *)cells->as.buffer.data;
    Obj *out = fem_mesh_wrap(v, nodes, cells, count);
    FemMesh mesh = mesh_view(v, objval(out));
    for (size_t i = 0; i < nodes->as.array.n; i++)
        fem_checked(v, mesh.x[i]);
    for (int e = 0; e < count; e++) {
        Val cell = elements.o->as.list.a[e];
        if (!isobj(cell, O_LIST) || cell.o->as.list.n != mesh.arity)
            vmerror_kind(v, "shape", "wrong FEM cell arity");
        uint32_t *ix = indices + (size_t)e * mesh.arity;
        for (int j = 0; j < mesh.arity; j++) {
            ix[j] = node_index(v, cell.o->as.list.a[j], n);
            used->as.buffer.data[ix[j]] = 1;
        }
        fem_geometry(v, mesh, ix);
    }
    for (int i = 0; i < n; i++)
        if (!used->as.buffer.data[i])
            vmerror_kind(v, "shape", "isolated FEM node");
    solver_workspace_done(v, used);
    fem_check_faces(v, mesh);
    return objval(out);
}

void fem_read_field(VM *v, Val value, size_t n, double *out, double lower, bool strict) {
    if (value.t == V_INT || value.t == V_NUM) {
        double x = sparse_finite(v, value);
        for (size_t i = 0; i < n; i++)
            out[i] = x;
    } else {
        solver_read_vector(v, value, n, out);
    }
    for (size_t i = 0; i < n; i++)
        if (out[i] < lower || (strict && out[i] == lower))
            vmerror_kind(v, "domain", "FEM coefficient outside its domain");
}

FemData fem_read_data(VM *v, FemMesh m, Val source, Val boundary, FemOptions opts, double *work) {
    FemData d = {work, work + m.nodes, work + 2 * m.nodes, work + 2 * m.nodes + m.count};
    fem_read_field(v, source, m.nodes, d.source, -INFINITY, false);
    fem_read_field(v, opts.diffusion, m.count, d.diffusion, 0, true);
    fem_read_field(v, opts.reaction, m.count, d.reaction, 0, false);
    for (int i = 0; i < m.nodes; i++)
        d.boundary[i] = NAN;
    if (!isobj(boundary, O_LIST))
        vmerror_kind(v, "type", "FEM Dirichlet constraints must be a list");
    for (int i = 0; i < boundary.o->as.list.n; i++) {
        Val pair = boundary.o->as.list.a[i];
        if (!isobj(pair, O_LIST) || pair.o->as.list.n != 2)
            vmerror_kind(v, "shape", "expected a FEM [node, value] constraint");
        uint32_t node = node_index(v, pair.o->as.list.a[0], m.nodes);
        if (!isnan(d.boundary[node]))
            vmerror_kind(v, "domain", "duplicate FEM Dirichlet node");
        d.boundary[node] = sparse_finite(v, pair.o->as.list.a[1]);
    }
    return d;
}

uint32_t fem_component(uint32_t *parent, uint32_t node) {
    while (node != parent[node]) {
        parent[node] = parent[parent[node]];
        node = parent[node];
    }
    return node;
}

static void check_coercive(VM *v, FemMesh m, FemData data, uint32_t *parent, bool *anchored) {
    for (int i = 0; i < m.nodes; i++) {
        parent[i] = (uint32_t)i;
        anchored[i] = false;
    }
    for (int e = 0; e < m.count; e++) {
        const uint32_t *cell = m.cells + (size_t)e * m.arity;
        uint32_t root = fem_component(parent, cell[0]);
        for (int j = 1; j < m.arity; j++)
            parent[fem_component(parent, cell[j])] = root;
    }
    for (int i = 0; i < m.nodes; i++)
        if (!isnan(data.boundary[i]))
            anchored[fem_component(parent, (uint32_t)i)] = true;
    for (int e = 0; e < m.count; e++)
        if (data.reaction[e] > 0)
            anchored[fem_component(parent, m.cells[(size_t)e * m.arity])] = true;
    for (int i = 0; i < m.nodes; i++)
        if (parent[i] == (uint32_t)i && !anchored[i])
            vmerror_kind(v, "domain",
                         "each FEM component needs Dirichlet data or positive reaction");
}

static Val assemble(VM *v, FemMesh m, Val *args, FemOptions opts) {
    size_t values = 2 * ((size_t)m.nodes + m.count);
    size_t bytes = values * sizeof(double) + (size_t)m.nodes * (sizeof(uint32_t) + sizeof(bool));
    Obj *work = fem_work(v, bytes, (size_t)m.nodes * sizeof(double), opts.workspace);
    double *storage = (double *)work->as.buffer.data;
    FemData data = fem_read_data(v, m, args[1], args[2], opts, storage);
    uint32_t *parent = (uint32_t *)(storage + values);
    check_coercive(v, m, data, parent, (bool *)(parent + m.nodes));
    Obj *b = array_new(v, 1, m.nodes, 1);
    double *rhs = b->as.array.data;
    sparse_begin(v);
    size_t capacity = (size_t)m.count * m.arity * m.arity + m.nodes;
    Obj *coo = sparse_builder_new(v, (uint32_t)m.nodes, (uint32_t)m.nodes, capacity, SIZE_MAX);
    for (int e = 0; e < m.count; e++) {
        const uint32_t *cell = m.cells + (size_t)e * m.arity;
        FemElement g = fem_geometry(v, m, cell);
        double mass = g.measure / (m.arity * (m.arity + 1));
        double scaled = sqrt(g.measure) * sqrt(data.diffusion[e]);
        for (int i = 0; i < m.arity; i++) {
            uint32_t row = cell[i];
            if (!isnan(data.boundary[row]))
                continue;
            double load = 0;
            for (int j = 0; j < m.arity; j++) {
                uint32_t col = cell[j];
                double weight = mass * (1 + (i == j));
                double stiffness = data.reaction[e] * weight;
                for (int d = 0; d < m.dimension; d++)
                    stiffness += (scaled * g.gradient[i][d]) * (scaled * g.gradient[j][d]);
                fem_checked(v, stiffness);
                load += weight * data.source[col];
                if (isnan(data.boundary[col])) {
                    if (stiffness != 0)
                        sparse_builder_write(coo, row, col, stiffness);
                } else {
                    load -= stiffness * data.boundary[col];
                }
            }
            rhs[row] = fem_checked(v, rhs[row] + load);
        }
    }
    for (int i = 0; i < m.nodes; i++)
        if (!isnan(data.boundary[i])) {
            sparse_builder_write(coo, (uint32_t)i, (uint32_t)i, 1);
            rhs[i] = data.boundary[i];
        }
    Obj *csr = sparse_builder_finish(v, coo);
    sparse_finish(v);
    solver_workspace_done(v, work);
    Obj *out = newobj(v, O_MAP);
    sparse_result_field(v, out, "matrix", objval(csr));
    sparse_result_field(v, out, "rhs", objval(b));
    return objval(out);
}

static Val field_operation(VM *v, FemMesh mesh, Val field, Val mobility, bool velocity) {
    size_t work_count = (size_t)mesh.nodes + (velocity ? mesh.count : 0);
    size_t out_count = velocity ? (size_t)mesh.dimension * mesh.count : 0;
    Obj *work = fem_work(v, work_count * sizeof(double), out_count * sizeof(double), SIZE_MAX);
    double *u = (double *)work->as.buffer.data, *k = u + mesh.nodes;
    solver_read_vector(v, field, mesh.nodes, u);
    if (velocity)
        fem_read_field(v, mobility, mesh.count, k, 0, true);
    Obj *out = velocity ? array_new(v, mesh.dimension, mesh.count, mesh.dimension) : NULL;
    long double integral = 0;
    for (int e = 0; e < mesh.count; e++) {
        const uint32_t *cell = mesh.cells + (size_t)e * mesh.arity;
        FemElement g = fem_geometry(v, mesh, cell);
        if (!velocity) {
            for (int i = 0; i < mesh.arity; i++)
                integral += (long double)g.measure * u[cell[i]] / mesh.arity;
            continue;
        }
        for (int d = 0; d < mesh.dimension; d++) {
            long double grad = 0;
            for (int i = 1; i < mesh.arity; i++)
                grad += ((long double)u[cell[i]] - u[cell[0]]) * g.gradient[i][d];
            out->as.array.data[(size_t)e * mesh.dimension + d] =
                fem_checked(v, (double)(-k[e] * grad));
        }
    }
    solver_workspace_done(v, work);
    return velocity ? objval(out) : number(fem_checked(v, (double)integral));
}

static Val hierarchy_report(VM *v, FemMesh mesh) {
    Obj *leaves = array_new(v, 1, mesh.count, 1);
    Obj *parents = array_new(v, 1, mesh.tree_count, 1);
    Obj *generations = array_new(v, 1, mesh.tree_count, 1);
    for (int i = 0; i < mesh.count; i++)
        leaves->as.array.data[i] = mesh.leaves ? mesh.leaves[i] : (uint32_t)i;
    for (int i = 0; i < mesh.tree_count; i++) {
        uint32_t p = mesh.tree ? mesh.tree[i].parent : UINT32_MAX;
        parents->as.array.data[i] = p == UINT32_MAX ? -1.0 : (double)p;
        generations->as.array.data[i] = mesh.tree ? mesh.tree[i].generation : 0;
    }
    Obj *result = newobj(v, O_MAP);
    sparse_result_field(v, result, "leaves", objval(leaves));
    sparse_result_field(v, result, "parents", objval(parents));
    sparse_result_field(v, result, "generations", objval(generations));
    return objval(result);
}

enum { MESH, NODES, ELEMENTS, ASSEMBLE, ESTIMATE, REFINE, INTEGRAL, DARCY, MIXED, HIERARCHY };
static const char *const names[] = {"mesh",        "nodes",    "elements", "assemble",
                                    "estimate",    "refine",   "integral", "darcy_velocity",
                                    "darcy_mixed", "hierarchy"};

const char *fem_name(int id) {
    int index = id - FEM_BASE;
    return index >= 0 && (size_t)index < sizeof names / sizeof *names ? names[index] : NULL;
}

Obj *fem_module(VM *v) {
    return named_native_module(v, "fem", names, sizeof names / sizeof *names, FEM_BASE);
}

static Val run(VM *v, int id, int argc, Val *args) {
    if (id == MESH && argc == 2)
        return mesh_new(v, args[0], args[1]);
    if (id == NODES && argc == 1) {
        mesh_view(v, args[0]);
        return objval(tensor_copy(v, args[0].o->as.fem.nodes));
    }
    if (id == ELEMENTS && argc == 1) {
        FemMesh mesh = mesh_view(v, args[0]);
        Obj *out = list(v);
        for (int e = 0; e < mesh.count; e++) {
            Obj *cell = list(v);
            for (int j = 0; j < mesh.arity; j++)
                append(v, cell, integer(mesh.cells[(size_t)e * mesh.arity + j]));
            append(v, out, objval(cell));
        }
        return objval(out);
    }
    if ((id == ASSEMBLE || id == REFINE) && (argc == 3 || argc == 4)) {
        FemMesh mesh = mesh_view(v, args[0]);
        FemOptions options = fem_options(v, argc == 4 ? args[3] : nil(), id == REFINE);
        return id == ASSEMBLE ? assemble(v, mesh, args, options)
                              : fem_refine(v, mesh, args, options);
    }
    if (id == MIXED && (argc == 4 || argc == 5))
        return fem_darcy_mixed(v, mesh_view(v, args[0]), args, argc == 5 ? args[4] : nil());
    if (id == HIERARCHY && argc == 1)
        return hierarchy_report(v, mesh_view(v, args[0]));
    if (id == ESTIMATE && (argc == 4 || argc == 5))
        return fem_estimate(v, mesh_view(v, args[0]), args,
                            fem_options(v, argc == 5 ? args[4] : nil(), false));
    if ((id == INTEGRAL && argc == 2) || (id == DARCY && argc == 3))
        return field_operation(v, mesh_view(v, args[0]), args[1], argc == 3 ? args[2] : nil(),
                               id == DARCY);
    vmerror_kind(v, "arity", "invalid FEM API arity");
}

Val fem_call(VM *v, int id, int argc, Val *args) {
    v->solver_checkpoint = v->objects;
    Val result = run(v, id - FEM_BASE, argc, args);
    v->solver_checkpoint = NULL;
    return result;
}
