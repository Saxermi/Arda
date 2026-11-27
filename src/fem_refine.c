#include "fem.h"
#include "sort_records.h"

typedef struct {
    double energy;
    int cell;
} Mark;

static int compare_faces(const void *left, const void *right) {
    const FemFace *a = left, *b = right;
    if (a->a != b->a)
        return a->a < b->a ? -1 : 1;
    if (a->b != b->b)
        return a->b < b->b ? -1 : 1;
    return (a->slot > b->slot) - (a->slot < b->slot);
}

void fem_sort_faces(FemMesh mesh, FemFace *faces) {
    for (int e = 0; e < mesh.count; e++) {
        const uint32_t *cell = mesh.cells + (size_t)e * mesh.arity;
        for (int j = 0; j < mesh.arity; j++) {
            uint32_t a = cell[(j + 1) % mesh.arity];
            uint32_t b = mesh.dimension == 1 ? a : cell[(j + 2) % mesh.arity];
            uint32_t slot = (uint32_t)(e * mesh.arity + j);
            faces[slot] = (FemFace){a < b ? a : b, a < b ? b : a, slot};
        }
    }
    FemFace temp;
    sort_records(faces, (size_t)mesh.count * mesh.arity, sizeof *faces, compare_faces, &temp);
}

size_t fem_face_end(VM *v, const FemFace *faces, size_t first, size_t count) {
    size_t last = first + 1;
    while (last < count && faces[first].a == faces[last].a && faces[first].b == faces[last].b)
        last++;
    if (last - first > 2)
        vmerror_kind(v, "domain", "non-manifold FEM face");
    return last;
}

static void face_normal(VM *v, FemMesh mesh, uint32_t slot, double *normal) {
    const uint32_t *cell = mesh.cells + (size_t)(slot / mesh.arity) * mesh.arity;
    FemElement g = fem_geometry(v, mesh, cell);
    const double *gradient = g.gradient[slot % mesh.arity];
    double magnitude = hypot(gradient[0], gradient[1]);
    normal[0] = gradient[0] / magnitude;
    normal[1] = gradient[1] / magnitude;
}

void fem_check_faces(VM *v, FemMesh mesh) {
    size_t count = (size_t)mesh.count * mesh.arity;
    Obj *work = buffer_new(v, count * sizeof(FemFace));
    FemFace *faces = (FemFace *)work->as.buffer.data;
    fem_sort_faces(mesh, faces);
    for (size_t first = 0, last; first < count; first = last) {
        last = fem_face_end(v, faces, first, count);
        if (last - first == 2) {
            double a[2], b[2];
            face_normal(v, mesh, faces[first].slot, a);
            face_normal(v, mesh, faces[first + 1].slot, b);
            if (a[0] * b[0] + a[1] * b[1] >= 0)
                vmerror_kind(v, "domain", "overlapping FEM cells at a shared face");
        }
    }
    solver_workspace_done(v, work);
}

static double flux(VM *v, FemMesh mesh, FemData data, const double *u, uint32_t slot,
                   double *scale) {
    unsigned e = slot / mesh.arity, opposite = slot % mesh.arity;
    const uint32_t *cell = mesh.cells + (size_t)e * mesh.arity;
    FemElement g = fem_geometry(v, mesh, cell);
    double norm = hypot(g.gradient[opposite][0], g.gradient[opposite][1]);
    *scale = mesh.dimension == 1 ? g.measure : 2 * (g.measure * norm);
    long double result = 0;
    for (int d = 0; d < mesh.dimension; d++) {
        long double derivative = 0;
        for (int j = 1; j < mesh.arity; j++)
            derivative += ((long double)u[cell[j]] - u[cell[0]]) * g.gradient[j][d];
        result -= derivative * (g.gradient[opposite][d] / norm);
    }
    return fem_checked(v, (double)(result * data.diffusion[e]));
}

Val fem_estimate(VM *v, FemMesh mesh, Val *args, FemOptions opts) {
    size_t values = 3 * (size_t)mesh.nodes + 2 * (size_t)mesh.count;
    size_t nfaces = (size_t)mesh.count * mesh.arity;
    Obj *work = fem_work(v, values * sizeof(double) + nfaces * sizeof(FemFace),
                         (size_t)mesh.count * sizeof(double), opts.workspace);
    double *storage = (double *)work->as.buffer.data;
    FemData data = fem_read_data(v, mesh, args[1], args[2], opts, storage);
    double *u = storage + values - mesh.nodes;
    solver_read_vector(v, args[3], mesh.nodes, u);
    FemFace *faces = (FemFace *)(storage + values);
    fem_sort_faces(mesh, faces);
    Obj *out = array_new(v, 1, mesh.count, 1);
    double *energy = out->as.array.data;
    for (int e = 0; e < mesh.count; e++) {
        const uint32_t *cell = mesh.cells + (size_t)e * mesh.arity;
        FemElement g = fem_geometry(v, mesh, cell);
        long double sum = 0, squares = 0;
        for (int j = 0; j < mesh.arity; j++) {
            long double residual =
                (long double)data.source[cell[j]] - (long double)data.reaction[e] * u[cell[j]];
            sum += residual;
            squares += residual * residual;
        }
        energy[e] = fem_checked(v, (double)((sum * sum + squares) * g.measure * g.diameter *
                                            g.diameter / (mesh.arity * (mesh.arity + 1))));
    }
    for (size_t first = 0, last; first < nfaces; first = last) {
        last = fem_face_end(v, faces, first, nfaces);
        FemFace a = faces[first];
        bool interior = last - first == 2;
        if (!interior && !isnan(data.boundary[a.a]) && !isnan(data.boundary[a.b]))
            continue;
        double h, other_h;
        long double jump = flux(v, mesh, data, u, a.slot, &h);
        FemFace b = faces[last - 1];
        if (interior) {
            jump += flux(v, mesh, data, u, b.slot, &other_h);
            h = fmin(h, other_h);
        }
        long double term = jump * jump * h;
        if (mesh.dimension == 2)
            term *= h;
        double share = fem_checked(v, (double)(term / (interior ? 2 : 1)));
        energy[a.slot / mesh.arity] = fem_checked(v, energy[a.slot / mesh.arity] + share);
        if (interior)
            energy[b.slot / mesh.arity] = fem_checked(v, energy[b.slot / mesh.arity] + share);
    }
    solver_workspace_done(v, work);
    return objval(out);
}

static int largest_first(const void *left, const void *right) {
    const Mark *a = left, *b = right;
    if (a->energy != b->energy)
        return a->energy > b->energy ? -1 : 1;
    return (a->cell > b->cell) - (a->cell < b->cell);
}

static void mark_elements(FemMesh mesh, const double *energy, Mark *order, unsigned char *marked,
                          double fraction) {
    long double total = 0;
    for (int i = 0; i < mesh.count; i++) {
        order[i] = (Mark){energy[i], i};
        total += energy[i];
    }
    Mark temp;
    sort_records(order, (size_t)mesh.count, sizeof *order, largest_first, &temp);
    long double selected = 0;
    for (int i = 0; i < mesh.count && order[i].energy > 0; i++) {
        if (fraction < 1 && selected >= fraction * total)
            break;
        marked[order[i].cell] = 1;
        selected += order[i].energy;
    }
}

static void split_triangle(VM *v, uint32_t children[4][3], int *count, uint32_t a, uint32_t b,
                           uint32_t mid) {
    for (int k = 0; k < *count; k++) {
        int ia = -1, ib = -1;
        for (int i = 0; i < 3; i++) {
            if (children[k][i] == a)
                ia = i;
            if (children[k][i] == b)
                ib = i;
        }
        if (ia >= 0 && ib >= 0) {
            memcpy(children[*count], children[k], sizeof children[k]);
            children[k][ib] = mid;
            children[(*count)++][ia] = mid;
            return;
        }
    }
    vmerror_kind(v, "domain", "invalid FEM subdivision");
}

static bool split_face(FemMesh mesh, const unsigned char *marked, FemFace a, FemFace b) {
    return marked[a.slot / mesh.arity] || marked[b.slot / mesh.arity];
}

/* Choosing the longest initial edge gives reproducible labels, including ties.
 * On a descendant vertex zero is the midpoint from its most recent bisection. */
static uint32_t reference_slot(FemMesh mesh, int e) {
    if (mesh.tree)
        return 3 * (uint32_t)e;
    const uint32_t *cell = mesh.cells + 3 * (size_t)e;
    double longest = -1;
    uint32_t lo = UINT32_MAX, hi = UINT32_MAX;
    int best = 0;
    for (int j = 0; j < 3; j++) {
        uint32_t a = cell[(j + 1) % 3], b = cell[(j + 2) % 3];
        if (b < a) {
            uint32_t swap = a;
            a = b;
            b = swap;
        }
        const double *x = mesh.x + 2 * (size_t)a, *y = mesh.x + 2 * (size_t)b;
        double length = hypot(x[0] - y[0], x[1] - y[1]);
        if (length > longest || (length == longest && (a < lo || (a == lo && b < hi)))) {
            best = j;
            longest = length;
            lo = a;
            hi = b;
        }
    }
    return 3 * (uint32_t)e + (uint32_t)best;
}

static void require_edge(uint32_t slot, uint32_t *selected, uint32_t *queue, size_t *tail) {
    if (selected[slot] == UINT32_MAX) {
        selected[slot] = 0;
        queue[(*tail)++] = slot;
    }
}

static void nvb_close(VM *v, FemMesh mesh, const FemFace *faces, const unsigned char *marked,
                      uint32_t *mid, uint32_t *refs, uint32_t *peer, uint32_t *queue) {
    size_t length = 3 * (size_t)mesh.count;
    for (size_t first = 0, last; first < length; first = last) {
        last = fem_face_end(v, faces, first, length);
        uint32_t a = faces[first].slot, b = faces[last - 1].slot;
        peer[a] = b;
        peer[b] = a;
    }
    size_t tail = 0;
    for (int e = 0; e < mesh.count; e++) {
        refs[e] = reference_slot(mesh, e);
        if (marked[e])
            require_edge(refs[e], mid, queue, &tail);
    }
    // Every requested old edge also requires the reference edge of its owner.
    // Its counterpart on the adjacent triangle shares the same midpoint.
    for (size_t head = 0; head < tail; head++) {
        uint32_t slot = queue[head];
        require_edge(peer[slot], mid, queue, &tail);
        require_edge(refs[slot / 3], mid, queue, &tail);
    }
}

static void nvb_bisect(uint32_t left[3], uint32_t right[3], uint32_t mid, uint32_t *left_id,
                       uint32_t *right_id, FemBranch *tree, int *used) {
    uint32_t c = left[0], a = left[1], b = left[2], parent = *left_id;
    left[0] = right[0] = mid;
    left[1] = c;
    left[2] = a;
    right[1] = b;
    right[2] = c;
    FemBranch branch = {parent, tree[parent].generation + 1};
    tree[*used] = tree[*used + 1] = branch;
    *left_id = (uint32_t)(*used)++;
    *right_id = (uint32_t)(*used)++;
}

static int nvb_children(const uint32_t *cell, const uint32_t *mid, int ref, uint32_t local[4][3],
                        uint32_t ids[4], FemBranch *tree, int *used) {
    for (int j = 0; j < 3; j++)
        local[0][j] = cell[(ref + j) % 3];
    if (mid[ref] == UINT32_MAX)
        return 1;
    nvb_bisect(local[0], local[1], mid[ref], ids, ids + 1, tree, used);
    int count = 2;
    // The remaining two original edges are the children's reference edges.
    uint32_t left = mid[(ref + 2) % 3], right = mid[(ref + 1) % 3];
    if (left != UINT32_MAX) {
        nvb_bisect(local[0], local[count], left, ids, ids + count, tree, used);
        count++;
    }
    if (right != UINT32_MAX) {
        nvb_bisect(local[1], local[count], right, ids + 1, ids + count, tree, used);
        count++;
    }
    return count;
}

Val fem_refine(VM *v, FemMesh mesh, Val *args, FemOptions opts) {
    if (opts.nvb && mesh.dimension != 2)
        vmerror_kind(v, "shape", "newest-vertex bisection requires a triangular mesh");
    size_t length = (size_t)mesh.count * mesh.arity;
    size_t values = (size_t)mesh.nodes + mesh.count;
    size_t face_count = mesh.dimension == 2 ? length : 0;
    size_t bytes = values * sizeof(double) + (size_t)mesh.count * sizeof(Mark) +
                   face_count * (sizeof(FemFace) + sizeof(uint32_t)) + mesh.count;
    if (opts.nvb)
        bytes += (2 * length + (size_t)mesh.count) * sizeof(uint32_t);
    Obj *work = fem_work(v, bytes, 0, opts.workspace);
    double *u = (double *)work->as.buffer.data, *energy = u + mesh.nodes;
    Mark *order = (Mark *)(u + values);
    FemFace *faces = (FemFace *)(order + mesh.count);
    uint32_t *mid = (uint32_t *)(faces + face_count);
    uint32_t *refs = mid + face_count, *peer = refs, *queue = refs;
    if (opts.nvb) {
        peer = refs + mesh.count;
        queue = peer + length;
    }
    unsigned char *marked = (unsigned char *)(queue + (opts.nvb ? length : 0));
    memset(marked, 0, (size_t)mesh.count);
    memset(mid, 0xff, face_count * sizeof *mid);
    solver_read_vector(v, args[1], mesh.nodes, u);
    solver_read_vector(v, args[2], mesh.count, energy);
    for (int e = 0; e < mesh.count; e++)
        if (energy[e] < 0)
            vmerror_kind(v, "domain", "FEM indicators must be nonnegative");
    mark_elements(mesh, energy, order, marked, opts.fraction);

    int added = 0, children = mesh.count;
    if (mesh.dimension == 1) {
        for (int e = 0; e < mesh.count; e++)
            added += marked[e];
        children += added;
    } else {
        fem_sort_faces(mesh, faces);
        if (opts.nvb)
            nvb_close(v, mesh, faces, marked, mid, refs, peer, queue);
        for (size_t first = 0, last; first < length; first = last) {
            last = fem_face_end(v, faces, first, length);
            bool split = opts.nvb ? mid[faces[first].slot] != UINT32_MAX
                                  : split_face(mesh, marked, faces[first], faces[last - 1]);
            if (split) {
                for (size_t i = first; i < last; i++)
                    mid[faces[i].slot] = (uint32_t)(mesh.nodes + added);
                added++;
                children += (int)(last - first);
            }
        }
    }
    int n = mesh.nodes + added;
    size_t branches = opts.nvb ? (size_t)mesh.tree_count + 2 * (children - mesh.count) : 0;
    size_t tree_bytes =
        opts.nvb ? branches * sizeof(FemBranch) + (size_t)children * sizeof(uint32_t) : 0;
    size_t child_entries = (size_t)children * mesh.arity;
    size_t output =
        ((size_t)n * (mesh.dimension + 1) + (size_t)2 * added + children) * sizeof(double) +
        child_entries * sizeof(uint32_t);
    if (n > opts.max_nodes || children > opts.max_elements || branches > ARDA_MAX_ELEMENTS ||
        (size_t)n * mesh.dimension > ARDA_MAX_ELEMENTS || (size_t)2 * added > ARDA_MAX_ELEMENTS ||
        output > tensor_memory_available(v) || tree_bytes > tensor_memory_available(v) - output)
        vmerror_kind(v, "memory_budget", "FEM refinement exceeds output budget");
    Obj *nodes = array_new(v, mesh.dimension, n, mesh.dimension);
    Obj *cells = buffer_new(v, child_entries * sizeof(uint32_t));
    Obj *field = array_new(v, 1, n, 1);
    Obj *parents = array_new(v, 1, children, 1);
    Obj *edges = array_new(v, 2, added, 2);
    Obj *history = opts.nvb ? buffer_new(v, tree_bytes) : NULL;
    FemBranch *tree = history ? (FemBranch *)history->as.buffer.data : NULL;
    uint32_t *leaves = tree ? (uint32_t *)(tree + branches) : NULL;
    int used = mesh.tree_count;
    if (history) {
        if (mesh.tree)
            memcpy(tree, mesh.tree, (size_t)used * sizeof *tree);
        else
            for (int e = 0; e < used; e++)
                tree[e] = (FemBranch){UINT32_MAX, 0};
    }
    double *x = nodes->as.array.data, *fine = field->as.array.data;
    uint32_t *ix = (uint32_t *)cells->as.buffer.data;
    memcpy(x, mesh.x, (size_t)mesh.nodes * mesh.dimension * sizeof *x);
    memcpy(fine, u, (size_t)mesh.nodes * sizeof *fine);

    int node = mesh.nodes;
    size_t first = 0;
    for (int k = 0; k < added; k++, node++) {
        uint32_t a, b;
        if (mesh.dimension == 1) {
            while (!marked[first])
                first++;
            a = mesh.cells[2 * first];
            b = mesh.cells[2 * first + 1];
            first++;
        } else {
            size_t last = fem_face_end(v, faces, first, length);
            while (mid[faces[first].slot] == UINT32_MAX) {
                first = last;
                last = fem_face_end(v, faces, first, length);
            }
            a = faces[first].a;
            b = faces[first].b;
            first = last;
        }
        edges->as.array.data[2 * (size_t)k] = a;
        edges->as.array.data[2 * (size_t)k + 1] = b;
        for (int d = 0; d < mesh.dimension; d++) {
            double xa = mesh.x[(size_t)a * mesh.dimension + d],
                   xb = mesh.x[(size_t)b * mesh.dimension + d];
            x[(size_t)node * mesh.dimension + d] = xa + (xb - xa) * 0.5;
        }
        fine[node] = u[a] * 0.5 + u[b] * 0.5;
    }

    int child = 0;
    node = mesh.nodes;
    FemMesh refined = {.x = x,
                       .cells = ix,
                       .nodes = n,
                       .count = children,
                       .dimension = mesh.dimension,
                       .arity = mesh.arity};
    for (int e = 0; e < mesh.count; e++) {
        const uint32_t *cell = mesh.cells + (size_t)e * mesh.arity;
        uint32_t local[4][3];
        memcpy(local[0], cell, (size_t)mesh.arity * sizeof *cell);
        int count = 1;
        uint32_t ids[4] = {mesh.leaves ? mesh.leaves[e] : (uint32_t)e};
        if (opts.nvb)
            count = nvb_children(cell, mid + 3 * (size_t)e, (int)(refs[e] % 3), local, ids, tree,
                                 &used);
        else if (mesh.dimension == 1 && marked[e]) {
            local[0][1] = local[1][0] = (uint32_t)node++;
            local[1][1] = cell[1];
            count++;
        } else if (mesh.dimension == 2) {
            const uint32_t *m = mid + 3 * (size_t)e;
            if (m[0] != UINT32_MAX && m[1] != UINT32_MAX && m[2] != UINT32_MAX) {
                const uint32_t red[4][3] = {{cell[0], m[2], m[1]},
                                            {m[2], cell[1], m[0]},
                                            {m[1], m[0], cell[2]},
                                            {m[0], m[1], m[2]}};
                memcpy(local, red, sizeof red);
                count = 4;
            } else {
                for (int j = 0; j < 3; j++)
                    if (m[j] != UINT32_MAX)
                        split_triangle(v, local, &count, cell[(j + 1) % 3], cell[(j + 2) % 3],
                                       m[j]);
            }
        }
        for (int j = 0; j < count; j++, child++) {
            fem_geometry(v, refined, local[j]);
            memcpy(ix + (size_t)child * mesh.arity, local[j], (size_t)mesh.arity * sizeof *ix);
            parents->as.array.data[child] = e;
            if (leaves)
                leaves[child] = ids[j];
        }
    }

    solver_workspace_done(v, work);
    Obj *result = newobj(v, O_MAP);
    Obj *out = fem_mesh_wrap(v, nodes, cells, children);
    if (history) {
        out->as.fem.hierarchy = history;
        out->as.fem.tree_count = used;
        barrier(v, out, objval(history));
    }
    sparse_result_field(v, result, "mesh", objval(out));
    sparse_result_field(v, result, "field", objval(field));
    sparse_result_field(v, result, "parents", objval(parents));
    sparse_result_field(v, result, "edge_midpoints", objval(edges));
    return objval(result);
}
