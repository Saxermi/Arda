#include "fem.h"

typedef struct {
    double mass[3][3], area, pressure_scale;
} DarcyCell;

typedef struct {
    uint32_t dof;
    int sign;
} DarcySide;

/* The RT0 basis for side i is (x - vertex[i]) / (2*area) and has unit
 * integrated outward normal flux there. Three points integrate its mass exactly. */
static DarcyCell cell_mass(VM *v, FemMesh mesh, const uint32_t *cell, double mobility) {
    DarcyCell out = {.area = fem_geometry(v, mesh, cell).measure};
    long double sum[3][3] = {{0}};
    for (int q = 0; q < 3; q++) {
        long double delta[3][2] = {{0}};
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int d = 0; d < 2; d++)
                    delta[i][d] += ((long double)mesh.x[2 * (size_t)cell[j] + d] -
                                    mesh.x[2 * (size_t)cell[i] + d]) *
                                   (q == j ? 4 : 1) / 6;
        for (int i = 0; i < 3; i++)
            for (int j = i; j < 3; j++)
                sum[i][j] += delta[i][0] * delta[j][0] + delta[i][1] * delta[j][1];
    }
    for (int i = 0; i < 3; i++)
        for (int j = i; j < 3; j++) {
            double a = fem_checked(v, (double)(sum[i][j] / 12 / out.area / mobility));
            out.mass[i][j] = out.mass[j][i] = a;
        }
    return out;
}

static uint32_t number_sides(VM *v, FemMesh mesh, const FemFace *faces, FemData data,
                             DarcySide *sides, uint32_t *parents, bool *anchored) {
    uint32_t dofs = 0;
    for (int e = 0; e < mesh.count; e++)
        parents[e] = (uint32_t)e;
    size_t count = 3 * (size_t)mesh.count;
    for (size_t first = 0, last; first < count; first = last) {
        last = fem_face_end(v, faces, first, count);
        FemFace a = faces[first];
        if (last - first == 2) {
            uint32_t b = faces[first + 1].slot;
            sides[a.slot] = (DarcySide){dofs, 1};
            sides[b] = (DarcySide){dofs++, -1};
            parents[fem_component(parents, b / 3)] = fem_component(parents, a.slot / 3);
        } else if (!isnan(data.boundary[a.a]) && !isnan(data.boundary[a.b])) {
            sides[a.slot] = (DarcySide){dofs++, 1};
        } else {
            sides[a.slot] = (DarcySide){UINT32_MAX, 0};
        }
    }
    for (size_t first = 0, last; first < count; first = last) {
        last = fem_face_end(v, faces, first, count);
        uint32_t slot = faces[first].slot;
        if (last - first == 1 && sides[slot].sign)
            anchored[fem_component(parents, slot / 3)] = true;
    }
    for (int e = 0; e < mesh.count; e++)
        if (parents[e] == (uint32_t)e && !anchored[e])
            vmerror_kind(v, "domain", "each mixed Darcy component needs a pressure boundary edge");
    return dofs;
}

static long double source_integral(const uint32_t *cell, FemData data, double area) {
    return ((long double)data.source[cell[0]] + data.source[cell[1]] + data.source[cell[2]]) *
           area / 3;
}

static Obj *assemble(VM *v, FemMesh mesh, FemData data, DarcyCell *cells, const DarcySide *sides,
                     uint32_t dofs, const double *scale, double *rhs) {
    uint32_t n = dofs + (uint32_t)mesh.count;
    Obj *coo = sparse_builder_new(v, n, n, 15 * (size_t)mesh.count, SIZE_MAX);
    for (int e = 0; e < mesh.count; e++) {
        DarcyCell *c = cells + e;
        const uint32_t *vertices = mesh.cells + 3 * (size_t)e;
        const DarcySide *local = sides + 3 * (size_t)e;
        uint32_t p = dofs + (uint32_t)e;
        long double schur_diag = 0;
        for (int i = 0; i < 3; i++)
            if (local[i].sign) {
                long double d = scale[local[i].dof];
                schur_diag += d * d;
            }
        c->pressure_scale = fem_checked(v, (double)(1 / sqrtl(schur_diag)));
        rhs[p] =
            fem_checked(v, (double)(-source_integral(vertices, data, c->area) * c->pressure_scale));
        for (int i = 0; i < 3; i++) {
            DarcySide a = local[i];
            if (!a.sign)
                continue;
            long double row = (long double)a.sign * scale[a.dof];
            double coupling = fem_checked(v, (double)(-row * c->pressure_scale));
            sparse_builder_write(coo, p, a.dof, coupling);
            sparse_builder_write(coo, a.dof, p, coupling);
            for (int j = i; j < 3; j++) {
                DarcySide b = local[j];
                if (!b.sign)
                    continue;
                double value =
                    fem_checked(v, (double)(row * c->mass[i][j] * b.sign * scale[b.dof]));
                if (value == 0)
                    continue;
                sparse_builder_write(coo, a.dof, b.dof, value);
                if (i != j)
                    sparse_builder_write(coo, b.dof, a.dof, value);
            }
        }
    }
    return sparse_builder_finish(v, coo);
}

Val fem_darcy_mixed(VM *v, FemMesh mesh, Val *args, Val settings) {
    if (mesh.dimension != 2)
        vmerror_kind(v, "shape", "mixed Darcy requires a triangular mesh");
    if (3 * (size_t)mesh.count > ARDA_MAX_ELEMENTS)
        vmerror_kind(v, "memory_budget", "mixed Darcy flux output exceeds the array limit");
    SolverOptions solver = solver_options(v, settings);
    size_t length = 3 * (size_t)mesh.count;
    size_t values = 2 * ((size_t)mesh.nodes + mesh.count) + length;
    size_t bytes = values * sizeof(double) + (size_t)mesh.count * sizeof(DarcyCell) +
                   length * (sizeof(FemFace) + sizeof(DarcySide)) +
                   (size_t)mesh.count * (sizeof(uint32_t) + sizeof(bool));
    Obj *work = fem_work(v, bytes, 0, solver.workspace_limit);
    double *data_buffer = (double *)work->as.buffer.data;
    double *scale = data_buffer + values - length;
    DarcyCell *cells = (DarcyCell *)(data_buffer + values);
    FemFace *faces = (FemFace *)(cells + mesh.count);
    DarcySide *sides = (DarcySide *)(faces + length);
    uint32_t *parents = (uint32_t *)(sides + length);
    bool *anchored = (bool *)(parents + mesh.count);
    FemOptions material = {.diffusion = args[3], .reaction = number(0)};
    FemData data = fem_read_data(v, mesh, args[1], args[2], material, data_buffer);
    fem_sort_faces(mesh, faces);
    uint32_t dofs = number_sides(v, mesh, faces, data, sides, parents, anchored);
    size_t n = dofs + (size_t)mesh.count;
    if (n > ARDA_MAX_ELEMENTS)
        vmerror_kind(v, "memory_budget", "mixed Darcy system exceeds the array limit");
    for (int e = 0; e < mesh.count; e++) {
        const uint32_t *vertices = mesh.cells + 3 * (size_t)e;
        cells[e] = cell_mass(v, mesh, vertices, data.diffusion[e]);
        for (int i = 0; i < 3; i++) {
            DarcySide side = sides[3 * (size_t)e + i];
            if (side.sign)
                scale[side.dof] = fem_checked(v, scale[side.dof] + cells[e].mass[i][i]);
        }
    }
    for (uint32_t i = 0; i < dofs; i++)
        scale[i] = fem_checked(v, 1 / sqrt(scale[i]));
    Obj *b = array_new(v, 1, (int)n, 1);
    double *rhs = b->as.array.data;
    for (size_t first = 0, last; first < length; first = last) {
        last = fem_face_end(v, faces, first, length);
        FemFace a = faces[first];
        DarcySide side = sides[a.slot];
        if (last - first == 1 && side.sign)
            rhs[side.dof] = fem_checked(v, -(0.5 * data.boundary[a.a] + 0.5 * data.boundary[a.b]) *
                                               scale[side.dof]);
    }
    sparse_begin(v);
    Obj *mat = assemble(v, mesh, data, cells, sides, dofs, scale, rhs);
    Val report = solver_solve_minres(v, objval(mat), objval(b), settings);
    sparse_drop_temporary(v, mat, NULL);
    sparse_finish(v);
    tensor_buffer_discard(v, b->as.array.buffer);

    Obj *pressure = array_new(v, 1, mesh.count, 1);
    Obj *velocity = array_new(v, 2, mesh.count, 2);
    Obj *flux = array_new(v, 2, mesh.count, 3);
    Obj *solution = tablefind(&report.o->as.map.tab, "solution")->val.o;
    const double *x = solution->as.array.data;
    double max_imbalance = 0;
    for (int e = 0; e < mesh.count; e++) {
        DarcyCell *c = cells + e;
        const uint32_t *cell = mesh.cells + 3 * (size_t)e;
        pressure->as.array.data[e] = fem_checked(v, x[dofs + e] * c->pressure_scale);
        long double balance = 0, u[2] = {0};
        for (int i = 0; i < 3; i++) {
            DarcySide side = sides[3 * (size_t)e + i];
            double q = side.sign ? fem_checked(v, side.sign * x[side.dof] * scale[side.dof]) : 0;
            flux->as.array.data[3 * (size_t)e + i] = q;
            balance += q;
            for (int d = 0; d < 2; d++) {
                long double offset = 0;
                for (int j = 0; j < 3; j++)
                    offset += ((long double)mesh.x[2 * (size_t)cell[j] + d] -
                               mesh.x[2 * (size_t)cell[i] + d]) /
                              3;
                u[d] += offset * q / (2 * (long double)c->area);
            }
        }
        double error = fem_checked(v, (double)(balance - source_integral(cell, data, c->area)));
        max_imbalance = fmax(max_imbalance, fabs(error));
        for (int d = 0; d < 2; d++)
            velocity->as.array.data[2 * (size_t)e + d] = fem_checked(v, (double)u[d]);
    }
    tensor_buffer_discard(v, solution->as.array.buffer);
    solver_workspace_done(v, work);
    Obj *out = newobj(v, O_MAP);
    sparse_result_field(v, out, "pressure", objval(pressure));
    sparse_result_field(v, out, "velocity", objval(velocity));
    sparse_result_field(v, out, "flux", objval(flux));
    sparse_result_field(v, out, "face_dofs", integer(dofs));
    sparse_result_field(v, out, "max_cell_imbalance", number(max_imbalance));
    const char *const keys[] = {"converged",       "iterations", "reason",         "residual",
                                "workspace_bytes", "trace",      "trace_truncated"};
    for (size_t i = 0; i < sizeof keys / sizeof *keys; i++) {
        Entry *entry = tablefind(&report.o->as.map.tab, keys[i]);
        if (entry)
            sparse_result_field(v, out, keys[i], entry->val);
    }
    return objval(out);
}
