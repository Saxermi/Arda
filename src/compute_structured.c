#include "compute_structured.h"
#include "lazy_graph.h"
#include "linear_operator.h"
#include "typed_tensor.h"

#include <time.h>

typedef struct {
    Obj *op;
    TypedTensor input;
    uint64_t version;
    int rows;
    size_t workspace, output;
} StructuredPlan;

static StructuredPlan *plan_data(Obj *object) {
    return (StructuredPlan *)(object + 1);
}

static void put(VM *v, Obj *map, const char *key, Val value) {
    tableset(v, map, &map->as.map.tab, key, value);
}

static void check_budget(VM *v, size_t output, size_t workspace) {
    size_t available = tensor_memory_available(v);
    if (output > available || workspace > available - output)
        vmerror_kind(v, "memory_budget", "structured plan exceeds available tensor memory");
}

Val structured_plan(VM *v, Val op, Val value) {
    if (!isobj(op, O_LINOP) && !(isobj(op, O_SPARSE) && op.o->as.sparse.csr))
        vmerror_kind(v, "type", "structured plan requires a linop or canonical CSR");
    TypedTensor x = typed_tensor(v, value);
    size_t rows, cols;
    linop_shape(v, op.o, &rows, &cols);
    if (x.dtype != DT_F64 || x.rank != 1 || !typed_contiguous(&x) || x.n != cols ||
        rows > ARDA_MAX_ELEMENTS)
        vmerror_kind(v, "shape",
                     "structured plan requires a packed Float64 vector of matching size");
    size_t workspace = linop_workspace(v, op.o), output = rows * sizeof(double);
    check_budget(v, output, workspace);
    Obj *object = newobj_extra(v, O_STRUCTURED_PLAN, sizeof(StructuredPlan));
    *plan_data(object) =
        (StructuredPlan){op.o, x, x.buffer->as.buffer.version, (int)rows, workspace, output};
    barrier(v, object, op);
    barrier(v, object, objval(x.buffer));
    return objval(object);
}

Val structured_explain(VM *v, Val value) {
    const StructuredPlan *p = plan_data(value.o);
    Obj *report = newobj(v, O_MAP);
    put(v, report, "algorithm", objval(string(v, "Structured matrix-vector application")));
    put(v, report, "kernel", objval(string(v, "structured")));
    put(v, report, "storage",
        objval(string(v, p->input.buffer->as.buffer.mapped ? "mapped" : "heap")));
    put(v, report, "workspace_bytes", integer((int64_t)p->workspace));
    put(v, report, "output_bytes", integer((int64_t)p->output));
    put(v, report, "estimated_seconds", nil());
    put(v, report, "flops",
        p->op->type == O_SPARSE                  ? number(2.0 * (double)p->op->as.sparse.nnz)
        : p->op->as.linop.kind == LINOP_DIAGONAL ? integer(p->rows)
                                                 : nil());
    return objval(report);
}

Val structured_execute(VM *v, Val plan, int trace_limit) {
    const StructuredPlan *p = plan_data(plan.o);
    const TypedTensor *x = &p->input;
    if (x->buffer->as.buffer.mapped)
        vm_require_file_access(v);
    if (p->version != x->buffer->as.buffer.version)
        vmerror_kind(v, "mutation", "structured input changed after planning");
    check_budget(v, p->output, p->workspace);
    const double *data = (const double *)typed_element(x, (ptrdiff_t)x->offset);
    for (size_t i = 0; i < x->n; i++)
        if (!isfinite(data[i]))
            vmerror_kind(v, "domain", "nonfinite structured input");
    struct timespec start, end;
    bool timed = trace_limit >= 0 && clock_gettime(CLOCK_MONOTONIC, &start) == 0;
    Obj *result = typed_new(v, DT_F64, 1, &p->rows);
    double *out = (double *)typed_element(&result->as.tensor, 0);
    LinopContext context;
    LinearOperator op = linop_open(v, p->op, &context);
    if (x->n && p->rows)
        op.apply(op.context, data, 1, out, 1);
    linop_close(v, &context);
    if (v->interrupted && *v->interrupted) {
        tensor_buffer_discard(v, result->as.tensor.buffer);
        vmerror_kind(v, "interrupt", "structured execution did not complete");
    }
    for (int i = 0; i < p->rows; i++) {
        if (!isfinite(out[i])) {
            tensor_buffer_discard(v, result->as.tensor.buffer);
            vmerror_kind(v, "domain", "nonfinite structured result");
        }
    }
    if (trace_limit < 0)
        return objval(result);
    Val elapsed = nil();
    if (timed && clock_gettime(CLOCK_MONOTONIC, &end) == 0)
        elapsed = number((double)(end.tv_sec - start.tv_sec) +
                         1e-9 * (double)(end.tv_nsec - start.tv_nsec));
    Obj *report = structured_explain(v, plan).o, *trace = list(v);
    if (trace_limit) {
        Obj *event = newobj(v, O_MAP);
        put(v, event, "start", integer(0));
        put(v, event, "elements", integer(p->rows));
        append(v, trace, objval(event));
    }
    put(v, report, "value", objval(result));
    put(v, report, "workspace_peak_bytes", integer((int64_t)p->workspace));
    put(v, report, "elapsed_seconds", elapsed);
    put(v, report, "estimate_error_seconds", nil());
    put(v, report, "blocks", integer(1));
    put(v, report, "trace", objval(trace));
    put(v, report, "trace_omitted", integer(!trace_limit));
    return graph_make_run(v, report, "Structured matrix-vector application");
}

void structured_trace(VM *v, Obj *object) {
    const StructuredPlan *p = plan_data(object);
    markobj(v, p->op);
    markobj(v, p->input.buffer);
}
