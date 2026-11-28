#ifndef ARDA_LAZY_GRAPH_H
#define ARDA_LAZY_GRAPH_H

#include "arda.h"

#define LAZY_BASE 2700

Obj *lazy_module(VM *v);
Val lazy_call(VM *v, int id, int argc, Val *args);
Val graph_plan(VM *v, Val root, Val options);
Val graph_explain(VM *v, Val plan);
Val graph_execute(VM *v, Val plan, int trace_limit);
Val graph_make_run(VM *v, Obj *report, const char *algorithm);
Val graph_run_info(VM *v, Val run);
const char *graph_run_algorithm(Val run);
void graph_trace_object(VM *v, Obj *object);
void graph_release_object(Obj *object);

#endif
