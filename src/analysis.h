#ifndef ARDA_ANALYSIS_H
#define ARDA_ANALYSIS_H

#include "arda.h"

#define CONTRACT_BASE 2800

struct FunctionAnalysis *function_analyze(const Node *function);
Val function_cost(VM *v, Val function, Val specification);
Val function_plan(VM *v, Val function, Val specification);
const char *function_algorithm(VM *v, Val fn);
Obj *contract_module(VM *v);
Val contract_call(VM *v, int id, int argc, Val *args);

#endif
