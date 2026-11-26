#ifndef ARDA_COMPUTE_PLAN_H
#define ARDA_COMPUTE_PLAN_H

#include "compute_ops.h"

Val compute_plan(VM *v, Val a, Val b, Val options);
Val compute_explain(VM *v, Val plan);
Val compute_execute(VM *v, Val plan, Val a, Val b);
Val compute_last_run(VM *v);

#endif
