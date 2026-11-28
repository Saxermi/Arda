#ifndef ARDA_COMPUTE_STRUCTURED_H
#define ARDA_COMPUTE_STRUCTURED_H

#include "arda.h"

Val structured_plan(VM *v, Val op, Val input);
Val structured_explain(VM *v, Val plan);
Val structured_execute(VM *v, Val plan, int trace_limit);
void structured_trace(VM *v, Obj *object);

#endif
