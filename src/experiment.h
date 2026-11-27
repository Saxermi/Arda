#ifndef ARDA_EXPERIMENT_H
#define ARDA_EXPERIMENT_H

#include "arda.h"

#define EXPERIMENT_BASE 2600
Val experiment_call(VM *v, int id, int argc, Val *args);
Obj *experiment_module(VM *v);

#endif
