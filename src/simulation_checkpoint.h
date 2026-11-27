#ifndef ARDA_SIMULATION_CHECKPOINT_H
#define ARDA_SIMULATION_CHECKPOINT_H

#include "arda.h"

enum { CHECKPOINT_WORDS = 16 };
typedef enum { CHECKPOINT_RK45 = 1, CHECKPOINT_HEAT } CheckpointKind;

uint64_t checkpoint_real(VM *v, Val value);
double checkpoint_double(uint64_t word);
Val checkpoint_save_state(VM *v, Val path, CheckpointKind kind, Val state,
                          const uint64_t words[CHECKPOINT_WORDS], Val budget);
Val checkpoint_load_state(VM *v, Val path, CheckpointKind kind, uint64_t words[CHECKPOINT_WORDS]);
Obj *checkpoint_module(VM *v);
Val checkpoint_call(VM *v, int id, int argc, Val *args);

#endif
