#ifndef ARDA_COMPUTE_TOPOLOGY_H
#define ARDA_COMPUTE_TOPOLOGY_H

#include "arda.h"

enum { COMPUTE_AFFINITY_WORDS = 16 };
bool compute_tuning_placement(uint64_t mask[COMPUTE_AFFINITY_WORDS]);
Val compute_topology(VM *v);
Val compute_affinity(VM *v, int argc, Val *args);
void compute_release_affinity(VM *v);
Val compute_first_touch(VM *v, Val shape);

#endif
