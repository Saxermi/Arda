#ifndef ARDA_MATHDOC_REPORT_H
#define ARDA_MATHDOC_REPORT_H

#include "arda.h"

Val mathdoc_describe(VM *v, Val target, Val level);
Val mathdoc_export(VM *v, int argc, Val *args);

#endif
