#ifndef ARDA_METADATA_H
#define ARDA_METADATA_H

#include "record_io.h"

enum { METADATA_BYTES = 1024 * 1024 };
bool metadata_binary(RecordOutput *out, Val value);
bool metadata_json(RecordOutput *out, Val value);
bool metadata_decode(VM *v, const unsigned char *data, size_t size, Val *value);

#endif
