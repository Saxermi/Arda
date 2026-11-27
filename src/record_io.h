#ifndef ARDA_RECORD_IO_H
#define ARDA_RECORD_IO_H

#include "tensor_file.h"
#include "sha256.h"

uint64_t record_u64(const unsigned char *p);
void record_put_u64(unsigned char *p, uint64_t word);
typedef struct {
    int fd;
    size_t size, limit, used;
    Sha256 hash;
    bool checksummed;
    unsigned char buffer[4096];
} RecordOutput;

bool record_emit(RecordOutput *out, const void *data, size_t bytes);
bool record_text(RecordOutput *out, const char *text);
bool record_flush(RecordOutput *out);
size_t record_budget(VM *v, Val value, size_t limit);
_Noreturn void record_error(VM *v, const char *operation, int error);

#endif
