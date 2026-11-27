#ifndef ARDA_TENSOR_FILE_H
#define ARDA_TENSOR_FILE_H

#include "typed_tensor.h"

void tensor_file_require_little_endian(VM *v);
bool tensor_file_size(const TypedTensor *layout, size_t *size);
bool tensor_file_fits_budget(Val limit, size_t size);
bool tensor_file_write(int fd, const void *data, size_t bytes);
int tensor_publish_payload(const char *path, const TypedTensor *layout,
                           bool (*writer)(int, const void *), const void *context);
Val tensor_import_npy(VM *v, Val input, Val output, Val budget);
Val tensor_import_delimited(VM *v, Val input, Val output, Val shape, Val options);

#endif
