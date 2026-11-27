#ifndef ARDA_TENSOR_FILE_H
#define ARDA_TENSOR_FILE_H

#include "typed_tensor.h"

typedef struct {
    TypedTensor layout;
    size_t bytes;
    int fd;
} TensorFile;

void tensor_file_require_little_endian(VM *v);
bool tensor_file_size(const TypedTensor *layout, size_t *size);
bool tensor_file_fits_budget(Val limit, size_t size);
int tensor_file_open(const char *path, TensorFile *file);
bool tensor_file_read(const TensorFile *file, size_t start, size_t count, void *data);
bool tensor_file_write(int fd, const void *data, size_t bytes);
bool tensor_file_patch(int fd, const TypedTensor *layout, size_t start, size_t count,
                       const void *data);
int tensor_publish_file(const char *path, bool (*writer)(int, const void *), const void *context);
int tensor_publish_payload(const char *path, const TypedTensor *layout,
                           bool (*writer)(int, const void *), const void *context);
Val tensor_import_npy(VM *v, Val input, Val output, Val budget);
Val tensor_import_delimited(VM *v, Val input, Val output, Val shape, Val options);
Val tensor_file_info(VM *v, Val source);
Val tensor_write_block(VM *v, Val output, Val start, Val tile);
Val tensor_truncate_output(VM *v, Val output, Val count);

#endif
