#include "activation_store.h"

#include <sys/stat.h>
#include <unistd.h>

static uint64_t saturating_add(uint64_t current, uint64_t delta) {
    return delta > (uint64_t)INT64_MAX - current ? INT64_MAX : current + delta;
}

ADOffloadPlan *offload_plan(VM *v, Val value) {
    if (!isobj(value, O_OFFLOAD_PLAN))
        vmerror_kind(v, "type", "expected an activation offload plan");
    return value.o->as.offload_plan;
}

Val offload_plan_new(VM *v, Val directory, Val budget, Val block) {
    vm_require_file_access(v);
    const char *path = asstr(v, directory);
    struct stat info;
    if (!*path || stat(path, &info) || !S_ISDIR(info.st_mode))
        vmerror_kind(v, "io", "activation offload directory must exist");
    if (budget.t != V_INT || budget.i < 0 || (uint64_t)budget.i > SIZE_MAX)
        vmerror_kind(v, "disk_budget", "offload budget must be a nonnegative byte count");
    if (block.t != V_INT || block.i < 1 || block.i > ARDA_MAX_ELEMENTS)
        vmerror_kind(v, "shape", "offload block must be a positive tensor element count");

    size_t length = strlen(path) + 1;
    if (length > SIZE_MAX - sizeof(ADOffloadPlan))
        vmerror_kind(v, "io", "offload directory name is too long");
    size_t allocation = sizeof(ADOffloadPlan) + length;
    Obj *out = newobj(v, O_OFFLOAD_PLAN);
    ADOffloadPlan *plan = mem(allocation);
    plan->references = 1;
    plan->limit = (size_t)budget.i;
    plan->block_bytes = (size_t)block.i * sizeof(double);
    memcpy(plan->directory, path, length);
    out->as.offload_plan = plan;
    account(v, out, 0, allocation);
    return objval(out);
}

void offload_plan_release(ADOffloadPlan *plan) {
    if (plan && !--plan->references)
        free(plan);
}

void offload_file_release(Buffer *buffer) {
    ADOffloadPlan *plan = buffer->offload;
    if (!plan)
        return;
    close(buffer->spill_fd);
    plan->live_bytes -= buffer->allocation_bytes;
    plan->live_files--;
    offload_plan_release(plan);
    buffer->offload = NULL;
}

static void field(VM *v, Obj *out, const char *name, uint64_t value) {
    tableset(v, out, &out->as.map.tab, name, integer((int64_t)value));
}

Val offload_stats(VM *v, const ADOffloadPlan *plan) {
    Obj *out = newobj(v, O_MAP);
    field(v, out, "disk_budget_bytes", plan->limit);
    field(v, out, "live_disk_bytes", plan->live_bytes);
    field(v, out, "peak_disk_bytes", plan->peak_bytes);
    field(v, out, "live_files", plan->live_files);
    field(v, out, "files_written", plan->written_files);
    field(v, out, "bytes_written", plan->written_bytes);
    field(v, out, "bytes_read", plan->read_bytes);
    field(v, out, "read_chunks", plan->read_chunks);
    field(v, out, "block_elements", plan->block_bytes / sizeof(double));
    field(v, out, "vm_checkpoint_replay_operations", v->ad_replay_operations);
    return objval(out);
}

void offload_add_read(ADReadSpec *spec, const Buffer *buffer) {
    if (!buffer || !buffer->offload || !buffer->bytes)
        return;
    for (size_t i = 0; i < spec->count; i++)
        if (spec->buffers[i] == buffer)
            return;
    if (spec->count < AD_READ_SLOTS)
        spec->buffers[spec->count++] = buffer;
    size_t width = buffer->offload->block_bytes;
    if (width > buffer->bytes)
        width = buffer->bytes;
    if (width > spec->block_bytes)
        spec->block_bytes = width;
}

size_t offload_read_bytes(ADReadSpec spec) {
    return spec.count * spec.block_bytes;
}

void offload_read_start(VM *v, ADReadCache *cache, ADReadSpec spec) {
    size_t bytes = offload_read_bytes(spec);
    if (!bytes)
        return;
    unsigned char *work = tensor_workspace_acquire(v, bytes);
    if (!work) {
        gcfull(v);
        vmerror_kind(v, "memory_budget",
                     "offloaded activation read windows exceed the tensor memory budget");
    }
    *cache = (ADReadCache){.slot_bytes = spec.block_bytes, .slots = spec.count};
    v->ad_read_memory = work;
    v->ad_read_cache = cache;
}

void offload_read_finish(VM *v) {
    if (v->ad_read_memory)
        tensor_workspace_release(v, v->ad_read_memory);
    v->ad_read_memory = NULL;
    v->ad_read_cache = NULL;
}

double offload_read_float64(VM *v, const Buffer *buffer, size_t byte_offset) {
    double value;
    ADReadCache *cache = v->ad_read_cache;
    if (!cache || !buffer->offload) {
        memcpy(&value, buffer->data + byte_offset, sizeof value);
        return value;
    }
    ADOffloadPlan *plan = buffer->offload;
    size_t offset = byte_offset - byte_offset % plan->block_bytes;
    size_t slot = 0;
    for (; slot < cache->slots; slot++)
        if (cache->entries[slot].buffer == buffer && cache->entries[slot].offset == offset)
            break;
    if (slot == cache->slots) {
        slot = cache->next;
        cache->next = (slot + 1) % cache->slots;
        size_t count = buffer->bytes - offset;
        if (count > plan->block_bytes)
            count = plan->block_bytes;
        unsigned char *window = v->ad_read_memory + slot * cache->slot_bytes;
        if (count > cache->slot_bytes || !offload_read_chunk(buffer, offset, window, count)) {
            int cause = count > cache->slot_bytes ? EINVAL : errno;
            gcfull(v);
            vmerror_kind(v, "io", "cannot read offloaded activation: %s", strerror(cause));
        }
        cache->entries[slot].buffer = buffer;
        cache->entries[slot].offset = offset;
        plan->read_bytes = saturating_add(plan->read_bytes, count);
        plan->read_chunks = saturating_add(plan->read_chunks, 1);
    }
    memcpy(&value, v->ad_read_memory + slot * cache->slot_bytes + byte_offset - offset,
           sizeof value);
    return value;
}

void offload_record_replay(VM *v, size_t elements, size_t operations) {
    uint64_t count = operations && elements > (uint64_t)INT64_MAX / operations
                         ? INT64_MAX
                         : (uint64_t)elements * operations;
    v->ad_replay_operations = saturating_add(v->ad_replay_operations, count);
}
