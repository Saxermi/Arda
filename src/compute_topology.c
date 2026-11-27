#define _GNU_SOURCE
#include "compute_topology.h"
#include "compute_plan.h"
#include "cpu_kernels.h"
#include <unistd.h>

#if defined(_OPENMP) && !defined(ARDA_BLAS)
#include <omp.h>
#endif
#ifdef __linux__
#include <dirent.h>
#include <sched.h>
#endif
#ifdef ARDA_HWLOC
#include <hwloc.h>
#endif

static void field(VM *v, Obj *map, const char *name, Val value) {
    tableset(v, map, &map->as.map.tab, name, value);
}

#if defined(__linux__) || defined(ARDA_HWLOC)
static Val id_value(int id) {
    return id >= 0 ? integer(id) : nil();
}

static void cache_record(VM *v, Obj *caches, int level, const char *type, uint64_t bytes) {
    Obj *cache = newobj(v, O_MAP);
    field(v, cache, "level", integer(level));
    field(v, cache, "type", objval(string(v, type)));
    field(v, cache, "bytes", bytes <= INT64_MAX ? integer((int64_t)bytes) : nil());
    append(v, caches, objval(cache));
}

static void cpu_record(VM *v, Obj *cpus, int cpu, int core, int package, int node, Obj *caches) {
    Obj *record = newobj(v, O_MAP);
    field(v, record, "cpu", integer(cpu));
    field(v, record, "core", id_value(core));
    field(v, record, "package", id_value(package));
    field(v, record, "numa_node", id_value(node));
    field(v, record, "caches", objval(caches));
    append(v, cpus, objval(record));
}

#endif

#ifdef __linux__
struct ComputeAffinity {
    cpu_set_t initial;
    bool changed;
};

static bool read_affinity(cpu_set_t *set) {
    CPU_ZERO(set);
    return sched_getaffinity(0, sizeof *set, set) == 0;
}

static Val affinity_list(VM *v, const cpu_set_t *set) {
    Obj *result = list(v);
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++)
        if (CPU_ISSET(cpu, set))
            append(v, result, integer(cpu));
    return objval(result);
}

static long sysfs_number(const char *file, bool size) {
    FILE *stream = fopen(file, "r");
    if (!stream)
        return -1;
    long value = -1;
    char unit = 0;
    int fields = fscanf(stream, "%ld%c", &value, &unit);
    fclose(stream);
    if (fields < 1 || value < 0)
        return -1;
    long multiplier = !size ? 1 : unit == 'K' ? 1024L : unit == 'M' ? 1048576L : 1;
    return value <= LONG_MAX / multiplier ? value * multiplier : -1;
}

static int sysfs_node(int cpu) {
    char path[128];
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d", cpu);
    DIR *dir = opendir(path);
    if (!dir)
        return -1;
    int id = -1;
    struct dirent *entry;
    while ((entry = readdir(dir))) {
        char extra;
        if (sscanf(entry->d_name, "node%d%c", &id, &extra) == 1 && id >= 0)
            break;
        id = -1;
    }
    closedir(dir);
    return id;
}

static void sysfs_cpu(VM *v, Obj *cpus, int cpu) {
    char path[128];
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
    int core = (int)sysfs_number(path, false);
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
    int package = (int)sysfs_number(path, false);
    Obj *caches = list(v);
    for (int index = 0; index < 32; index++) {
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cache/index%d/level", cpu,
                 index);
        long level = sysfs_number(path, false);
        if (level < 0)
            break;
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cache/index%d/size", cpu, index);
        long bytes = sysfs_number(path, true);
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cache/index%d/type", cpu, index);
        FILE *stream = fopen(path, "r");
        if (stream) {
            char type[16];
            int fields = fscanf(stream, "%15s", type);
            fclose(stream);
            if (fields == 1 && bytes >= 0) {
                type[0] = (char)tolower((unsigned char)type[0]);
                cache_record(v, caches, (int)level, type, (uint64_t)bytes);
            }
        }
    }
    cpu_record(v, cpus, cpu, core, package, sysfs_node(cpu), caches);
}
#endif

#ifdef ARDA_HWLOC
static int hwloc_id(hwloc_obj_t object) {
    return object && object->os_index <= INT_MAX ? (int)object->os_index : -1;
}

static void hwloc_cpus(VM *v, Obj *cpus, hwloc_topology_t topology) {
    hwloc_bitmap_t allowed = hwloc_bitmap_dup(hwloc_topology_get_allowed_cpuset(topology));
    if (!allowed)
        return;
    hwloc_bitmap_t binding = hwloc_bitmap_alloc();
    if (binding && !hwloc_get_cpubind(topology, binding, HWLOC_CPUBIND_THREAD))
        hwloc_bitmap_and(allowed, allowed, binding);
    hwloc_bitmap_free(binding);
    int cpu = -1;
    while ((cpu = hwloc_bitmap_next(allowed, cpu)) >= 0) {
        hwloc_obj_t pu = hwloc_get_pu_obj_by_os_index(topology, (unsigned)cpu);
        if (!pu)
            continue;
        int node = -1;
        hwloc_obj_t object = NULL;
        while ((object = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_NUMANODE, object))) {
            if (object->cpuset && hwloc_bitmap_isset(object->cpuset, (unsigned)cpu)) {
                node = hwloc_id(object);
                break;
            }
        }
        Obj *caches = list(v);
        for (object = pu->parent; object; object = object->parent) {
            if (!hwloc_obj_type_is_cache(object->type))
                continue;
            const struct hwloc_cache_attr_s *cache = &object->attr->cache;
            const char *type = cache->type == HWLOC_OBJ_CACHE_DATA          ? "data"
                               : cache->type == HWLOC_OBJ_CACHE_INSTRUCTION ? "instruction"
                                                                            : "unified";
            cache_record(v, caches, (int)cache->depth, type, cache->size);
        }
        int core = hwloc_id(hwloc_get_ancestor_obj_by_type(topology, HWLOC_OBJ_CORE, pu));
        int package = hwloc_id(hwloc_get_ancestor_obj_by_type(topology, HWLOC_OBJ_PACKAGE, pu));
        cpu_record(v, cpus, cpu, core, package, node, caches);
    }
    hwloc_bitmap_free(allowed);
}
#endif

Val compute_topology(VM *v) {
    Obj *cpus = list(v);
    const char *source = "unavailable";
#ifdef ARDA_HWLOC
    hwloc_topology_t topology;
    if (!hwloc_topology_init(&topology)) {
        hwloc_topology_set_icache_types_filter(topology, HWLOC_TYPE_FILTER_KEEP_ALL);
        if (!hwloc_topology_load(topology)) {
            hwloc_cpus(v, cpus, topology);
            if (cpus->as.list.n)
                source = "hwloc";
        }
        hwloc_topology_destroy(topology);
    }
#endif
#ifdef __linux__
    cpu_set_t affinity;
    if (!cpus->as.list.n && read_affinity(&affinity)) {
        source = "linux";
        for (int cpu = 0; cpu < CPU_SETSIZE; cpu++)
            if (CPU_ISSET(cpu, &affinity))
                sysfs_cpu(v, cpus, cpu);
    }
#endif
    Obj *result = newobj(v, O_MAP);
    field(v, result, "source", objval(string(v, source)));
    field(v, result, "cpus", objval(cpus));
    return objval(result);
}

Val compute_affinity(VM *v, int argc, Val *args) {
#ifdef __linux__
    cpu_set_t current;
    if (!read_affinity(&current))
        vmerror_kind(v, "backend", "unable to read calling thread CPU affinity");
    if (!argc)
        return affinity_list(v, &current);
    struct ComputeAffinity *state = v->compute_affinity;
    cpu_set_t initial = state && state->changed ? state->initial : current, desired = initial;
    if (args[0].t != V_NIL) {
        if (!isobj(args[0], O_LIST) || !args[0].o->as.list.n)
            vmerror_kind(v, "type", "compute affinity needs a nonempty CPU list or mun");
        Obj *ids = args[0].o;
        CPU_ZERO(&desired);
        for (int i = 0; i < ids->as.list.n; i++) {
            Val id = ids->as.list.a[i];
            if (id.t != V_INT || id.i < 0 || id.i >= CPU_SETSIZE || !CPU_ISSET((int)id.i, &initial))
                vmerror_kind(v, "domain", "requested CPU is outside the initial affinity");
            CPU_SET((int)id.i, &desired);
        }
    }
    if (sched_setaffinity(0, sizeof desired, &desired))
        vmerror_kind(v, "backend", "unable to set calling thread CPU affinity");
    cpu_set_t actual;
    if (!read_affinity(&actual) || !CPU_EQUAL(&desired, &actual)) {
        sched_setaffinity(0, sizeof current, &current);
        vmerror_kind(v, "backend", "operating system did not accept the complete CPU affinity");
    }
    if (!state)
        state = v->compute_affinity = mem(sizeof *state);
    state->initial = initial;
    state->changed = args[0].t != V_NIL;
    v->compute_thread_cap = state->changed ? CPU_COUNT(&actual) : 0;
    compute_clear_tuning(v);
    return affinity_list(v, &actual);
#else
    (void)argc;
    (void)args;
    vmerror_kind(v, "backend", "calling thread CPU affinity requires Linux");
#endif
}

void compute_release_affinity(VM *v) {
#ifdef __linux__
    struct ComputeAffinity *state = v->compute_affinity;
    if (state && state->changed)
        sched_setaffinity(0, sizeof state->initial, &state->initial);
    free(state);
#else
    (void)v;
#endif
}

Val compute_first_touch(VM *v, Val value) {
    int shape[8];
    int rank = typed_dimensions(v, value, shape);
    Obj *tensor = typed_new(v, DT_F64, rank, shape);
    size_t bytes = tensor->as.tensor.n * sizeof(double);
    unsigned char *data = typed_element(&tensor->as.tensor, 0);
#if defined(_OPENMP) && !defined(ARDA_BLAS)
    long page = sysconf(_SC_PAGESIZE);
    if (page > 0 && bytes / (size_t)page >= 128) {
        size_t width = (size_t)page, offset = (uintptr_t)data % width;
        size_t pages = (bytes + offset + width - 1) / width;
        int team = cpu_parallel_team(v, (int)(pages / 64));
        if (team > 1) {
#pragma omp parallel num_threads(team) default(none) firstprivate(data, bytes, pages, width, offset)
            {
                size_t worker = (size_t)omp_get_thread_num(), count = (size_t)omp_get_num_threads();
                size_t first = pages * worker / count, last = pages * (worker + 1) / count;
                size_t begin = first ? first * width - offset : 0;
                size_t end = last == pages ? bytes : last * width - offset;
                memset(data + begin, 0, end - begin);
            }
            return objval(tensor);
        }
    }
#endif
    memset(data, 0, bytes);
    return objval(tensor);
}
