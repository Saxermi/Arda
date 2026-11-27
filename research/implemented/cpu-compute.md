# Bounded and explainable CPU compute

## 0.18.4 bounded kernels

`compute.tiled_matmul(A, B[, block])` accepts a Float64 matrix and a compatible Float64 matrix or
vector. The default tile width is 64, with supported overrides from 1 to 256. A strided matrix RHS
may use one packed tile from the tensor workspace pool. The kernel visits the terms of each dot
product in increasing index order and rounds multiplication and addition separately. Native teams are
optional and divide disjoint rows. A packed RHS tile is reused in a serial path.

`compute.stream_stats(path[, block[, {"prefetch":true}]])` retains ordered online moments under
prefetch. A single POSIX worker reads the next chunk using [pread](https://pubs.opengroup.org/onlinepubs/9799919799/functions/pread.html)
while the caller processes the current chunk. The plan reports the effective mode and pooled
workspace. Files of at most one chunk need one buffer. The reported I/O bytes are logical payload
bytes and may come from the OS page cache.

The explicit Float64 regions `compute.axpy(x, y, alpha)`, `compute.mul_add(a, b, c)` and
`compute.fused_products(a, b, c, d, e)` evaluate `alpha*x+y`, `a*b+c` and `(a*b+c*d)+e`
in one pass on same-shaped tensors. Contracting FMA is disabled. `compute.map_reduce(map, x)`
supports `"sum"`, `"sum_squares"` and `"sum_sin"`, rounds the mapped values as Float64, then
accumulates in traversal order in a C `long double`, like Float64 tensor sum. These regional
kernels are explicit entry points. General automatic expression rewrites are outside this scope.

## 0.18.5 ACE v1 plans

```arda
import compute as c
let plan = c.plan("matmul", A, B, {"kernel":"tiled", "block":64, "threads":1})
print(c.explain(plan))
let result = c.execute(plan, A, B)
print(c.last_run())
```

Plans hold shape, strides, storage class and execution choices. They can be reused with compatible
inputs and read their current values. An execution rechecks its output and Arda workspace against
the current memory budget. `"auto"` selects a packed native kernel for eligible matrices in
non-BLAS builds and the strict tiled path otherwise. Explicit SIMD/FMA and CBLAS kernel names
are subject to build and CPU support. CBLAS controls its own workspace and threads, so Arda reports
its opaque block and team size as `mun`. The native reference backend also does not expose a
configurable block. Native thread requests are capped by the worker runtime and workload size.

The [Roofline model](https://doi.org/10.1145/1498765.1498785) motivates the optimistic lower bound
`max(flops/compute_rate, logical_ram_bytes/ram_rate, logical_file_bytes/disk_rate)`. Optional
`calibration` fields are `flops_per_second`, `ram_bytes_per_second`, `disk_bytes_per_second` and
`overhead_seconds`. Supply measured rates for each active tier to get a time estimate. The
report adds the supplied fixed overhead separately. Logical bytes count operands and output once.
The file tier estimates cold mapped-input traffic and does not measure cache misses or disk reads.

`{"tune":true}` times two tiled block widths and the default packed native kernel when eligible.
Tuning accepts nonempty heap inputs with at most 64 million multiply-add terms. It runs one
warm-up and three timed trials per candidate with a reused output and includes clearing that
output. A per-VM cache retains the last eight measurements keyed by shape, strides, storage
class, eligible candidate set, effective candidate teams, native dispatch and CPU affinity. Explicit kernel and block overrides bypass tuning. The tuning
median models the sampled kernel on this machine and is reported separately from the analytical
estimate. Tuning does not modify the VM-wide kernel or thread controls.

## 0.18.6 topology and backend comparison

`compute.topology()` reports the caller's visible logical CPUs with core, package, NUMA node and
cache sizes where available. The default Linux build reads the caller's affinity and sysfs.
`make HWLOC=1` optionally links hwloc 2.x for topology discovery. Missing topology fields use
`mun`. Linux affinity controls use the platform CPU-set limit. `compute.affinity([id, ...])`
pins the calling thread within its initial mask and caps Arda-managed native teams during
the pin. Use `compute.affinity(mun)` to restore the original mask. OpenMP controls worker
placement through `OMP_PLACES` and `OMP_PROC_BIND`. `OMP_THREAD_LIMIT` can match the CPU quota of a
container whose visible CPU count exceeds its quota. CBLAS uses its provider's settings.
[hwloc](https://www.open-mpi.org/projects/hwloc/) · [Linux affinity](https://man7.org/linux/man-pages/man2/sched_setaffinity.2.html)

`compute.first_touch(shape)` makes an ordinary zeroed Float64 tensor. For large allocations
with native OpenMP teams enabled, workers write disjoint ranges at page boundaries. Placement
depends on fresh page faults, the OS memory policy, allocator reuse and worker affinity.
[Linux NUMA memory policy](https://docs.kernel.org/admin-guide/mm/numa_memory_policy.html)

`compute.compare(A, B, options)` compares two independent plan options. `first` and `second` take
the same fields as `compute.plan("matmul", ...)`. The top-level options are `warmup` (0–5, default
1), `repeats` (1–9, default 5), `atol` (default 1e-12) and `rtol` (default 1e-10). It alternates
which backend runs first, reuses one output per backend and includes output clearing in both timings.
It reports the median and actual execution choices for each backend. The finite-output check uses
`abs(x-y) <= atol + rtol * max(abs(x),abs(y))` per element. Nonfinite results fail the check and
give `mun` for maximum absolute error. Any caller-requested tuning occurs before comparison timing.

```arda
import compute as c
let report = c.compare(A, B, {"first":{"kernel":"tiled","block":32},
                              "second":{"kernel":"tiled","block":64},
                              "warmup":2, "repeats":7})
print(report["equivalent"], report["max_abs_error"])
print(report["first"], report["second"])
```

The default path checks tensor type and layout once. Typed kernels use native buffers.
AMDE collection runs during compilation, and trace, unit and AD objects require explicit
use of their corresponding APIs. The hot loops for these ordinary Float64 kernels have no
per-element AMDE, tracing, unit or AD branches. See the existing LTO and PGO build recipes
for compiler-specific controls. [Build recipes](transforms-and-kernels.md#0182-controlled-cpu-parallelism)

### Implementation provenance

The tiled product computes the ordinary GEMM or GEMV definition with ascending inner index.
Work is `O(mkn)` for GEMM and `O(mk)` for GEMV. A non-unit-stride RHS can use at most
`8 * min(k, b) * min(n, b)` bytes of panel scratch for tile width `b`. Fused expressions and mapped
sums are linear in the element count with constant scratch beyond the output. The stream reduction
retains the earlier shifted Welford update and uses one or two bounded buffers. Implementations
are in [compute_ops.c](../../src/compute_ops.c) and [tensor_storage.c](../../src/tensor_storage.c).
ACE and topology controls are in [compute_plan.c](../../src/compute_plan.c) and
[compute_topology.c](../../src/compute_topology.c).

Recent work on [locality-aware composition](https://doi.org/10.1145/3729292) motivates revisiting
general graph composition after Arda has dependency and effect contracts. The implemented
regions are hand-written kernels for fixed, pure numerical expressions. The implementations
use original code and the POSIX, OpenMP, CBLAS and optional hwloc interfaces.

### Local validation and measurements

The test host exposed an Intel Xeon Platinum 8573C, three logical CPUs on one NUMA node and a
two-CPU cgroup quota. Arda used GCC 14.2.0 with `-std=c11 -O2`. The reference packages were
NumPy 2.3.5 and SciPy 1.17.0. Timed regions exclude input creation and include output allocation
unless stated otherwise. The fusion and scaling harness records raw timing samples and checks the
result independently. [Benchmark harness](../../benchmarks/cpu_compute.py)

| Workload | Baseline | Changed path | Result check |
| --- | ---: | ---: | --- |
| 10 million Float64 elements, `a*b+c*d+e` | 1,311.521 ms | 79.508 ms fused | Both sums were exactly 320,000,000 |
| 64 MiB tensor-file statistics, warm page cache | 83.329 ms serial | 80.094 ms prefetched | Count, mean and population variance matched NumPy |
| 768 × 768 GEMM | 30.281 ms, one worker | 16.502 ms, two workers | Max absolute error 2.70e-13 against NumPy |
| 4096 × 2048 GEMV | 5.172 ms, one worker | 2.665 ms, two workers | Max absolute error 4.26e-13 against NumPy |
| 262,144-row tridiagonal CSR SpMV | 0.632 ms SciPy | 1.725 ms Arda, one worker | All outputs matched exactly |

These are medians of seven samples. Single-worker fusion was pinned to CPU 0. Large product and
stream tests used CPUs 0 and 2. Packed GEMM with 128 rows and 128 columns and matching-size GEMV
stayed serial for every request. Large GEMM and GEMV used two workers for requests of 2, 4 or 8
under `OMP_THREAD_LIMIT=2`. Arda's CSR SpMV stayed serial across all requests. The dense
comparison required `rtol=1e-10` and `atol=1e-11` for every element.

The fused expression removes three 80,000,000-byte intermediate allocations. Median whole-process
peak RSS was 704,324 KiB for the plain expression and 469,956 KiB for the fused path.
The tensor budget peak, which includes fixture construction, was 720,000,000 bytes and
640,000,000 bytes respectively. Those are separate measurements. The statistics run used
65,536-element chunks and 512 KiB serial or 1 MiB double-buffered scratch. The small median
prefetch gain reflects a warm file test on this host. The serial and prefetched ranges overlapped.

The single-threaded release gate compared unchanged kernels against 0.17.4 and 0.18.3.
Eleven rotated rounds on CPU 0 covered GEMV, SpMV, AXPY, reductions, libm, copies and VM loops.
Legacy dense GEMM matched NumPy bit-for-bit on the original binary-fraction fixture.

| Square GEMM order | 0.17.4 | 0.18.3 | 0.18.6 portable | 0.18.6 OpenBLAS | NumPy 2.3.5 | Eigen 3.4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.347 ms | 0.114 ms | 0.118 ms | 0.060 ms | 0.060 ms | 0.053 ms |
| 512 | 23.473 ms | 9.001 ms | 8.897 ms | 3.361 ms | 3.311 ms | 3.893 ms |
| 1024 | 173.475 ms | 80.981 ms | 79.734 ms | 25.368 ms | 23.857 ms | 25.641 ms |

The 0.17.4 comparison ran separately with the same fixture, CPU pin and compiler optimization
level. OpenBLAS 0.3.29 was linked for the external Arda path. The fastest available backend
depends on shape, accuracy requirements and build. The targeted gate met the 10% budget
against 0.17.4. A short packed-copy sample was 9.5% slower than 0.17.4 and 10.1% slower than
0.18.3 at approximately 0.03 ms. A focused rerun used 21 rotated rounds with one discarded
warm-up batch and three batches of 60 copies each. The medians were 22.69, 22.76 and 22.37 µs
for 0.17.4, 0.18.3 and 0.18.6 respectively. The short-copy slowdown did not reproduce.
Other unchanged workloads were within the 10% target in the interleaved run. [Baseline harness](../../benchmarks/cpu_baseline.py)

The AMDE benchmark ran eleven interleaved processes per mode. Dense GEMM medians were
16.198 ms with metadata disabled, 17.850 ms in compact mode and 16.882 ms for 0.18.3.
These wall times include startup and compilation. Runtime inspection confirms that the ordinary
Float64 kernel bodies contain no per-element AMDE, trace, unit or AD hooks. The timing
comparison doesn't isolate those features individually. [AMDE benchmark](../../benchmarks/amde_bench.py)

The native matrix passed 254 runtime tests in each of eight configurations. Each
OpenBLAS/LAPACK configuration also passed all 69 independent reference tests. The other
configurations passed 60, with nine LAPACK-specific skips. ASan and UBSan included leak
checks. Clang x86-64 and AArch64 under QEMU passed the same ten focused compute tests.
A separate GCC PGO build collected profiles with the runtime suite, then passed all 254
runtime tests again with the generated profiles.
Native cross-node page placement, physical SSD throughput, hardware counters and
laptop energy or thermal measurements remain hardware-dependent validation items.

```sh
make
python3 benchmarks/cpu_compute.py --suite fusion --arda ./arda --cpus 0
python3 benchmarks/cpu_compute.py --suite stream --arda ./arda --cpus 0 2
make OPENMP=1
python3 benchmarks/cpu_compute.py --suite scaling --arda ./arda --cpus 0 2 --thread-limit 2
```

Use the appropriate CPU IDs for the test machine. The Python harness needs NumPy, SciPy
for scaling and GNU `time` for the fusion RSS.
