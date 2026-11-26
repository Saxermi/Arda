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
class and requested threads. Explicit kernel and block overrides bypass tuning. The tuning
median models the sampled kernel on this machine and is reported separately from the analytical
estimate. Tuning does not modify the VM-wide kernel or thread controls.
