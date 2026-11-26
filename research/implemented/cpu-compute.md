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
