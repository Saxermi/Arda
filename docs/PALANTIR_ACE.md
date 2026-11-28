# Palantír adaptive execution and mathematical introspection

## 0.22.2 Lazy graphs

`lazy.input(tensor)` captures a Float64 view and its shared buffer version.
All operands of a binary operation must have the same shape. The available
pointwise operations are `add`, `sub`, `mul`, `div`, `neg`, `sin` and `exp`.
`lazy.sum(node)` is a terminal reduction. `lazy.materialize(node[, options])`
plans and runs the graph. Existing eager expressions still execute immediately.

The graph retains each leaf, including its view layout. Arda writes through any
alias invalidate the version. A subsequent run checks all inputs before it
allocates the result. Build a new input node after writing to the buffer.
`lazy.file(path)` retains an open read-only descriptor for a validated Float64
`.atn` file. Atomic path replacement leaves that descriptor on the original file.
Arda checks size and modification time around a scan. Keep external inputs
immutable during execution. For memory mappings, that requirement also prevents
faults when an external process truncates a mapped file.

A stable topological schedule evaluates each shared node once per block. A
scratch slot can return to the free list after its final consumer. Packed
inputs can be borrowed, while strided inputs are packed in logical index order.
The fused recipes cover `a*b+c` and `sum(sin(x)*exp(-x))` with leaf inputs. The
former rounds multiplication and addition separately. The latter accumulates
Float64 products in source order in a `long double` accumulator. Results are
deterministic for fixed inputs and build. Platform `libm` and `long double` can
affect cross-platform bits. The graph limits are 64 levels and 128 distinct nodes.

## 0.22.3 Plan selection

`compute.plan("graph", node[, options])` considers `in_ram`, `tiled` and `fused`.
Storage is a separate field, with values `heap`, `mapped` and `file_blocks`.
`block_elements` is an upper bound on tiled and fused blocks. The planner can
shrink a block to fit the tensor budget. If the result itself exceeds the
heap tensor limit, reduce the graph before materializing it.

`compute.explain(plan)` reports workspace and output separately and explains
each candidate. Optional `calibration` supplies `flops_per_second`,
`specials_per_second`, `ram_bytes_per_second`, `disk_bytes_per_second` and
`overhead_seconds` per block. The model adds the block overhead to the maximum of
modeled CPU, RAM and file transfer times. Estimates can be absent when a needed
rate is missing. Logical traffic estimates describe the schedule and exclude
cache misses and OS read-ahead. The plan is reusable as long as its inputs and
budget remain valid. Changing the budget can make execution fail so that callers
can explicitly replan.

`compute.plan("apply", operator, vector)` wraps the native matrix-free or
canonical CSR apply. It accepts a matching packed Float64 vector and returns a
typed tensor. It accounts for the operator workspace and checks finite inputs
and results. The structured plan reports estimated arithmetic work for CSR and
diagonal operators. For other operators that field is `mun`.

## 0.22.4 Contracts and AMDE

`contract.require`, `contract.ensure` and `contract.assume` accept a Boolean and
an optional message. They evaluate at execution time and raise respectively
`precondition`, `postcondition` and `assumption` on false. They do not supply
compiler facts or rewrite executable code.

`mathdoc.getPlan(function[, spec])` describes a supported function and includes
`getCost` when bounds are available. Both work on a compile-time summary. A
specification can set `{"parameters":{"n":[2,4],"k":3}}`. Each supplied loop
parameter is an Int64 or a closed Int64 interval. Native half-open ranges use
constant strides, including negative ones. Nested loops can use independent
parameter and literal bounds. Counts cover source-level scalar expression
operations under numeric operands, excluding argument evaluation for literal
range bounds, VM dispatch, iteration and call overhead. Branches have intervals.
Short-circuiting and early returns can broaden the lower bound to zero.
Explicitly bounded direct recursion accepts `recursion_depth` up to 512.
That depth applies to the root-inclusive call tree and is a caller assumption.

Unsupported dynamic calls, effects, nonlocal values, variable loop strides and
indirect recursion return `mun`. An incomplete specification may leave only
the cost unknown. Static graph queries plan without reading the numeric file
payload. `compute.capabilities()` reports CPU and RAM execution, host permission
for file I/O under `ssd`, and reserved device and network entries. It does not
probe disk hardware or compete with the host for benchmarks.

`compute.execute(plan)` returns its numerical value. With
`compute.run(plan[, trace_limit])` the run object also exposes a report via
`compute.report(run)`. It includes measured elapsed time and when an estimate
exists, measured-minus-estimated time. The trace records starts and sizes of up
to 256 completed blocks, with one aggregate count for omitted blocks.
`mathdoc.getAlgorithm(run)` and `run.getAlgorithm()` name the algorithm that ran.
Run the [example](../examples/palantir_ace.arda) with
`./arda examples/palantir_ace.arda`.

## Cloud validation

Use `make test`, `make test-reference` and `make sanitize` for correctness.
The independent references cover seeded NumPy arrays and a SciPy CSR operator.
Tests exercise version invalidation, file replacement and corruption, memory
revalidation, plan choices, branch bounds and side-effect-free queries under GC
stress. A native SIGINT test stopped a file scan before it produced a value.
A separate test opened and reclaimed 200 file-owning graphs under a limit of
40 open descriptors. The example ran under GC stress. Format and generated
header checks passed.

| Cloud configurations | Runtime tests per build | Independent references per build |
| --- | ---: | ---: |
| GCC portable, OpenMP, Clang, GNU OpenMP/vector math/HWLOC/LTO, AArch64 under QEMU | 347 | 88 passed, 12 LAPACK-specific skips |
| GCC ASan/UBSan and OpenMP with ASan/UBSan | 347 | 88 passed, 12 LAPACK-specific skips |
| OpenBLAS/LAPACK with and without ASan/UBSan | 347 | 100 passed |

The sanitizer configurations enabled leak detection. References used NumPy
2.3.5, SciPy 1.17.0 and the CPU JAX suite. AArch64 ran under emulation.

### Performance and memory

The [benchmark](../benchmarks/ace_v2.py) used GCC 14.2 with the default
`-O2` portable build, pinned to one logical CPU on a Xeon Platinum 8573C.
Each in-memory workload used 4,194,304 Float64 elements. Seven measured
runs followed two warmups. Modes rotated between rounds and ran in
separate processes. Timing includes output allocation and execution.
Input creation, planning and verification are outside the execution
window. Every run checked the exact expected checksum.

| Workload and mode | Median time (ms) | Observed range (ms) | Intermediate storage | Median peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| `a*b+c`, 0.22.1 eager | 266.05 | 249.77–285.17 | 32 MiB | 162.0 |
| `a*b+c`, 0.22.1 explicit `compute.mul_add` | 19.60 | 16.68–24.80 | 0 | 130.0 |
| `a*b+c`, 0.22.4 graph fusion | 19.27 | 18.39–27.51 | 0 | 130.0 |
| `(a+b)*(a-b)`, 0.22.4 whole-array | 53.62 | 50.36–79.25 | 64 MiB | 162.0 |
| `(a+b)*(a-b)`, 0.22.4 tiled | 19.56 | 17.18–21.09 | 128 KiB | 98.0 |

The baseline was commit `cd2e637`. Graph planning took a median 6.4–7.0 µs
for the measured plans. The execution ranges overlap for explicit
and graph fusion. The tiled schedule used 512 times less intermediate
storage than the whole-array schedule on this workload. Intermediate
storage excludes inputs and output. Process peak RSS includes those
inputs, the output and the runtime.

The file experiment wrote and synced a 2 GiB Float64 payload. Arda
processed 4,096 blocks with a 128 MiB process address-space limit, no
configured swap, 512 KiB numeric workspace and 2.63 MiB measured peak
RSS. The fused sine-decay reduction took 2.98 s with uncontrolled page
cache state. Its relative error was `8.78e-13` against the independent
four-value analytic repetition evaluated with Python `math.fsum`.
The acceptance tolerance was `2e-11`. This measures a scan over a real
file under a process cap. A cold-storage bandwidth measurement needs
separate cache control.

```sh
taskset -c 2 python3 benchmarks/ace_v2.py --arda ./arda \
  --baseline /path/to/arda-0.22.1 --file-mib 2048
```

Use an available CPU ID and a GNU `time` binary. The Linux file check also
uses `prlimit`. The 100 GB fixture remains pending because the cloud
filesystem had about 4.5 GiB free.
