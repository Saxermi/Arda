# Arda 0.22.0–0.22.1 — Language and runtime foundations

*Development codename: Palantír*

## 0.22.0 Language contracts

| Feature | Contract |
| --- | --- |
| Tuples | `(a,b)`, `(a,)` and `()` are immutable containers. They support zero-based and negative indexing, slices, iteration and structural equality. A tuple can contain objects that remain mutable. |
| Sets | `{a,b}`, `set()` and `set(source)` build immutable sets. Equality ignores order. Iteration preserves the order of first appearance. The hash index uses Arda equality, including equivalent Int64 and Float64 keys. `{}` remains an empty map. |
| Membership | `value in source` checks lists, tuples, sets, map keys, string substrings and Int64 ranges. It leaves iterators untouched. |
| Iterators | `iter(source)` makes a cursor. Calling it on a cursor returns that cursor. `next(cursor[,default])` consumes one value, including nil. At exhaustion it returns the supplied default or raises an error with code `stop_iteration`. `for` and comprehensions consume the same cursor when passed one. |
| Generators | A function containing `yield` returns a single-use generator. Its body and omitted defaults start on the first advance. `yield` suspends the current frame, including loop state and handlers. Falling off the function or returning ends iteration. An uncaught error ends it and propagates. |
| Named arguments | `f(a, scale: 2)` binds by parameter name. Positional arguments come first, supplied expressions are evaluated in source order, and omitted defaults run in declaration order. Duplicate, missing and unknown arguments raise `arity`. Arda functions, bound methods and constructors support the syntax. |
| Type annotations | `caro f(x: Float64) -> Float64:` records optional named types. `signature(f)` returns parameter and return metadata. Annotations are descriptive and leave runtime argument values and evaluation unchanged. |
| Formatted strings | `f"sum = {expression}"` evaluates fields left to right using the existing `str` conversions. Use `{{` and `}}` for literal braces. Strings are single-line and support expression interpolation. |
| Context managers | `with manager as value:` resolves the exit method, then calls `__enter__()` and binds its result. On fallthrough, return, break or continue that exits the scope, it calls `__exit__(nil)`. On an error it calls `__exit__(error)` with the caught error map. A truthy result suppresses that error. |

Each cursor over `range` or `tensor.blocks` has an independent position.
`close_iterator(cursor)` discards a cursor's position and reference to the
source. A held `tensor.blocks` source can still be used to create new cursors.
Use `tensor.close_blocks(source)` to release its file descriptor explicitly.
A generator is itself an iterator. Closing it ends that continuation
and frees its suspended state. `yield` inside a `with` body is rejected so
closing a suspended generator cannot silently skip a user-defined exit hook.
`set(source)` accepts synchronous sources. To build a set from a generator,
consume the generator with a `for` or comprehension first. Formatted strings
use expression fields without a separate format-specification language.
[Runnable example](../examples/palantir_runtime.arda) · [Regression tests](../tests/test_palantir_runtime.py)

## 0.22.1 VM controls

| Option | Contract |
| --- | --- |
| `--max-instructions=N` | Stop after N bytecode instructions per evaluation. A single native operation can do substantial work before returning to dispatch. |
| `--max-stack=N` | Cap operand slots, including locals stored on the operand stack. The default ceiling is 1,000,000 slots. |
| `--max-frames=N` | Cap active VM frames, including a resumed generator. The default ceiling is 512. |
| `--max-vm-bytes=N` | Check accounted managed bytes at instruction boundaries. Collect before rejecting live state over the limit. Native scratch storage, frontend memory and process RSS have separate accounting. A single instruction can temporarily exceed the cap. |
| `--interruptible` | Observe SIGINT at the next dispatch boundary. Native work already in progress can delay the stop. |
| `--no-file-io` | Deny script-level file operations and filesystem imports. Embedded standard-library imports and the host's initial program read remain available. |
| `--gc-profile` | Report sweep count, total and maximum sweep duration, maximum objects visited in a sweep and peak accounted VM bytes at instruction boundaries. |

Instruction, managed-memory and interrupt termination end the current
evaluation without running user callbacks. Stack-limit errors can be caught.
The controls operate inside the interpreter. Process limits and isolation are
the system boundary for workloads that require them.

The compiler assigns parameters and locals to storage per lexical binding.
A binding that can be reached from a nested function lives in the shared
heap environment for its scope. The other bindings use operand-stack
slots, including when the same function creates a closure. Closure reads
can pass through several nested functions and can see a later declaration
in an enclosing scope. The compiler preserves each reachable binding when
the answer depends on whether that declaration has run. A scope allocates
a heap environment only when its bindings need it. Exception handlers
restore the enclosing environment and clear discarded operand slots. Attribute lookup uses table layout
stamps and fresh values, so a method overridden on the instance is found on
the following lookup. An immediate instance method call carries the resolved
function and receiver on the operand stack. An attribute evaluated as a
value still produces a bound method. Integer less-than and its consuming
branch can share one instruction. A suspended generator roots its bytecode
and locals until it finishes, is closed, or becomes unreachable.
[Compiler](../src/compiler.c) · [VM](../src/vm.c) · [Collector](../src/gc.c)

## Profiling and validation

The profiling harness uses GNU gprof with separately instrumented binaries and
GNU time for peak RSS on normal binaries. It compares identical source and
checks every expected output before recording timing. Builds share the same
optimization flags. Each workload runs in two alternating orders with two
warmups and seven measured executions per binary. The calls workload
contains 3.5 million calls with a default parameter, the attribute workload
contains 2.5 million instance calls and the allocation workload creates
1.5 million short-lived lists and maps. The numerical control runs 24 dense
Float64 products at 256 × 256.
[Profiling harness](../benchmarks/palantir_vm.py)

```sh
make -j2 BUILD=build/profile CFLAGS='-std=c11 -O2 -g -pg -Wall -Wextra -Wpedantic' LDFLAGS=-pg
make -j2 BUILD=build/plain
python3 benchmarks/palantir_vm.py --baseline /path/to/0.21.4/arda \
  --arda build/plain/arda --cpu 0
```

### Measured results

The baseline is the merged 0.21.4 source at commit `8ac8a6f`. Both builds used
GCC 14.2 with `-O2 -g`, the portable numerical backend and one pinned logical
CPU on an Intel Xeon Platinum 8573C cloud host. Wall time includes process
startup. Peak RSS is the highest of the seven measured processes.

| Workload | 0.21.4 median [range], ms | 0.22.1 median [range], ms | Peak RSS, KiB before / after |
| --- | ---: | ---: | ---: |
| 12 million local-variable loop iterations | 1,393.65 [1,366.82, 1,416.54] | 688.74 [680.70, 763.05] | 1,276 / 1,296 |
| 3.5 million function calls | 1,156.82 [1,132.62, 1,177.55] | 387.08 [381.11, 412.06] | 1,272 / 1,304 |
| 2.5 million instance method calls | 951.78 [928.07, 961.68] | 288.06 [280.15, 292.05] | 1,276 / 1,296 |
| 1.5 million list/map iterations | 952.38 [924.98, 1,000.84] | 543.90 [537.43, 575.90] | 1,268 / 1,420 |
| 24 native Float64 matrix products | 35.08 [32.69, 39.30] | 36.34 [32.17, 41.63] | 3,980 / 4,060 |
| 192 native Float64 matrix products | 245.88 [242.61, 269.15] | 248.88 [246.45, 262.32] | 4,012 / 4,020 |

GNU gprof counted 72,000,006 calls to `envslot` in the baseline local
loop and 2 in the candidate. Baseline `envslot` and `tablefind_cached`
together accounted for about 28% of the instrumented local-loop samples.
The function and method loops reduced `tableset` calls from 3,500,032 to 36
and from 2,500,035 to 39, respectively. The allocation case reduced
`gcstep` invocations from 63,829 to 39,999. The remaining allocation and
interpreter costs still appear in the profile.

The two numerical-control medians differed by 3.6% and 1.2%, respectively.
Their measured ranges overlap. Both long numerical profiles sampled the
same `gemm_avx2` kernel and counted 1,572,864 kernel calls. The normal-build
machine code for that kernel was byte-identical. The cause of the small timing difference remains unresolved.

The allocation run with `--gc-profile` recorded 20,000 sweeps, averaging
6.9 µs per sweep. Its observed maximum was 8.56 ms. An earlier run on the
same workload observed a maximum of 0.54 ms. In three additional runs,
a function retained 100,000 one-element lists, called `collect()`, and
returned their count. The longest sweeps ranged from 3.07 to 4.10 ms
with up to 100,040 objects. Sweep time is elapsed time and includes host
scheduling. The GC still reclaims an entire sweep atomically.

### Cloud validation

All 332 regression tests passed across nine build configurations. These
included portable GCC, OpenMP, OpenBLAS/LAPACK, ASan/UBSan with leak
detection, OpenMP with vector math/LTO/hwloc, Clang 19 and GCC AArch64
under QEMU. The native affinity test was rerun with multiple CPUs allowed.
All 98 independent reference tests passed on the two LAPACK builds.
Other builds passed 86 and skipped the 12 LAPACK-dependent cases.
Generated sources and C formatting also passed.

### Per-binding capture analysis

The comparison uses pushed commit `d141c57` as the baseline. Both builds ran
on the same pinned Xeon Platinum 8573C CPU using GCC 14.2 with `-O2 -g`.
Each median uses seven executions after two warmups. Outputs were checked
on every run.

| Workload | Function-wide capture decision, ms | Per-binding capture decision, ms | Peak RSS, KiB before / after |
| --- | ---: | ---: | ---: |
| 12 million local-variable iterations with a captured parameter | 1,615.38 [1,592.89, 1,638.16] | 628.13 [620.49, 635.20] | 1,412 / 1,256 |
| 3.5 million calls to a closure from a loop | 1,020.10 [1,013.85, 1,052.53] | 342.36 [340.29, 349.74] | 1,372 / 1,352 |
| Local-variable control with no closure | 644.73 [640.18, 692.37] | 632.60 [627.19, 641.07] | 1,420 / 1,288 |
| Function-call control with no closure | 365.26 [357.56, 376.06] | 371.33 [368.47, 373.32] | 1,400 / 1,256 |

Times are median [minimum, maximum]. A separate run created a 20,000-element
local list and returned a closure that increments a captured parameter.
After collection with the closure still live, Arda's accounted heap had
grown by 525,118 bytes on the baseline and by 383 bytes with per-binding
storage. Both closures returned 41 and 42 on the first two calls.
[Timing harness](../benchmarks/palantir_vm.py) · [Capture and lifetime tests](../tests/test_palantir_runtime.py)

The capture change passed 336 regression tests in each of nine cloud build
configurations, including sanitized builds, OpenBLAS/LAPACK, Clang and
emulated AArch64. Both LAPACK configurations passed all 98 reference tests.
Each other configuration passed 86 and skipped the 12 LAPACK-dependent cases.
A deterministic comparison of 100 mixed-scope closure programs produced
identical output on both builds under GC stress.
