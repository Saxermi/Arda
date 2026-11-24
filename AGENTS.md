# AGENTS.md — Working on Arda

## Mission and scope

Arda is an experimental, standalone scientific programming language implemented in C11. Its motto is **Compute. Optimize. Understand.** The goal is high-performance, resource-aware, explainable and reproducible scientific computing, from ordinary CPU-only laptops to research clusters.

Arda is CPU-first, not CPU-only. Preserve the small portable runtime while enabling optional optimized numerical libraries and future accelerators. Favor exploiting mathematical structure and reducing data movement over simply adding hardware.

The Elvish-inspired syntax and Tolkien release codenames are part of the project's identity. Preserve the English keyword aliases, which map to the same tokens. Use conventional English names for scientific APIs and write public-facing documentation in English. The inspired keywords are not authoritative translations.

## Establish the current state before editing

- Read [README.md](README.md), [CONTRIBUTING.md](CONTRIBUTING.md), [CHANGELOG.md](CHANGELOG.md), and the relevant section of [ROADMAP.md](ROADMAP.md).
- The source and tests describe implemented behavior; the roadmap describes proposals, not promises. Inspect the latest repository state rather than assuming a planned API exists.
- At the time this guide was introduced, the baseline was 0.13.1 (Scientific Maintenance), with dense rank-1–8 Float64 tensors, scalar reverse-mode AD, selected SIMD kernels, numerical methods and a specialized matrix-free Poisson solver.
- The next milestone at that baseline is 0.14 (Moria), native COO/CSR sparse computing. See [ROADMAP_0.14.md](ROADMAP_0.14.md) and the consolidated master roadmap for its scope.
- Do not implement unrelated future milestones, invent benchmark results, or bump the version unless the requested task warrants it.

## Repository map

| Path | Responsibility |
| --- | --- |
| `src/arda.h` | Shared runtime types, objects and interfaces |
| `src/lexer.c`, `src/parser.c`, `src/compiler.c`, `src/frontend.c` | Source preparation, parsing and bytecode compilation |
| `src/vm.c`, `src/gc.c`, `src/native.c` | Execution, object lifetime and native module dispatch |
| `src/tensor.c`, `src/scientific.c` | Tensor representation and numerical kernels |
| `src/calculus.c`, `src/autodiff.c`, `src/highdim.c` | Numerical methods, scalar AD and structured solvers |
| `lib/` | Arda standard-library sources, including ODE and optimization algorithms |
| `tools/` | Embedded-library generators |
| `tests/` | Python unittest regression tests and optional numerical reference tests |
| `examples/`, `benchmarks/`, `docs/` | Usage, performance workloads and documentation |

## Non-negotiable engineering principles

- Numerical correctness and memory safety come before a speed claim. Check allocation sizes, integer conversions, shapes, strides, index bounds, aliasing, ownership and cleanup on error paths.
- Keep a portable C implementation. BLAS/LAPACK and future SIMD, GPU and MPI backends must remain optional; do not add mandatory heavyweight dependencies.
- Do not silently densify sparse operators, lower precision, introduce lossy approximations, or change the mathematical problem. Exact algebraic transformations can still change floating-point rounding; preserve documented numerical semantics.
- Prefer native typed buffers, direct algorithms, reusable workspaces, cache locality, matrix-free operators and minimal copying. Do not introduce per-element heap objects or unnecessary temporaries in numerical kernels.
- Keep mathematical metadata, tracing, verification, units and AD graph construction outside unused execution paths. No per-element logging or optional mode checks in hot loops. "Zero-cost scientific introspection" is a design target, not a measured guarantee.
- Respect memory budgets, account for transient workspaces, and distinguish logical size, allocated storage and measured process peak RSS. Offloading is not free; consider I/O costs.
- Preserve existing language behavior and GC lifetime rules. When adding a native object, review allocation, GC tracing, ownership, release, dispatch, error paths and GC-stress behavior together. Do not retain dangling pointers into frontend or collected objects.
- Use clear C11 code, existing conventions and `.clang-format`. Avoid unrelated refactors. Keep warnings, errors and public API behavior explicit.
- Prefer incremental changes over rebuilding the interpreter or adding speculative abstractions. Reuse the existing infrastructure.

## Scientific validation — local only

**Do not add GitHub Actions, hosted CI workflows or remote test jobs. Do not trigger hosted runners.** The project owner explicitly wants to preserve GitHub runner minutes. Tests, sanitizers, reference comparisons and benchmarks run locally on available machines; cluster testing requires explicitly available resources.

Build and test on Linux (Ubuntu is the primary target):

```sh
make -j
./arda --version
make test
make check-generated
make sanitize
```

Optional backends (install the appropriate development libraries locally):

```sh
make BLAS=1 LAPACK=1 test
```

Optional independent NumPy/SciPy reference validation:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
make test-reference
```

Use the Makefile's separate build configurations; no clean build is required merely to switch BLAS/LAPACK/sanitizer settings. Use `make format-check` when clang-format is available. Do not hand-edit the generated `src/ode_source.h` and `src/opt_source.h`; update the corresponding `lib/*.arda` source and regenerate using the Makefile.

Add focused regression tests for fixes and numerical tests for new algorithms. Include edge cases, invalid inputs, overflow, empty shapes, error cleanup and GC stress where applicable. Compare numerical outputs against independent references using justified tolerances, residuals and conditioning information. Report precisely which tests actually ran and which were skipped; do not imply that a local run establishes general correctness or cross-platform support.

For performance changes, compare equivalent workloads and accuracy against the previous version and suitable optimized references. Record compiler, build flags, hardware, thread count, input sizes, runtime and peak memory; distinguish theoretical storage estimates from measurements. Do not fabricate results or bypass checks to improve a benchmark.

## Roadmap guidance

The master roadmap describes the proposed AME (memory engine), AKE (kernel engine), ACE (adaptive compute engine), AMDE/Rúmil (mathematical documentation), AML (machine learning) and distributed HPC layer. Develop these incrementally.

- Sparse: direct COO/CSR conversion, canonical CSR, checked indices, duplicate summation, explicit zero policy, and no dense intermediates. Prepare compact local indices separately from future global HPC indices.
- MathDoc: `getTex()` and static mathematical descriptions must not execute the underlying numerical function. Distinguish mathematical definitions, backend algorithms and observed run data. Unsupported expressions or external implementations must be reported honestly. Trace collection is opt-in and bounded.
- HPC: optimize single-node CPU execution first, then expand multicore, NUMA and MPI; preserve portable fallbacks and avoid hidden communication or replication.
- Beyond RAM and ML: design for bounded working sets, efficient data movement and explicit approximations. Very large model inference and advanced training methods are research goals, not current capabilities.

Read the relevant release specification for detailed acceptance criteria rather than attempting the entire roadmap in one change.

## Change checklist

1. Inspect the latest source, tests and release specification relevant to the request.
2. Make the smallest coherent change; preserve existing interfaces unless a change is explicitly intended.
3. Add regression/reference tests and run appropriate local validation if possible.
4. Update examples, README, changelog and roadmap status where behavior actually changes. Do not mark an unimplemented feature as completed.
5. Summarize changed files, actual test results, numerical/performance implications and known limitations.

Do not commit binaries, build directories, virtual environments, credentials, private datasets or unverified results. Preserve the MIT license. Do not force-push, rewrite repository history, change repository settings or publish changes without the appropriate user authorization. Keep this guide aligned with actual commands and project policy.
