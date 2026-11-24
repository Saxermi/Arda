# AGENTS.md — Working on Arda

## Mission and scope

Arda is an independent, research-driven scientific programming language implemented in C11 and distributed under the MIT license. Its motto is **Compute. Optimize. Understand.** The goal is high-performance, resource-aware, explainable and reproducible scientific computing, from ordinary CPU-only laptops to research clusters.

Arda is CPU-first, not CPU-only. Preserve the small portable runtime while enabling optional optimized numerical libraries and future accelerators. Favor exploiting mathematical structure and reducing data movement over simply adding hardware.

## Scientific identity and naming conventions

- ARDA stands for **Adaptive Runtime for Data and Algorithms**. Use `Arda` for the language, `arda` for the executable, and `.arda` for source files.
- Lead public documentation with Arda's research-driven scientific purpose, numerical accuracy, and reproducible results. Write public-facing documentation in English and distinguish implemented capabilities from architectural goals.
- Title roadmap milestones `Arda [version] — [scientific or technical milestone]`. Put the existing literary codename on a secondary `Development codename` line and preserve historical release names.
- Preserve the Elvish-inspired keywords and English aliases. Use conventional scientific terms for mathematical APIs and the established technical names and abbreviations for subsystems.
- Use original scientific, mathematical, and geometric branding. Keep literary inspiration secondary and preserve the existing non-affiliation statement in the README.
- Follow the scientific research, provenance, intellectual property, and local validation policies below.

## Establish the current state before editing

- Read [README.md](README.md), [CONTRIBUTING.md](CONTRIBUTING.md), [CHANGELOG.md](CHANGELOG.md), and the relevant section of [ROADMAP.md](ROADMAP.md).
- The source and tests describe implemented behavior; the roadmap describes proposals, not promises. Inspect the latest repository state rather than assuming a planned API exists.
- At the time this guide was introduced, the baseline was 0.13.1 (Scientific Maintenance), with dense rank-1–8 Float64 tensors, scalar reverse-mode AD, selected SIMD kernels, numerical methods and a specialized matrix-free Poisson solver.
- Arda 0.14.3 (development codename Moria) implements native COO/CSR, typed assembly, sparse algebra, Matrix Market I/O, CG/PCG, and sparse memory budgets. Read [docs/MORIA_0.14.3.md](docs/MORIA_0.14.3.md) and [the foundations report](docs/MORIA_0.14.0.md). Arda 0.14.4 closes sparse release validation. See [the sparse release report](docs/MORIA_0.14.4.md).
- Arda 0.15.0 (development codename Rúmil) adds reusable dense factorizations, matrix utilities and structured direct solvers. See [the release report](docs/RUMIL_0.15.0.md). Next is 0.15.1 SVD, eigenvalues, and conditioning.
- Do not implement unrelated future milestones, invent benchmark results, or bump the version unless the requested task warrants it.

## Repository map

| Path | Responsibility |
| --- | --- |
| `src/arda.h` | Shared runtime types, objects and interfaces |
| `src/lexer.c`, `src/parser.c`, `src/compiler.c`, `src/frontend.c` | Source preparation, parsing and bytecode compilation |
| `src/vm.c`, `src/gc.c`, `src/native.c` | Execution, object lifetime and native module dispatch |
| `src/tensor.c`, `src/scientific.c` | Tensor representation and numerical kernels |
| `src/linalg.c` | Dense factorizations and reusable triangular solves |
| `src/sparse.c`, `src/sparse.h` | Native COO/CSR storage, conversion, assembly, and budget accounting |
| `src/sparse_algebra.c`, `src/sparse_market.c`, `src/sparse_solver.c` | Sparse operators, Matrix Market I/O, and CG/PCG |
| `src/calculus.c`, `src/autodiff.c`, `src/highdim.c` | Numerical methods, scalar AD and structured solvers |
| `lib/` | Arda standard-library sources, including ODE and optimization algorithms |
| `tools/` | Embedded-library generators |
| `tests/` | Python unittest regression tests and optional numerical reference tests |
| `examples/`, `benchmarks/`, `docs/` | Usage, performance workloads and documentation |
| `research/`, `DECISIONS.md` | Scientific references, algorithm provenance, and engineering decisions |

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

## Scientific research and provenance

Apply this policy when adding, replacing or substantially optimizing scientific libraries, mathematical algorithms or computational kernels. Routine fixes and unrelated documentation changes need only the research relevant to their impact.

- Review primary literature, authoritative numerical documentation and relevant work from the preceding 24 months when available. Search sources such as Crossref, OpenAlex, arXiv, SIAM, ACM, IEEE and university archives. Check corrections, retractions, limitations and independent evaluations where available. Use lawful access, summarize in original words and link to papers. Report unavailable internet or source access explicitly.
- Compare suitable baseline and newer methods before choosing. Evaluate correctness, forward and backward error, conditioning, convergence, work and workspace, cache locality, data movement, SIMD, CPU/GPU portability, scaling and maintenance cost. Keep a tested baseline when useful. Consider multiple implementations for ACE when evidence identifies distinct operating regimes. Adopt precision changes, approximations and convergence changes only under explicit numerical contracts. Preserve safety checks and avoid implicit densification.
- Update the relevant entry under [research/](research/README.md), add verified entries to [REFERENCES.bib](research/REFERENCES.bib), and record significant choices or rejected alternatives in [DECISIONS.md](DECISIONS.md). Record source functions, mathematical assumptions, complexity, backend, provenance, validation and known limits. Mark uncertain provenance explicitly. Distinguish historical references, sources consulted during implementation, external code reuse, estimates and measured results. Move or consolidate planned notes when the implementation lands.
- Validate locally with appropriate analytical and independent reference cases, including numerical edge cases, convergence or residual checks, and memory safety checks. Compare performance at equivalent accuracy and record dimensions, hardware, compiler flags, thread counts, timing method and peak memory. Keep scientific metadata outside hot paths and preserve static introspection without executing the target function.

## Intellectual property and project identity

- Prefer original code from mathematical definitions and published principles. Before copying or translating third-party code or protected expression, verify permission, the exact license and its compatibility with the intended MIT distribution. Preserve required notices, attribute origins and document modifications when required. Record reused code, optional runtime backends and development dependencies in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- For a concrete novel method, perform a proportionate preliminary patent search, especially for machine learning, quantization, compression, hardware kernels, fusion, compiler optimization, memory management, adaptive execution and distributed computing. Use public resources such as Google Patents, Espacenet and WIPO PATENTSCOPE. Compare relevant claim elements with the intended implementation and record credible concerns and applicable jurisdictions or status when verified. Escalate concrete unresolved concerns before integration. A preliminary search cannot establish freedom to operate. Keep the process practical for an independent project.
- Avoid reproducing protected literary passages, official illustrations, logos or substantial linguistic material without permission.

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

The master roadmap describes the planned Arda Memory Engine (AME), Arda Kernel Engine (AKE), Adaptive Compute Engine (ACE), Arda Mathematical Documentation Engine (AMDE, codename Rúmil), Arda Machine Learning (AML), and distributed computing and HPC infrastructure. Develop these incrementally.

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
