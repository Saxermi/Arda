# Arda 0.15.4 — Mathematical Documentation (AMDE v1)

*Development codename: Rúmil*

**Status: development candidate, not a validated release.** The source changes add bounded user-function MathIR, preserve the native registry and cost models, and provide a compilation metadata switch. Accepted policy: compact metadata by default, opaque dynamic calls, and limited display-only simplification at level 0. Source-level views and executable numerical semantics remain unchanged. See the [contracts](../research/implemented/mathdoc.md) and [example](../examples/mathdoc_queries.arda).

## Local release gate

The owner supplied a successful local portable build and regression transcript on a ThinkPad T14 Gen 2 AMD running Linux on 2026-11-24. The initial corrected source at `1eb5d53` compiled and linked. After switching Python environments, all 177 regression tests passed in 0.706 s, including all 13 MathDoc tests. The previous `zip(strict=True)` failure was a Python interpreter compatibility issue, not a solver failure; the temporary compatibility patch was reverted, restoring the original test blob. The exact Python/compiler versions were not included in the transcript. This result is owner-reported, not an independent rerun. Reference tests, sanitizer builds, format checks, performance measurements and final-artifact validation are still pending.

Run the following on a local Linux machine from the final checkout; do not use hosted CI. Use Python 3.10 or newer for the regression suite.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
make -j
make test test-reference check-generated
./arda --gc-stress examples/mathdoc_queries.arda
./arda --mathdoc=none examples/mathdoc_queries.arda
make BLAS=1 LAPACK=1 test test-reference
make sanitize
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    make BLAS=1 LAPACK=1 SANITIZE=1 test test-reference
make format-check
```

Record machine/compiler/backend versions, pass/fail/skip counts, and warnings. Measure compile time, startup, RSS and repeated numerical/VM workloads against 0.15.3 and the same candidate with metadata disabled. Report statistical variation; do not claim zero overhead without evidence. Re-run from the final release artifact before updating version numbers, tagging or declaring the milestone complete. Tests should verify that static queries do not execute functions, including closures and opaque calls, and that metadata is freed on compiler failure and GC collection.

Loops remain in 0.15.5 (limited static recognition), while complex control flow and execution traces remain in 0.22.4.

## Introspection overhead measurement

Run `python3 benchmarks/amde_bench.py --rounds 11` against the portable candidate. Add `--baseline /absolute/path/to/arda-0.15.3` for a separately built release with matched compiler/backend flags. Each workload verifies the exact output. The script interleaves modes with a fixed seed, uses a warmup, reports median and range of process wall time and Linux GNU time peak RSS, and prints raw samples. The compile-heavy workload measures startup, compilation and definition execution together, not isolated compiler timing. VM calls and dense matmul workloads also include process startup and setup; interpret results with that limitation. No timing or RSS result is implied by this script. Record CPU, compiler, flags and backend before comparing releases. Targets are design goals, not proof of equivalence from a single run.
