# Arda 0.15.4 — Mathematical Documentation (AMDE v1)

*Development codename: Rúmil*

**Status: development candidate, not a validated release.** The source changes add bounded user-function MathIR, preserve the native registry and cost models, and provide a compile-time metadata switch. See the [contracts](../research/implemented/mathdoc.md) and [example](../examples/mathdoc_queries.arda).

## Local release gate

No build, sanitizer, benchmark, or numerical reference result is claimed for this candidate. Run the following on a local Linux machine from the final checkout; do not use hosted CI.

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
