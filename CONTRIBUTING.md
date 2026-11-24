# Contributing to Arda

Arda is an experimental C11 scientific language. Correctness, resource efficiency, performance and transparent mathematics are the main development priorities.

## Local validation only

Do not add GitHub Actions or hosted CI workflows. The repository owner's current policy is to avoid consuming hosted runtime minutes.

Once the source is imported, use the Makefile:

```sh
make -j
make test
make sanitize
make check-generated
make BLAS=1 LAPACK=1 test
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
make test-reference
```

Run benchmarks locally and include machine, compiler, build flags, workload, precision, timings and peak memory. Compare with optimized baselines, not deliberately inefficient implementations.

Keep portable C paths, check index/size overflow and ownership, and test error paths and GC stress. Do not silently densify sparse data, lower precision or introduce unrequested tracing overhead. Update documentation and the changelog for changes. Distinguish implemented capabilities from proposals in the roadmap.

Use clear, conventional English scientific API names; preserve both Tolkien-inspired and English keyword aliases. Avoid committing build outputs, virtual environments, datasets, secrets and generated benchmark results without provenance.
