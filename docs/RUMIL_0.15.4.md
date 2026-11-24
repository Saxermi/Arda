# Arda 0.15.4 — Mathematical Documentation (AMDE v1)

*Development codename: Rúmil*

AMDE v1 documents selected native functions and supported straight-line user functions without executing them. The native registry covers LaTeX, algorithms, assumptions, complexity and shape-based cost estimates where those fields are available. The compiler collects bounded user-function MathIR by default. Level 0 applies limited display identities, while levels 1 and 3 show source-level operations. Dynamic calls are shown as unresolved. [Query contracts](../research/implemented/mathdoc.md) · [Runnable example](../examples/mathdoc_queries.arda)

## Local release gate

The final source archive was extracted and built locally on Debian 13 x86-64 with GCC 14.2.0, Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0. The optional backend used OpenBLAS 0.3.29 with one thread. The 177 regression methods include 13 MathDoc tests. The five portable reference skips cover routines that require LAPACK.

| Build | Regression passed | Reference passed | Reference skipped |
| --- | ---: | ---: | ---: |
| Portable | 177 | 26 | 5 |
| BLAS/LAPACK | 177 | 31 | 0 |
| Portable with ASan, UBSan and leak detection | 177 | 26 | 5 |
| BLAS/LAPACK with ASan, UBSan and leak detection | 177 | 31 | 0 |

The generated-header and clang-format 19.1.7 checks passed. The example passed with GC stress in both metadata modes. Its six LaTeX fragments compiled with `pdflatex`. A deliberately invalid `break` after an eligible function definition exercised cleanup after MathIR extraction with both sanitizer builds. Both reported the expected compile error. GCC emitted the existing `-Wclobbered` warning for `self` in `src/vm.c` in the sanitizer configurations.

Run the checks from a freshly extracted source archive on Linux. Use Python 3.10 or newer with the reference dependencies, and install clang-format and the optional BLAS/LAPACK development libraries locally.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
make test test-reference check-generated
./arda --gc-stress examples/mathdoc_queries.arda
./arda --gc-stress --mathdoc=none examples/mathdoc_queries.arda
make BLAS=1 LAPACK=1 test test-reference
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    make SANITIZE=1 test test-reference
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    make BLAS=1 LAPACK=1 SANITIZE=1 test test-reference
make format-check
```

The roadmap puts limited static loop recognition in 0.15.5 and expanded control-flow analysis and execution traces in 0.22.4. [Roadmap](../ROADMAP.md)

## Introspection overhead measurement

The benchmark used one logical CPU on an Intel Xeon Platinum 8573C. Both portable executables used GCC 14.2.0 with `-std=c11 -O2 -g -Wall -Wextra -Wpedantic`, `_XOPEN_SOURCE=700` and `libm`. Each mode and workload had one warmup and 11 interleaved runs. The harness verifies every output and prints timing and peak RSS samples using GNU time.

| Workload | 0.15.4 compact ms, median [range] | 0.15.4 none ms, median [range] | 0.15.3 ms, median [range] |
| --- | ---: | ---: | ---: |
| Startup and `pedo(1)` | 4.565 [3.906, 5.669] | 4.618 [3.293, 6.829] | 4.455 [3.166, 5.189] |
| 2,000 function definitions | 34.322 [31.716, 40.162] | 31.652 [28.896, 36.366] | 32.315 [29.639, 35.917] |
| 500,000 VM calls | 96.154 [90.584, 107.387] | 96.167 [91.863, 102.135] | 99.020 [95.347, 110.669] |
| Nine 256 × 256 matrix multiplications | 54.533 [48.942, 65.251] | 53.138 [48.052, 59.943] | 51.632 [48.426, 59.676] |

The definition workload includes process launch, compilation and execution. Compact metadata increased its median wall time by 8.4% and median peak RSS by 768 KiB compared with `--mathdoc=none`. In compact, none and 0.15.3 order, median peak RSS was 1,056 / 1,048 / 1,052 KiB for startup, 14,756 / 13,988 / 13,852 KiB for definitions, 1,060 / 1,060 / 1,052 KiB for VM calls, and 4,688 / 4,700 / 4,564 KiB for matrix multiplication. The timing ranges overlap across modes for each workload. These are process-level results from a single machine.

```sh
python3 benchmarks/amde_bench.py --rounds 11 --cpu 1 \
    --baseline /absolute/path/to/arda-0.15.3
```
