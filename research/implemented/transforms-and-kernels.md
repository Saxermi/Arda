# Fourier transforms, tensors and CPU kernels

**Status** — Implemented in 0.14.3.

## Radix-2 FFT and IFFT

The forward discrete Fourier transform is $X_k=\sum_{j=0}^{N-1}x_j e^{-2\pi i jk/N}$, with positive exponent and $1/N$ scaling for the inverse. Arda's iterative radix-2 decimation-in-time FFT bit-reverses the input, combines even/odd subproblems through butterflies $(u+\omega v,u-\omega v)$ and advances stage twiddles by complex recurrence. It accepts power-of-two lengths up to $2^{20}$, returns an $N\times2$ real/imaginary tensor and computes in-place on that output buffer. Work is $O(N\log N)$, storage $O(N)$ for output with constant stage scratch. Recurrence drift and floating-point cancellation can affect long transforms. Mixed radices, arbitrary lengths and multidimensional transforms require different paths. [`fft`](../../src/calculus.c) · [Cooley and Tukey 1965](https://doi.org/10.1090/S0025-5718-1965-0178586-1) · [Frigo and Johnson 2005 for modern FFT design](https://doi.org/10.1109/JPROC.2004.840301)

## Tensor views, broadcasting and reductions

Arda stores dense Float64 values in row-major buffers with shape, strides and owner references. An index $i$ addresses $o+\sum_j i_j s_j$. Trailing axes broadcast when equal or when one extent is one, with effective stride zero in the repeated operand. Result construction costs $O(R+N)$ for rank $R$ and $N$ output elements. Shape and stride permutations or supported views use $O(R)$ metadata and share storage. Copying or packing uses $O(N)$ output storage. The implementation iterates outer axes with an odometer and specializes the inner contiguous or broadcast segment. Non-contiguous inputs can be packed for algorithms requiring dense access. Views can alias their owners and layout interacts with cache traffic. The storage cap is 16 million elements and the maximum rank is eight. [`tensor_binary`, `tensor_copy`, `tensor_call`](../../src/tensor.c) · [Harris et al. 2020 for array semantics](https://doi.org/10.1038/s41586-020-2649-2)

## Blocked SIMD and fused vector updates

Packed elementwise add, subtract, multiply and divide use SSE2 in two-double lanes where supported, plus a scalar tail. Dense GEMM uses accumulation $C_{ij}\gets C_{ij}+A_{ik} B_{kj}$ in 32-element blocks on the portable/SSE2 path and blocks of 64 in the runtime-dispatched AVX2 path. The explicit AVX2 instructions do separate multiply and add. A build with CBLAS delegates the product and backend precision details. Timing depends on packing, cache locality, vector width, alignment and tails. `high.axpy` computes $y+\alpha x$ into one output in $O(N)$ work and storage, avoiding a standalone scaled intermediate. The expression name “fused” here refers to combining array operations and does not promise a hardware FMA. [`dense`](../../src/tensor.c) · [`multiply` and `matmul_avx2`](../../src/scientific.c) · [`high_call`](../../src/highdim.c) · [Goto and van de Geijn 2008](https://doi.org/10.1145/1356052.1356053) · [Williams et al. 2009](https://doi.org/10.1145/1498765.1498785)

**Validation and provenance** — The regression and independent suites cover FFT/IFFT against direct DFT and SciPy, broadcasting, aliased and strided views, and dense products. The historical x86-64 report measured 0.13.0 medians of 31.55 ms for its 512 × 512 product and 11.78 ms for its two-million-element strided copy on the stated machine. Those are workload-specific measurements. No matching FFT throughput data was found. Source arrived by `e2ba505` and initial literature provenance is unknown. External CBLAS is optional. NumPy is used for validation. [Reference suite](../../tests/reference_numerics.py) · [Historical benchmark](../../benchmarks/highdim-results.txt) · [Licenses](../../THIRD_PARTY_NOTICES.md)

**Bibliography keys** — `CooleyTukey1965`, `FrigoJohnson2005`, `GotoVanDeGeijn2008`, `HarrisEtAl2020`, `VirtanenEtAl2020`, `WilliamsEtAl2009`.

## 0.18.0 CPU baseline

Run these commands with development headers for Eigen and an optional BLAS. Set the normal compiler and linker search paths for a provider installed outside the system search path.

```sh
make -j2
make -j2 BLAS=1
python3 benchmarks/cpu_baseline.py --arda portable=build/blas0-lapack0-sanitize0/arda \
  --arda openblas=build/blas1-lapack0-sanitize0/arda --eigen-include /usr/include/eigen3 --micro
```

The harness uses identical row-major Float64 inputs from a fixed seed, with integer entries divided by 128. Every result matches an independent Int64 product at 289 coordinates and the full NumPy output exactly. Products and partial sums are exactly representable for these fixtures. Each process warms up once at each size and reports seven batches of timed operations including output allocation. The time excludes input generation, process startup and output validation. Reported dispersion is median absolute deviation (MAD).

One pinned CPU on an Intel Xeon Platinum 8573C under Linux 6.18.57 used GCC 14.2 (`-O2`) for Arda, GCC 14.2 (`-O3 -march=native`) for Eigen 3.4.0, Debian OpenBLAS 0.3.29 for Arda and NumPy 2.3.5's bundled OpenBLAS. The harness requests one thread in every provider. The reported machine had 48 KiB L1 data and 2 MiB L2 per core and a reported 260 MiB shared L3. Add `--perf` on an installation with Linux `perf stat` and access to hardware counters. Those counters cover the external processes, including startup. The current environment has no installed `perf`; the copy figures below measure logical read and write bandwidth, including allocation, in a cache-sized workload.

| Size | Arda portable (ms) | Arda OpenBLAS (ms) | Eigen (ms) | NumPy (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.3379 | 0.0584 | 0.0518 | 0.0607 |
| 512 | 21.7700 | 3.2166 | 3.6281 | 3.2143 |
| 1024 | 188.1972 | 24.1029 | 24.3113 | 23.4945 |

Portable medians in milliseconds for the accompanying microbenchmarks were GEMV 0.1227, SpMV 0.4219, AXPY 0.0615, reduction 0.4314, sine 0.5692, cosine 0.4882, exponential 0.4791 and a 20,000-iteration VM loop 3.5536. Copying the 256-by-256 Float64 fixture gave 50.5 GB/s packed and 1.27 GB/s transposed. The GEMV uses a matrix with one output column. SpMV uses a 65,536-row tridiagonal CSR matrix. `compute.cpu()` reports architecture, SSE2, AVX2, FMA and AArch64 NEON support.

Optional LTO uses `make LTO=1 CC=gcc` or `make LTO=1 CC=clang` with the corresponding linker toolchain. It uses a separate build directory and preserves the normal architecture target. For profile-guided optimization, keep the same build directory between training and profile use so the compiler can match the profiles. For GCC use `-fprofile-generate=/absolute/profile/path` in both `CFLAGS` and `LDFLAGS`, run representative benchmarks with that binary, then rebuild in that same `BUILD` directory with `-fprofile-use=/absolute/profile/path -fprofile-correction` in `CFLAGS`. For Clang use `-fprofile-instr-generate` on compilation and linkage, set `LLVM_PROFILE_FILE` while running training workloads, merge the raw profiles with `llvm-profdata merge`, and rebuild with `-fprofile-instr-use=/absolute/merged.profdata`. Preserve the normal C11 and warning flags when overriding `CFLAGS`. [Benchmark harness](../../benchmarks/cpu_baseline.py) · [GCC instrumentation](https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html) · [Clang PGO](https://clang.llvm.org/docs/UsersManual.html#profile-guided-optimization)

## 0.18.1 register-blocked matrix multiplication

`dense_matmul_into` in `src/dense_gemm.c` applies 64-entry column and inner panels and 32-row blocks. A microkernel holds four output rows and two vectors per row in registers. It reuses each pair of right-hand vectors across those four rows. AVX2 processes eight columns per tile. SSE2 and AArch64 NEON process four. Unaligned loads support contiguous views at arbitrary element offsets. Short rows and columns use bounded scalar tails. Empty products keep their initialized output. The algorithm uses $O(mkn)$ work, an $O(mn)$ output and constant per-worker kernel scratch. Inputs remain in their row-major storage. [Goto and van de Geijn](https://doi.org/10.1145/1356052.1356053)

`compute.matmul_kernel()` reports the requested packed rank-two Float64 policy. Use `compute.matmul_kernel("reference")` for the original kernel, or request `"sse2"`, `"avx2"`, `"avx2-fma"` or `"neon"` when supported by the build and CPU. `"auto"` restores selection based on CPU capability and shape. An unavailable choice raises `backend`. With BLAS enabled, the packed Float64 path uses the linked CBLAS and reports `"cblas"`. The existing typed paths cover strided, batched, mixed-type and complex cases.

The default portable kernel keeps separate multiplication and addition in increasing inner-index order. Explicit `"avx2-fma"` uses fused rounding for complete tiles and scalar tails. Optional Linux x86-64 `make VECMATH=1` and `compute.vector_math(true)` enable libmvec for ordinary contiguous Float64 `sin`, `cos` and `exp` with at least 16 elements. Tracked tensor operations, ordinary scalar operations and vector tails use the scalar routines. Accuracy and special values depend on the system libmvec provider. The opt-in is scoped to the VM.

Timings use the same machine, compiler settings, matrices and harness as the 0.18.0 baseline. Both binaries were run in the same comparison. Exact sampled checks and full output agreement passed. The lowest observed baseline sample exceeded the highest measured tiled sample at all three sizes.

| Size | Reference median ± MAD (ms) | 0.18.1 auto median ± MAD (ms) | Speedup |
| ---: | ---: | ---: | ---: |
| 128 | 0.5014 ± 0.0068 | 0.1785 ± 0.0261 | 2.81× |
| 512 | 22.9536 ± 0.7016 | 9.0492 ± 0.3408 | 2.54× |
| 1024 | 201.8388 ± 6.3935 | 77.5752 ± 1.8144 | 2.60× |

On the constant 65,536-element fixture, opt-in libmvec took 0.0721 ms for sine, 0.0667 ms for cosine and 0.0691 ms for exponential. Scalar mode in the same binary took 0.5820, 0.4938 and 0.4757 ms. This input exercises contiguous access and moderate arguments. [Numerical tests](../../tests/reference_cpu_kernels.py) · [Dispatch, tail and FMA tests](../../tests/test_cpu_kernels.py)
