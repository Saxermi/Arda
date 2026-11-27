# Arda 0.18.3 — Block Quantization and Mixed-Precision CPU Products

The `quant` module provides weight-only Q4 and Q8 for rank-two matrices and an explicit Float16/BFloat16 multiplication path. See [the runnable example](../../examples/quantization.arda).

## Formats and kernels

| API | Contract |
| --- | --- |
| `quant.quantize(W, "q4" or "q8")` | Take an immutable snapshot of a rank-two real floating-point tensor. Read strided weights in logical order. Reject nonfinite weights and overflow in the Float32 scale. |
| `quant.info(Q)` | Return the logical shape, encoding, block size, block count, and separate code, scale and total payload sizes in bytes. Object overhead is additional. |
| `quant.dequantize(Q)` | Allocate and return the reconstructed Float64 matrix under the existing tensor memory budget. |
| `quant.gemv(Q, x[, mode])` | Multiply by a matching Float64 vector. The Float64 result has one entry per weight row. |
| `quant.gemm(Q, X[, mode])` | Multiply by a Float64 matrix with shape `[weight_columns, output_columns]`. The output is Float64. |
| `quant.lowp_matmul(A, B[, mode])` | Multiply matching Float16 matrices or matching BFloat16 matrices. Widen on load and use Float32 products, accumulation and output. |
| `quant.backends()` | Report available SIMD implementations for packed products, Float16 inputs and BFloat16 inputs. |

The product mode is `"auto"` by default. `"reference"` selects portable C for comparisons. `"simd"` explicitly requires a supported CPU and compatible strides. Packed GEMM requires unit activation column stride for SIMD, and GEMV requires unit vector stride. The portable path supports signed strides. The low-precision SIMD path requires contiguous inputs. SIMD handles complete vector tiles with bounded scalar tails. These operations accept ordinary tensors. Differentiated inputs raise `autodiff`; explicit detachment is available through `ad.detach`.

Each weight row is divided into blocks of 32 along the contracted axis. For a block with maximum magnitude $a$, the Float32 scale $s$ is the least positive Float32 value at least $a / q_{\max}$ when $a$ is nonzero. Zero blocks use a zero scale. Quantization rounds $w/s$ to the nearest integer with ties away from zero and clips to the signed symmetric range. The maximum code is 7 for Q4 and 127 for Q8. The representation uses stored codes offset by 8 or 128. Q4 puts the first value in each byte's low nibble. Float64 reconstruction is $s q$. Rounding error per value is at most about half a scale, subject to floating-point arithmetic. Outliers can increase the error for other values in the same block. Tiny values can round to zero.

The packed buffer has a four-byte scale and either 16 Q4 code bytes or 32 Q8 code bytes per block. Each row's last block is padded. For complete blocks, Q4 uses 5 bits per weight and Q8 uses 9 bits. A theoretical 70 billion weights in complete Q4 blocks would occupy **43.75 GB** of payload. The current API uses Arda's 16-million-element matrix limit and an in-memory format.

`src/quant.c` handles encoding, budgets and logical tensor indexing. `src/quant_kernels.c` selects AVX2 or AArch64 NEON for packed products. The SIMD kernels decode complete blocks and consume at most one reconstructed block at a time. GEMV uses four partial sums per block and adds the block results in order. GEMM keeps output columns in SIMD registers through one decoded block and accumulates each output in increasing inner-index order. This retains normal rounding without fused multiply-add. GEMV and GEMM can differ in their last bits because their reduction orders differ. Packed products allocate only the output from the tensor budget and use bounded stack scratch for the decoded block. Their work is $O(mk)$ for a vector and $O(mkn)$ for a matrix with $n$ columns.

The low-precision path uses F16C with AVX2 on x86 or FP16-capable NEON on AArch64 for Float16 conversion. BFloat16 SIMD expands its bits into Float32 lanes on AVX2 or NEON. Both paths perform arithmetic in Float32. SIMD selection is checked at runtime, with the same arithmetic contract in portable C. [Arm ACLE vector intrinsics](https://arm-software.github.io/acle/neon_intrinsics/advsimd.html) · [Intel intrinsics guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)

This is an original symmetric linear quantizer and CPU implementation. Calibrated methods such as GPTQ use layer reconstruction information to improve quality. More recent edge-CPU work such as PolyQ co-designs quantization with inference kernels. Those approaches motivate later measurements on real models. Arda owns the in-memory format and implementation. The cited projects served as research references. [Quantization concepts](https://huggingface.co/docs/transformers/main/quantization/concept_guide) · [GPTQ](https://arxiv.org/abs/2210.17323) · [PolyQ](https://arxiv.org/abs/2607.14618)

## Local evidence

```sh
make -j2
python3 benchmarks/quantization.py
```

The run used a pinned core on an Intel Xeon Platinum 8573C, Linux 6.18.57, GCC 14.2 portable `-O2` and NumPy 2.3.5 for references. The 1,024-by-1,024 Gaussian weight matrix used a fixed seed. The table shows the median of seven batches after one warm-up per mode, with median absolute deviation in parentheses. Timed work includes output allocation. Weights were encoded or reconstructed during setup. Inputs, validation and output export were outside the timed intervals. Packed and reconstructed-dense results agreed with NumPy multiplication of the same reconstructed weights.

| Mode | Weight payload | GEMV ms (MAD) | GEMM, 32 columns ms (MAD) |
| --- | ---: | ---: | ---: |
| Original Float64 | 8 MiB | 2.072 (0.031) | 1.781 (0.012) |
| Q8 portable | 1.125 MiB | 1.338 (0.022) | 16.260 (0.084) |
| Q8 with AVX2 | 1.125 MiB | 0.349 (0.010) | 2.870 (0.031) |
| Dense from Q8 | 8 MiB | 2.043 (0.040) | 1.843 (0.054) |
| Q4 portable | 0.625 MiB | 1.734 (0.022) | 16.558 (0.085) |
| Q4 with AVX2 | 0.625 MiB | 0.381 (0.023) | 2.890 (0.102) |
| Dense from Q4 | 8 MiB | 2.069 (0.042) | 1.817 (0.025) |

Packed SIMD GEMV took about one sixth of the time of multiplying with the corresponding reconstructed dense matrix. At 32 columns the dense register-blocked kernels were faster, while packed multiplication used less weight memory. In that GEMM run the packed Q4 and Q8 paths peaked at 1,179,648 and 1,703,936 budgeted bytes. The dense reconstruction paths peaked at 9,043,968 and 9,568,256 bytes, including encoding and reconstruction during setup. The original dense mode mapped its weight file separately from the tensor heap.

Relative output L2 error against the original Float64 weights was 0.51–0.53% for Q8 and 9.56–9.65% for Q4. In a separate synthetic 16-class linear classifier with 128 inputs and 1,024 held-out examples, top-1 agreement with the original Float64 classifier was 1,021/1,024 for Q8 and 924/1,024 for Q4. Logit relative L2 error was 0.55% and 10.17%, respectively. These are results for the seeded synthetic model. [Benchmark runner](../../benchmarks/quantization.py) · [NumPy references](../../tests/reference_quant.py) · [Budget and runtime tests](../../tests/test_quant.py)

All 244 runtime tests passed in the portable, OpenMP/libmvec and OpenBLAS/LAPACK configurations, with and without ASan/UBSan. The LAPACK configurations passed all 68 reference tests. The others passed 59 with nine LAPACK cases skipped. Clang 19 passed the seven focused runtime and reference tests. An AArch64 GCC build selected NEON and NEON FP16 widening under QEMU and passed the same seven tests.
