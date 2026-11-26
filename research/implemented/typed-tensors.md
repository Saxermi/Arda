# Typed tensors and storage

Arda 0.16 provides CPU tensors with explicit dtypes and synchronous storage access. Version 0.16.3 adds memory budgets, tiled I/O and streaming statistics. The `numc` interface uses Float64 arrays and its established numerical kernels. The `tensor` module accepts legacy arrays as Float64 inputs. [Typed operations](../../src/typed_tensor.c) · [Buffer layout](../../src/arda.h)

## Buffers and views

The collector traces a shared buffer from each tensor and view. A buffer tracks allocation ownership, byte extent, alignment, and a release callback. Layouts carry their own shape, strides, offset, element count, and read-only status. Native code uses buffers synchronously while the VM retains the call operands. Retaining raw addresses across VM calls requires a future pinning API.

`numc.readonly(x)` shares storage. Changes through an existing writable alias remain visible. `tensor.view(x,shape,strides,offset)` uses strides and an offset in elements relative to a contiguous source. Signed and zero strides are supported. The checked view is read-only, including when multiple coordinates alias one element. `tensor.permute(x,axes)` preserves the input's access mode. `tensor.cast(x,dtype)` creates writable contiguous storage and `tensor.to_numc(x)` explicitly materializes a Float64 array.

## Dtypes and arithmetic

The registry supports `bool`, `uint8`, `int32`, `int64`, `float16`, `bfloat16`, `float32`, `float64`, `complex64` and `complex128`. Construct a tensor from a flat list with `tensor.array(values,dtype[,shape])` or construct a zero tensor with `tensor.zeros(shape,dtype)`. Use `shape`, `dtype`, `bytes`, `at`, `put`, and `tolist` for inspection and access. Heap-backed tensors allow one to eight axes and at most 16,000,000 elements. `bytes` reports logical values in bytes.

Elementwise `+ - * /` broadcast over trailing axes. Arithmetic promotes half formats to Float32 and bool to Int32. Integer division uses Float64. Float32 and Complex64 widen when mixed with Int32, Int64, or wider floating components. Arda integer, real, and complex scalars enter promotion as Int64, Float64, and Complex128. Integer casts require integral, in-range values. Bool accepts 0 and 1. A conversion to real storage rejects nonzero imaginary parts. Finite floating casts outside the target range raise an error. Binary16 and BFloat16 use direct rounding from Float64 with ties to even. [Registry and conversions](../../src/dtype.c) · [NumPy promotion comparison](https://numpy.org/doc/stable/reference/arrays.promotion.html)

`tensor.sum(x,axis)`, `tensor.matmul(a,b)`, `tensor.tensordot(a,b,axis_a,axis_b)`, and `tensor.einsum(spec,a,b)` accumulate and return Int64, Float64, or Complex128, with checked integer arithmetic. Matrix multiplication broadcasts batch axes. `tensordot` contracts one axis per input. `einsum` accepts two explicit operand label lists and an explicit output such as `"bij,jk->bik"`. Each lowercase label appears at most once per operand. Shared axes require equal lengths. `gather` and `scatter` take zero-based flat indices in logical order. `scatter` returns a copy and applies duplicate writes in input order. `mask` takes an equally shaped bool tensor. `fft` and `ifft` use the existing radix-2 kernel for power-of-two vectors up to 1,048,576 elements. These are portable typed paths. Block quantization is scheduled for 0.18.3.

## Tensor files

`tensor.save(path,x[,max_bytes])` writes a complete little-endian file in logical row-major order. The optional budget includes the header. It checks the budget before opening a temporary file, writes bounded chunks, flushes the file with `fsync`, and renames it over the destination after closing. Write or rename errors remove the temporary file. Directory-entry durability across a system crash depends on the filesystem.

`tensor.mmap(path)` maps a regular file read-only. Existing mappings keep their original inode when `save` replaces the path. Use only stable inodes with `mmap`. An external in-place truncate can raise SIGBUS. `tensor.read_block(path,start,count)` returns a writable one-dimensional tensor using `pread`. It validates the header, reads only the requested flat element range, and errors on short or interrupted reads. This is the bounded API for files that can be truncated externally. File mapping size is separate from resident memory and the VM heap counter. [POSIX mapping semantics](https://pubs.opengroup.org/onlinepubs/9799919799/functions/mmap.html) · [Implementation](../../src/tensor_storage.c)

The `ARDATEN1` header is 128 bytes. Byte offsets 8–10 store the dtype code, rank and little-endian marker (1). Dtype codes 1–10 follow the registry order listed above. Unsigned 64-bit fields at offsets 16, 24, and 32 store the payload offset (128), payload byte count, and the first of eight shape dimensions. Unused dimensions and reserved bytes are zero. Payload length must match both the checked shape product and the regular file's remaining size. Complex values store adjacent real and imaginary components. File APIs currently run on little-endian hosts.

```arda
tol tensor as t
esta x = t.array([1,2,3,4],"int32",[2,2])
t.save("sample.atn",x,144)
esta mapped = t.mmap("sample.atn")
pedo(t.shape(mapped),t.tolist(t.read_block("sample.atn",1,2)))
```

The program prints `[2, 2] [2, 3]`. The file size is 144 bytes.

## Memory budgets and streaming

`compute.memory_budget()` returns the configured per-VM byte limit. Pass a nonnegative byte count to set it or `mun` to restore the default. The limit applies to dense and typed heap-buffer payloads and managed native workspace, including idle workspace buffers. Mappings, VM object headers, sparse payloads, stack storage and scratch owned by external libraries have separate accounting. Heap-buffer payload stays charged until garbage collection. Setting a limit below live payload and active workspace raises an error. An idle workspace buffer may be evicted to make room.

`compute.memory_stats()` reports the memory tracked by the runtime.

| Field | Meaning |
| --- | --- |
| `vm_heap_bytes` | Heap bytes tracked by the VM collector. |
| `tensor_heap_bytes` | Heap backing payload. Views sharing a backing buffer contribute once. |
| `mapped_bytes` | Length of live mapped tensor files, including headers. |
| `workspace_reserved_bytes` | Scratch bytes checked out by a native operation. |
| `workspace_cached_bytes` | Idle scratch capacity available for reuse. |
| `peak_budgeted_bytes` | High-water mark of heap payload plus active and idle scratch since VM initialization. |
| `budget_bytes` | Explicit byte limit or `mun`. |

Pass a tensor as `compute.memory_stats(x)` for its `logical_bytes`, backing payload `backing_bytes`, and `storage` class. These counters describe allocations and mappings. Peak resident memory requires an operating-system measurement. The single-slot workspace pool retains up to 8 MiB and reuses a buffer for equal-size reservations. `compute.trim_pool()` releases idle capacity and returns the released bytes.

`compute.plan("stream_stats",path[,block_elements])` opens the header and reports `elements`, `block_elements`, `workspace_bytes`, `available_bytes`, `output_bytes`, and `fits`. Streaming returns a scalar report, so `output_bytes` is zero tensor payload. The plan reserves no buffer. `compute.stream_stats(path[,block_elements])` checks the budget again and reserves one block before reading the payload. The default is 65,536 elements, capped at the number of elements in the file. Pass a smaller block to fit a lower budget.

Streaming stats scan the file once in row-major order and return `count`, `mean`, and population `variance` computed with a shifted Welford accumulator. The method uses constant accumulator storage and one reusable input block. Supported inputs are bool, integer, and finite real dtypes. Mean and variance are Float64 outputs. Empty files return zero count and `mun` for both statistics. Nonfinite inputs and unrepresentable results raise errors. [Welford 1962](https://doi.org/10.1080/00401706.1962.10490022)

`tensor.blocks(path,block_elements)` returns a repeatable iterable over one open file descriptor. Each iteration yields an independently writable flat tensor, with a shorter final block as needed. Use `tensor.close_blocks(source)` to release the descriptor immediately. Garbage collection also closes it. A replacement of the path preserves the descriptor's old inode. In-place truncation causes short reads to fail.

`tensor.read_tile(path,origin,shape)` reads a rectangular region into a same-dtype heap tensor. Origins are zero-based. Each dimension must fit inside the file shape. The heap element cap and memory budget are checked before allocating. `tensor.write_tile(path,origin,tile[,max_bytes])` requires matching dtype and a tile inside the file shape. It copies the source to a temporary file in bounded chunks, patches the requested rows, flushes, and atomically replaces the target. Its scratch reservation is at most 64 KiB. Work and disk writes scale with the full file size. The disk budget covers the replacement file, and publication requires space for both versions during the write. Serialize writers to the same path to retain the results of every edit. Existing mappings and block iterators keep their old inode. [POSIX position-based reads](https://pubs.opengroup.org/onlinepubs/9799919799/functions/read.html) · [POSIX rename](https://pubs.opengroup.org/onlinepubs/9799919799/functions/rename.html)

```arda
tol tensor as t
tol compute as c
t.save("values.atn",t.array([1,2,3,4],"float64",[2,2]))
collect()
c.memory_budget(16)
esta plan = c.plan("stream_stats","values.atn",2)
esta stats = c.stream_stats("values.atn",2)
pedo(plan["fits"],plan["workspace_bytes"],stats["count"],stats["mean"],stats["variance"])
```

The program prints `nae 16 4 2.5 1.25`.

## Contiguous Float64 matrix products

Vilya dispatches contiguous rank-two Float64 tensors to the same blocked or CBLAS kernel as `numc.matmul`. Mixed dtypes, strided matrices, and batched products retain the typed contraction path.

A local single-core check used an Intel Xeon Platinum 8573C and GCC 14.2 with the default `-O2` build and BLAS disabled. The two input matrices were reshaped linspaces from −1 to 1 and 2 to −3. Each run used two warmups and reported the median of seven products, including output allocation and excluding setup and collection. The 128×128 product took 44.95 ms on 0.16.5 and 0.39 ms with Vilya. The 256×256 product took 370.53 ms and 3.24 ms, respectively. The maximum absolute difference from NumPy across both builds and sizes was 5.69 × 10⁻¹³. The measurements cover these contiguous products on this machine. [Shared numerical kernel](../../src/scientific.c)

## Local validation

All 203 regression tests passed in the portable and BLAS/LAPACK builds, both with and without ASan, UBSan and leak detection. All 40 NumPy/SciPy reference tests passed with LAPACK. The portable configurations passed 35 and skipped the five LAPACK-dependent cases. Generated headers, formatting and the two documentation examples passed.

The automated tests cover GC view lifetime, read-only writes, signed strides, type overflow, corrupt headers, budget rejection, failed partial writes, atomic replacement and temporary-file cleanup. Sparse-file fixtures verify block and tile indexing beyond the heap cap. NumPy references cover typed arithmetic, complex FFTs, streamed mean and variance, and multidimensional tile reads. The 20 GB and 80 GB performance gates await real-file measurements. [Storage tests](../../tests/test_tensor_storage.py) · [Budget and tile tests](../../tests/test_tensor_memory.py) · [Streaming references](../../tests/reference_tensor_memory.py)
