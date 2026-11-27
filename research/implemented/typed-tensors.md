# Typed tensors and storage

Arda 0.16 provides CPU tensors with explicit dtypes and synchronous storage access. Version 0.16.3 adds memory budgets, tiled I/O and streaming statistics. The `numc` interface uses Float64 arrays and its established numerical kernels. The `tensor` module accepts legacy arrays as Float64 inputs. [Typed operations](../../src/typed_tensor.c) · [Buffer layout](../../src/arda.h)

## Buffers and views

The collector traces a shared buffer from each tensor and view. A buffer tracks allocation ownership, byte extent, alignment, and a release callback. Layouts carry their own shape, strides, offset, element count, and read-only status. Native code uses buffers synchronously while the VM retains the call operands. Retaining raw addresses across VM calls requires a future pinning API.

`numc.readonly(x)` shares storage. Changes through an existing writable alias remain visible. `tensor.view(x,shape,strides,offset)` uses strides and an offset in elements relative to a contiguous source. Signed and zero strides are supported. The checked view is read-only, including when multiple coordinates alias one element. `tensor.permute(x,axes)` preserves the input's access mode. `tensor.cast(x,dtype)` creates writable contiguous storage and `tensor.to_numc(x)` explicitly materializes a Float64 array.

## Dtypes and arithmetic

The registry supports `bool`, `uint8`, `int32`, `int64`, `float16`, `bfloat16`, `float32`, `float64`, `complex64` and `complex128`. Construct a tensor from a flat list with `tensor.array(values,dtype[,shape])` or construct a zero tensor with `tensor.zeros(shape,dtype)`. Use `shape`, `dtype`, `bytes`, `at`, `put`, and `tolist` for inspection and access. Heap-backed tensors allow one to eight axes and at most 16,000,000 elements. `bytes` reports logical values in bytes.

Elementwise `+ - * /` broadcast over trailing axes. Arithmetic promotes half formats to Float32 and bool to Int32. Integer division uses Float64. Float32 and Complex64 widen when mixed with Int32, Int64, or wider floating components. Arda integer, real, and complex scalars enter promotion as Int64, Float64, and Complex128. Integer casts require integral, in-range values. Bool accepts 0 and 1. A conversion to real storage rejects nonzero imaginary parts. Finite floating casts outside the target range raise an error. Binary16 and BFloat16 use direct rounding from Float64 with ties to even. [Registry and conversions](../../src/dtype.c) · [NumPy promotion comparison](https://numpy.org/doc/stable/reference/arrays.promotion.html)

`tensor.sum(x,axis)`, `tensor.matmul(a,b)`, `tensor.tensordot(a,b,axis_a,axis_b)`, and `tensor.einsum(spec,a,b)` accumulate and return Int64, Float64, or Complex128, with checked integer arithmetic. Matrix multiplication broadcasts batch axes. `tensordot` contracts one axis per input. `einsum` accepts two explicit operand label lists and an explicit output such as `"bij,jk->bik"`. Each lowercase label appears at most once per operand. Shared axes require equal lengths. `gather` and `scatter` take zero-based flat indices in logical order. `scatter` returns a copy and applies duplicate writes in input order. `mask` takes an equally shaped bool tensor. `fft` and `ifft` use the existing radix-2 kernel for power-of-two vectors up to 1,048,576 elements. These are portable typed paths. The explicit `quant` module provides block quantization and mixed-precision products. [Quantization contracts](quantization.md)

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

## Streaming imports

Both imports write the existing `ARDATEN1` layout in row-major order using a 64 KiB data buffer. Input must be a regular file. An optional logical disk budget covers the 128-byte header and the complete payload. The destination is published after successful validation and writing. On errors, the previously published file stays available. The imported file works with `tensor.mmap`, `tensor.read_tile`, `tensor.blocks`, and other supported file operations.

| API | Input contract |
| --- | --- |
| `tensor.import_npy(source,destination[,disk_budget])` | NPY 1.0, 2.0, or 3.0 up to a 4 KiB header. Dense C-order arrays of rank 1–8 with Boolean, UInt8, Int32, Int64, Float16, Float32, Float64, Complex64, or Complex128 elements. Multibyte values must be little-endian. The payload length must match the header exactly. |
| `tensor.import_delimited(source,destination,[rows,columns][,options])` | Finite Float64 decimal data with the exact supplied shape. Options are `delimiter` (comma by default or tab), `header` (Boolean, false by default), and `disk_budget`. Fields may be double-quoted and have up to 256 decoded bytes. Records end with LF or CRLF, with an optional final line ending. A header, when requested, must have the specified column count. |

Matrix Market data uses the existing sparse I/O API. JSON, NPZ, VTK, HDF5, NPY Fortran order, NPY scalar arrays, structured or object dtypes, and endian conversion remain outside these import contracts. [NumPy file format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html) · [RFC 4180 quoting conventions](https://www.rfc-editor.org/rfc/rfc4180) · [Sparse I/O](../../docs/MORIA_0.14.3.md)

For a tab-separated input with a header and two data rows of three columns:

```arda
tol tensor as t
t.import_delimited("samples.tsv","samples.atn",[2,3],
    {"delimiter":"\t","header":nae,"disk_budget":176})
pedo(t.tolist(t.read_tile("samples.atn",[0,0],[1,3])))
```

For a first data row of 1, 2, and 3, the program prints `[[1, 2, 3]]`.

## Staged tensor outputs

`tensor.open_output(path,shape,dtype[,disk_budget])` opens a zero-initialized temporary file beside the destination. Filesystems with sparse files can represent those zeros without writing every block. The logical disk budget includes the header and all elements, including zeros, and applies per file. It does not reserve physical space, so operating-system I/O errors can still occur.

`tensor.write_tile(output,origin,tile)` patches the staged file at zero-based coordinates. Tiles must match the output dtype and fit its shape. Noncontiguous tiles use at most 64 KiB of budgeted packing workspace. `tensor.read_tile(output,origin,shape)` materializes only the requested tile and checks the heap and memory limits. Multiple writes may overlap, with later writes overwriting earlier writes.

`tensor.commit_output(output)` flushes the file and atomically publishes it under the destination path. Existing open readers keep their previous file. `tensor.abort_output(output)` discards the temporary file, and collection also discards abandoned handles. Handles become inactive after commit, abort, or a failed write or commit caused by I/O. Serialize separate writers to the same destination to retain every edit. Forced termination can leave a temporary file, and a system crash can affect directory-entry durability. A chunk cache and restart protocol are pending. [Implementation](../../src/tensor_storage.c)

```arda
tol tensor as t
esta output = t.open_output("result.atn",[2,3],"float32",152)
t.write_tile(output,[0,0],t.array([1,2,3],"float32",[1,3]))
t.write_tile(output,[1,1],t.array([4,5],"float32",[1,2]))
t.commit_output(output)
pedo(t.tolist(t.read_tile("result.atn",[0,0],[2,3])))
```

The program prints `[[1, 2, 3], [0, 4, 5]]`.

For a local comparison, both paths wrote the same three 2 × 2 Float64 tiles at `[1,3]`, `[1000,31]`, and `[5998,998]` in a zero-filled 6,000 × 1,000 output. Both produced byte-identical 48,000,128-byte files. The path-based API started from a precreated zero file and copied the full file on each call. The staged timing included initialization and commit, with 96 bytes of tile data written. The Linux cloud run used one Arda thread on an AMD EPYC 9V74, GCC 14.2 with the default `-O2` build, BLAS disabled, and an overlay filesystem. Timings are end-to-end wall-clock medians of seven fresh invocations after two warmups, including the file flushes. The path-based calls took 77.11 ms with 1,504 KiB median peak RSS. The staged run took 3.65 ms with 1,372 KiB median peak RSS. These measurements cover sparse tile updates at this size. The 20 GB and 80 GB real-file gates are pending. [Roadmap](../../ROADMAP.md#arda-021--beyond-ram-data-processing-and-visualization-ame-v2)

## Contiguous Float64 matrix products

Vilya dispatches contiguous rank-two Float64 tensors to the same blocked or CBLAS kernel as `numc.matmul`. Mixed dtypes, strided matrices, and batched products retain the typed contraction path.

A local single-core check used an Intel Xeon Platinum 8573C and GCC 14.2 with the default `-O2` build and BLAS disabled. The two input matrices were reshaped linspaces from −1 to 1 and 2 to −3. Each run used two warmups and reported the median of seven products, including output allocation and excluding setup and collection. The 128×128 product took 44.95 ms on 0.16.5 and 0.39 ms with Vilya. The 256×256 product took 370.53 ms and 3.24 ms, respectively. The maximum absolute difference from NumPy across both builds and sizes was 5.69 × 10⁻¹³. The measurements cover these contiguous products on this machine. [Shared numerical kernel](../../src/scientific.c)

## Local validation

All 203 regression tests passed in the portable and BLAS/LAPACK builds, both with and without ASan, UBSan and leak detection. All 40 NumPy/SciPy reference tests passed with LAPACK. The portable configurations passed 35 and skipped the five LAPACK-dependent cases. Generated headers, formatting and the two documentation examples passed.

The automated tests cover GC view lifetime, read-only writes, signed strides, type overflow, corrupt headers, budget rejection, failed partial writes, atomic replacement and temporary-file cleanup. Sparse-file fixtures verify block and tile indexing beyond the heap cap. NumPy references cover typed arithmetic, complex FFTs, streamed mean and variance, and multidimensional tile reads. The 20 GB and 80 GB performance gates await real-file measurements. [Storage tests](../../tests/test_tensor_storage.py) · [Budget and tile tests](../../tests/test_tensor_memory.py) · [Streaming references](../../tests/reference_tensor_memory.py)
