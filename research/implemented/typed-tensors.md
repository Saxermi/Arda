# Typed tensors and storage

Arda 0.16.2 provides CPU tensors with explicit dtypes and synchronous storage access. The existing `numc` interface still uses Float64 arrays and its established numerical kernels. The `tensor` module operates on typed tensors and accepts legacy arrays as Float64 inputs. [Typed operations](../../src/typed_tensor.c) · [Buffer layout](../../src/arda.h)

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

The `ARDATEN1` header is 128 bytes. Byte offsets 8–10 store the dtype code, rank and little-endian marker (1). Dtype codes 1–10 follow the registry order listed above. Unsigned 64-bit fields at offsets 16, 24, and 32 store the payload offset (128), payload byte count, and the first of eight shape dimensions. Unused dimensions and reserved bytes are zero. Payload length must match both the checked shape product and the regular file's remaining size. Complex values store adjacent real and imaginary components. File APIs currently run on little-endian hosts. Batch allocation and random-write pooling are separate roadmap tasks.

```arda
tol tensor as t
esta x = t.array([1,2,3,4],"int32",[2,2])
t.save("sample.atn",x,144)
esta mapped = t.mmap("sample.atn")
pedo(t.shape(mapped),t.tolist(t.read_block("sample.atn",1,2)))
```

The program prints `[2, 2] [2, 3]`. The file size is 144 bytes.

## Local validation

The regression tests cover GC view lifetime, read-only writes, signed strides, type overflow, corrupt headers, atomic replacement, and temporary-file cleanup. They also read a 16,000,003-element file using a two-element block. This sparse-file fixture checks indexing beyond the heap cap. NumPy references cover half rounding, complex FFT, broadcasting and contractions. [Storage tests](../../tests/test_tensor_storage.py) · [NumPy references](../../tests/reference_typed_tensor.py)
