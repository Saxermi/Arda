# Optimization and scalar automatic differentiation

**Status** — Implemented in 0.14.3.

## L-BFGS with Armijo backtracking

`opt.lbfgs` minimizes a scalar objective whose callback returns a map with a scalar `value` and a same-shape `gradient` tensor. For $s_i=x_{i+1}-x_i$, $y_i=g_{i+1}-g_i$, it retains up to $m$ recent pairs and $\rho_i=1/(s_i^T y_i)$. The two-loop inverse-Hessian apply uses a backward $q\gets q-\rho_i(s_i^Tq)y_i$ and a forward correction with the stored scalars, starting with scale $(s^Ty)/(y^Ty)$ from the latest pair. Backtracking halves an initial unit trial until $f(x+\alpha p)\le f(x)+10^{-4}\alpha g^Tp$, for at most 40 trials. Unsafe curvature pairs are skipped. A non-descent direction resets history and falls back to negative gradient. For $n$ parameters, work is $O(mn)$ plus callback and $O(n)$ vector work per trial. Live history takes $O(mn)$ space. Smoothness, accurate gradients and useful curvature control performance. The code's Armijo-only search and termination do not give the guarantees of all strong-Wolfe or convex L-BFGS variants. [`lbfgs`](../../lib/opt.arda) · [Nocedal 1980](https://doi.org/10.1090/S0025-5718-1980-0572855-7) · [Liu and Nocedal 1989](https://doi.org/10.1007/BF01589116) · [Armijo 1966](https://doi.org/10.2140/pjm.1966.16.1)

## SGD and Adam

`opt.sgd` applies $\theta^+=\theta-\eta g$ to vectors. The caller supplies the gradient and controls sampling, batches and step schedules. `opt.adam` carries first and uncentered second moments $m_t=.9m_{t-1}+.1g_t$, $v_t=.999v_{t-1}+.001g_t^2$ and computes $\theta_t=\theta_{t-1}-\eta(m_t/(1-.9^t))/(\sqrt{v_t/(1-.999^t)}+10^{-8})$. It computes bias corrections with `expm1`, requires step numbering from one and returns a map with `parameters`, `first_moment`, and `second_moment`. Both cost $O(n)$ work and $O(n)$ returned storage. Adam squares gradients in Float64. The kernel checks nonfinite data and rejects negative second-moment entries, but convergence depends on the problem, sampling and step sequence. The optimizer accepts explicit gradients and leaves the training loop to the caller. [Optimizer kernels](../../src/optimizer.c) · [Robbins and Monro 1951](https://doi.org/10.1214/aoms/1177729586) · [Kingma and Ba, ICLR 2015](https://arxiv.org/abs/1412.6980)

## Reverse-mode autodiff

`ad.variable(x)` accepts a finite real scalar or a dense heap-backed Float64 tensor. The tape stores one node per executed supported operation. A node has its primal result, parent links, and the values and metadata needed to apply a local derivative. Each value uses the existing typed layout for accessing tensor data. A read-only tensor view shares the mutation version of its backing buffer.

The sweep traverses the reachable graph in reverse topological order. For each edge, it adds the local vector–Jacobian product `Jᵀg` to the corresponding input gradient. Repeated parents add their contributions. Broadcast adjoints sum over repeated axes. Slices scatter to source positions, and overlapping signed-stride views can add multiple contributions at a single source position. Batched matrix products use the two transposed matrix products for the input adjoints and sum across broadcast batches.

Run `ad.backward(loss)` to seed a scalar loss with one. A tensor result requires an explicit finite Float64 seed with the same shape. `ad.grad(x)` returns `mun` until a successful sweep. Later sweeps replace the gradients of nodes they visit. Gradient tensors are returned through read-only views. `ad.value(x)` exposes the primal. Modifying a buffer used as a saved value raises a mutation error during backward. Linear rules that depend only on layout do not inspect payload versions. Scalar and tensor inputs and outputs must be finite. Differentiated log and square root require positive inputs.

Supported tensor operations are `+ - * /`, elementary functions `sin`, `cos`, `exp`, `log`, `sqrt`, `ad.tanh`, constant-exponent `ad.pow`, `tensor.sum`, `tensor.mean`, `tensor.matmul`, `tensor.reshape`, `tensor.permute`, `tensor.slice`, `tensor.view`, and `tensor.at`. Use the tensor API for supported tensor calls. Mapped storage and non-Float64 tensor dtypes are outside the tape.

Topology and graph metadata take linear space in the number of nodes. Each sweep stages the gradients under the tensor memory budget before publishing them. A retry can require room for both previous and new gradients. The reverse matrix product currently uses direct loops. Tape liveness, replay, and offloading belong to later milestones. MathIR remains independent of runtime AD. [Implementation](../../src/autodiff.c) · [Analytic and finite-difference checks](../../tests/reference_vilya_autodiff.py) · [Baydin et al.](https://www.jmlr.org/papers/v18/17-468.html) · [PyTorch version-counter reference](https://docs.pytorch.org/docs/stable/notes/autograd.html#in-place-correctness-checks)

## Derivative queries

For a recorded operation with an input coordinate $x_i$, an output coordinate $y_j$, and local partial $s_{ji}$, a forward sweep adds $s_{ji}\dot x_i$ to $\dot y_j$. A reverse sweep adds $s_{ji}\bar y_j$ to $\bar x_i$. The same rule traversal covers broadcast, views, reductions, dense matmul and CSR matvec. No callback is replayed. A query uses only contributions downstream of the selected node. [JAX JVP and VJP reference](https://docs.jax.dev/en/latest/301/cookbook.html)

For a scalar loss, differentiate the reverse recurrence in a chosen direction to add $s_{ji}\dot{\bar y}_j+\dot s_{ji}\bar y_j$ to $\dot{\bar x}_i$. A Hessian applies this recurrence to each input basis direction. A Jacobian uses either an input basis with forward sweeps or an output basis with reverse sweeps, whichever basis is smaller. For $N$ input values and $M$ output values, these require $N$ Hessian directional sweeps or $\min(N,M)$ Jacobian basis sweeps. Published gradients stay outside the query. Work buffers are reused across basis sweeps. Allocations and limits are checked before the sweep. Dense Hessians require $O(N^2)$ result storage. [Pearlmutter 1994](https://doi.org/10.1162/neco.1994.6.1.147) · [Source](../../src/autodiff.c)

For a CSR entry $a_k$ at row $r$ and column $c$, matvec adds $a_kx_c$ to $y_r$. The adjoints add $x_c\bar y_r$ to $\bar a_k$ and $a_k\bar y_r$ to $\bar x_c$. The pattern remains immutable, and explicit values can be zero. Derivative products use $O(\mathrm{nnz})$ edge work and ordinary dense derivative vectors. The implementation keeps the CSR pattern compressed throughout. [JAX BCOO reference](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.sparse.BCOO.html)

Local validation uses JAX 0.6.2 on CPU with Float64 enabled. Tests compare backward on a batched matrix product, seeded JVP and VJP, a Jacobian through a signed-stride view, a nonlinear dense Hessian, and derivative results of a fixed sparse pattern against JAX BCOO. The default runtime and test dependencies remain portable. The optional reference tests run when JAX is installed. [JAX suite](../../tests/reference_vilya_jax.py) · [Runtime regressions](../../tests/test_vilya_runtime.py)

## Checkpointing and gradient liveness

The selected-gradient sweep traverses the graph from the output in reverse topological order. After a node propagates its adjoint to its parents, its scratch can be released or reused at that slot's last use. The selected results are staged until success. An optional output-tape release clears the root's references and collects unreachable history. The collector keeps history that other outputs and intermediate nodes still own.

For batch means, Arda combines the existing mean gradient $g$ from $n$ samples with the new mean gradient $h$ from $m$ samples as $(n/(n+m))g+(m/(n+m))h$. Weighting each mean equally would change the objective when batch sizes differ. The running sample count and published gradient update together.

An elementwise checkpoint stores operation recipes and their shape-compatible boundaries, plus references to the recorded constants and random sample values. Replay evaluates each output coordinate through the recipe. It derives local first-order slopes and, for Hessians, the directional changes of those slopes. For a segment of $L$ operations on $N$ entries, replay takes $O(LN)$ work and $O(L)$ private replay storage, in addition to retained tensors and adjoints. It leaves application code and random generators outside replay. [Checkpointing baseline for deeper graphs](https://arxiv.org/abs/1604.06174) · [JAX rematerialization](https://docs.jax.dev/en/latest/301/remat.html) · [PyTorch RNG replay tradeoffs](https://docs.pytorch.org/docs/stable/checkpoint.html)

### Local memory and timing check

A Float64 vector of 16,384 entries started at 0.2. The program applied `sin(state*0.97 + 0.02)` 48 times and differentiated the sum of the final state. The checkpoint mode packed blocks of eight steps and collected the unpacked segment after each block. All modes agreed within a relative tolerance of 2 × 10⁻¹³. The median of five fresh single-core runs on an Intel Xeon Platinum 8573C used GCC 14.2, the default `-O2` portable build and one thread.

| Mode | Budgeted tensor peak | Live tensor memory after backward and collection | Median backward |
| --- | ---: | ---: | ---: |
| Standard backward | 36.25 MiB | 36.25 MiB | 32.3 ms |
| Retain input gradient and consume output tape | 18.38 MiB | 0.25 MiB | 21.9 ms |
| Checkpoint every eight steps with retained backward | 3.75 MiB | 0.25 MiB | 52.2 ms |

The peak is Arda's tensor budget counter across forward and backward. Timing starts after forward and includes work done inside backward, including consumption. The final column excludes collection called separately after backward. [JAX checkpoint comparison](../../tests/reference_vilya_jax.py) · [Failure and memory-budget regressions](../../tests/test_vilya_runtime.py)

## Activation offloading

`offload_activation` in `src/autodiff.c` replaces a selected activation with an immutable, logically contiguous Float64 snapshot while preserving its parent edges. `offload_snapshot` in `src/tensor_storage.c` writes it with the existing little-endian tensor format and maps the unlinked file. Derivative rules request their saved operands through `src/activation_store.c`. A sweep holds up to three aligned read windows for a tape node and reuses the reservation across nodes. A snapshot of $N$ values writes $8N + 128$ bytes. With $B$ entries per window, a sequential traversal of one snapshot needs $\lceil N/B\rceil$ chunk fills. Interleaved or strided accesses can refill evicted windows. Forward builds each activation normally before spilling. The runtime accounts for disk, tensor heap and scratch separately from the OS page cache.

PyTorch saved tensor hooks demonstrate explicit packing and unpacking of activations that backward needs. A newer PyTorch CPU offloading design schedules transfers with graphs and supports prefetching. ZeRO-Infinity addresses GPU, CPU and storage tiering in distributed training. Arda reuses its own tensor file format and uses synchronous transfers and user-selected placement in its single-process CPU runtime. No source code was reused from these references. [PyTorch autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html#hooks-for-saved-tensors) · [PyTorch CPU offloading](https://docs.pytorch.org/devlogs/distributed/2026-06-23-cpu-offloading/) · [ZeRO-Infinity](https://arxiv.org/abs/2104.07857)

A CPU JAX Float64 reference checks the loss, VJP, JVP and Hessian for mixed spill and checkpoint graphs. Runtime tests check exact storage boundaries, mutation, read-only aliasing, strided inputs, collection and transactional gradient failures. [JAX reference](../../tests/reference_vilya_jax.py) · [Runtime regressions](../../tests/test_vilya_runtime.py)

### Local activation storage benchmark

A 16,384-entry Float64 vector started at 0.2 and ran `sin(state * 0.97 + 0.02)` for 48 steps. The loss was the sum of the final vector. Every run used retained-gradient backward with tape consumption. The full spill mode offloaded all three intermediates per step and collected after each step. The checkpoint mode packed eight steps per segment and collected after each segment. The combined mode spilled only those checkpoint boundaries. The spill plan used a 32 MiB disk quota and 64 KiB windows.

The medians come from five fresh processes per mode after one warm-up per mode. The process was pinned to one core on an Intel Xeon Platinum 8573C. Both versions used GCC 14.2, the default portable `-O2` build and one thread. Timing covers the forward loop and the backward call separately, with collection and file activity inside those intervals. Process peak RSS is Linux `wait4`'s maximum resident set size. All loss and gradient comparisons against 0.17.2 satisfied a relative tolerance of 2 × 10⁻¹³.

| Mode | Forward (ms) | Backward (ms) | Peak budgeted memory (MiB) | Peak process RSS (MiB) | Written / read (MiB) | Scalar replay operations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.17.2 retained | 67.91 | 22.55 | 18.38 | 19.71 | 0 / 0 | 0 |
| 0.17.3 retained | 66.73 | 23.66 | 18.38 | 19.73 | 0 / 0 | 0 |
| 0.17.3 spill every intermediate | 71.53 | 31.22 | 0.75 | 20.66 | 18.02 / 6.00 | 0 |
| 0.17.3 checkpoints | 65.27 | 53.88 | 3.75 | 11.88 | 0 / 0 | 2,359,296 |
| 0.17.3 spilled checkpoints | 66.23 | 53.36 | 3.13 | 11.88 | 0.75 / 0.63 | 2,359,296 |

Full spilling brought budgeted tensor memory down by 96% in this graph. Process RSS rose slightly because the mapped pages could remain resident. The checkpoint modes reduced RSS and spent more time replaying operations. File I/O figures count logical traffic through Arda on the local overlay filesystem. Device throughput and larger-than-RAM RSS behavior require separate measurements.
