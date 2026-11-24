# Arda

**Adaptive Runtime for Data and Algorithms**

*Compute. Optimize. Understand.*

Arda is an independent, open-source, research-driven scientific programming language for numerical mathematics, resource-aware execution, and mathematical transparency. Its standalone C11 implementation is distributed under the MIT license.

Its long-term goal is to combine scientifically validated numerical algorithms with adaptive execution, from ordinary laptops to distributed research clusters. The language brings native numerical data structures and algorithms to an indentation-based runtime for physics, engineering, and other computational research.

**Current version** — Arda 0.16.2 — Typed Tensors and File Storage

*Development codename: Mithril*

**Next milestone** — Arda 0.16.3 — Buffer Pools, Workspace Reservations, and Memory Budgets

**Wiki** — [Quickstart, language and module guides, and AMDE examples](https://github.com/Saxermi/Arda/wiki)

Arda supports typed tensors, read-only tensor file mapping, static mathematical queries, numerical solvers, and compact matrix operators. The optional LAPACK build adds randomized SVD for dense, sparse, and structured operators. [Tensor API and validation](research/implemented/typed-tensors.md)

> **Development status:** Arda is an early-stage scientific language with dense and sparse tensors, bounded tensor-file reads, selected optimized CPU kernels, and scalar automatic differentiation. Automatic out-of-core planning, general mathematical introspection, and distributed HPC are roadmap goals.

## What is Arda?

The standalone `arda` executable compiles `.arda` source files to bytecode and runs them in a stack-based virtual machine. The runtime manages objects, function calls, modules and garbage collection, while performance-sensitive numerical operations are implemented in native C.

This design aims to combine the convenience of a high-level scripting language with the efficiency of specialized numerical kernels. A user can write a differential equation, manipulate a tensor, define a function or implement an optimization algorithm without manually managing every memory allocation. At the same time, the numerical layer can operate directly on contiguous native buffers instead of interpreting every scalar operation as a separate language instruction.

Arda is **CPU-first, not CPU-only**. Its portable implementation does not require a dedicated GPU or a large external numerical framework. Optimized hardware-specific kernels and optional libraries complement the portable implementation. GPU acceleration, multi-core scheduling and distributed execution are intended to extend this foundation, not replace it.

## Why develop another scientific language?

Python, Julia, MATLAB, Mathematica, C++, Fortran and their ecosystems already offer excellent scientific tools. Arda does not assume that existing numerical libraries are inefficient, nor does it claim to outperform them. Instead, it explores how several capabilities can be designed to work together from the beginning:

| Design goal | What it means for Arda |
| --- | --- |
| **High performance** | Native numerical kernels, SIMD, cache-aware algorithms, and optional optimized backends; eventually multicore execution, fusion and specialized compilation. |
| **Resource awareness** | Make memory consumption, computation cost and data movement explicit; choose suitable representations and algorithms within available resources. |
| **High-dimensional mathematics** | Exploit sparsity, matrix-free operators, low-rank factorizations and compressed tensors instead of blindly materializing enormous dense arrays. |
| **Beyond-RAM computing** | Eventually process file-backed data and models in bounded chunks, with controlled I/O, prefetching and offloading. |
| **Mathematical transparency** | Eventually describe equations, assumptions, algorithms and costs directly from functions without requiring a numerical run. |
| **Numerical reliability** | Independent reference tests today; planned verification, uncertainty propagation, units and reproducible experiment records. |
| **Scalable execution** | Preserve efficient local CPU execution while developing optional GPU, NUMA and MPI support for larger systems. |
| **Accessibility** | A compact standalone runtime with approachable syntax and a portable baseline, suitable for education, research and embedded numerical applications. |

The intended distinction is the integration of these capabilities, not the invention of every underlying technique. A scientific function should ultimately be something that can be **executed, analyzed, optimized, verified and documented** using consistent mathematical objects and interfaces.

## The mathematical idea: exploit structure before adding hardware

A faster processor cannot fix every inefficient mathematical representation. Consider a tridiagonal linear system with one million unknowns. A conventional dense Float64 matrix requires 8 TB of storage, even though only 2,999,998 coefficients are nonzero. A CSR representation with 32-bit column indices and 64-bit row offsets requires approximately 44 MB for its three principal buffers.

Similarly, a regular-grid differential operator can often be applied directly to a vector using a stencil without storing any matrix at all. Arda already has a specialized matrix-free Poisson solver in its `high` module. For a grid of 128 × 128 × 128 points, a hypothetical dense operator would occupy 32 TiB; six Float64 work vectors occupy only 96 MiB. These figures describe theoretical storage, not measured peak memory or a universal solver workspace requirement.

Arda's development strategy is to exploit such mathematical structure first, then optimize the resulting kernels using cache locality, SIMD, parallelism and suitable external libraries. Moria provides native COO/CSR storage, typed assembly, sparse algebra, and a shared internal operator interface for dense, CSR, and matrix-free Poisson kernels.

This approach applies equally to scientific simulation and machine learning. Quantization, low-rank approximations, activation checkpointing and selective data loading can reduce the resources needed by suitable models. They do not eliminate the computational cost of arbitrary problems, and approximate methods must be explicitly permitted.

## Long-term vision: from a laptop to a research cluster

Arda is intended to support multiple levels of computing without forcing users to abandon the mathematical model when their hardware changes.

| Environment | Intended execution strategy |
| --- | --- |
| Everyday laptop | Efficient CPU kernels, compact data structures, bounded memory and optional SSD streaming |
| Multicore workstation | Parallel numerical kernels, cache-aware scheduling, NUMA-aware placement and optional accelerators |
| Research cluster | MPI, distributed tensors and sparse matrices, domain decomposition, distributed solvers and parallel I/O |

The planned **Adaptive Compute Engine (ACE)** will use problem structure, shape, precision requirements, available memory, compute capacity and communication costs to select among supported execution strategies. It will not silently lower precision, introduce a lossy approximation or allocate a dense representation of a huge sparse operator.

Beyond-RAM execution is not a promise that an SSD performs like RAM. Disk-backed algorithms must explicitly control data movement and working storage, and their practical speed depends on I/O bandwidth and access patterns. Very large quantized model inference is an experimental research goal, not a guarantee of interactive performance; full pretraining of a 70B model on an ordinary laptop is not a project target.

## Mathematical documentation and introspection

The **Arda Mathematical Documentation Engine (AMDE)**, development codename *Rúmil*, documents selected native functions, straight-line user functions and supported single range loops. `nc.solve.getTex(3)` shows pivoted LU, and `nc.solve.getCost({"shape":[10,10]})` estimates arithmetic and buffer sizes from dimensions. [Query contracts](research/implemented/mathdoc.md) · [Runnable example](examples/mathdoc_queries.arda)

A simple energy function can be written today using existing language syntax:

```arda
caro energy(m, v, h):
    esta kinetic = 0.5 * m * v * v
    esta potential = m * 9.81 * h
    nan kinetic + potential
```

A query such as `energy.getTex(0)` describes the source expression as LaTeX. Its mathematical meaning (shown here in simplified notation) is:

$$
E(m,v,h)=\frac{1}{2}mv^2+9.81mh
$$

For supported user functions, levels 0–3 describe a compact expression, intermediate relations, known assumptions and source-level operations respectively. No dynamic call target is inferred. Level 0 applies limited formal scalar-real display simplification to call-free functions without external dependencies; levels 1 and 3 preserve source operations. Compact metadata is enabled by default; `--mathdoc=none` disables user-function metadata collection while retaining native descriptors. See the query contracts for limits and unsupported constructs.

The intended levels range from a compact equation, through intermediate expressions and assumptions, to the mathematical algorithm used by a supported native function. An optional recorded run could also provide selected numerical intermediate results and convergence data. Static documentation must not execute the numerical function or pretend that an unexecuted calculation has converged.

The broader proposed introspection interface includes:

| API | Question it answers |
| --- | --- |
| `getTex(level)` / `getMath()` | What mathematics does this function express? |
| `getAlgorithm()` / `getAssumptions()` | Which mathematical method and prerequisites apply? |
| `getCost()` / `getPlan()` | What resources might it need, and how could it execute? |
| `getUnits()` / `getSensitivity()` | What physical dimensions and parameter sensitivities matter? |
| `verify()` | Does a computed result satisfy the relevant numerical criteria? |
| `trace()` / `reproduce()` | What happened during a recorded run, and can it be repeated? |

The linked contracts list the implemented queries. Other entries in the table describe future work, including methods that require a concrete numerical run. Native algorithm descriptions identify the applicable backend.

A key architectural requirement is **zero-cost scientific introspection as a design goal**: optional tracing, symbolic analysis and documentation must not insert logging, allocations or additional checks into numerical hot loops when unused. Trace collection should have explicit limits on depth, event count and storage. Basic safety checks remain enabled in every execution mode.

## Architecture and development philosophy

Arda compares established methods with relevant research before making substantial numerical changes. The research notes record mathematical assumptions, implementation provenance, and local validation. Algorithm choices prioritize accuracy, numerical reliability, portability, and resource use under equivalent workloads. [Scientific research](research/README.md)

The current runtime is deliberately small. The roadmap expands it incrementally into the following logical subsystems, rather than replacing the existing interpreter:

```text
Arda source (.arda)
  |
  +-- Lexer / Parser / Bytecode compiler
  |       |
  |       +-- VM / Garbage collector / Native modules
  |
  +-- Planned scientific infrastructure
          |
          +-- AMDE           Mathematical descriptions and reports
          +-- AML            Differentiation and machine learning
          +-- ACE            Adaptive planning and scheduling
                 |
                 +-- AME     Buffers, memory budgets and file storage
                 +-- AKE     Numerical kernels, SIMD and parallelism
                 +-- HPC     Partitioning, MPI and distributed operators
```

The name describes Arda's architectural direction.

- **Adaptive** refers to algorithm selection and resource-aware planning through the planned Adaptive Compute Engine (ACE).
- **Runtime** refers to the standalone compiler, virtual machine, and numerical execution environment.
- **Data** includes dense tensors and sparse matrices today, with additional structured representations and beyond-RAM storage planned through the Arda Memory Engine (AME). AME will distinguish logical tensor size from physical storage.
- **Algorithms** include native numerical kernels and research-driven methods. The planned Arda Kernel Engine (AKE) will manage optimized execution. Mathematical documentation and Arda Machine Learning (AML) will build on the same representations.

The planned distributed computing and HPC infrastructure will extend the shared operator and storage abstractions to distributed memory. The portable C11 runtime is the development baseline.

Performance is measured, not assumed. Comparisons should use equivalent problems, precision, convergence criteria, hardware and optimized reference implementations. Runtime, peak memory, data movement, numerical error and scientific productivity all matter. The detailed [master roadmap](ROADMAP.md) sets out the release milestones and validation goals.

## Who is Arda for?

The project is intended for students and educators learning numerical methods, researchers building reproducible simulations, engineers solving scientific and physical problems, developers working with resource-constrained numerical applications, and eventually ML and HPC practitioners. Early releases focus on a small, reliable numerical core rather than trying to match the breadth of mature scientific ecosystems.

Arda is currently a prototype suitable for exploration and experimentation, not production-validated scientific software or a security sandbox. Planned capabilities are not guarantees of eventual performance or support for every mathematical workload.

## Quick start (Ubuntu/Linux)

```sh
sudo apt install build-essential python3
git clone https://github.com/Saxermi/Arda.git
cd Arda
make -j
./arda --version
./arda examples/hello.arda
./arda examples/science.arda
```

Python 3.10 or newer is required for the regression suite. Python is used by development tools, tests and embedded-library generators; it is not required to run compiled Arda programs through the runtime. The default build requires no external numerical libraries. Optional BLAS/LAPACK:

```sh
sudo apt install libblas-dev liblapack-dev
make BLAS=1 LAPACK=1
./arda examples/highdim.arda
```

Switching build configurations does not require `make clean`. The `arda` symlink selects the most recently built configuration. To install the selected executable system-wide: `sudo install -m 755 arda /usr/local/bin/arda`.

## A taste of Arda

```arda
tol numc as nc

caro magnitude(x, y):
    nan nc.sqrt(x*x + y*y)

pedo(magnitude(3, 4))
esta A = nc.array([[3, 2], [1, 2]])
esta b = nc.array([7, 5])
pedo(nc.solve(A, b))
```

Expected mathematical results: 5 and the vector [1, 2]. English aliases can be used instead:

```arda
import numc as nc

def magnitude(x, y):
    return nc.sqrt(x*x + y*y)

print(magnitude(3, 4))
```

## Implemented features

| Area | Current capabilities |
| --- | --- |
| Language | Dynamic typing, indentation, lexical scopes, first-class functions, closures, recursion, default parameters, lambdas, list comprehensions, classes, single inheritance, exceptions, modules, UTF-8 strings |
| Runtime | Bytecode compiler and stack VM, incremental marking GC, weak string interning, bytecode-root collection, recoverable frontend errors in REPL/imports |
| Numerical arrays | Float64 dense tensors (rank 1–8), trailing-axis broadcasting, strided views, reductions, rank-two matrix multiplication and linear solves |
| CPU kernels | Portable C, SSE2 elementwise arithmetic, blocked matrix multiplication with runtime-dispatched AVX2 on supported x86 CPUs; optional CBLAS and LAPACK |
| Numerical methods | Composite quadrature, radix-2 FFT/IFFT, RK4, adaptive Dormand–Prince RK45, L-BFGS |
| Differentiation | Scalar reverse-mode AD, SGD and Adam vector updates |
| Large structured problems | Dense and sparse Jacobi-PCG, sparse CG, CGLS, specialized matrix-free 1–8D Cartesian Poisson solver |
| Sparse matrices | Immutable COO/CSR, typed assembly, CSR algebra, Matrix Market I/O, sparse memory budget |

Tensor AD, general multithreading, GPU execution, MPI, disk-backed tensors and general mathematical introspection remain roadmap goals.

## Native sparse matrices

```arda
tol sparse as sp
esta A = sp.coo(3,4,[2,0,0,2,0],[3,1,1,0,3],[7,2,3,4,-1])
esta B = sp.to_csr(A)
pedo(sp.components(B))
pedo(sp.components(sp.to_coo(B)))

tol numc as nc
esta K = sp.tridiag(5,-1,4,-1)
esta rhs = sp.matvec(K,nc.array([1,2,3,4,5]))
esta result = sp.pcg(K,rhs,1e-10,100)
pedo(result["reason"],result["residual"],nc.tolist(result["solution"]))
```

Expected CSR components: `[[0, 2, 2, 4], [1, 3, 0, 3], [5, -1, 4, 7]]`.

COO allows duplicates and explicit zeros; CSR uses strictly sorted columns within each row, combines duplicates in input order, and omits exact zeros. No dense intermediate is required. Explicit `sp.todense` and `sp.from_dense` bridges are available, along with shape, nnz, bytes, format, checked get, copied components, and static `sp.describe` conversion metadata. See the [0.14.3 sparse computing guide](docs/MORIA_0.14.3.md) for assembly, algebra, solver and budget contracts, plus local validation and benchmarks. The [0.14.0 report](docs/MORIA_0.14.0.md) covers the original storage layout and conversion measurements.

## Numerical standard library

The `tensor` module supports typed CPU arrays and file-backed storage. Use `tensor.array(values,dtype[,shape])` for flat input, `tensor.cast` for explicit conversion, and `tensor.to_numc` to materialize a Float64 array for the existing kernels. Use `tensor.mmap` for read-only views or `tensor.read_block` for bounded reads that handle truncated files with an error. [Typed tensor and file contracts](research/implemented/typed-tensors.md)

### `numc`: dense arrays and linear algebra

```arda
tol numc as nc
esta x = nc.linspace(0, 4, 5)
esta y = 2*x + 1
pedo(y, nc.mean(y))
esta A = nc.array([[3, 2], [1, 2]])
pedo(nc.solve(A, nc.array([7, 5])))
```

| API | Purpose |
| --- | --- |
| `array`, `zeros`, `ones`, `linspace` | Dense tensor construction |
| `shape`, `bytes`, `tolist`, `backend`, `simd` | Metadata and conversion |
| `sum`, `mean`, `var`, `std`, `min`, `max` | Full reductions; population variance/std |
| `dot`, `norm`, `matmul`, `transpose`, `solve` | Dense vector/matrix algebra |
| `sin`, `cos`, `exp`, `log`, `sqrt` | Elementwise mathematics |
| `reshape`, `view`, `permute`, `copy`, `contiguous` | Tensor layout and views |
| `get`, `set`, `at`, `put` | Element access and mutation |

Arrays are row-major Float64, with GC-traced ownership. Arithmetic `+ - * /` is elementwise with broadcasting; `matmul` is explicit and currently rank two. Reshape requires contiguous input with the same element count; views use positive steps; permutation and views share the backing buffer. Row indexing and `transpose` return copies. Equality is object identity, not an elementwise comparison. `bytes` measures logical data, not exclusive allocated storage. Some operations materialize non-contiguous inputs. The one-shot dense `solve` uses partial-pivoted LU and returns the solution.

### `linalg`: matrix factorizations and direct solvers

The portable build includes reusable LU, Cholesky and QR, dense utilities, and direct tridiagonal and banded solves. Build with `make LAPACK=1` for `svd`, `pinv`, `rank`, `cond`, `eigh`, and `eig`. `calc.backend()` reports `"lapack"` when these operations are available. [Portable examples](examples/factorizations.arda) · [SVD contracts](research/implemented/linear-algebra.md#lapack-reduced-svd) · [Eigenvalue example](examples/eigenpairs.arda)

```arda
tol numc as nc
tol linalg as la
esta a = nc.array([[4,0],[0,1],[0,0]])
esta parts = la.svd(a)  # [U, s, Vt]
pedo(nc.tolist(parts[1]), la.rank(a), la.cond(a))  # [4, 1] 2 4
pedo(nc.tolist(la.pinv(a)))  # [[0.25, 0, 0], [0, 1, 0]]
```

### `random` and `stats`

`random.generator(seed[,stream])` creates an explicit PCG32 generator for uniform, normal, exponential, integer and Poisson draws, as well as choice and shuffle. `stats` provides quantiles, covariance, correlation, regression, histograms, variance and standard deviation, plus normal and exponential probability functions. `stats.online()` accumulates a count, mean and population variance without storing samples. [Base API contracts](research/implemented/random-and-statistics.md) · [Distribution and dispersion APIs](research/implemented/mathematical-library.md#distributions-and-dispersion)

### `complex`, `linop`, and randomized SVD

`complex.new(re,im)` creates a scalar with real and imaginary components. The module exposes conjugation, magnitude, phase and elementary functions. `linop` builds compact low-rank, diagonal, banded, Toeplitz and Kronecker operators and applies them or their transposes to vectors or blocks. With `LAPACK=1`, `linalg.rsvd(A,rank[,oversampling[,power_iterations[,seed]]])` returns `[U,s,Vt,estimated_relative_error]`. [Runnable example](examples/structured_computing.arda) · [API contracts](research/implemented/structured-computing.md)

### `calc`, `ode`, `opt`: numerical algorithms

| API | Algorithm |
| --- | --- |
| `calc.trapz(y,h)`, `calc.simpson(y,h)` | Composite rules on uniform samples |
| `calc.fft(a)`, `calc.ifft(a)` | Radix-2 Cooley–Tukey, N×2 real/imaginary data |
| `calc.rk4_linear(A,y0,h,steps)` | Native RK4 for y'=Ay |
| `calc.lapack_solve(A,b)`, `calc.eigh(A)` | Optional LAPACK DGESV/DSYEV |
| `ode.integrate(f,a,b,n=100)` | Composite Simpson quadrature with callback |
| `ode.rk4(f,t0,y0,h,steps)` | Classical explicit RK4, scalar/tensor state |
| `ode.rk45(f,t0,y0,t1,h,rtol=1e-6,atol=1e-9,maxiter=100000)` | Adaptive Dormand–Prince 5(4), returns `[final_state,accepted_steps,rejected_steps]` |
| `opt.lbfgs(f,x0,tolerance=1e-6,maxiter=500,memory=10)` | User-supplied gradient, L-BFGS with Armijo backtracking; returns `[x,iterations,gradient_norm,converged,value]` |

The `ode` and `opt` modules are written in Arda and embedded at build time. Callback functions should not mutate their inputs. FFT lengths must be powers of two up to 1,048,576. The optional LAPACK routines require a LAPACK build; `nc.solve` is available in the portable build. RK45 controls estimated local error, not a guaranteed global error. There is no general stiff solver or event detection yet.

### `special` and `numerics`

The `special` module provides real Gamma, Beta, integer-order Bessel and error functions. Use `numerics.brent(f,a,b)` for a sign bracket, `numerics.newton(f,df,a,b)` when a derivative is available, and `numerics.quad(f,a,b)` for adaptive scalar integration. `numerics.linear(x,y)` and `numerics.spline(x,y)` return reusable interpolants. [API contracts](research/implemented/mathematical-library.md) · [Runnable example](examples/mathematical_library.arda)

### `ad`: scalar automatic differentiation

```arda
tol ad
esta x = ad.variable(3)
esta y = ad.variable(2)
esta f = x*x + x*y
ad.backward(f)
pedo(ad.value(f), ad.grad(x), ad.grad(y))
```

Expected output: `15 8 3`. Scalar reverse-mode AD records a GC-managed DAG and accumulates gradients in reverse topological order. Supported arithmetic and elementary functions include `sin`, `cos`, `exp`, `log`, `sqrt`, `tanh`, and constant-exponent `pow`. `sgd` and `adam` return updated vectors; these are not general neural-network layers or tensor AD. See [learning example](examples/learning.arda).

### `high`: structured and high-dimensional numerical methods

| API | Purpose |
| --- | --- |
| `cg(A,b,tolerance,maxiter)` | Dense Jacobi-preconditioned conjugate gradient; symmetric SPD matrix required |
| `poisson(b,tolerance,maxiter)` | Matrix-free negative discrete Laplacian with homogeneous exterior Dirichlet conditions and unit spacing |
| `laplace(x)` | Apply the same negative discrete Laplacian |
| `lstsq(A,b,tolerance,maxiter)` | Rectangular CGLS without explicitly forming AᵀA |
| `axpy(x,y,a)`, `inner(x,y)`, `sum_axis(x,axis)` | Fused scale/add, tensor inner product, axis reduction |
| `finite(x)`, `rkerror(y,z,low,rtol,atol)` | Finite checks and weighted RMS error estimator |

The iterative solvers return `[solution,iterations,relative_residual,converged]`. CG/Poisson use the relative data residual; CGLS uses the relative normal-equation residual. SPD is a caller assumption, with symmetry checks and numerical breakdown detection; convergence and accuracy are not universally guaranteed. The matrix-free Poisson solver does not allocate an N×N matrix: its per-iteration work is O(dN) for d axes and N grid points, and its working storage is O(N). See [high-dimensional examples](examples/highdim.arda).

## Language identity

Both keyword spellings are supported and can be mixed. Each pair maps to the same internal token. The optional spellings draw on Quenya and Sindarin as artistic inspiration, with keyword semantics defined by Arda.

| Arda | English | Meaning |
| --- | --- | --- |
| `esta` | `let` | Declare a variable |
| `caro` | `fn`, `def` | Define a function |
| `nan` | `return` | Return |
| `qui` | `if` | Conditional |
| `ai` | `else` | Alternative |
| `tenna` | `while` | Loop |
| `an` | `for` | Iteration |
| `pedo` | `print` | Output |
| `nae` | `true` | True |
| `law` | `false` | False |
| `mun` | `nil` | Null value |
| `nost` | `class` | Class |
| `tol` | `import` | Module import |

Blocks use `:` and spaces for indentation; semicolons and indentation tabs are rejected. Strings are UTF-8 and index Unicode code points. Integer arithmetic is checked signed 64-bit; floating-point arithmetic uses double precision. Modules are resolved relative to the importing file and cached per VM. Use ordinary English scientific API names regardless of keyword dialect.

## Tools and project layout

```sh
./arda --tokens examples/fibonacci.arda
./arda --bytecode examples/fibonacci.arda
./arda --gc-stress --gc-stats examples/highdim.arda
./arda
```

The REPL accepts one statement per line and blocks terminated by a blank line. `:quit` exits. Since 0.13.1, syntax errors no longer terminate the REPL. Unhandled errors in scripts return nonzero status. No stable serialized bytecode or public embedding API yet.

| Path | Purpose |
| --- | --- |
| `src/` | C11 lexer, parser, frontend, compiler, VM, GC and numerical modules |
| `lib/` | Embedded Arda standard library |
| `examples/` | Language and scientific examples |
| `tests/` | Local regression and independent reference tests |
| `benchmarks/` | Benchmark programs and historical results |
| `tools/` | Embedded source generators |
| `docs/` | Historical reference documentation |
| `ROADMAP.md` | Full development roadmap (German, with English technical terminology) |
| `ROADMAP_0.14.md` | Moria sparse milestone scope |
| [`DECISIONS.md`](DECISIONS.md) | Engineering decisions, evidence, and when to revisit them |
| [`research/`](research/README.md) | Scientific references, implementation provenance, and future algorithm comparisons |
| `docs/MORIA_0.14.0.md` | Sparse storage and conversion foundations |
| `docs/MORIA_0.14.3.md` | Assembly, algebra, solvers, memory budgets and local validation |
| `docs/MORIA_0.14.4.md` | Sparse release validation and benchmark results |
| `docs/RUMIL_0.15.0.md` | Dense factorization and direct solver release validation |
| `docs/RUMIL_0.15.1.md` | SVD and eigenvalue release validation |
| `docs/RUMIL_0.15.2.md` | Random generator and statistics release validation |
| `docs/RUMIL_0.15.3.md` | Complex scalars, matrix operators and derivative-free optimization |
| `docs/RUMIL_0.15.4.md` | Mathematical documentation and AMDE overhead measurements |
| `docs/RUMIL_0.15.5.md` | Scalar mathematical library and static range-loop validation |
| `CHANGELOG.md` | Release history |
| `MAINTENANCE.md` | 0.13.1 changes and measured validation |
| `CONTRIBUTING.md` | Local development and validation policy |

## Tests — local only

This project deliberately does **not** use GitHub Actions or hosted CI. Run tests on your own machine; no GitHub runner time is consumed.

```sh
make test
make sanitize
make check-generated
make BLAS=1 LAPACK=1 test
```

Optional independent numerical tests (NumPy/SciPy):

```sh
sudo apt install python3-venv
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
make test-reference
```

`make sanitize` enables ASan, UBSan and leak detection. `make format` and `make format-check` require clang-format. Generated module headers should not be formatted by hand.

0.14.0 passed 136 regular and nine independent reference methods locally in all three build configurations; see the [0.14.0 sparse foundations report](docs/MORIA_0.14.0.md). The 0.13.1 maintenance report records 126 regular and eight independent reference test methods passing in default, BLAS/LAPACK and sanitizer builds on Debian 13/GCC 14.2. These are historical local results, not GitHub CI results or a general numerical correctness guarantee. Historical performance measurements are in `benchmarks/`; compare only under matching configurations and accuracy requirements.

## Roadmap: from laptops to HPC

| Version | Scientific or technical milestone | Development codename |
| --- | --- | --- |
| 0.14 | [Resource-Efficient Sparse Computing](ROADMAP.md#arda-014--resource-efficient-sparse-computing) | Moria |
| 0.15 | [Numerical Methods and Mathematical Documentation](ROADMAP.md#arda-015--numerical-methods-and-mathematical-documentation) | Rúmil |
| 0.16 | [Virtual Tensors and Memory Management (AME v1)](ROADMAP.md#arda-016--virtual-tensors-and-memory-management-ame-v1) | Mithril |
| 0.17 | [Memory-Efficient Differentiable Programming](ROADMAP.md#arda-017--memory-efficient-differentiable-programming) | Vilya |
| 0.18 | [High-Performance CPU Execution and Adaptive Planning (AKE v1, ACE v1)](ROADMAP.md#arda-018--high-performance-cpu-execution-and-adaptive-planning-ake-v1-ace-v1) | Shadowfax II |
| 0.19 | [Advanced Numerical Solvers](ROADMAP.md#arda-019--advanced-numerical-solvers) | Aulë |
| 0.20 | [Scientific Simulation](ROADMAP.md#arda-020--scientific-simulation) | Ulmo |
| 0.21 | [Beyond-RAM Data Processing and Visualization (AME v2)](ROADMAP.md#arda-021--beyond-ram-data-processing-and-visualization-ame-v2) | Khazad-dûm |
| 0.22 | [Language Runtime and Adaptive Execution (ACE v2)](ROADMAP.md#arda-022--language-runtime-and-adaptive-execution-ace-v2) | Palantír |
| 0.23 | [Scientific Interoperability and MPI Foundations](ROADMAP.md#arda-023--scientific-interoperability-and-mpi-foundations) | Eärendil |
| 0.24 | [Optimizing Compiler](ROADMAP.md#arda-024--optimizing-compiler) | Fëanor |
| 0.25 | [Accelerated, Hybrid, and Distributed HPC](ROADMAP.md#arda-025--accelerated-hybrid-and-distributed-hpc) | Ancalagon |
| 0.26 | [Machine Learning Inference and Training](ROADMAP.md#arda-026--machine-learning-inference-and-training) | Ainulindalë |
| 0.27 | [Scientific Developer Tools and Ecosystem](ROADMAP.md#arda-027--scientific-developer-tools-and-ecosystem) | Imladris |
| 0.28 | [Scientific and Performance Validation](ROADMAP.md#arda-028--scientific-and-performance-validation) | Mandos |
| 0.29 | [Release Candidate](ROADMAP.md#arda-029--release-candidate) | Valinor |
| 1.0 | [Stable Scientific Computing Language](ROADMAP.md#arda-10--stable-scientific-computing-language) | Eä |

Read the [full master roadmap](ROADMAP.md) for detailed subversions, architecture, benchmark targets, research references, and development codenames.

The roadmap extends mathematical queries with nested loop and control-flow analysis, `getPlan()`, `verify()` and bounded `trace()`. Execution tracing and verification will be optional, with no per-element logging in the normal high-performance path. Planned memory and compute engines will combine sparse and structured representations, SIMD, parallel execution, memory mapping, streaming, quantization and optional MPI/GPU backends. These are goals, not present features. Full 70B-model training on ordinary laptops is not a project promise.

## Limitations and safety

- The runtime is not sandboxed; run only trusted scripts. Arda code can access files with the process's permissions.
- Heap-backed dense tensors have a 16,000,000-element cap and a rank limit of eight. File-backed tensors may be larger and support bounded block reads. [File access limits](research/implemented/typed-tensors.md#tensor-files)
- Numerical error and convergence depend on conditioning and problem structure; independent reference tests are not formal guarantees.
- The portable runtime is single-threaded; external BLAS may manage its own threads. No general GPU, MPI, JIT or package manager yet.
- The C API, numerical APIs and language semantics are experimental and may change before 1.0.

## License and attribution

Arda is distributed under the [MIT License](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) for optional backends and development dependencies. For research citations, use [CITATION.cff](CITATION.cff).

Arda's naming conventions and optional Elvish-inspired syntax are inspired by the works of J. R. R. Tolkien. Arda is an independent project and is not affiliated with or endorsed by the Tolkien Estate or any associated rights holders.
