# Arda

**Compute. Optimize. Understand.**

Arda is an experimental, standalone scientific programming language written in C11. Its indentation-based syntax offers Tolkien-inspired keywords and conventional English aliases. The project aims to make high-performance, resource-aware and explainable scientific computing accessible on everyday laptops, while scaling to workstations and HPC clusters.

**Current version: 0.13.1 — Scientific Maintenance. Next planned release: 0.14 — Moria.**

Arda is an early-stage prototype, not production-validated scientific software or a sandbox. Features in the roadmap are proposals, not implemented capabilities.

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

Python is used by development tools, tests and embedded-library generators; it is not required to run compiled Arda programs through the runtime. The default build requires no external numerical libraries. Optional BLAS/LAPACK:

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
| Large structured problems | Dense Jacobi-PCG, CGLS, specialized matrix-free 1–8D Cartesian Poisson solver |

General sparse matrices, tensor AD, general multithreading, GPU execution, MPI, disk-backed tensors and mathematical introspection are **not implemented yet**.

## Numerical standard library

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

Arrays are row-major Float64, with GC-traced ownership. Arithmetic `+ - * /` is elementwise with broadcasting; `matmul` is explicit and currently rank two. Reshape requires contiguous input with the same element count; views use positive steps; permutation and views share the backing buffer. Row indexing and `transpose` return copies. Equality is object identity, not an elementwise comparison. `bytes` measures logical data, not exclusive allocated storage. Some operations materialize non-contiguous inputs. Dense `solve` uses partial-pivot Gaussian elimination, not an explicit inverse; condition/error estimation is not implemented.

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

The keywords are artistically inspired by Quenya and Sindarin, not authoritative linguistic translations. Both spelling sets map to identical internal tokens and can be mixed.

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
| `ROADMAP_0.14.md` | Initial sparse specification proposal |
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

The 0.13.1 maintenance report records 126 regular and eight independent reference test methods passing in default, BLAS/LAPACK and sanitizer builds on Debian 13/GCC 14.2. These are historical local results, not GitHub CI results or a general numerical correctness guarantee. Historical performance measurements are in `benchmarks/`; compare only under matching configurations and accuracy requirements.

## Roadmap: from laptops to HPC

| Version | Codename | Focus |
| --- | --- | --- |
| 0.14 | Moria | COO/CSR, sparse algebra and memory accounting |
| 0.15 | Rúmil | Scientific foundations and mathematical documentation |
| 0.16 | Mithril | Virtual tensors, file-backed storage and compact representations |
| 0.17 | Vilya | Memory-efficient tensor autodiff |
| 0.18 | Shadowfax II | High-performance CPU execution, SIMD, multicore and ACE v1 |
| 0.19 | Aulë | Advanced numerical solvers and mixed precision |
| 0.20 | Ulmo | Scientific simulation, adaptive ODE/PDE and FEM |
| 0.21 | Khazad-dûm | Beyond-RAM algorithms and streaming |
| 0.22 | Palantír | Adaptive compute engine and language runtime |
| 0.23 | Eärendil | Interoperability and MPI foundations |
| 0.24 | Fëanor | Optimizing compiler |
| 0.25 | Ancalagon | Hybrid and distributed HPC |
| 0.26 | Ainulindalë | Advanced machine learning |
| 0.27 | Imladris | Developer ecosystem |
| 0.28 | Mandos | Scientific and performance validation |
| 0.29 | Valinor | Release candidate |
| 1.0 | Eä | Stable scientific runtime |

Read the [full master roadmap](ROADMAP.md) for detailed subversions, architecture, benchmark targets, reference research and Tolkien codenames.

The planned mathematical introspection API includes `getTex()`, `getCost()`, `getPlan()`, `getAssumptions()`, `verify()` and bounded `trace()`. Formula export is intended to work without running the numerical calculation. Execution tracing and verification will be optional, with no per-element logging in the normal high-performance path. Planned memory and compute engines will combine sparse and structured representations, SIMD, parallel execution, memory mapping, streaming, quantization and optional MPI/GPU backends. These are goals, not present features. Full 70B-model training on ordinary laptops is not a project promise.

## Limitations and safety

- The runtime is not sandboxed; run only trusted scripts. Arda code can access files with the process's permissions.
- Dense arrays have an existing 16,000,000-element cap and rank limit of eight; this is not a total process memory limit.
- Numerical error and convergence depend on conditioning and problem structure; independent reference tests are not formal guarantees.
- The portable runtime is single-threaded; external BLAS may manage its own threads. No general GPU, MPI, JIT or package manager yet.
- The C API, numerical APIs and language semantics are experimental and may change before 1.0.

## License and attribution

[MIT License](LICENSE). Arda's names, release codenames and keyword inspiration are artistic references to Tolkien's legendarium. This is an independent project, with no official affiliation. No rights to third-party trademarks or illustrations are granted by the code license.
