# Arda

**Compute. Optimize. Understand.**

Arda is an experimental, standalone scientific programming language implemented in C11. It combines an indentation-based syntax with Tolkien-inspired keywords and conventional English aliases.

The project aims to make high-performance, resource-aware and explainable scientific computing accessible on everyday laptops, while scaling to workstations and HPC clusters.

**Current source version: 0.13.1 — Scientific Maintenance.** The next planned release is **0.14 — Moria**. The source import into this new repository is pending; the documentation is being published first.

## Current capabilities

- Bytecode compiler, stack-based VM, garbage collection, functions, classes, modules, exceptions and a REPL.
- Native Float64 tensors (rank 1–8), broadcasting, strided views, reductions and dense linear algebra.
- Portable C numerical kernels, SSE2 elementwise arithmetic and runtime-dispatched AVX2 matrix multiplication on supported CPUs; optional BLAS and LAPACK.
- Numerical integration, FFT/IFFT, RK4, adaptive RK45 and L-BFGS.
- Scalar reverse-mode automatic differentiation and SGD/Adam updates.
- Iterative CG/CGLS and specialized matrix-free multidimensional Poisson solvers.

Native COO/CSR, general tensor autodiff, memory mapping, GPU execution, MPI and mathematical introspection are **planned, not implemented** in 0.13.1.

## Quick start (Ubuntu/Linux)

Once the source import is complete:

```sh
sudo apt install build-essential python3
git clone https://github.com/Saxermi/Arda.git
cd Arda
make -j
./arda --version
./arda examples/hello.arda
make test
```

Python is used by development tools and tests, not by the Arda runtime.

Optional numerical libraries:

```sh
sudo apt install libblas-dev liblapack-dev
make BLAS=1 LAPACK=1
```

## A taste of Arda

```python
tol numc as nc

caro magnitude(x, y):
    nan nc.sqrt(x*x + y*y)

pedo(magnitude(3, 4))

esta A = nc.array([[3, 2], [1, 2]])
esta b = nc.array([7, 5])
pedo(nc.solve(A, b))
```

Expected results: 5 for the magnitude and the vector [1, 2] for the linear system.

English alternatives are equivalent: `import`, `def`, `return`, `print`, and `let`.

## Long-term direction

| Release | Codename | Focus |
| --- | --- | --- |
| 0.14 | Moria | Sparse COO/CSR and memory accounting |
| 0.15 | Rúmil | Numerical foundations, mathematical documentation |
| 0.16 | Mithril | Tensor and memory engine, file-backed storage |
| 0.17 | Vilya | Memory-efficient tensor autodiff |
| 0.18 | Shadowfax II | High-performance CPU execution |
| 0.19 | Aulë | Advanced numerical solvers |
| 0.20 | Ulmo | Scientific simulation |
| 0.21 | Khazad-dûm | Beyond-RAM computing |
| 0.22 | Palantír | Adaptive compute engine |
| 0.23 | Eärendil | Interoperability and MPI foundations |
| 0.24 | Fëanor | Optimizing compiler |
| 0.25 | Ancalagon | Hybrid and distributed HPC |
| 0.26 | Ainulindalë | Machine learning |
| 0.27 | Imladris | Developer ecosystem |
| 0.28 | Mandos | Scientific and performance validation |
| 0.29 | Valinor | Release candidate |
| 1.0 | Eä | Stable scientific runtime |

Read the [full roadmap](ROADMAP.md) for architecture, release gates and research goals. Future APIs such as `getTex()`, `getCost()`, `getPlan()`, bounded tracing and reproducible experiments will be optional and designed to avoid overhead in numerical hot loops.

## Local testing (no hosted CI)

All tests are intended to run locally. This repository does not use GitHub Actions or hosted test runners.

```sh
make test
make sanitize
make check-generated
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
make test-reference
make BLAS=1 LAPACK=1 test
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development process and [CHANGELOG.md](CHANGELOG.md) for release history.

## Validation and limitations

The original 0.13.1 maintenance report records 126 regression and eight independent NumPy/SciPy reference test methods passing in default, BLAS/LAPACK and sanitizer builds on Debian 13/GCC 14.2. These are historical results, not a GitHub CI run or a general correctness guarantee.

Arda is a prototype, not a security sandbox or production-validated scientific solver. Dense arrays are currently limited to 16 million elements. Performance claims require reproducible comparisons with optimized reference libraries at equal accuracy.

## Tolkien-inspired identity

The syntax and release codenames are artistic references to Tolkien's legendarium, not authoritative translations or an official affiliation. Scientific APIs use conventional English names.

## License

[MIT](LICENSE).
