# Scientific research in Arda

This directory records the mathematics behind Arda's scientific features and compares candidates for future milestones. The [roadmap](../ROADMAP.md) tracks scope and sequencing. The [decision log](../DECISIONS.md) records the outcome of project discussions. The source and tests establish implemented behavior. This inventory covers Arda through 0.16.5 and the implemented Vilya changes on the 0.17 development branch.

| Implemented | Planned or experimental candidates |
| --- | --- |
| [Dense linear algebra and reductions](implemented/linear-algebra.md) | [Factorizations, statistics, quadrature and special functions](planned/numerical-libraries.md) |
| [Typed tensors, shared buffers and file storage](implemented/typed-tensors.md) | |
| [Block quantization and mixed-precision products](implemented/quantization.md) | |
| [Low-rank tensors](implemented/low-rank-tensors.md) | |
| [Scientific quantities and shape contracts](implemented/scientific-types.md) | |
| [Random generation and statistics](implemented/random-and-statistics.md) | |
| [Complex scalars, structured operators and randomized SVD](implemented/structured-computing.md) | |
| [Static mathematical documentation](implemented/mathdoc.md) | |
| [Scalar special functions, roots, interpolation, quadrature and distributions](implemented/mathematical-library.md) | |
| [Sparse storage, algebra, assembly and solvers](implemented/sparse-matrices.md) | [Krylov methods, preconditioning and verification](planned/sparse-solvers.md) |
| [Integration and differential equations](implemented/integration-and-odes.md) | [Simulation, discretization and adjoints](planned/simulation.md) |
| [Optimization and automatic differentiation](implemented/optimization-and-ad.md) | [Tensor compression, tensor AD and mathematical introspection](planned/tensors-and-symbolics.md) |
| [Aulë numerical solvers](implemented/aule-solvers.md) | [Advanced sparse solver extensions](planned/sparse-solvers.md) |
| [Fourier transforms, tensors and numerical kernels](implemented/transforms-and-kernels.md) | [Memory, I/O, ACE, CPU, GPU and distributed execution](planned/computing-systems.md) |
| [Tiled kernels, fusion, ACE plans and CPU controls](implemented/cpu-compute.md) | [Automatic scheduling and accelerators](planned/computing-systems.md) |
| | [Scientific ML, quantization and memory-efficient training](planned/machine-learning.md) |

## Read and update an entry

Record the mathematical problem and assumptions, the actual algorithm and backend, source entry points, complexity and workspace, known limitations, validation, provenance, and applicable licensing. Express formulas in LaTeX. Use stable section titles and bibliography keys from [REFERENCES.bib](REFERENCES.bib). A historical reference explains a method's lineage. Mark a source as consulted during implementation only when the repository or a maintainer can substantiate it. Observed backend calls establish the interface used. The exact implementation inside a system library depends on the linked provider.

Use **Implemented**, **Candidate**, or **Experimental** for the capability under discussion. Record an accepted or rejected choice with its rationale in the decision log. When a candidate becomes implemented, move or consolidate the research and link the implementation and tests. Preserve useful negative results with the conditions under which they were obtained.

For a substantial change, start with primary publications and authoritative documentation. Search for relevant work from the preceding 24 months and check publisher updates, corrections and retractions where available. Look for independent evidence and report access gaps. Compare algorithms on matched workloads, numerical tolerances and hardware. Report wall time, memory and data movement where relevant. Treat estimates, published results and Arda measurements as distinct evidence. Keep the research review proportionate to the change. [Development policy](../AGENTS.md#scientific-research-and-provenance)

## Research priorities

1. **Sparse solver validation.** Keep the [Moria residual and breakdown contract](../DECISIONS.md#2026-11-24--sparse-pcg-performance-and-numerical-checks). Expand the reproducible SPD corpus beyond the well-conditioned tridiagonal case. Include badly scaled and cancellation-heavy inputs, and distinguish solve time from structural checks and independent residual evaluation.
2. **0.15 scientific foundations.** Build on reusable pivoted LU, Cholesky and Householder QR, with documented backward error checks and assumptions. Compare an optional LAPACK path before selecting a portable eigensolver or a randomized approximation strategy. [Numerical libraries](planned/numerical-libraries.md)
3. **0.18 kernel research.** Profile matched dense and sparse workloads before changing accumulation. Test FP64 versus extended accumulation on difficult cases, and measure setup reuse for immutable sparse matrices. Evaluate FFT twiddle refresh or a more accurate recurrence if long-transform error grows measurably. Evaluate safe reuse of Dormand–Prince stages only after testing callback and numerical semantics. [Kernels](implemented/transforms-and-kernels.md) · [ODE methods](implemented/integration-and-odes.md)
4. **Longer-term research.** Stage tensor compression, GPU, MPI and model formats behind portable baselines and measured memory and error contracts. Capture metadata and reproducible experiment manifests through the 0.27–1.0 reporting and validation milestones. [Computing systems](planned/computing-systems.md) · [Roadmap](../ROADMAP.md)

The initial review used publisher and Crossref records for DOI metadata, arXiv records for preprints, and official references for library and format behavior. Recent papers are candidates or comparators. Publication results have yet to be reproduced specifically for Arda. Original source consultation for the historical native implementations remains unknown.

## Metadata for AMDE / Rúmil

Development-time entries can inform future descriptors. Use independent fields for **mathematics**, **algorithm**, **backend**, **references**, **assumptions**, **complexity**, **limitations**, **validation**, and **provenance**. Where relevant, record a workspace estimate separately from measured peak RSS and identify the input and environment. Distinguish mathematical equivalence from floating-point equivalence, and static material from actual traces. Static documentation queries should never execute user code as part of describing it. The Markdown and BibTeX require zero runtime work. The [native descriptor contracts](implemented/mathdoc.md) cover the initial queries. [AMDE roadmap](../ROADMAP.md#36-mathematical-introspection-und-provenance-amde--rúmil)

Store bibliographic records and links only. For external code or backends, record the applicable revision, license and notices. See [third-party notices](../THIRD_PARTY_NOTICES.md).
