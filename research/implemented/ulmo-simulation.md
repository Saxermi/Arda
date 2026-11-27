# Ulmo scientific simulation

## 0.20.0 RK45 output, events and restart

The existing `ode.rk4` and default `ode.rk45` entry points remain available. RK45 uses
the Dormand–Prince 5(4) pair and the weighted RMS local error with scalar `rtol` and
`atol`. The final optional `options` map enables output features.

| Option | Contract |
| --- | --- |
| `saveat` | A list of finite times within the interval, strictly ordered in the integration direction. Returns `times` and `states`, including outputs between accepted steps. The schedule does not determine step sizes. |
| `dense_output` | When true, return a callable `at(t)` over the computed interval. It uses a binary search over accepted segments. |
| `stream` | A synchronous `stream(t,y)` callback. Receives each scheduled output, or the initial and accepted endpoints when there is no schedule. The result leaves `times` and `states` empty. |
| `events` | A list of maps with finite scalar `fn(t,y)`, optional `direction` of -1, 0 or 1 in increasing physical time, and optional Boolean `terminal`. |
| `event_tol` | Root bracketing tolerance in time units, default 1e-10. |
| `checkpoint` | When true, return a versioned in-memory checkpoint containing a state snapshot, time, next step magnitude, tolerances and step limit. |

The dense interpolant is Shampine's fourth-order extension through the accepted
step, evaluated using its existing Dormand–Prince stages. It needs no additional
RHS evaluations. Retained dense output stores data per accepted step. Streaming
alone holds no accumulated trajectory in the solver. Stream arguments and returned
samples are independent snapshots. With events enabled, event records accumulate in
the result even when the sampling callback is streaming.

Event detection checks sign changes and zeros at accepted step endpoints. When a
crossing is bracketed, bisection on the interpolant locates it to the requested
time tolerance or local floating-point resolution. Events use zero-based indices
into the definitions, are reported in integration order, and a terminal event
truncates both the trajectory and the schedule. A zero at the beginning of a step
is attributed to the previous crossing, so an initial zero does not fire. Grazing
roots and multiple crossings of the same event within one accepted step can be missed.

`ode.rk45_resume(f,checkpoint,t1[,options])` starts a new segment with the checkpoint's
settings. Output callbacks and event definitions are provided for each segment.
Adaptive counters measure the new segment. Persistent serialization belongs to a
later milestone.

`ode.verlet(acceleration,t0,q,p,h,steps)` is the kick–drift–kick leapfrog form for
unit mass, with acceleration depending on time and position. It supports finite
scalar or matching `numc` tensor states, and a signed fixed step. Returns `time`,
`position`, `momentum` and `steps`. For a separable autonomous Hamiltonian with
constant step size the method is second-order, symplectic and time-reversible.
[ODE source](../../lib/ode.arda) · [Continuous extension](https://doi.org/10.1090/S0025-5718-1986-0815836-3) · [Störmer–Verlet](https://doi.org/10.1007/3-540-30666-8)

## 0.20.1 Implicit Euler, BDF2 and Radau IIA

All three calls take `(f,t0,y0,t1,h[,options])` for a scalar or nonempty rank-one
Float64 `numc` state, and return `solution`, `time`, `accepted_steps` and
`newton_iterations`. They use a fixed nominal step and shorten the last step.
Positive `h` is a magnitude, as in RK45. These first variants provide step control
to the caller and Newton error tolerances to the integrator.

| Entry point | Method |
| --- | --- |
| `ode.implicit_euler` | Backward Euler, order one. |
| `ode.bdf2` | Backward differentiation, order two after a backward-Euler startup. The final clipped step uses its actual step-size ratio. |
| `ode.radau` | Two-stage right Radau IIA collocation, order three. Solves the coupled stage system and returns the final stage. |

The Newton line search accepts a correction when it reduces the Euclidean residual.
Convergence uses `atol + rtol * max(||x_initial||₂, ||x||₂)`. Options include
`atol`, `rtol`, `maxiter` for Newton, and `max_steps` for integration. Use a
`jacobian(t,y)` callback returning a scalar derivative or dense Float64 matrix, or a
`jvp(t,y,v)` returning the derivative applied to a direction. The latter constructs
Newton columns via directional products, and the default uses finite differences
of the residual. The implementation assembles a dense Newton matrix. It reuses the
existing LU solver by default or `solver.solve("gmres")` with optional Jacobi
preconditioning. Dense factorization costs O(n³) for an n-component state and
O(n²) matrix storage. Radau solves a system of size 2n. Adaptive variable-order BDF,
fifth-order Radau and matrix-free Newton–Krylov are follow-ups.
[Implicit implementation](../../lib/ode.arda) · [Stiff methods](https://doi.org/10.1007/978-3-642-05221-7)

## 0.20.2 Cartesian-grid PDEs

`pde.grid(shape,spacing[,boundary])` creates a one- to three-dimensional
matrix-free operator for the negative Laplacian. Spacing and boundary can be
scalars or lists, one value per axis. Grid states are flattened row-major Float64
vectors. `linop.apply`, `linop.apply_transpose` and `solver.verify` work with this
operator. The existing `linop.poisson` operator also covers ranks up to eight.

| Boundary name | Discretization |
| --- | --- |
| `"dirichlet"` | Zero values at exterior grid neighbors. For an axis of length n this corresponds to n interior nodes with the walls at indices 0 and n+1 in one-based indexing. |
| `"neumann"` | Zero diffusive normal derivative at exterior cell faces by equal-value ghosts. Advective flux is zero at those faces. |
| `"periodic"` | Wrap opposite neighbors and share numerical flux across the seam. |

The boundary applies to both sides of its axis. Mixed boundary types across axes
are supported. Grid coefficients are constant per axis.

| Entry point | Scheme and conditions |
| --- | --- |
| `pde.poisson(grid,b[,options])` | Conjugate gradients from the common solver. With only periodic and Neumann axes, the constant nullspace requires a zero-mean RHS and a zero-mean returned solution. Compatibility accepts cancellation within a floating-point bound. The reported residual is recomputed after centering. |
| `pde.heat(grid,u,diffusivity,duration,dt[,options])` | Centered diffusion in space and forward Euler in time. Stability bound is `2 diffusivity dt Σ 1/hᵢ² ≤ 1`. |
| `pde.wave(grid,u,v,speed,duration,dt[,options])` | Centered spatial Laplacian and velocity Verlet, returning displacement in `solution` and `velocity`. Stability bound is `speed dt √(Σ 1/hᵢ²) ≤ 1`. |
| `pde.advection_diffusion(grid,u,velocity,diffusivity,duration,dt[,options])` | Conservative donor-cell upwind advection with centered diffusion and forward Euler, using a constant velocity per axis. The sufficient bound is `dt Σ (abs(vᵢ)/hᵢ + 2 diffusivity/hᵢ²) ≤ 1`. |

Time-dependent calls return `solution`, `time`, `steps` and `cfl`. They clip the last
time step and use bounded numeric workspace. Boundary fluxes telescope for periodic
and closed Neumann transport, conserving the unweighted sum on a uniform Cartesian
mesh up to roundoff. Stencil sweeps traverse tiles of 4,096 outputs with contiguous
runs between boundary cases. Each apply takes O(dN) work for d axes and N points.
Time stepping uses O(N) scratch and the ordinary tensor memory budget. These methods
cover linear PDEs with constant coefficients and homogeneous prescribed boundaries.
[Stencil and time stepping](../../src/pde.c) · [Finite-volume methods](https://doi.org/10.1017/CBO9780511791253)

## Numerical validation

All 275 runtime tests passed in five local configurations, including the
portable and OpenMP builds and ASan/UBSan. Both OpenBLAS/LAPACK configurations
passed all 80 numerical reference tests. The other configurations passed 70
with ten LAPACK-specific skips. Clang x86-64 and GCC AArch64 under QEMU each
passed all 14 focused Ulmo tests.

The independent reference suite compares irregular RK45 samples to SciPy, fixed-step
Radau to SciPy on the stiff Robertson test, and a mixed-boundary Poisson solve to an
independently assembled Kronecker matrix. Step refinement checks orders one, two and
three for the three implicit methods respectively. Grid refinement checks second-order
heat convergence when the time step scales with the square of the spacing. Wave time
refinement checks second order against an exact semidiscrete periodic mode. Other tests
cover reverse integration, terminal event ordering, in-memory restart, Verlet reversal,
closed-boundary mass conservation, unstable step rejection and budget failure cleanup.
[Reference tests](../../tests/reference_ulmo.py) · [Runtime tests](../../tests/test_ulmo.py)

The [Ulmo example](../../examples/ulmo.arda) ran under GC stress and was timed three times
on CPU 0 on the Intel Xeon Platinum 8573C host, with GCC 14.2.0 and the default
`-std=c11 -O2` build. The 256×256 heat example took 200 steps, had a relative
error of **1.48e-8** against the exact continuous solution, and ran in **51.2 ms**
median for the stepping call. Median whole-process peak RSS was **6.9 MiB**
(7,044 KiB), including the ODE examples. The stiff forced decay example used
50 Radau steps and had a final absolute error of **2.86e-8**. The FEM example and its memory measurements are below.

```sh
make
./arda --gc-stress examples/ulmo.arda
/usr/bin/time -f '%M KiB peak RSS' taskset -c 0 ./arda examples/ulmo.arda
```

The code is original. Published methods supply the numerical formulas. Recent
mixed-precision Runge–Kutta results motivate further study of stage precision
and local error estimates. This implementation uses Float64 states and arithmetic.
[Mixed-precision Runge–Kutta research](https://doi.org/10.1007/s40324-026-00428-9)

## 0.20.3 Linear finite elements

The scalar elliptic model is `-div(k grad u) + c u = f`, with positive diffusivity
`k` and nonnegative reaction `c`. Geometry and assembly use Float64. Source and
reaction products are integrated exactly for the given P1 interpolant of the source,
using the simplex mass entry `measure * (1 + [i == j]) / (arity * (arity + 1))`.
Element gradients are constant and their products give the diffusion stiffness.
An analytic source supplied as nodal samples therefore incurs an interpolation error.

| API | Contract |
| --- | --- |
| `fem.mesh(nodes,cells)` | Snapshots rank-one coordinates for segments or an N×2 NumC array for triangles. Each cell is a list of two or three zero-based node indices. Either orientation works. The caller supplies a conforming, nonoverlapping mesh. Shape, finite geometry, isolated vertices, degeneracy and invalid shared faces are checked. Nonlocal intersections and hanging nodes in input meshes are outside these checks. |
| `fem.nodes(mesh)`, `fem.elements(mesh)` | Independent coordinate and connectivity copies. The mesh itself is immutable. |
| `fem.assemble(mesh,source,dirichlet[,options])` | Returns `matrix` (canonical sparse CSR) and `rhs` (NumC). Source is a finite scalar or nodal Float64 vector. Coefficients `diffusivity` and `reaction` in the options are scalars or Float64 vectors in element order, defaulting to 1 and 0. |
| `fem.estimate(mesh,source,dirichlet,field[,options])` | One squared residual indicator per element. Uses the (h_T^2)-weighted volume residual, jumps of normal diffusive flux on shared faces and zero-flux residual on natural boundary faces. A Dirichlet boundary face in 2D has both its endpoints prescribed. |
| `fem.refine(mesh,field,indicators[,options])` | Bulk marking in decreasing indicator order with element-index tie breaking. `fraction` is in (0,1], default 0.5. Zero total error yields an equivalent mesh. Returns `mesh`, `field`, zero-based element `parents` and `edge_midpoints`, whose row gives the two old endpoints of each appended node. `max_nodes`, `max_elements` and `workspace_bytes` bound refinement. |
| `fem.integral(mesh,field)` | Integrates the P1 interpolant over the mesh. |
| `fem.darcy_velocity(mesh,pressure,mobility)` | Returns `-mobility * grad pressure` per element, in an array of shape E or E×2. Mobility is strictly positive and can be scalar or cellwise. |

Dirichlet values are a list of unique `[node,value]` pairs. Prescribed values are
eliminated from free rows and the corresponding columns. Prescribed rows then
have a unit diagonal, preserving symmetry. Unspecified boundary faces have
homogeneous natural Neumann data. Every connected component needs at least
one prescribed node or positive reaction on a cell. For example, solve with
`solver.solve("cg", system["matrix"], system["rhs"])`. General Neumann loads,
3D elements, and transient or nonlinear CFD are outside the implemented scope.

Marking bisects marked segments or splits all three edges of marked triangles.
Neighboring triangles that share a split edge also get that midpoint. A shared
midpoint uses one global node. Fully split triangles use four similar subtriangles.
The transferred P1 field is the same piecewise-linear function on the refined mesh,
which preserves its integral to roundoff. The caller uses `parents` to transfer
cell coefficients and the edge map to update its boundary conditions. Assign
midpoint Dirichlet data only to edges on the relevant boundary. The implementation
does not claim arbitrary-contrast estimator robustness or optimal adaptive complexity.

For N nodes and E elements, temporary COO storage is O(E+N). CSR canonicalization
sorts within each row and merges duplicate entries. Mesh validation, marking and
face matching use O(E log E) time with in-place record sorting, O(N+E) checked
buffers and no dense matrix intermediates. Assembly and estimate accept
`workspace_bytes` for their dense scratch. The process tensor and sparse budgets
also apply, including which allocator accounts for the COO-to-CSR workspace. Input
lists and caller-owned outputs still contribute to process RSS.

The steady Darcy example uses an affine pressure and zero normal flux along the
top and bottom. The per-element gradient is a velocity diagnostic. It need not
produce a locally conservative normal flux at interfaces with general data. The opt-in mixed formulation below enforces local flux balance.
[Example](../../examples/ulmo_fem.arda) · [Assembly](../../src/fem.c) ·
[Estimator and refinement](../../src/fem_refine.c)

The estimator follows the classical residual approach and bulk marking follows
Dörfler. The optional newest-vertex bisection implementation below provides a persistent refinement hierarchy. Literature convergence results for specific
adaptive schemes do not establish those guarantees for this subdivision.
[Babuška and Rheinboldt](https://doi.org/10.1137/0715049) ·
[Dörfler](https://doi.org/10.1137/0733054) ·
[Recent work on adaptive optimality (preprint)](https://arxiv.org/abs/2610.39719)

## 0.20.4 Differentiable simulation

| API | Contract |
| --- | --- |
| `sim.rk4(rhs)` | Builds an RK4 model from a deterministic `rhs(t,y,p)` using AD-supported operations. Scalar and nonempty rank-one NumC Float64 states and parameters are supported. Do not mutate inputs or depend on external side effects in the RHS. |
| `sim.heat(grid)` | Builds a forward Euler model from an existing Cartesian PDE grid. State is a flat NumC vector. The scalar parameter is the nonnegative diffusivity and the existing CFL check applies. `pde.grid_size(grid)` reports the required vector length. |
| `sim.solve(model,t0,y0,p,t1,h[,options])` | Returns `solution` and `steps`. Uses positive nominal step `h`, clips the final step and supports descending times for RK4. |
| `sim.jvp(model,t0,y0,p,t1,h,dy0,dp[,options])` | Adds the terminal `tangent` for the supplied joint direction in initial state and parameter. |
| `sim.adjoint(model,t0,y0,p,t1,h,seed[,options])` | Returns the terminal state, `initial_gradient`, `parameter_gradient`, `steps`, checkpoint count and recomputed step count. The seed is the terminal cotangent, with the same shape as the state. |
| `sim.fit_terminal(model,t0,y0,p0,t1,h,target[,options,tolerance,maxiter])` | Fits a parameter to a terminal observation using half the squared Euclidean discrepancy and the existing L-BFGS optimizer. Returns its convergence report plus `parameter` and `terminal_state`. The starting heat diffusivity must be feasible. Heat trials outside the allowed interval are backtracked. It does not guarantee identifiability or a global optimum. |

ODE step derivatives use the existing local AD graph, including the four RK stages.
The heat update is `z = (I - h k A)y` for the existing matrix-free negative
Laplacian. The JVP combines `(I - h k A) dy` and `-h A y dk`. The reverse step
applies the transpose to the seed and accumulates `-h <seed, A y>` for the
diffusivity. All derivatives are for the fixed numerical step schedule. Adaptive
RK45, implicit solvers, FEM shape derivatives and event times are outside this API.

Options are `checkpoint_stride` (default 32), `max_steps` (100,000),
`max_saved_states` (4,096 explicit checkpoint/replay states) and `stream(t,y)`.
The stream receives independent snapshots of the initial and accepted endpoint
values during the forward pass. It is called once per endpoint, including during
a parameter fit's final forward run only. Reverse replay performs no stream calls.
External callbacks may retain snapshots separately. The tensor memory budget
also applies to checkpoint, replay and temporary tensors; the state-count limit
does not count every stage or VM allocation. Checkpoints reside in memory.

For S steps with a block size B, storage for saved states is
O(ceil(S/B) + B) including one replay block. With `B=1` there is no block replay.
Other strides replay at most S minus the count of stored checkpoints. Each RK4
derivative still evaluates the four-stage calculation for the local AD graph.
Multistage and dependency-aware scheduling can use a fixed byte budget more
efficiently when step costs vary. DJ4Earth uses differentiated models, compiler
transformations and checkpointing for substantially broader model classes.
[Simulation source](../../lib/sim.arda) · [Step-based checkpointing](https://doi.org/10.1016/j.jocs.2024.102405) ·
[Multistage adjoint scheduling](https://doi.org/10.1016/j.jocs.2022.101913) ·
[DJ4Earth](https://doi.org/10.1029/2025MS005615)

### FEM and sensitivity validation

All 284 runtime tests passed in each of five local configurations: portable GCC,
GCC with ASan/UBSan, OpenMP, OpenBLAS/LAPACK, and OpenBLAS/LAPACK with ASan/UBSan.
Both LAPACK builds passed all 86 numerical references. The other builds passed 76
with ten LAPACK-specific skips. Clang x86-64 and GCC AArch64 under QEMU each
passed all 29 focused Ulmo tests.

Independent SciPy assembly uses triangle quadrature and a sparse direct reference
for irregular geometry, heterogeneous materials and nonzero Dirichlet data. Line
refinement produces a fourfold reduction in integrated L² error per halving of
uniform spacing. The triangular manufactured Poisson problem shows near fourfold
nodal error reduction. The adaptive suite checks shared-face conformity, integrated
field conservation and zero or negative indicator handling. Darcy interface flux
is checked against a two-material 1D analytical solution. Simulation references
check the ODE primal and initial/parameter derivatives against closed forms, the
heat gradient against a discrete Fourier multiplier, JVP/adjoint duality,
centered finite differences, mass conservation and two parameter-recovery cases.
Runtime tests also exercise error cleanup, memory limits and GC stress.
[Independent references](../../tests/reference_ulmo_fem_ad.py) ·
[Runtime coverage](../../tests/test_ulmo_fem_ad.py)

The implementation and benchmarks use project-authored code. The independent
reference checks require NumPy/SciPy, and the runtime keeps its portable dependency
contract.

### Performance and memory

Five separate runs per case were pinned to CPU 0 of an Intel Xeon Platinum 8573C,
with GCC 14.2.0 and the portable `-std=c11 -O2` build, or NumPy 2.3.5 and
SciPy 1.17.0 with one BLAS thread. Times are medians of monotonic in-program
measurements. GNU time reports whole-process peak RSS, including interpreter imports,
mesh construction, and any example diagnostics.

On a 4,096-node, 7,938-triangle square, both implementations formed 19,716 CSR
entries and took 93 Jacobi-preconditioned CG iterations with relative tolerance
1e-10. Arda's mesh phase includes topology validation. SciPy uses vectorized
construction of the regular input and uses the known geometry of the generated
triangles. The median for mesh construction, assembly and CG together was 18.23 ms
for Arda and 11.36 ms for NumPy/SciPy on this affine Darcy benchmark.

| Implementation | Mesh phase | Assembly | CG | Relative residual | Relative nodal error | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Arda | 12.0 ms | 2.54 ms | 3.70 ms | 1.90e-11 | 1.15e-12 | 8.39 MiB |
| NumPy/SciPy | 0.40 ms | 6.22 ms | 4.56 ms | 2.48e-11 | 1.35e-12 | 48.9 MiB |

The 64×64 heat adjoint took 256 steps at dt = 2⁻¹² and diffusivity 0.1. Both
schedules gave a parameter gradient of -1.92835033157075, with absolute error
2.26e-14 against the analytical discrete Fourier derivative. Each streamed 257
snapshots. Peak RSS includes the preceding scalar fit.

| Checkpoint stride | Stored checkpoints | Recomputed steps | Adjoint time | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| 1 | 256 | 0 | 28.7 ms | 17.6 MiB |
| 16 | 16 | 240 | 29.2 ms | 4.47 MiB |

```sh
make
/usr/bin/time -f '%M KiB peak RSS' taskset -c 0 ./arda examples/ulmo_fem.arda
OPENBLAS_NUM_THREADS=1 /usr/bin/time -f '%M KiB peak RSS' taskset -c 0 python3 benchmarks/ulmo_fem_scipy.py
scratch=$(mktemp -d)
for stride in 1 16; do
    sed "s/\[1,16\]/[$stride]/" examples/ulmo_sensitivity.arda >"$scratch/heat-$stride.arda"
    /usr/bin/time -f '%M KiB peak RSS' taskset -c 0 ./arda "$scratch/heat-$stride.arda"
done
```

The Cartesian heat, FEM and differentiable simulation examples also passed with
GC stress under a 1 GiB virtual-memory limit on this host. A physical laptop was
outside the measurement environment. [SciPy baseline](../../benchmarks/ulmo_fem_scipy.py) ·
[Sensitivity example](../../examples/ulmo_sensitivity.arda)

## Alternative Ulmo methods

These methods run alongside the original midpoint refinement, P1 pressure with
gradient recovery, and block checkpointing. All use Float64. The implementation
uses project-authored code and adds no runtime dependencies.

### Newest-vertex bisection

Use `fem.refine(mesh,field,indicators,{"method":"nvb"})` on a conforming triangular
mesh. The first call labels the longest edge deterministically, with node-index
tie breaking. Descendants preserve the bisection rule by keeping the newest
vertex first. When a marked triangle's reference edge needs to split, both its
owners split that edge and any reference edges needed to reach it. A queue
closes these requirements over the original faces, then each triangle applies
the resulting legal bisections. The edge midpoint is shared once. The same P1
prolongation and `parents` mapping used by midpoint refinement apply.

`fem.hierarchy(mesh)` returns `leaves` in element order and arrays of `parents`
and `generations` indexed by tree node. A root has parent -1 and generation 0.
IDs are stable along successive NVB calls. Request `"method":"nvb"` at each
refinement to carry forward the tree. Refining by the midpoint method produces a
mesh without NVB ancestry. The tree stores labels, parents and generations for the
resulting mesh and is independent of old mesh snapshots. The usual `max_nodes`,
`max_elements`, `workspace_bytes` and global tensor budget apply.

With E input triangles, edge sorting costs O(E log E) and the queue is O(E).
Copying the persistent tree is O(T) for T tree entries. NVB preserves the
shape classes generated by the bisection rule. It does not improve an arbitrary
poor initial mesh or guarantee this benchmark's minimum angle on other meshes.
[Kossaczký's bisection strategy](https://doi.org/10.1016/0377-0427(94)90034-5) ·
[Recent mesh grading analysis](https://doi.org/10.1090/mcom/4102) ·
[Refinement source](../../src/fem_refine.c)

### Mixed triangular Darcy flow

`fem.darcy_mixed(mesh,source,dirichlet,mobility[,solver_options])` solves
`q = -k grad p` and `div q = f` with positive scalar or cellwise
mobility. The source is scalar or nodal P1. Each exterior edge with both endpoints
prescribed uses the exact mean of the linearly interpolated pressure values.
Other exterior edges have zero normal flux. Every edge-connected component needs
at least one pressure edge. Source integrals use the nodal mean times cell area.

There is one pressure unknown per triangle and one integrated normal flux degree
of freedom per interior or pressure edge. The local RT0 mass is integrated by a
three-point degree-two rule. A symmetric sparse saddle system couples flux mass
and discrete divergence. Diagonal scaling uses the flux-mass diagonal and the
diagonal of the corresponding approximate pressure Schur complement. MINRES
solves this scaled system. The sparse matrix is internal and its storage is
released when the call returns. Assembly before canonicalization and each
MINRES iteration are O(E) on a triangular mesh. The iteration count depends on
mesh resolution, shape and mobility.

The result contains `pressure` (cell averages), `velocity` (E×2 samples at
centroids), `flux` (E×3 integrated outward values for the edges opposite local
vertices 0, 1 and 2), and `face_dofs`. Interior neighbors use opposite signs of
the same degree of freedom. `max_cell_imbalance` is the maximum absolute
difference between summed outward flux and integrated source. `converged`,
`iterations`, `reason` and `residual` describe MINRES on the scaled system.
Its `workspace_bytes` report refers to the Krylov solve. The option of that
name bounds assembly scratch and Krylov scratch individually, and the global
tensor and sparse budgets also apply. Higher-order, transient, and nonlinear
flow are beyond this solver's contract.
[Raviart and Thomas](https://doi.org/10.1007/BFb0064470) ·
[Recent RT boundary work](https://doi.org/10.1051/m2an/2025028) ·
[Alternative nonconforming solver](https://doi.org/10.1137/0722029) ·
[Mixed solver source](../../src/fem_darcy.c)

### RK4 stage-aware checkpointing

Set `"checkpoint_strategy":"stages"` in the options to `sim.adjoint` or
`sim.fit_terminal` for an RK4 model. Both strategies save restarting states at
the usual `checkpoint_stride`. The stage-aware reverse pass replays the last
eligible steps of each block as AD tapes, preserving the four RK stages. The
reverse sweep uses those tapes directly, removing a duplicate RHS evaluation for
each retained stage. Stages are released as their tapes are consumed. The default
`"block"` strategy recomputes each local VJP from the restart states.

The tail size is chosen from the available logical `max_saved_states` slots and
the `max_stage_tapes` cap per block (default 2). A tape is charged four extra
slots, one per stage. The actual bytes depend on the
RHS and include intermediate AD arrays. Use the existing tensor byte budget to
cap allocations. Reports include `peak_checkpoint_slots`,
`reused_step_tapes` and, for RK4, `rhs_evaluations`. Both strategies require a
deterministic, AD-compatible RHS. The stage replay runs no streaming callbacks
and supports clipped final steps and backward integration. The implementation
applies the stage-retention principle to block replay. The broader CAMS
strategy optimizes stage placement across the whole time horizon.
[Multistage adjoint checkpointing](https://doi.org/10.1016/j.jocs.2022.101913) ·
[Simulation source](../../lib/sim.arda)

### Comparison with the previous implementation

The benchmark uses the original 0.20.4 binary from commit `528c17d` for the
midpoint, P1 and block configurations. The corresponding alternatives use the
current binary. Both builds use GCC 14.2.0 with the default portable
`-std=c11 -O2` flags on one pinned Xeon Platinum 8573C core. Times are medians
of five separate runs. GNU time records whole-process peak RSS. Reported
tensor peaks use Arda's allocator counters.

```sh
make
baseline=$(mktemp -d)
git archive 528c17d | tar -x -C "$baseline"
make -C "$baseline"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/ulmo_alternatives.py \
    --cpu 0 --baseline "$baseline/arda"
```

**Refinement.** Twelve local marking rounds on the same 16×16-vertex grid selected
the current triangle whose centroid was closest to a fixed point. Only refinement
calls were timed. Both methods preserved the P1 field integral at the printed
precision and covered the same unit square. The runs compare the same number of
marking rounds. They end with different meshes and element counts.

| Method | Final vertices | Final triangles | Minimum angle | Refinement time | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Midpoint | 292 | 522 | 0.158° | 2.12 ms | 2,500 KiB |
| NVB | 315 | 568 | 45.0° | 2.37 ms | 2,520 KiB |

**Darcy.** Both methods used 1,922 triangles, unit mobility, source 2, pressure
0 and 1 on the left and right, homogeneous no-flow top and bottom, and solver
tolerance 1e-10. The exact pressure is `2x - x²` and the exact velocity is
`(2x - 2, 0)`. Timing covers assembly, solve and velocity/flux recovery. The
cell balance and pointwise velocity error are distinct metrics.

| Method | Unknowns | Iterations | Maximum cell imbalance | Relative centroid velocity error | Time | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P1 pressure and gradient | 1,024 | 43 | 1.04e-3 | 0.931% | 1.46 ms | 3,264 KiB |
| RT0 flux and P0 pressure | 4,805 | 536 | 1.64e-10 | 1.317% | 30.81 ms | 4,448 KiB |

**Checkpointing.** The nonautonomous nonlinear RK4 workload used 1,024 state
entries and 256 steps, with 16-step blocks. The paired runs used the same
global tensor byte cap and logical state cap. The 4 MiB trial used 33 slots
and at most one retained stage tape per block. The 16 MiB trial used 64 slots
and up to 16 tapes per block. The initial-state and parameter gradients agreed
within 1e-13. The stage mode used fewer RHS calls and showed slightly higher
wall time and higher memory use on this model.

| Tensor cap | Strategy | RHS calls | Tapes reused | Time | Peak budgeted tensors | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4 MiB | Block | 3,008 | 0 | 611.0 ms | 1.79 MiB | 3,648 KiB |
| 4 MiB | Stage-aware | 2,952 | 14 | 624.5 ms | 2.13 MiB | 3,952 KiB |
| 16 MiB | Block | 3,008 | 0 | 605.3 ms | 1.79 MiB | 3,680 KiB |
| 16 MiB | Stage-aware | 2,384 | 156 | 625.0 ms | 9.50 MiB | 12,068 KiB |

All 288 runtime tests passed in portable GCC, OpenMP, OpenBLAS/LAPACK, GCC
ASan/UBSan, and OpenBLAS/LAPACK with ASan/UBSan. Both LAPACK configurations passed
all 89 reference tests. The other builds passed 79 with ten LAPACK-specific skips.
Clang and AArch64 under QEMU each passed all 36 focused Ulmo tests. Independent
SciPy assembly and a direct sparse solve agreed with RT0 pressure and flux on
heterogeneous irregular triangles. The manufactured Darcy velocity error fell
by about two per halving of the mesh width. Tests also check NVB ancestry and
edge conformity, pressure-boundary validation, allocation rollback, checkpoint
reuse counts, clipped reverse-time integration and unchanged stream ordering.
[Benchmark driver](../../benchmarks/ulmo_alternatives.py) ·
[Runtime tests](../../tests/test_ulmo_alternatives.py) ·
[Independent references](../../tests/reference_ulmo_alternatives.py)
