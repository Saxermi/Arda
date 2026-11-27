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
50 Radau steps and had a final absolute error of **2.86e-8**. The full 0.20
acceptance gate also requires the FEM example from a later milestone.

```sh
make
./arda --gc-stress examples/ulmo.arda
/usr/bin/time -f '%M KiB peak RSS' taskset -c 0 ./arda examples/ulmo.arda
```

The code is original. Published methods supply the numerical formulas. Recent
mixed-precision Runge–Kutta results motivate further study of stage precision
and local error estimates. This implementation uses Float64 states and arithmetic.
[Mixed-precision Runge–Kutta research](https://doi.org/10.1007/s40324-026-00428-9)
