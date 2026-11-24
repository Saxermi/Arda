# Scalar mathematical library

Arda 0.15.5 adds portable Linux scalar methods. Special functions and distribution formulas run in native C. Callback-based solvers, quadrature and interpolation live in the embedded Arda module. [Native special functions](../../src/special.c) · [Arda routines](../../lib/numerics.arda) · [Runnable example](../../examples/mathematical_library.arda)

## Special functions

| API | Contract and implementation |
| --- | --- |
| `special.gamma(x)`, `special.lgamma(x)` | Real gamma and \(\log\lvert\Gamma(x)\rvert\) via libm `tgammal` and `lgammal`. Nonpositive integer poles raise errors. |
| `special.beta(a,b)`, `special.lbeta(a,b)` | Euler beta and its logarithm for positive inputs. Moderate arguments use the log-gamma identity. For a large argument, combine Stirling terms with `log1p` before subtraction to reduce cancellation. |
| `special.bessel_j(n,x)`, `special.bessel_y(n,x)` | Integer-order Bessel functions of the first and second kind via POSIX libm `jn` and `yn`. Orders range from −10,000 to 10,000. The real \(Y\) interface requires \(x>0\). |
| `special.erf(x)`, `special.erfc(x)` | System libm error function and complementary error function. `erfc` evaluates the tail directly. |

Inputs and results use finite real scalars. Out-of-range Float64 results raise an error. Underflow may round to zero or a subnormal. The provider determines the internal algorithms of the libm calls. Arda's log-Beta uses a fixed number of operations and a truncated Bernoulli expansion for the large terms. It uses constant workspace. [DLMF gamma and Beta](https://dlmf.nist.gov/5) · [DLMF error functions](https://dlmf.nist.gov/7) · [DLMF Bessel functions](https://dlmf.nist.gov/10) · [POSIX Bessel provider interface](https://pubs.opengroup.org/onlinepubs/9799919799/functions/j0.html)

## Root finding

- `numerics.brent(f,a,b[,xtol,rtol,maxiter])` uses secant and inverse quadratic interpolation inside a sign bracket, with bisection when interpolation fails the progress checks. Each update evaluates `f` once.
- `numerics.newton(f,df,a,b[,xtol,rtol,maxiter])` takes a user-supplied derivative. A proposed step must stay inside the bracket. A zero derivative, an out-of-bracket step, or insufficient bracket shrinkage leads to bisection.

The methods accept either endpoint order. A continuous scalar `f` must have opposite signs at the endpoints or a zero at an endpoint. The span and callback results must be finite in Float64. A zero callback value terminates immediately. Otherwise convergence uses bracket width at most `max(xtol,rtol*abs(x))` at the returned endpoint `x`. The defaults are \(10^{-12}\) for both tolerances and 100 updates. Require `xtol > 0` and `0 <= rtol < 1`. Iteration or floating-point resolution exhaustion raises an error. For a discontinuous callback, a shrinking sign bracket may locate a jump. Both methods have \(O(k)\) work outside the callbacks for \(k\) updates and constant workspace. [Brent 1971](https://maths-people.anu.edu.au/~brent/pub/pub005.html)

## Interpolation

`numerics.linear(x,y)` and `numerics.spline(x,y)` return reusable scalar interpolants with private copies of their samples. Supply real lists or rank-1 vectors of equal length with at least two points. Knots must increase strictly with finite spacing. Query points must be inside the closed knot interval.

The linear interpolant blends its neighboring samples. The natural spline solves the tridiagonal system for the second derivatives, with zero second derivatives at the endpoints. It stores cubic corrections in normalized segment coordinates. Construction uses \(O(n)\) work and memory. Binary search makes evaluation \(O(\log n)\) with constant workspace. Bad scaling can make spline slopes or coefficients unrepresentable, in which case the routine raises an error. Natural splines may overshoot. [SciPy reference boundary conditions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html)

## Adaptive quadrature

`numerics.quad(f,a,b[,rtol,atol,max_intervals])` integrates a real scalar callback on a finite interval with finite span. It returns `[value, estimated_error, evaluations]`. A reversed interval negates the value. A zero-width interval returns `[0,0,0]` without sampling.

Each panel evaluates the 15-point Kronrod rule and its embedded seven-point Gauss rule at shared nodes. The panel error uses their difference, the sampled absolute deviation and a roundoff floor scaled by an estimate of the integral of \(\lvert f\rvert\). A max-heap selects the largest estimated error for refinement. Stop when the sum of panel estimates meets `max(atol,rtol*abs(value))`. Defaults are `rtol=1e-9`, `atol=1e-12` and 1,024 panels. Tolerances must be nonnegative with `rtol < 1` and at least one positive. The panel budget is an integer from 1 to 1,000,000. Budget exhaustion and unrepresentable subdivision raise errors. The estimate depends on sampled values and can miss unresolved spikes or oscillations. For \(p\) final panels there are \(15+30(p-1)\) evaluations, \(O(p\log p)\) work outside the callback and \(O(p)\) storage. The implementation uses published nodes and weights with original Arda control and heap code. [QUADPACK reference pair](https://www.netlib.org/quadpack/dqk15.f) · [SciPy quadrature reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html)

## Distributions and dispersion

- `stats.normal_pdf/cdf/sf(x,mean,deviation)` require a positive deviation. The CDF and survival function evaluate opposite `erfc` tails directly.
- `stats.exponential_pdf/cdf/sf(x,rate)` require a positive rate. The CDF uses `expm1` near zero. The density and CDF are zero below the support, where the survival probability is one.
- `random.exponential(generator,rate[,count])` uses the explicit PCG32 stream and inverse CDF `-log1p(-u)/rate`. A batch produces a Float64 vector. Invalid parameters or unrepresentable samples leave the generator unchanged.
- `stats.variance(x[,ddof])` and `stats.stddev(x[,ddof])` take a finite Float64 vector and divide the scaled sum of squared deviations by `len(x)-ddof`. The integer correction defaults to zero and must be smaller than the vector length. The implementation shifts and scales the samples before Welford accumulation. Both statistics take linear work and constant workspace.

## Method comparisons

System libm supplies the established special functions. The stable log-Beta expression follows the DLMF log-gamma and Euler beta identities. Boost's TOMS 748 method uses additional interpolation and is a comparator for future root evaluation-count measurements. The 2026 hybrid Simpson and Gaussian rule is an alternative for integrands with localized irregularities. The existing Simpson integrator provides a fixed-grid baseline. This first adaptive implementation uses shared Gauss–Kronrod samples and an interval budget. A shape-preserving spline would require a separate monotonicity contract. [Boost.Math root-method comparison](https://www.boost.org/doc/libs/1_88_0/libs/math/doc/html/root_finding.html) · [Hybrid quadrature study](https://doi.org/10.1371/journal.pone.0335582)

Bibliography keys include `NISTDLMF`, `PosixBessel2024`, `Brent1971`, `SciPyCubicSpline`, `PiessensEtAl1983`, `HybridQuadrature2026`, `Welford1962`, and `ONeill2014`. [Bibliography](../REFERENCES.bib)
