# Random generation and statistics

**Status** — Implemented in Arda 0.15.2. Both modules use portable C. [Source archive validation](../../docs/RUMIL_0.15.2.md)

## Random generator and distributions

`random.generator(seed[,stream])` takes a signed 64-bit integer seed and an optional nonnegative 63-bit integer stream, defaulting to zero. A generator has 64 bits of PCG state and the odd increment $2\cdot\mathrm{stream}+1$. Seeding follows PCG's two initialization steps with the seed interpreted modulo $2^{64}$. The state update uses multiplier `6364136223846793005`, and the XSH-RR output permutes the old state into a 32-bit word. `random.uint32(g)` exposes that word as an Arda integer. A generator belongs to the caller, and two generators initialized alike advance independently. The raw stream is defined for this version. Float64 distribution draws also depend on the platform math library and floating-point evaluation. This generator is intended for simulation. [PCG paper](https://www.pcg-random.org/paper.html) · [Source](../../src/random.c)

| API | Contract |
| --- | --- |
| `uniform(g,low,high[,count])` | Float64 in `[low,high)` for finite `low < high`. Two PCG words give 53 bits of uniform fraction. Interpolation uses a weighted sum and bounds the rounded result below `high`. |
| `normal(g,mean,sd[,count])` | Float64 normal draw with finite mean and nonnegative finite standard deviation. Uses the cosine output of Box–Muller. An unrepresentable draw raises an error. |
| `integer(g,low,high[,count])` | Exact Arda integer in `[low,high)` over signed 64-bit bounds. Rejection removes modulo bias from the 64-bit sampling step before taking the remainder. |
| `poisson(g,mean[,count])` | Integer draw for a finite mean in `[0,1e8]`. Zero returns zero without advancing. Means below 10 use products of uniforms, and larger means use Hörmann's transformed rejection with a stable log-mass calculation. |
| `choice(g,sequence[,count])` | Samples a nonempty list or Float64 vector with replacement. Vector views keep their logical element order. |
| `shuffle(g,sequence)` | Fisher–Yates permutation of a copied list or Float64 vector. The input keeps its original order. |

Omitting `count` returns a single sample. A count up to the tensor element limit returns a vector for floating distributions, a list for integer and Poisson draws, and the input sequence kind for `choice`. An explicit zero count consumes no draws. Calls error before committing a failed draw's generator state. The generator and sampling scratch have constant size. Batch and shuffle storage is linear in the output count. The small-mean Poisson path uses an expected $1+\lambda$ uniforms. The transformed path has an acceptance shortcut and uses rejection with no fixed iteration count. [Box and Muller 1958](https://doi.org/10.1214/aoms/1177706645) · [Hörmann 1993](https://doi.org/10.1016/0167-6687(93)90997-4)

## Descriptive and streaming statistics

The vector functions take finite rank-one Float64 tensors, including strided views. `clip` also accepts higher ranks. Inputs are preserved. [Source](../../src/stats.c)

| API | Contract |
| --- | --- |
| `sum(x)`, `cumsum(x)` | Neumaier compensated sum and prefixes. Empty inputs return zero and an empty vector, respectively. |
| `median(x)`, `quantile(x,q)` | Sort a copy and linearly interpolate at index $(n-1)q$ for finite $q\in[0,1]$. Median uses $q=1/2$. The vector must be nonempty. |
| `covariance(x,y)` | Sample covariance with denominator $n-1$ for equal-length vectors of at least two elements. |
| `correlation(x,y)` | Pearson correlation with the same length requirement and two nonconstant vectors. |
| `regression(x,y)` | Ordinary least-squares line returned as `[slope,intercept]`. The predictor must be nonconstant. |
| `histogram(x,edges)` | Float64 bin counts, with strictly increasing finite edges. Bins are left-closed and right-open, with the final edge included. Values outside the edges are skipped. |
| `argmin(x)`, `argmax(x)` | Zero-based first index of the extremum in a nonempty vector. |
| `clip(x,lo,hi)` | Tensor of the same shape with entries clamped to finite bounds satisfying `lo <= hi`. |
| `online()`, `update(state,x)`, `summary(state)` | Shifted Welford updates. The summary is `[count,mean,population_variance]` and returns `[0,mun,mun]` before the first update. |

The paired functions shift each series by its first value and scale deviations before accumulating second and cross moments with a bivariate Welford update. The accumulator for streaming observations keeps only a count, an origin, a shifted mean and a centered sum of squares. Each update takes constant work. Paired statistics and compensated sums take $O(n)$ work and constant scalar scratch. Quantile sorting uses $O(n\log n)$ work on the tested library and a copied vector. Binary search for $b$ histogram bins uses $O(n\log b+b)$ work including validation and setup. Many output operations require one tensor allocation, and values beyond Float64 range are reported. Extended accumulator precision follows the platform's `long double`. Recent work on parallel compensated summation motivates testing pairwise accumulation when adding parallel reductions. [Welford 1962](https://doi.org/10.1080/00401706.1962.10490022) · [Neumaier 1974](https://doi.org/10.1002/zamm.19740540106) · [Dmitruk et al. 2026](https://doi.org/10.1002/cpe.70886)

## Local performance

Measurements used the portable `-O2` build with GCC 14.2 on one pinned logical CPU of an Intel Xeon Platinum 8573C and NumPy 2.3.5 with one thread. Each time is the median of seven calls after one warm-up, excluding setup and accuracy checks. Linux peak RSS includes the entire process and imports.

| Workload | Arda ms | NumPy ms | Arda / NumPy peak RSS (MiB) |
| --- | ---: | ---: | ---: |
| 100,003 standard normal draws, seed 1521 per implementation | 22.2 | 1.46 | 11.1 / 49.0 |
| Median of 100,003 values | 5.25 | 0.839 | 11.1 / 49.2 |

The normal samples had means 0.00181 and 0.000341 and population variances 0.99648 and 0.99755 for Arda and NumPy, respectively. Both used unit variance, with Arda using PCG32 and Box–Muller and NumPy using its own default generator and sampler. The median input was $x_i=((8191i)\bmod 100003)/100003$ for $0\leq i<100003$. Both medians agreed with $50001/100003$ within $10^{-14}$. NumPy uses selection for the median. [Accuracy tests](../../tests/reference_probability.py) · [NumPy median implementation](https://github.com/numpy/numpy/blob/v2.3.5/numpy/lib/_function_base_impl.py)

**Bibliography keys** — `BoxMuller1958`, `DmitrukEtAl2026`, `Hoermann1993`, `Neumaier1974`, `ONeill2014`, `Welford1962`.
