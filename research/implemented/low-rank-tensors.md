# Low-rank tensors

The `lowrank` module stores real Float64 tensors as factors or cores. Constructors snapshot the inputs, including strided inputs. Physical dimensions can be empty, while latent and bond ranks must be positive. The represented size can exceed the heap tensor element cap.

| Constructor | Representation |
| --- | --- |
| `cp(factors[,weights])` | One $n_d \times r$ factor per mode. Optional weights are a list or a vector of length $r$. |
| `tucker(core,factors)` | A dense latent core with dimensions $r_d$ and one $n_d \times r_d$ factor per mode. |
| `tt(cores)` | One $r_{d-1} \times n_d \times r_d$ core per mode, with compatible bond ranks and both boundary ranks equal to one. |

`shape(x)`, `ranks(x)`, and `at(x,indices)` inspect the representation. `contract(x,axis,vector)` removes one physical mode using a real Float64 vector and returns a scalar for the last mode. `dot(a,b)` requires matching formats and physical shapes. It accepts different latent ranks. `norm(x)` computes the Frobenius norm. `materialize(x)` explicitly allocates a dense Float64 tensor subject to the usual heap element cap and memory budget.

CP dot products form pairwise products of factor column inner products. A cancellation-dominated CP Gram sum uses bounded direct element evaluation when the input is small enough. Larger cancellation-dominated cases raise an error. Tucker dots apply the factor cross-Gram matrices along latent dimensions. Tensor train dots propagate a contraction of the two bond spaces along the cores. The scratch allocations for Tucker and tensor trains are checked under the tensor memory budget. These contraction paths use storage proportional to factors, cores, and latent workspace. [CP and Tucker definitions](https://doi.org/10.1137/07070111X) · [Tensor train definitions](https://doi.org/10.1137/090752286)

## TT-SVD and rounding

`tt_svd(dense,relative_tol[,max_rank])` applies thin SVDs to successive unfoldings of a heap-sized Float64 tensor. `round(train,relative_tol[,max_rank])` performs a right orthogonalization sweep, then truncates in the opposite direction. Both return `[tensor,relative_tail]`. The relative tolerance lies in `[0,1]`. An explicit rank cap can make the reported relative tail exceed the requested tolerance. At least one bond component is retained. The tail combines discarded singular values in quadrature relative to the input norm. Floating-point roundoff is separate from the truncation estimate.

Both routines reuse Arda's thin LAPACK SVD. Tensor train rounding works with core unfoldings and never constructs the represented dense tensor. Its cost depends on physical dimensions and bond ranks. The portable build provides all explicit constructors and contractions. Deterministic sweeps establish the reference path for later comparison with randomized rounding. TT-cross and CP or Tucker fitting remain outside these construction APIs. [Deterministic TT algorithms](https://users.math.msu.edu/users/iwenmark/Teaching/CMSE890/TENSOR_oseledets2011.pdf) · [Randomized TT rounding comparison](https://doi.org/10.1137/25M181708X)
