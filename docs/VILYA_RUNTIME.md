# Vilya runtime contracts

This document describes the changes on the 0.17 development branch.

## Iteration and imports

`range(stop)` and `range(start, stop[, step])` produce an immutable, half-open sequence with constant storage. The step can be negative. Endpoints and length must fit signed Int64. `len`, indexing, `for`, and comprehensions work with ranges. `list(range(...))` explicitly materializes up to 1,000,000 elements. Arda can iterate larger ranges without allocating the sequence up front. Static MathIR loop documentation still uses its bounded range model.

A bare standard module name such as `import tensor as t` selects the standard library. Bare names outside the standard library are looked up next to the importing file. `import "std:tensor" as t` requires a standard module. `import "./tensor.arda" as local` selects a local file. Quoted imports require an alias.

## Numeric rules and diagnostics

Integer addition, subtraction, and multiplication check signed Int64 overflow. Integer remainder by zero raises an error. Ordinary real floating-point arithmetic follows IEEE behavior. Division by zero can produce infinity or NaN, and `sqrt(-1)` produces NaN. Real `/` produces floating-point division even when both inputs are integers. Scientific routines can impose stricter domains. Differentiation and scientific quantities check for finite numerical values.

Tensors, quantities, and differentiated values require an explicit scalar condition. For example, use `tensor.shape(x)[0] != 0` to test the first axis. Call `ad.value` explicitly when using a tracked scalar as a control-flow input. The resulting branch choice is ordinary runtime control flow.

`catch` binds a map. Runtime errors provide `code`, `message`, `file`, and `line`. Error codes used here include `shape`, `dtype`, `memory_budget`, `mutation`, `nonfinite`, and `ambiguous_truth`. Older error sites use the generic `runtime_error` code. `throw` requires a map with string `code` and `message` fields. Additional fields survive the throw.

```arda
try:
    throw {"code": "shape", "message": "invalid sample", "expected": [2, 3]}
catch issue:
    pedo(issue["message"], issue["expected"])
```

## Solver reports and numerical names

| API | Result fields |
| --- | --- |
| `solver.solve`, `solver.multigrid` | `solution`, `iterations`, `residual`, `converged`, `reason` |
| `ode.rk45` | `solution`, `accepted_steps`, `rejected_steps` |
| `opt.lbfgs` | `solution`, `iterations`, `gradient_norm`, `converged`, `value`, `reason` |
| Zeroth-order optimizers | The L-BFGS fields and `evaluations` |
| `opt.adam` | `parameters`, `first_moment`, `second_moment` |

An L-BFGS objective callback returns a map with `value` and `gradient`. Direct solves return the solution value. Iterative solves report the relative data residual. LSQR and LSMR also report the relative normal-equation residual as `normal_residual`. [Solver contracts](../research/implemented/aule-solvers.md)

Use `at(x,[i,j])` and `put(x,[i,j],value)` for dense element access. Use `tensor.sum(x[,axis])` for typed full and axis sums. Float64 reductions use extended accumulation where `long double` provides it. The optional axis removes that axis. A rank-one axis reduction returns a scalar. Use `opt.sgd` and `opt.adam` for optimizer updates, `sparse.csr` and `sparse.to_csr` for CSR construction and conversion, and `linalg.eigh` for symmetric eigenpairs.

## Dense Float64 autodiff

Follow [the runnable example](../examples/tensor_autodiff.arda) and [the mathematical and memory contracts](../research/implemented/optimization-and-ad.md#reverse-mode-autodiff).

## Gradient queries and sparse matvec

| Call | Behavior |
| --- | --- |
| `ad.zero_grad(node)` | Clear its published gradient. The next `ad.grad(node)` returns `mun`. |
| `ad.detach(node)` | Return its ordinary primal by alias. Mutating the alias still changes the shared buffer version. |
| `ad.no_grad(callable, ...)` | Run the callable and its nested calls with recording disabled. The caller's mode survives exceptions. Returning an existing node directly preserves that node. |
| `ad.vjp(output,input[,cotangent])` | Return `Jᵀ·cotangent` with the input shape. A scalar output defaults to cotangent one. |
| `ad.jvp(output,input,tangent)` | Return `J·tangent` with the output shape. |
| `ad.jacobian(output,input)` | Return a dense Jacobian with output axes followed by input axes. |
| `ad.hessian(loss,input)` | Return a scalar-loss Hessian with two copies of the input axes. |

For derivative queries, `output` and `input` are tracked nodes. An intermediate input is treated as independent from its ancestors. A disconnected tracked input has zero derivative. Supply finite, exact-shape Float64 seeds. Queries leave the stored gradients untouched. Dense Jacobians and Hessians are subject to the eight-axis, heap element and memory-budget limits. Split large derivative work into seeded products.

`sparse.matvec(A,x)` differentiates a tracked dense Float64 vector with a fixed sparse matrix. `sparse.matvec(A,values,x)` accepts replacement Float64 values in canonical CSR storage order. It can differentiate the values, the dense vector, or both. CSR offsets and column indices remain fixed. A supplied zero retains its structural position and its gradient. The operation traverses CSR rows and keeps the matrix compressed.

[Runnable 0.17.1 example](../examples/gradient_queries.arda) · [Numerical validation](../research/implemented/optimization-and-ad.md#derivative-queries)

## Memory-efficient backward and checkpoints

`ad.backward(output, seed, options)` takes a map with a nonempty `retain` list of connected tracked nodes. Use `mun` for the seed of a scalar loss. The sweep allocates adjoints when needed and retires temporary adjoints after their last use. One private buffer can be reused for another adjoint of the same size. Published gradients for the listed nodes change only after success. Existing gradients of other nodes keep their previous values.

Set `consume: true` in the options map to release the output's ownership of its tape after a successful backward pass. Arda collects history that no other node owns. The output primal and published gradients remain accessible. Later differentiation through that consumed output raises an error. Leave the flag unset for repeated backward passes over the same graph.

Set `samples: count` for the gradient of a batch **mean**. Each selected node publishes the running sample-weighted mean. For example, gradients from batches of sizes two and six contribute one quarter and three quarters of the result. `ad.zero_grad` clears the stored gradient and its count. Clear an existing unweighted gradient before beginning weighted accumulation.

`ad.checkpoint(output, input[, second_input])` packs the pure elementwise tape segment between the specified tracked boundaries and its result. Tensors in the segment have the output shape, with broadcasting allowed for scalars. The resulting node supports backward, JVP, VJP and dense derivative queries. Use ordinary tape operations at segment boundaries for reductions, matrix products and views. Release references to the unpacked segment so its intermediate tensors can be collected.

The checkpoint retains ordinary input constants and already computed random samples. Replay runs only the packed numerical operations. It leaves callbacks, I/O and generator state outside replay. Saved constants and boundaries still use shared-buffer version checks. Replay requires scratch proportional to the number of recorded operations and spends extra CPU time to avoid holding intermediate tensor payloads. [Runnable example](../examples/checkpoint_training.arda) · [Algorithm and validation](../research/implemented/optimization-and-ad.md#checkpointing-and-gradient-liveness)

## Activation offloading

| Call | Behavior |
| --- | --- |
| `ad.offload_plan(directory, disk_budget[, block_elements])` | Use an existing writable directory and set a quota for all live files owned by this plan, including a 128-byte header per activation. The default block is 8,192 Float64 entries (64 KiB). |
| `ad.offload(intermediate, plan)` | Return a tracked node with an immutable file-backed snapshot and the same parents and derivative rule. Rebind the intermediate to the result so the original activation can be collected. |
| `ad.offload_stats(plan)` | Report quota, current and peak disk bytes, current and cumulative file counts, cumulative bytes written and read, number of logical read chunks, block size, and the VM-wide checkpoint replay operation count. |

Offloading applies to recorded dense Float64 tensor intermediates, including packed checkpoints. It creates a new node with its own gradient slot. Leave parameters on the heap and pass the returned node to downstream operations. An empty tensor needs no file. To offload the result of an arbitrary-stride `tensor.view`, materialize it first with an operation such as `view + 0`. A live original node, another operation that saved its primal, or a view alias can keep its original buffer alive. `collect()` makes reclamation synchronous after those references are dropped.

The selected directory can reside on an SSD. Arda unlinks each temporary file as soon as it opens it and releases the mapping and descriptor once the last reference is collected. Forward kernels read the immutable mapping. Derivative rules read saved activations through up to three bounded windows per tape node. Window and replay scratch count against `compute.memory_budget`. `compute.memory_stats()["mapped_bytes"]` accounts for live file mappings separately. An activation must fit while it is produced before it can be spilled, and outputs and adjoints still need their usual space. The operating system controls page-cache residency.

A quota failure raises `disk_budget`, and failures from the underlying filesystem raise `io`. A failed derivative sweep preserves previously published gradients. The file counters include successful transfers through the offload writer and derivative read windows, with headers included for writes. They exclude accesses through the forward mapping. The checkpoint counter measures scalar primal operations replayed across completed tape nodes, with a multiply and a sine each counting as one operation. Arda reports neither hardware FLOP counts nor physical SSD traffic from these counters. [Runnable example](../examples/activation_offload.arda) · [Measurement and algorithm provenance](../research/implemented/optimization-and-ad.md#activation-offloading)

## Sensitivity and uncertainty

- `f.getSensitivity([a,b,...])` and `mathdoc.getSensitivity(f,[a,b,...])` execute a callable once with fresh AD variables for each finite real scalar or dense Float64 tensor argument. The model must return a tracked scalar. The report contains `value`, `absolute` and `relative` fields. The last two are lists in argument order. Absolute sensitivity is the signed derivative in the original units. Relative sensitivity scales each derivative by `input / output`. A zero output leaves every relative entry undefined (`mun`). A scalar zero input also yields `mun`. A tensor input returns a map with `values` and a Boolean `defined` tensor of the input shape, with zero placeholders where the mask is false. An otherwise tracked output has derivative zero for an unused argument. The model runs with ordinary callback effects and differentiates the branch taken at that point.
- `f.getDerivativeTex("a")` and `mathdoc.getDerivativeTex(f,"a")` query the static MathIR. They support scalar straight-line literals, parameters, local bindings, basic arithmetic and calls that resolve to the original built-in `sin`, `cos` or `sqrt`. Formulas assume the original real domain and the differentiability of participating operations. The query returns `mun` when the body, binding, or size falls outside this subset. It never calls the model. With static metadata disabled, numerical sensitivity is still available.
- `ad.hvp(loss, input, direction)` returns the input-shaped product of the Hessian of a tracked scalar loss with the supplied finite, exact-shape direction. The sweep uses the recorded graph, supports checkpoints and spills, and leaves published gradients untouched. Derivative workspace grows with the active tape. [Hessian-vector product benchmark](../research/implemented/optimization-and-ad.md#hessian-vector-products)

For uncertainty propagation, package the inputs into one rank-one Float64 tensor. Provide the covariance as either `{"stddev": s}` with a nonnegative standard deviation for each independent input, or `{"factor": L}` with a Float64 matrix whose rows correspond to inputs. The factor encodes `L Lᵀ` and can be rectangular or rank deficient.

| Call | Result |
| --- | --- |
| `uncertainty.linear(output,input,covariance)` | The tracked scalar output's nominal `value`, first-order `variance` and `stddev`. |
| `uncertainty.covariance_product(output,input,covariance,cotangent)` | Apply the linearized output covariance to a same-shape cotangent. The result has the scalar or tensor output shape. |
| `uncertainty.monte_carlo(model,mean,covariance,generator,samples)` | Sample Gaussian inputs with the supplied covariance. The callback receives each ordinary Float64 vector and returns a finite scalar. The report gives `count`, `mean`, sample `variance` and `stddev` from at least two draws. The generator advances as inputs are drawn, including a draw whose callback fails. |

Linear propagation is a local approximation. For example, a squared Gaussian input centered at zero has zero first-order uncertainty and a nonzero Monte Carlo spread. Monte Carlo uses an explicit generator and retains running moments. Its output summarizes sampling variation in the model, with finite-sample error. Results, inputs and derivatives follow the local tape's ownership. The draft distributed contract specifies proposed parameter identity, ownership and aggregation rules for future MPI work. [Runnable example](../examples/sensitivity_uncertainty.arda) · [Algorithms and evidence](../research/implemented/optimization-and-ad.md#structured-uncertainty-propagation)

[Draft distributed contract](../research/planned/computing-systems.md#draft-distributed-parameter-and-gradient-ownership-contract)
