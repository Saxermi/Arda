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
| `high.cg`, `high.poisson`, `high.lstsq`, `sparse.cg`, `sparse.pcg` | `solution`, `iterations`, `residual`, `converged`, `reason` |
| `ode.rk45` | `solution`, `accepted_steps`, `rejected_steps` |
| `opt.lbfgs` | `solution`, `iterations`, `gradient_norm`, `converged`, `value`, `reason` |
| Zeroth-order optimizers | The L-BFGS fields and `evaluations` |
| `opt.adam` | `parameters`, `first_moment`, `second_moment` |

An L-BFGS objective callback returns a map with `value` and `gradient`. Direct solves return the solution value. CG and Poisson report the relative data residual. CGLS reports the relative normal-equation residual.

Use `at(x,[i,j])` and `put(x,[i,j],value)` for dense element access. Use `tensor.sum(x[,axis])` for typed full and axis sums. Float64 reductions use extended accumulation where `long double` provides it. The optional axis removes that axis. A rank-one axis reduction returns a scalar. Use `opt.sgd` and `opt.adam` for optimizer updates, `sparse.csr` and `sparse.to_csr` for CSR construction and conversion, and `linalg.eigh` for symmetric eigenpairs.

## Dense Float64 autodiff

Follow [the runnable example](../examples/tensor_autodiff.arda) and [the mathematical and memory contracts](../research/implemented/optimization-and-ad.md#reverse-mode-autodiff).
