"""Aulë solver contracts, stop reasons, workspace, and error recovery."""
import unittest

import test_arda

IMPORTS = """
import solver as s
import numc as n
import linop as l
import sparse as sp
import tensor as t
import compute as c
"""


class SolverTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_matrix_free_and_csr_under_the_same_contract(self):
        self.run_code(IMPORTS + """
let a = l.poisson([3])
let b = sp.tridiag(3,-1,2,-1)
let rhs = n.array([0,0,4])
let current = s.solve("cg",a,rhs)
let csr = s.solve("cg",b,rhs)
print(l.info(a)["dtype"],l.info(a)["rows"],l.info(a)["workspace_bytes"])
print(n.tolist(l.apply(a,n.array([1,2,3]))),n.tolist(l.apply_transpose(a,n.array([1,2,3]))))
print(current["reason"]==csr["reason"],current["iterations"]==csr["iterations"])
print(abs(n.at(current["solution"],[2])-3) < 1e-12,s.verify(b,rhs,csr["solution"])["verified"])
""", "float64 3 0\n[0, 0, 4] [0, 0, 4]\nnae nae\nnae nae", stress=True)

    def test_typed_negative_strides_and_structured_operators(self):
        self.run_code(IMPORTS + """
let data = t.array([4,-1,-1,4],"float64",[2,2])
let backwards = t.view(data,[2,2],[-2,-1],3)
let rhs = t.view(t.array([3,6],"float64"),[2],[-1],1)
let report = s.solve("cg",backwards,rhs)
print(report["converged"],s.verify(backwards,rhs,report["solution"])["verified"])
let factor = n.array([[1,0],[0,2]])
let kron = l.kron(factor,factor)
let r = s.solve("cg",kron,n.ones(4),{"assume":{"positive_definite":true}})
print(r["converged"],l.info(kron)["workspace_bytes"])
let low = l.lowrank(factor,n.array([[1,0],[0,1]]))
let r = s.solve("gmres",low,n.array([2,4]))
print(r["converged"],s.verify(low,n.array([2,4]),r["solution"])["verified"])
try:
    s.solve("cg",kron,n.ones(4))
catch err:
    print(err["code"])
""", "nae nae\nnae 32\nnae nae\ndomain", stress=True)

    def test_tolerance_failure_and_trace_limits(self):
        self.run_code(IMPORTS + """
let indefinite = n.array([[1,2],[2,1]])
let b = n.array([1,-1])
let bad = s.solve("cg",indefinite,b)
let good = s.solve("minres",indefinite,b,{"trace":5})
print(bad["reason"],bad["breakdown"],good["converged"],len(good["trace"]))
let a = sp.tridiag(5,-1,2,-1)
let limited = s.solve("gmres",a,n.array([0,1,0,0,0]),{"restart":1,"maxiter":2,"trace":1})
print(limited["reason"],limited["converged"],limited["iterations"],limited["trace_truncated"])
let zero = s.solve("cg",a,n.zeros(5))
print(zero["reason"],zero["residual"],zero["iterations"])
""", "nonpositive_curvature nae nae 1\nmax_iterations law 2 nae\nconverged 0 0", stress=True)

    def test_budget_and_unwind_leave_existing_tensors_usable(self):
        self.run_code(IMPORTS + """
let a = n.array([[2,1],[0,2]])
let rhs = n.ones(2)
let live = c.memory_stats()["tensor_heap_bytes"]
let good = s.solve("gmres",a,rhs)
let work = good["workspace_bytes"]
good = mun
collect()
c.memory_budget(live+work+16)
try:
    s.solve("gmres",a,rhs,{"workspace_bytes":work-1})
catch err:
    print(err["code"])
try:
    s.solve("cg",a,rhs)
catch err:
    print(err["code"],c.memory_stats()["tensor_heap_bytes"]==live)
let again = s.solve("gmres",a,rhs,{"workspace_bytes":work})
print(again["converged"],s.verify(a,rhs,again["solution"])["verified"])
""", "memory_budget\ndomain nae\nnae nae", stress=True)

    def test_preconditioning_and_coo_failure_releases_payloads(self):
        self.run_code(IMPORTS + """
let a = sp.tridiag(5,-1,4,-1)
let b = n.array([1,2,4,5,3])
for preconditioner in ["jacobi","block_jacobi","ic0"]:
    let r = s.solve("cg",a,b,{"preconditioner":preconditioner})
    print(r["converged"],r["preconditioner"],s.verify(a,b,r["solution"])["verified"])
let non = sp.coo(2,2,[0,0,1,1],[0,1,0,1],[3,1,-1,2])
for method in ["gmres","bicgstab"]:
    let r = s.solve(method,non,n.array([3,1]),{"preconditioner":"ilu0"})
    print(r["converged"],r["iterations"])
try:
    s.solve("cg",a,b,{"preconditioner":"ilu0"})
catch err:
    print(err["code"])
let missing = sp.coo(2,2,[0,1],[1,0],[1,1])
let live = c.memory_stats()["tensor_heap_bytes"]
let rhs = n.ones(2)
try:
    s.solve("gmres",missing,rhs,{"preconditioner":"ilu0"})
catch err:
    print(err["code"],c.memory_stats()["tensor_heap_bytes"]==live+16)
""", "nae jacobi nae\nnae block_jacobi nae\nnae ic0 nae\nnae 1\nnae 1\ndomain\nsolver_breakdown nae",
                      stress=True)

    def test_inconsistent_least_squares_uses_normal_residual(self):
        self.run_code(IMPORTS + """
let a = n.array([[1,0],[1,1],[1,2]])
let b = n.array([1,4,3])
for method in ["lsqr","lsmr"]:
    let r = s.solve(method,a,b)
    let x = r["solution"]
    print(r["reason"],r["residual"]>0.3,r["normal_residual"]<1e-10,
          abs(n.at(x,[0])-5.0/3.0)<1e-12,abs(n.at(x,[1])-1)<1e-12)
""", "least_squares nae nae nae nae\nleast_squares nae nae nae nae", stress=True)

    def test_multigrid_bounds_and_weighted_rectangular_grid(self):
        self.run_code(IMPORTS + """
let op = l.poisson([3,4,5],[2,3,1])
let b = n.ones(60)
let good = s.multigrid(op,b,{"trace":2})
print(good["converged"],s.verify(op,b,good["solution"])["verified"],good["trace_truncated"])
let short = s.multigrid(op,b,{"maxiter":1})
print(short["reason"],short["iterations"])
try:
    s.multigrid(op,b,{"workspace_bytes":1})
catch err:
    print(err["code"])
""", "nae nae nae\nmax_iterations 1\nmemory_budget", stress=True)

    def test_mixed_refinement_and_float64_fallback(self):
        self.run_code(IMPORTS + """
let a = n.array([[4,1],[1,3]])
let b = n.array([6,7])
let r = s.refine(a,b,{"tol":1e-14})
print(r["converged"],r["fallback"],r["factor_dtype"],s.verify(a,b,r["solution"])["verified"])
let difficult = n.array([[1,1],[1,1+1e-8]])
let rhs = n.array([3,3+2e-8])
let precise = s.refine(difficult,rhs,{"tol":1e-14})
print(precise["converged"],precise["fallback"],precise["fallback_reason"],precise["factor_dtype"])
let singular = s.refine(n.array([[1,1],[1,1]]),n.array([1,2]))
print(singular["converged"],singular["breakdown"],singular["fallback_reason"])
""", "nae law float32 nae\nnae nae float32_factorization float64\nlaw nae float64_singular",
                      stress=True)

    def test_lanczos_exact_and_invariant_subspace(self):
        self.run_code(IMPORTS + """
let a = l.diagonal(n.array([1,6,2]))
let report = s.lanczos(a,n.array([1,1,1]),2,{"basis":3})
print(report["converged"],abs(n.at(report["values"],[0])-6)<1e-12,
      abs(n.at(report["values"],[1])-2)<1e-12)
let partial = s.lanczos(a,n.array([0,1,0]),2,{"basis":3})
print(partial["converged"],partial["reason"],n.tolist(partial["values"]))
""", "nae nae nae\nlaw invariant_subspace [6]", stress=True)

    def test_verification_real_complex_and_svd(self):
        self.run_code(IMPORTS + """
import complex as z
let diagonal = n.array([[2,0],[0,4]])
let b = n.array([2,8])
let check = s.verify(diagonal,b,n.array([1,2]),{"condition":true})
print(check["verified"],check["condition_inf"],check["backward_error_inf"])
let rot = n.array([[0,-1],[1,0]])
let e = s.verify_eigen(rot,z.new(0,1),{"real":n.array([1,0]),"imag":n.array([0,-1])})
print(e["verified"],e["residual"])
let u = n.array([0,1])
print(s.verify_svd(diagonal,4,u,u)["verified"],s.verify_svd(diagonal,3,u,u)["verified"])
print(s.requirements("minres")["symmetric"],s.requirements("lsqr")["apply_transpose"])
""", "nae 2 0\nnae 0\nnae law\nnae nae", stress=True)

    def test_arnoldi_complex_pair_or_explicit_backend_error(self):
        self.run_code(IMPORTS + """
let a = n.array([[0,-2,1],[2,0,0],[0,0,1]])
let supported = false
try:
    let r = s.arnoldi(a,n.ones(3),2,{"basis":3,"tol":1e-12})
    let im = r["values_imag"]
    supported = r["converged"] and abs(abs(n.at(im,[0]))-2)<1e-12 and abs(n.at(im,[0])+n.at(im,[1]))<1e-12
catch err:
    supported = err["code"]=="backend"
print(supported)
""", "nae", stress=True)

    def test_retired_solver_exports_are_absent(self):
        imports = "tol high\ntol sparse as sp\n"
        for name in ("high.cg", "high.poisson", "high.laplace", "high.lstsq",
                     "sp.cg", "sp.pcg", "sp.verify", "sp.workspace"):
            with self.subTest(name=name):
                self.run_code(imports + f"pedo({name})", error="no attribute", stress=True)
