"""Runtime contracts for NVB, RT0 Darcy and RK4 stage checkpoints."""
import unittest

import test_arda


class UlmoAlternativesTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_nvb_neighbor_closure_lineage_and_prolongation(self):
        self.run_code("""
import fem
import numc as n
let x = n.array([[0,0],[2,0],[0.9,0.5],[0.2,-3]])
let m = fem.mesh(x,[[0,1,2],[1,0,3]])
let field = n.array([1,-2,3,5])
let f = fem.refine(m,field,n.array([1,0]),{"method":"nvb"})
let h = fem.hierarchy(f["mesh"])
print(n.tolist(f["edge_midpoints"]),n.tolist(f["parents"]))
print(n.tolist(h["leaves"]),n.tolist(h["parents"]),n.tolist(h["generations"]))
let g = fem.refine(f["mesh"],f["field"],n.ones(5),{"method":"nvb","fraction":1})
let newer = fem.hierarchy(g["mesh"])
let valid = true
for i in range(len(h["parents"])):
    if h["parents"][i] != newer["parents"][i] or h["generations"][i] != newer["generations"][i]:
        valid = false
for leaf in newer["leaves"]:
    let at = int(leaf)
    let depth = newer["generations"][at]
    while depth > 0:
        at = int(newer["parents"][at])
        depth = depth-1
        if newer["generations"][at] != depth:
            valid = false
    if at > 1 or newer["parents"][at] != -1:
        valid = false
let reset = fem.mesh(fem.nodes(g["mesh"]),fem.elements(g["mesh"]))
print(valid,len(fem.elements(reset)),abs(fem.integral(m,field)-fem.integral(g["mesh"],g["field"]))<1e-13)
try:
    fem.refine(m,field,n.array([1,0]),{"method":"nvb","max_elements":4})
catch e:
    print(e["code"])
""", """[[0, 1], [1, 3]] [0, 0, 1, 1, 1]
[2, 3, 4, 6, 7] [-1, -1, 0, 0, 1, 1, 5, 5] [0, 0, 1, 1, 1, 1, 2, 2]
nae 11 nae
memory_budget""", stress=True)

    def test_mixed_affine_solution_and_cleanup(self):
        self.run_code("""
import fem
import numc as n
import sparse
import compute
let points = n.array([[0,0],[1,0],[1,1],[0,1]])
let m = fem.mesh(points,[[0,2,1],[0,2,3]])
let fixed = [[0,0],[1,1],[2,1],[3,0]]
try:
    fem.darcy_mixed(m,0,[[0,0]],1)
catch e:
    print(e["code"])
try:
    fem.darcy_mixed(m,0,fixed,1,{"workspace_bytes":0})
catch e:
    print(e["code"])
collect()
let before = compute.memory_stats()["tensor_heap_bytes"]
sparse.set_budget(500)
try:
    fem.darcy_mixed(m,0,fixed,1)
catch e:
    print(sparse.memory()["used"]==0,compute.memory_stats()["tensor_heap_bytes"]==before)
sparse.set_budget(nil)
let r = fem.darcy_mixed(m,0,fixed,1,{"tol":1e-13})
print(r["converged"],r["face_dofs"],r["max_cell_imbalance"]<1e-14)
print(n.norm(r["pressure"]-n.array([2.0/3,1.0/3]))<1e-14,
      n.norm(r["velocity"]-n.array([[-1,0],[-1,0]]))<1e-14)
print(n.at(r["flux"],[0,2])+n.at(r["flux"],[1,2])==0,sparse.memory()["used"]==0)
""", "domain\nmemory_budget\nnae nae\nnae 5 nae\nnae nae\nnae nae", stress=True)

    def test_stage_checkpoints_match_block_with_budget_and_streaming(self):
        self.run_code("""
import sim
import numc as n
import pde
let visits = []
let evaluations = 0
def stream(t,y):
    push(visits,t)
    y[0] = 900
def rhs(t,y,p):
    evaluations = evaluations + 1
    return p*y*(1-y) + 0.01*t*y*y
let model = sim.rk4(rhs)
let y = n.array([0.4,0.2,0.7])
let p = n.array([0.4,0.5,0.6])
let seed = n.array([0.5,-1,2])
let a = sim.adjoint(model,1,y,p,0.03,0.05,seed,{"checkpoint_stride":5,"max_saved_states":20})
let counted = evaluations
evaluations = 0
let b = sim.adjoint(model,1,y,p,0.03,0.05,seed,{"checkpoint_stride":5,"max_saved_states":20,
                   "checkpoint_strategy":"stages","max_stage_tapes":1,"stream":stream})
print(len(visits),visits[0],visits[-1],a["rhs_evaluations"]==counted,
      b["rhs_evaluations"]==evaluations)
print(n.norm(a["solution"]-b["solution"])==0,
      n.norm(a["initial_gradient"]-b["initial_gradient"])<1e-13,
      n.norm(a["parameter_gradient"]-b["parameter_gradient"])<1e-13)
print(b["reused_step_tapes"]>0 and b["reused_step_tapes"]<=b["checkpoints"],
      b["peak_checkpoint_slots"]<=20,
      a["rhs_evaluations"]-b["rhs_evaluations"]==4*b["reused_step_tapes"])
let tight = sim.adjoint(model,1,y,p,0.03,0.05,seed,
                        {"checkpoint_stride":5,"max_saved_states":8,"checkpoint_strategy":"stages"})
print(tight["peak_checkpoint_slots"]<=8,n.norm(tight["parameter_gradient"]-a["parameter_gradient"])<1e-13)
try:
    sim.adjoint(sim.heat(pde.grid([3],1)),0,y,0.1,1,0.1,seed,{"checkpoint_strategy":"stages"})
catch e:
    print(e["code"])
""", "21 1 0.03 nae nae\nnae nae nae\nnae nae nae\nnae nae\ndomain", stress=True)

    def test_fit_and_zero_step_stage_checkpoints(self):
        self.run_code("""
import sim
def rhs(t,y,p):
    return -p*y
let model = sim.rk4(rhs)
let options = {"checkpoint_strategy":"stages","checkpoint_stride":4,"max_saved_states":30}
let target = sim.solve(model,0,1.0,0.6,1,0.04)["solution"]
let r = sim.fit_terminal(model,0,1.0,0.3,1,0.04,target,options,1e-10)
print(r["converged"],abs(r["parameter"]-0.6)<1e-8)
let zero = sim.adjoint(model,1,2.0,0.3,1,0.04,3,options)
print(zero["rhs_evaluations"],zero["reused_step_tapes"],zero["initial_gradient"],
      zero["parameter_gradient"],zero["peak_checkpoint_slots"])
""", "nae nae\n0 0 3 0 0", stress=True)
