"""FEM and differentiable simulation contracts, including errors under GC stress."""
import unittest

import test_arda


class FemTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_discontinuous_material_and_darcy_flux(self):
        self.run_code("""
import fem
import numc as n
import solver
let coords = n.array([0,0.5,1])
let mesh = fem.mesh(coords,[[1,0],[1,2]])
coords[0] = 12
let snapshot = fem.nodes(mesh)
snapshot[0] = -1
let k = n.array([1,4])
let bc = [[0,0],[2,1]]
let s = fem.assemble(mesh,0,bc,{"diffusivity":k})
let r = solver.solve("cg",s["matrix"],s["rhs"])
let u = r["solution"]
print(type(mesh),n.tolist(fem.nodes(mesh)),fem.elements(mesh))
print(r["converged"],abs(u[1]-0.8)<1e-14)
print(n.norm(fem.darcy_velocity(mesh,u,k)+1.6)<1e-14)
print(n.norm(fem.estimate(mesh,0,bc,u,{"diffusivity":k}))<1e-24)
let react = fem.assemble(mesh,n.array([2,2,2]),[],{"reaction":2})
let constant = solver.solve("cg",react["matrix"],react["rhs"])
print(constant["converged"],n.norm(constant["solution"]-1)<1e-12)
""", "fem_mesh [0, 0.5, 1] [[1, 0], [1, 2]]\nnae nae\nnae\nnae\nnae nae", stress=True)

    def test_conforming_refinement_and_exact_prolongation(self):
        self.run_code("""
import fem
import numc as n
let m = fem.mesh(n.array([[0,0],[1,0],[1,1],[0,1]]),[[0,2,1],[0,2,3]])
let u = n.array([1,3,-2,5])
let r = fem.refine(m,u,n.array([1,0]))
let new = r["mesh"]
print(fem.elements(new),n.tolist(r["parents"]))
print(n.tolist(r["edge_midpoints"]),n.tolist(r["field"]))
print(abs(fem.integral(m,u)-fem.integral(new,r["field"]))<1e-14)
let none = fem.refine(new,r["field"],n.zeros(6))
print(len(fem.elements(none["mesh"])),n.norm(none["field"]-r["field"]),n.shape(none["edge_midpoints"]))
let full = fem.refine(new,r["field"],n.ones(6),{"fraction":1})
print(len(fem.elements(full["mesh"])),abs(fem.integral(full["mesh"],full["field"])-fem.integral(m,u))<1e-14)
""", """[[0, 5, 4], [5, 2, 6], [4, 6, 1], [6, 4, 5], [0, 5, 3], [5, 2, 3]] [0, 0, 0, 0, 1, 1]
[[0, 1], [0, 2], [1, 2]] [1, 3, -2, 5, 2, -0.5, 0.5]
nae
6 0 [0, 2]
24 nae""", stress=True)

    def test_neighbor_with_two_split_edges(self):
        self.run_code("""
import fem
import numc as n
let m = fem.mesh(n.array([[0,0],[1,0],[0,1],[0,-1],[-1,0]]),
                 [[0,1,2],[0,3,1],[0,2,4]])
let u = n.array([1,-2,3,4,5])
let r = fem.refine(m,u,n.array([0,1,1]),{"fraction":1})
let p = r["parents"]
print(len(p),p[0],p[2],p[3],p[6],p[7],p[10])
print(abs(fem.integral(m,u)-fem.integral(r["mesh"],r["field"]))<1e-14)
# Reconstructing the mesh checks all new shared faces and their orientations.
let rebuilt = fem.mesh(fem.nodes(r["mesh"]),fem.elements(r["mesh"]))
print(len(fem.elements(rebuilt)))
""", "11 0 0 1 1 2 2\nnae\n11", stress=True)

    def test_invalid_mesh_options_and_recovery(self):
        self.run_code("""
import fem
import numc as n
let x = n.array([0,0.5,1])
let m = fem.mesh(x,[[0,1],[1,2]])
def reject(call):
    try:
        call()
    catch e:
        print(e["code"])
def duplicate():
    fem.mesh(x,[[0,1],[1,0],[1,2]])
def degenerate():
    fem.mesh(n.array([[0,0],[1,1],[2,2]]),[[0,1,2]])
def floating():
    fem.assemble(m,0,[])
def bad_diffusivity():
    fem.assemble(m,0,[[0,0]],{"diffusivity":n.array([1,0])})
def limited():
    fem.assemble(m,0,[[0,0]],{"workspace_bytes":0})
def negative_estimate():
    fem.refine(m,x,n.array([1,-2]))
def limited_refinement():
    fem.refine(m,x,n.ones(2),{"max_nodes":3})
def disjoint():
    let d = fem.mesh(n.array([0,1,2,3]),[[0,1],[2,3]])
    fem.assemble(d,0,[[0,0]])
reject(duplicate)
reject(degenerate)
reject(floating)
reject(bad_diffusivity)
reject(limited)
reject(negative_estimate)
reject(limited_refinement)
reject(disjoint)
collect()
print(fem.integral(m,x),n.tolist(fem.assemble(m,0,[[0,7]])["rhs"]))
""", "domain\ndomain\ndomain\ndomain\nmemory_budget\ndomain\nmemory_budget\ndomain\n0.5 [7, 14, 0]",
                      stress=True)

    def test_sparse_budget_and_successful_reuse(self):
        self.run_code("""
import fem
import sparse
import compute
import numc as n
let m = fem.mesh(n.linspace(0,1,18),[[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],
    [8,9],[9,10],[10,11],[11,12],[12,13],[13,14],[14,15],[15,16],[16,17]])
collect()
collect()
let before = compute.memory_stats()["tensor_heap_bytes"]
sparse.set_budget(2500)
try:
    fem.assemble(m,1,[[0,0],[17,0]])
catch e:
    print(e["message"],sparse.memory()["used"]==0,
          compute.memory_stats()["tensor_heap_bytes"]==before)
let tiny = fem.mesh(n.array([0,1]),[[0,1]])
print(n.tolist(fem.assemble(tiny,1,[[0,0],[1,1]])["rhs"]))
""", "sparse memory budget exceeded nae nae\n[0, 1]", stress=True)


class DifferentiableSimulationTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_checkpoint_replay_and_streaming(self):
        self.run_code("""
import sim
import numc as n
let visits = []
def stream(t,y):
    push(visits,t)
    y[0] = 500
def rhs(t,y,p):
    return (t-p)*y
let model = sim.rk4(rhs)
let y = n.array([2,3])
let seed = n.array([0.4,-1])
let a = sim.adjoint(model,0,y,0.8,1.01,0.1,seed,
                    {"checkpoint_stride":3,"max_saved_states":6,"stream":stream})
let fresh = sim.adjoint(model,0,y,0.8,1.01,0.1,seed,{"checkpoint_stride":1})
print(a["steps"],a["checkpoints"],a["recomputed_steps"],len(visits),visits[0],visits[-1])
print(n.norm(a["solution"]-fresh["solution"])==0,
      n.norm(a["initial_gradient"]-fresh["initial_gradient"])==0,
      a["parameter_gradient"]==fresh["parameter_gradient"],fresh["recomputed_steps"])
try:
    sim.adjoint(model,0,y,0.8,1.01,0.1,seed,{"checkpoint_stride":3,"max_saved_states":4})
catch e:
    print(e["code"])
let identity = sim.adjoint(model,1,y,0.8,1,0.1,seed)
print(identity["steps"],identity["checkpoints"],identity["parameter_gradient"],
      n.norm(identity["initial_gradient"]-seed)==0)
""", "11 4 7 12 0 1.01\nnae nae nae 0\nmemory_budget\n0 0 0 nae", stress=True)

    def test_rk4_backward_and_disconnected_parameter(self):
        self.run_code("""
import sim
import ad
def independent(t,y,p):
    return -y
let model = sim.rk4(independent)
let a = sim.adjoint(model,1,ad.exp(-1),2,0,0.025,1)
let j = sim.jvp(model,1,ad.exp(-1),2,0,0.025,1,10)
print(abs(a["solution"]-1)<4e-9,abs(a["initial_gradient"]-ad.exp(1))<1e-8)
print(a["parameter_gradient"],abs(j["tangent"]-a["initial_gradient"])<1e-13)
""", "nae nae\n0 nae", stress=True)

    def test_heat_validation_and_fit(self):
        self.run_code("""
import sim
import pde
import linop
import numc as n
try:
    sim.heat(linop.poisson([4]))
catch e:
    print(e["code"])
let heat = sim.heat(pde.grid([4],0.25,"periodic"))
let y = n.array([1,0,-1,0])
try:
    sim.solve(heat,0,n.ones(3),0.1,0,0.1)
catch e:
    print(e["code"])
try:
    sim.jvp(heat,0,y,1,1,0.1,n.zeros(4),1)
catch e:
    print(e["code"])
let target = sim.solve(heat,0,y,0.1,1,0.1)["solution"]
let fit = sim.fit_terminal(heat,0,y,0.12,1,0.1,target)
print(fit["converged"],abs(fit["parameter"]-0.1)<2e-9,
      n.norm(fit["terminal_state"]-target)<1e-9)
""", "type\ndomain\npde_stability\nnae nae nae", stress=True)

    def test_ode_fit_step_limit_and_validation(self):
        self.run_code("""
import sim
import ad
def rate(t,y,p):
    return -p*y
let m = sim.rk4(rate)
let r = sim.fit_terminal(m,0,1.0,0.2,1,0.02,ad.exp(-0.6))
print(r["converged"],abs(r["parameter"]-0.6)<2e-8)
try:
    sim.solve(m,0,1,0.6,1,0.02,{"max_steps":2})
catch e:
    print(e["code"])
try:
    sim.solve(m,10000000000000000,1,0.6,10000000000000002,0.02)
catch e:
    print(e["code"])
try:
    sim.adjoint(m,0,1,0.6,1,0.02,[],{})
catch e:
    print(e["code"])
""", "nae nae\ndomain\ndomain\nshape", stress=True)
