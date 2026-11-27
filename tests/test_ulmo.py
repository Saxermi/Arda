"""Simulation API, event and restart contracts, and PDE error cleanup."""
import unittest

import test_arda

IMPORTS = """
import ode
import ad
import pde
import numc as n
import linop as l
import compute as c
"""


class UlmoTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_dense_output_saveat_and_endpoints(self):
        self.run_code(IMPORTS + """
def decay(t,y):
    return -y
let r = ode.rk45(decay,0,1.0,1,0.9,1e-9,1e-12,1000,
                 {"saveat":[0,0.019,0.37,0.92,1],"dense_output":true})
let at = r["at"]
print(r["times"],len(r["states"]),r["rejected_steps"]>0)
print(abs(at(0.333)-ad.exp(-0.333))<2e-9,r["states"][0]==at(0),r["states"][4]==at(1))
try:
    at(1.001)
catch e:
    print(e["code"])
let back = ode.rk45(decay,1,ad.exp(-1),0,0.3,1e-9,1e-12,1000,
                    {"saveat":[1,0.123,0],"dense_output":true})
let reverse_at = back["at"]
print(back["times"],abs(reverse_at(0.87)-ad.exp(-0.87))<2e-9)
""", "[0, 0.019, 0.37, 0.92, 1] 5 nae\nnae nae nae\ndomain\n[1, 0.123, 0] nae", stress=True)

    def test_events_sorted_and_terminal_truncates_the_step(self):
        self.run_code(IMPORTS + """
def unit(t,y):
    return 1
def early(t,y):
    return y-0.2
def stop(t,y):
    return y-0.4
def late(t,y):
    return y-0.8
let r = ode.rk45(unit,0,0.0,1,1,1e-9,1e-12,100,
                 {"events":[{"fn":late},{"fn":stop,"terminal":true},{"fn":early}],
                  "saveat":[0,0.3,0.6],"dense_output":true})
let hits = r["events"]
print(r["reason"],len(hits),hits[0]["index"],hits[1]["index"],r["times"])
print(abs(r["time"]-0.4)<2e-10,abs(hits[0]["state"]-0.2)<2e-10)
let back = ode.rk45(unit,1,1.0,0,1,1e-9,1e-12,100,
                   {"events":[{"fn":stop,"direction":1},{"fn":early,"direction":-1}]})
print(len(back["events"]),abs(back["events"][0]["time"]-0.4)<2e-10)
""", "event 2 2 1 [0, 0.3]\nnae nae\n1 nae", stress=True)

    def test_stream_snapshots_and_resume(self):
        self.run_code(IMPORTS + """
def decay(t,y):
    return -y
let times = []
def sink(t,y):
    push(times,t)
    y[0] = 999
let r = ode.rk45(decay,0,n.array([1,2]),0.5,0.2,1e-9,1e-12,1000,
                 {"saveat":[0,0.13,0.5],"stream":sink,"checkpoint":true})
let checkpoint = r["checkpoint"]
r["solution"][0] = 123
let resumed = ode.rk45_resume(decay,checkpoint,1)
print(times,r["times"],r["states"])
print(n.norm(resumed["solution"]-ad.exp(-1)*n.array([1,2]))<3e-9)
try:
    ode.rk45(decay,0,1,1,0.1,1e-6,1e-9,100,{"saveat":[0.5,0.1]})
catch e:
    print(e["code"])
""", "[0, 0.13, 0.5] [] []\nnae\ndomain", stress=True)

    def test_verlet_energy_and_time_reversal(self):
        self.run_code(IMPORTS + """
def accel(t,q):
    return -q
let r = ode.verlet(accel,0,n.array([1,2]),n.zeros(2),0.02,1000)
let q = r["position"]
let p = r["momentum"]
print(abs(n.dot(q,q)+n.dot(p,p)-5)<0.001)
let reversed = ode.verlet(accel,r["time"],q,p,-0.02,1000)
print(reversed["time"],n.norm(reversed["position"]-n.array([1,2]))<1e-12,
      n.norm(reversed["momentum"])<1e-12)
""", "nae\n0 nae nae", stress=True)

    def test_implicit_euler_bdf2_and_radau(self):
        self.run_code(IMPORTS + """
def decay(t,y):
    return -y
def jac(t,y):
    return -1
let euler = ode.implicit_euler(decay,0,1.0,1,0.1,{"jacobian":jac})
let bdf = ode.bdf2(decay,0,1.0,1,0.1,{"jacobian":jac})
let radau = ode.radau(decay,0,1.0,1,0.1)
print(euler["accepted_steps"],bdf["accepted_steps"],radau["accepted_steps"])
print(abs(euler["solution"]-0.385543289429532)<1e-9)
print(abs(bdf["solution"]-ad.exp(-1))<0.002,abs(radau["solution"]-ad.exp(-1))<6e-6)
let reverse = ode.radau(decay,1,ad.exp(-1),0,0.04)
print(reverse["time"],abs(reverse["solution"]-1)<2e-6)
""", "10 10 10\nnae\nnae nae\n0 nae", stress=True)

    def test_implicit_vector_jvp_and_explicit_gmres_selection(self):
        self.run_code(IMPORTS + """
def rhs(t,y):
    return n.array([-100*y[0],3*y[0]-2*y[1]])
def jvp(t,y,v):
    return rhs(t,v)
let a = ode.implicit_euler(rhs,0,n.array([1,2]),0.2,0.02)
let b = ode.implicit_euler(rhs,0,n.array([1,2]),0.2,0.02,
                           {"jvp":jvp,"linear_solver":"gmres","preconditioner":"jacobi"})
let r = ode.radau(rhs,0,n.array([1,2]),0.2,0.02,{"jvp":jvp})
print(n.norm(a["solution"]-b["solution"])<2e-9,r["newton_iterations"]>0)
try:
    ode.implicit_euler(rhs,0,n.ones(2),1,0.01,{"max_steps":1})
catch e:
    print(e["code"])
""", "nae nae\node_limit", stress=True)

    def test_pde_operator_boundaries_and_poisson_gauge(self):
        self.run_code(IMPORTS + """
let periodic = pde.grid([4],1,"periodic")
let neumann = pde.grid([4],1,"neumann")
let dirichlet = pde.grid([4],1,"dirichlet")
let u = n.array([1,2,3,4])
print(n.tolist(l.apply(periodic,u)),n.tolist(l.apply(neumann,u)),n.tolist(l.apply(dirichlet,u)))
let r = pde.poisson(neumann,l.apply(neumann,u))
print(r["gauge"],r["converged"],n.norm(r["solution"]-(u-2.5))<1e-12)
let d = pde.poisson(dirichlet,l.apply(dirichlet,u))
print(d["gauge"],n.norm(d["solution"]-u)<1e-12)
try:
    pde.poisson(periodic,n.ones(4))
catch e:
    print(e["code"])
""", "[-4, 0, 0, 4] [-1, 0, 0, 1] [0, 0, 0, 5]\nzero_mean nae nae\nfixed_boundary nae\npde_compatibility",
                      stress=True)

    def test_pde_periodic_conservation_and_courant_limit(self):
        self.run_code(IMPORTS + """
let g = pde.grid([4],0.25,"periodic")
let u = n.array([1,2,3,4])
let a = pde.advection_diffusion(g,u,1,0,0.25,0.25)
let reverse = pde.advection_diffusion(g,a["solution"],-1,0,0.25,0.25)
let h = pde.heat(g,u,0.1,0.63,0.1)
let closed = pde.advection_diffusion(pde.grid([4],0.25,"neumann"),u,1,0.05,0.4,0.08)
print(n.tolist(a["solution"]),n.tolist(reverse["solution"]))
print(h["steps"],h["time"],abs(n.sum(h["solution"])-10)<1e-13,abs(n.sum(closed["solution"])-10)<1e-13)
try:
    pde.heat(g,u,1,0.1,0.1)
catch e:
    print(e["code"])
try:
    pde.wave(g,u,n.zeros(4),2,0.2,0.2)
catch e:
    print(e["code"])
""", "[4, 1, 2, 3] [1, 2, 3, 4]\n7 0.63 nae nae\npde_stability\npde_stability", stress=True)

    def test_pde_workspace_rejection_and_error_unwind(self):
        self.run_code(IMPORTS + """
let grid = pde.grid([4],1)
let input = n.ones(4)
let bad = n.ones(3)
let baseline = c.memory_stats()["tensor_heap_bytes"]
try:
    pde.heat(grid,bad,1,0.1,0.1)
catch e:
    print(e["code"],c.memory_stats()["tensor_heap_bytes"]==baseline)
try:
    pde.heat(grid,input,1,0.1,0.1,{"workspace_bytes":31})
catch e:
    print(e["code"],c.memory_stats()["tensor_heap_bytes"]==baseline)
let r = pde.heat(grid,input,1,0,0.1,{"workspace_bytes":0})
print(r["steps"],n.tolist(r["solution"]))
""", "shape nae\nmemory_budget nae\n0 [1, 1, 1, 1]", stress=True)
