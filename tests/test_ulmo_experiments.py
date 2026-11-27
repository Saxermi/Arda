"""Ulmo experiments, persistent restart, and local Cartesian decomposition."""
import struct
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

import test_arda


class ExperimentTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_sweep_reuses_factor_streams_in_order_and_bounds_errors(self):
        self.run_code("""
import compute as c
import linalg as l
import numc as n
let cases = [12, -1, -2, 18]
let factor = l.lu(n.array([[2]]))
def run(value, f):
    push(cases, 100)
    if value < 0:
        throw {"code":"case_error","message":"case failed"}
    return l.solve(f,n.array([value]))[0]
def sink(row):
    if row["ok"]:
        print(row["index"],row["result"])
    else:
        print(row["index"],row["error"]["code"])
let summary = c.sweep(cases,run,{"factor":factor,"stream":sink,"max_errors":1})
print(summary["succeeded"],summary["failed"],len(summary["results"]),
      summary["errors"][0]["index"],summary["errors_truncated"],len(cases))
print(l.solve(factor,n.array([8]))[0])
""", "0 6\n1 case_error\n2 case_error\n3 9\n2 2 0 1 nae 8\n4", stress=True)

    def test_sweep_collects_results_rejects_limits_and_propagates_sink_errors(self):
        self.run_code("""
import compute as c
def square(x):
    return x*x
let report = c.sweep(range(3),square)
print(report["results"][2]["result"],report["failed"])
let empty = c.sweep([],square,{"max_cases":0})
print(empty["succeeded"],empty["errors_truncated"])
try:
    c.sweep(range(1000000000000),square,{"max_cases":3})
catch e:
    print(e["code"])
def sink(row):
    throw {"code":"sink_error","message":"stream failed"}
try:
    c.sweep([1,2],square,{"stream":sink})
catch e:
    print(e["code"])
try:
    c.sweep([],square,{"factor":23})
catch e:
    print(e["code"])
""", "4 0\n0 law\nsweep_limit\nsink_error\ntype", stress=True)

    def test_amde_contracts_are_available_without_execution(self):
        self.run_code("""
import pde as p
import fem as f
import ode as o
import mathdoc as m
print(m.getTex(p.grid,2)!=nil,m.getTex(p.heat,3)!=nil,m.getAlgorithm(p.wave)!=nil)
print(f.assemble.getAssumptions()!=nil,f.refine.getTex()!=nil,m.getTex(f.integral,1)!=nil)
print(o.rk45.getAlgorithm()!=nil,m.getTex(p.apply_subdomain,1)!=nil)
""", "nae nae nae\nnae nae nae\nnae nae", stress=True, flags=["--mathdoc=none"])


class CheckpointTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def run_process(self, root, name, code):
        script = root / name
        script.write_text(textwrap.dedent(code), encoding="utf-8")
        result = subprocess.run([str(test_arda.BIN), "--gc-stress", str(script)], cwd=root,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def test_restart_in_a_new_process_preserves_rk45_state_and_heat_grid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.run_process(root, "save.arda", """
                import ode as o
                import pde as p
                import numc as n
                def flow(t,y):
                    return -y
                let run = o.rk45(flow,0,n.array([1,2]),0.25,0.02,1e-10,1e-12,1000,{"checkpoint":true})
                o.rk45_save("ode.ckpt",run["checkpoint"])
                let grid = p.grid([3,4],[1,2],["neumann","periodic"])
                let field = n.array([i for i in range(12)])
                let heat = p.heat(grid,field,0.25,0.5,0.125,{"checkpoint":true})
                heat["solution"][0] = 1234
                p.heat_save("heat.ckpt",heat["checkpoint"])
            """)
            output = self.run_process(root, "resume.arda", """
                import ode as o
                import pde as p
                import numc as n
                import ad
                def flow(t,y):
                    return -y
                let cp = o.rk45_load("ode.ckpt")
                let ode = o.rk45_resume(flow,cp,1)
                print(cp["maxiter"],abs(ode["solution"][0]-ad.exp(-1))<1e-9,
                      abs(ode["solution"][1]-2*ad.exp(-1))<2e-9)
                let grid = p.grid([3,4],[1,2],["neumann","periodic"])
                let field = n.array([i for i in range(12)])
                let whole = p.heat(grid,field,0.25,1,0.125)
                let resumed = p.heat_resume("heat.ckpt",0.5,{"checkpoint":true})
                print(resumed["time"],resumed["steps"],n.norm(whole["solution"]-resumed["solution"])==0)
                p.heat_save("heat2.ckpt",resumed["checkpoint"])
                let r = p.heat_resume("heat2.ckpt",0.125)
                let full = p.heat(grid,field,0.25,1.125,0.125)
                print(r["time"],n.norm(r["solution"]-full["solution"])==0)
            """)
            self.assertEqual(output, "1000 nae nae\n1 4 nae\n1.125 nae")
            self.assertEqual({file.name for file in root.glob("*.ckpt")},
                             {"ode.ckpt", "heat.ckpt", "heat2.ckpt"})

    def test_validation_and_disk_budget_preserve_existing_record(self):
        self.run_code("""
import ode as o
import pde as p
let cp = {"version":1,"time":0,"step":0.1,"rtol":1e-6,"atol":1e-9,
          "maxiter":9007199254740993,"state":9223372036854775807}
o.rk45_save("state.ckpt",cp)
try:
    cp["state"] = 43
    o.rk45_save("state.ckpt",cp,1)
catch e:
    print(e["code"])
let restored = o.rk45_load("state.ckpt")
print(restored["state"],restored["maxiter"])
try:
    p.heat_resume("state.ckpt",1)
catch e:
    print(e["code"])
try:
    cp["time"] = 9007199254740993
    o.rk45_save("state.ckpt",cp)
catch e:
    print(e["code"])
""", "disk_budget\n9223372036854775807 9007199254740993\ncheckpoint_format\ncheckpoint_format",
                      stress=True)

    def test_invalid_metadata_and_truncation_release_memory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.run_process(root, "save.arda", """
                import pde as p
                import numc as n
                let run = p.heat(p.grid([4],1),n.ones(4),0.25,1,0.25,{"checkpoint":true})
                p.heat_save("valid.ckpt",run["checkpoint"])
            """)
            data = (root / "valid.ckpt").read_bytes()
            invalid = bytearray(data)
            struct.pack_into("<Q", invalid, 96 + 8 * 3, 0)
            (root / "invalid.ckpt").write_bytes(invalid)
            (root / "short.ckpt").write_bytes(data[:-1])
            output = self.run_process(root, "invalid.arda", """
                import pde as p
                import compute as c
                let live = c.memory_stats()["tensor_heap_bytes"]
                for name in ["invalid.ckpt","short.ckpt"]:
                    try:
                        p.heat_resume(name,1)
                    catch e:
                        print(e["code"],c.memory_stats()["tensor_heap_bytes"]==live)
                c.memory_budget(live+31)
                try:
                    p.heat_resume("valid.ckpt",1)
                catch e:
                    print(e["code"],c.memory_stats()["tensor_heap_bytes"]==live)
            """)
            self.assertEqual(output, "checkpoint_format nae\ncheckpoint_format nae\nmemory_budget nae")


class SubdomainTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_periodic_single_cell_slabs_and_neumann_physical_faces(self):
        self.run_code("""
import pde as p
import linop as l
import numc as n
let grid = p.grid([3],1,"periodic")
let d = p.partition(grid,0,3,0)
print(d["origin"],d["shape"],d["face_shape"],d["face_elements"],d["lower"],d["upper"])
let a = n.array([2])
let b = n.array([4])
let z = n.array([8])
let low = p.subdomain_face(grid,0,3,2,z,"upper")
let up = p.subdomain_face(grid,0,3,1,b,"lower")
let v = p.apply_subdomain(grid,0,3,0,a,low,up)
print(v[0],l.apply(grid,n.array([2,4,8]))[0])
let closed = p.grid([2],1,"neumann")
print(p.apply_subdomain(closed,0,2,0,a,nil,b)[0])
let one = p.grid([1],1,"periodic")
print(p.apply_subdomain(one,0,1,0,a,nil,nil)[0])
try:
    p.apply_subdomain(grid,0,3,0,a,nil,up)
catch e:
    print(e["code"])
try:
    p.partition(grid,0,4,0)
catch e:
    print(e["code"])
""", "[0] [1] [] 1 2 1\n-8 -8\n-2\n0\nshape\ndomain", stress=True)
