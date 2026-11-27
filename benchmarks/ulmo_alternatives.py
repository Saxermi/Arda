"""Compare Ulmo's refinement, Darcy and checkpoint algorithms on one CPU."""
import argparse
import ast
import json
import math
import os
from pathlib import Path
from statistics import median
import subprocess
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]

GRID = """
import fem
import numc as n
let x = n.zeros(width*width,2)
let field = n.zeros(width*width)
let cells = []
let boundary = []
for j in range(width):
    for i in range(width):
        let a = j*width+i
        let px = i/(width-1)
        let py = j/(width-1)
        n.put(x,[a,0],px)
        n.put(x,[a,1],py)
        field[a] = px*px+py
        if i == 0 or i == width-1:
            push(boundary,[a,2*px-px*px])
        if i < width-1 and j < width-1:
            push(cells,[a,a+1,a+width+1])
            push(cells,[a,a+width+1,a+width])
let mesh = fem.mesh(x,cells)
"""

REFINE = """
let integral = fem.integral(mesh,field)
let ms = 0.0
for level in range(12):
    let points = fem.nodes(mesh)
    let triangles = fem.elements(mesh)
    let best = 0
    let distance = 2
    for e in range(len(triangles)):
        let c = triangles[e]
        let dx = (n.at(points,[c[0],0])+n.at(points,[c[1],0])+n.at(points,[c[2],0]))/3-0.2973
        let dy = (n.at(points,[c[0],1])+n.at(points,[c[1],1])+n.at(points,[c[2],1]))/3-0.4127
        let d = dx*dx+dy*dy
        if d < distance:
            distance = d
            best = e
    let eta = n.zeros(len(triangles))
    eta[best] = 1
    let start = monotonic()
    let options = {}
    if method == "nvb":
        options["method"] = "nvb"
    let next = fem.refine(mesh,field,eta,options)
    ms = ms + 1000*(monotonic()-start)
    mesh = next["mesh"]
    field = next["field"]
print([ms,abs(fem.integral(mesh,field)-integral)])
print(n.tolist(fem.nodes(mesh)))
print(fem.elements(mesh))
"""

DARCY = """
import solver
let result = nil
let velocity = nil
let imbalance = 0.0
let face_dofs = 0
let start = monotonic()
if method == "p1":
    let s = fem.assemble(mesh,2,boundary)
    result = solver.solve("cg",s["matrix"],s["rhs"],{"tol":1e-10,"preconditioner":"jacobi"})
    velocity = fem.darcy_velocity(mesh,result["solution"],1)
    # The divergence of a constant velocity on every triangle is zero.
    imbalance = 1.0/((width-1)*(width-1))
else:
    result = fem.darcy_mixed(mesh,2,boundary,1,{"tol":1e-10,"maxiter":4000})
    velocity = result["velocity"]
    imbalance = result["max_cell_imbalance"]
    face_dofs = result["face_dofs"]
let ms = 1000*(monotonic()-start)
if not result["converged"]:
    throw {"code":"benchmark","message":result["reason"]}
let error2 = 0.0
let exact2 = 0.0
for e in range(len(cells)):
    let c = cells[e]
    let px = (n.at(x,[c[0],0])+n.at(x,[c[1],0])+n.at(x,[c[2],0]))/3
    let exact = 2*px-2
    let dx = n.at(velocity,[e,0])-exact
    let dy = n.at(velocity,[e,1])
    error2 = error2 + dx*dx + dy*dy
    exact2 = exact2 + exact*exact
print([ms,len(x),len(cells),face_dofs,result["iterations"],result["residual"],
       imbalance,sqrt(error2/exact2)])
"""

CHECKPOINT = """
import sim
import numc as n
import compute
let state = n.linspace(0.1,0.8,1024)
let direction = n.ones(1024)/1024
let rhs_calls = 0
def rhs(t,y,p):
    rhs_calls = rhs_calls+1
    return p*y*(1-y) + 0.01*t*y*y
let model = sim.rk4(rhs)
compute.memory_budget(tensor_mib*1024*1024)
let options = {"checkpoint_stride":16,"max_saved_states":saved_slots}
if method == "stages":
    options["checkpoint_strategy"] = method
    options["max_stage_tapes"] = stage_tapes
let start = monotonic()
let answer = sim.adjoint(model,0,state,0.7,1,1.0/256,direction,options)
let ms = 1000*(monotonic()-start)
let reused = 0
let peak = 31
if method == "stages":
    reused = answer["reused_step_tapes"]
    peak = answer["peak_checkpoint_slots"]
print([ms,answer["steps"],peak,rhs_calls,reused,compute.memory_stats()["peak_budgeted_bytes"],
       answer["parameter_gradient"]])
print(n.tolist(answer["initial_gradient"]))
"""

CASES = {
    "refine": ("midpoint", "nvb"),
    "darcy": ("p1", "rt0"),
    "checkpoint-4mib": ("block", "stages"),
    "checkpoint-16mib": ("block", "stages"),
}
FIELDS = {
    "refine": ("ms", "integral_error", "vertices", "triangles", "min_angle_deg", "area"),
    "darcy": ("ms", "vertices", "triangles", "face_dofs", "iterations", "residual",
              "max_imbalance", "relative_centroid_velocity_error"),
    "checkpoint": ("ms", "steps", "peak_slots", "rhs_calls", "reused_tapes", "peak_tensor_bytes",
                   "parameter_gradient"),
}


def mesh_quality(points, triangles):
    angle, area = math.pi, 0
    for cell in triangles:
        for i, a in enumerate(cell):
            b, c = cell[(i+1) % 3], cell[(i+2) % 3]
            ux, uy = (d-s for d, s in zip(points[b], points[a]))
            vx, vy = (d-s for d, s in zip(points[c], points[a]))
            det = abs(ux*vy-uy*vx)
            angle = min(angle, math.atan2(det, ux*vx+uy*vy))
            if i == 0:
                area += det/2
    return math.degrees(angle), area


def run_one(options, scratch, case, method, baseline):
    front = "let method = " + json.dumps(method) + "\n"
    if case == "refine":
        source = front + "let width = 16\n" + GRID + REFINE
    elif case == "darcy":
        source = front + "let width = 32\n" + GRID + DARCY
    else:
        budget, slots, tapes = (4, 33, 1) if case == "checkpoint-4mib" else (16, 64, 16)
        source = (front + f"let tensor_mib = {budget}\nlet saved_slots = {slots}\n"
                  + f"let stage_tapes = {tapes}\n" + CHECKPOINT)
    program, rss = scratch/"trial.arda", scratch/"rss"
    program.write_text(source)
    binary = options.baseline if baseline and options.baseline else options.arda
    run = subprocess.run(
        [options.time_bin, "-f", "%M", "-o", str(rss), str(binary), str(program)],
        capture_output=True, text=True, timeout=120,
    )
    if run.returncode:
        raise RuntimeError(run.stderr.strip())
    lines = run.stdout.splitlines()
    measures = ast.literal_eval(lines[0])
    if case == "refine":
        points, triangles = map(ast.literal_eval, lines[1:])
        angle, area = mesh_quality(points, triangles)
        measures += [len(points), len(triangles), angle, area]
    gradient = ast.literal_eval(lines[1]) if case.startswith("checkpoint") else None
    return [int(rss.read_text()), *measures], gradient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arda", type=Path, default=ROOT/"arda")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--time-bin", default="/usr/bin/time")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--case", choices=("all", *CASES), default="all")
    options = parser.parse_args()
    options.arda = options.arda.resolve()
    if options.baseline:
        options.baseline = options.baseline.resolve()
    if options.cpu is not None:
        os.sched_setaffinity(0, {options.cpu})
    if options.runs < 1:
        parser.error("--runs must be positive")
    with TemporaryDirectory() as tmp:
        for case, methods in CASES.items():
            if options.case not in ("all", case):
                continue
            runs = {name: [] for name in methods}
            for _ in range(options.runs):
                gradients = []
                for i, name in enumerate(methods):
                    sample, gradient = run_one(options, Path(tmp), case, name, i == 0)
                    runs[name].append(sample)
                    gradients.append(gradient)
                if case.startswith("checkpoint"):
                    terminal_diff = abs(runs[methods[0]][-1][-1]-runs[methods[1]][-1][-1])
                    if terminal_diff >= 1e-13 or max(abs(a-b) for a, b in zip(*gradients)) >= 1e-13:
                        raise AssertionError("checkpoint strategies produced different derivatives")
            for name in methods:
                fields = "checkpoint" if case.startswith("checkpoint") else case
                metrics = dict(zip(("peak_rss_kib", *FIELDS[fields]),
                                   (median(values) for values in zip(*runs[name]))))
                print(case, name, json.dumps(metrics))


if __name__ == "__main__":
    main()
