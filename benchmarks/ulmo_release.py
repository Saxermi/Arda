"""Reproduce the heat, FEM and cross-process restart checks for Ulmo."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_SETUP = """\
import ode as o
import pde as p
import numc as n
let side = 256
let h = 1.0/(side+1)
let dt = 0.00048828125
let duration = 100*dt
let mode = n.sin(3.141592653589793*n.linspace(h,1-h,side))
let initial = n.reshape(n.reshape(mode,[side,1])*n.reshape(mode,[1,side]),[side*side])
let grid = p.grid([side,side],h)
def decay(t,y):
    return -y
"""
CHECKPOINT_SAVE = CHECKPOINT_SETUP + """\
let start = monotonic()
let ode = o.rk45(decay,0,1.0,0.25,0.02,1e-10,1e-12,1000,{"checkpoint":true})
let heat = p.heat(grid,initial,0.001,duration,dt,{"checkpoint":true})
o.rk45_save("ode.ckpt",ode["checkpoint"])
p.heat_save("heat.ckpt",heat["checkpoint"],1048576)
print(monotonic()-start)
"""
CHECKPOINT_RESUME = CHECKPOINT_SETUP + """\
import ad
let start = monotonic()
let ode = o.rk45_resume(decay,o.rk45_load("ode.ckpt"),1)
let heat = p.heat_resume("heat.ckpt",duration)
let elapsed = monotonic()-start
let whole = p.heat(grid,initial,0.001,2*duration,dt)
print(elapsed,heat["time"],n.norm(heat["solution"]-whole["solution"]),
      abs(ode["solution"]-ad.exp(-1)))
"""


def run(command, directory, time_bin):
    rss_file = directory / "rss.txt"
    start = perf_counter()
    result = subprocess.run([time_bin, "-f", "%M", "-o", str(rss_file), *command],
                            cwd=directory, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"{command[-1]} failed ({result.returncode}): {result.stderr.strip()}")
    wall = perf_counter() - start
    return result.stdout, wall * 1000, int(rss_file.read_text())


def number_after(pattern, output):
    match = re.search(pattern, output, re.MULTILINE)
    if match is None:
        raise ValueError(f"Missing measurement matching {pattern!r} in {output!r}")
    return tuple(float(value) for value in match.groups())


def checked(value, limit, name):
    if not math.isfinite(value) or not 0 <= value <= limit:
        raise ValueError(f"{name} exceeded {limit:g}: {value:g}")
    return value


def heat_metrics(output, wall, rss):
    error, solver = number_after(r"^Relative heat error (\S+) seconds (\S+)$", output)
    decay, = number_after(r"^Resumed decay error (\S+)$", output)
    radau, = number_after(r"^Stiff steps \d+ error (\S+)$", output)
    checked(decay, 1e-8, "RK45 decay error")
    checked(radau, 1e-6, "Radau error")
    return dict(wall_ms=wall, solver_ms=solver * 1000, peak_rss_kib=rss,
                relative_error=checked(error, 1e-6, "heat error"))


def fem_metrics(output, wall, rss):
    iterations, residual = number_after(
        r"^Converged, iterations, relative residual (?:nae|True) (\d+) (\S+)$", output)
    error, = number_after(r"^Relative nodal error (\S+)$", output)
    mesh, assembly, solver = number_after(r"^Mesh, assembly, CG milliseconds (\S+) (\S+) (\S+)$",
                                         output)
    return dict(wall_ms=wall, mesh_ms=mesh, assembly_ms=assembly, solver_ms=solver,
                peak_rss_kib=rss, iterations=iterations,
                relative_error=checked(error, 1e-8, "FEM nodal error"),
                relative_residual=checked(residual, 1e-10, "FEM residual"))


def checkpoint_metrics(binary, directory, time_bin, flags):
    for name in ("heat.ckpt", "ode.ckpt"):
        (directory / name).unlink(missing_ok=True)
    save, save_wall, save_rss = run([binary, *flags, "save.arda"], directory, time_bin)
    resumed, resume_wall, resume_rss = run([binary, *flags, "resume.arda"], directory, time_bin)
    save_seconds, = map(float, save.split())
    resume_seconds, time, heat_difference, ode_error = map(float, resumed.split())
    if time != 200 * 0.00048828125:
        raise ValueError(f"Unexpected resumed time: {time}")
    return dict(wall_ms=save_wall + resume_wall, save_ms=save_seconds * 1000,
                resume_ms=resume_seconds * 1000, peak_rss_kib=max(save_rss, resume_rss),
                heat_difference=checked(heat_difference, 0, "step-aligned heat difference"),
                ode_error=checked(ode_error, 1e-9, "persisted RK45 error"))


def summarize(samples):
    summary = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        summary[key] = dict(median=statistics.median(values), minimum=min(values), maximum=max(values))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arda", type=Path, default=ROOT / "arda")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--scipy", action="store_true")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--gc-stress", action="store_true")
    parser.add_argument("--time-bin", default="/usr/bin/time")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    flags = ["--gc-stress"] if args.gc_stress else []
    binaries = {"candidate": str(args.arda.resolve())}
    if args.baseline:
        binaries["baseline"] = str(args.baseline.resolve())
    cases = {}
    with tempfile.TemporaryDirectory(prefix="arda-ulmo-release-") as tmp:
        directory = Path(tmp)
        for label in binaries:
            root = directory / label
            root.mkdir()
            (root / "save.arda").write_text(CHECKPOINT_SAVE, encoding="utf-8")
            (root / "resume.arda").write_text(CHECKPOINT_RESUME, encoding="utf-8")
        labels = list(binaries)
        for round_index in range(args.runs + 1):
            order = labels if round_index % 2 == 0 else labels[::-1]
            for label in order:
                binary = binaries[label]
                root = directory / label
                heat = heat_metrics(*run([binary, *flags, str(ROOT / "examples/ulmo.arda")],
                                        root, args.time_bin))
                fem = fem_metrics(*run([binary, *flags, str(ROOT / "examples/ulmo_fem.arda")],
                                      root, args.time_bin))
                restart = checkpoint_metrics(binary, root, args.time_bin, flags)
                if round_index:
                    for name, result in (("heat", heat), ("fem", fem), ("restart", restart)):
                        cases.setdefault((label, name), []).append(result)
            if args.scipy:
                scipy = fem_metrics(*run([sys.executable, str(ROOT / "benchmarks/ulmo_fem_scipy.py")],
                                         directory, args.time_bin))
                if round_index:
                    cases.setdefault(("scipy", "fem"), []).append(scipy)
    print(json.dumps({f"{label}.{name}": summarize(samples)
                      for (label, name), samples in cases.items()}, indent=2))


if __name__ == "__main__":
    main()
