"""Measure planned Float64 graphs and a real file scan under a process limit."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
from statistics import median
import struct
import subprocess
import tempfile


IMPORTS = """
import tensor as t
import compute as c
"""
LAZY = "import lazy as l\n"
MODES = ("eager", "explicit_fusion", "lazy_fused", "in_ram", "tiled")


def array_source(n, mode):
    general = mode in ("in_ram", "tiled")
    source = IMPORTS + f"""
let a = t.zeros([{n}],"float64")+2.0
let b = t.zeros([{n}],"float64")+3.0
"""
    if not general:
        source += f'let d = t.zeros([{n}],"float64")+4.0\n'
    if mode in ("eager", "explicit_fusion"):
        expr = "a*b+d" if mode == "eager" else "c.mul_add(a,b,d)"
        source += "let planning = 0.0\nlet workspace = " + (
            str(n * 8) if mode == "eager" else "0") + "\n"
    else:
        source = LAZY + source + """
let x = l.input(a)
let y = l.input(b)
"""
        graph = "l.mul(l.add(x,y),l.sub(x,y))" if general else "l.add(l.mul(x,y),l.input(d))"
        kernel = mode if general else "fused"
        source += f"""
let g = {graph}
let start_plan = monotonic()
let p = c.plan("graph",g,{{"kernel":"{kernel}"}})
let planning = monotonic()-start_plan
let workspace = c.explain(p)["workspace_bytes"]
"""
        expr = "c.execute(p)"
    return source + f"""
collect()
let start = monotonic()
let out = {expr}
let elapsed = monotonic()-start
print(elapsed,planning,workspace,c.memory_stats()["peak_budgeted_bytes"],t.sum(out))
"""


def file_fixture(path, n):
    header = bytearray(128)
    header[:8] = b"ARDATEN1"
    header[8:11] = bytes((8, 2, 1))
    struct.pack_into("<QQQQ", header, 16, 128, n * 8, n // 65536, 65536)
    chunk = struct.pack("<4d", 0, 1, 2, 3) * 16384
    with path.open("wb") as out:
        out.write(header)
        for _ in range(n // 65536):
            out.write(chunk)
        out.flush()
        os.fsync(out.fileno())


def invoke(binary, directory, timer, source, *, limit=None):
    script = directory / "bench.arda"
    stats = directory / "rss.txt"
    script.write_text(source, encoding="utf-8")
    prefix = ["prlimit", f"--as={limit}:{limit}", "--"] if limit else []
    result = subprocess.run([timer, "-f", "%M", "-o", stats, *prefix, binary, script],
                            cwd=directory, capture_output=True, text=True, check=True, timeout=300)
    return result.stdout.split(), int(stats.read_text(encoding="utf-8"))


def measure_file(args, directory):
    n = args.file_mib * 1024 * 1024 // 8
    path = directory / "input.atn"
    file_fixture(path, n)
    source = LAZY + IMPORTS + """
c.memory_budget(2097152)
let x = l.file("input.atn")
let g = l.sum(l.mul(l.sin(x),l.exp(l.neg(x))))
let p = c.plan("graph",g,{"block_elements":65536})
let info = c.explain(p)
let r = c.report(c.run(p,2))
print(info["kernel"],r["elapsed_seconds"],r["workspace_peak_bytes"],r["blocks"],
      c.memory_stats()["peak_budgeted_bytes"],r["value"],len(r["trace"]))
"""
    limit = 128 * 1024**2
    (kernel, secs, work, blocks, peak, actual, trace), rss = invoke(
        args.arda, directory, args.time, source, limit=limit)
    expected = math.fsum(math.sin(x) * math.exp(-x) for x in range(4)) * (n // 4)
    if kernel != "fused" or int(trace) != 2 or int(blocks) != n // 65536:
        raise AssertionError((kernel, blocks, trace))
    if not math.isclose(float(actual), expected, rel_tol=2e-11, abs_tol=1e-12):
        raise AssertionError((actual, expected))
    return dict(bytes=n * 8, file_blocks=path.stat().st_blocks, kernel=kernel,
                elapsed_seconds=float(secs), workspace_bytes=int(work), blocks=int(blocks),
                peak_budgeted_bytes=int(peak), rss_kib=rss, address_limit_bytes=limit,
                result=float(actual), reference=expected,
                relative_error=abs(float(actual) - expected) / abs(expected))


def measure_arrays(args, directory):
    rows = {name: [] for name in MODES}
    for round_index in range(args.rounds + args.warmups):
        order = MODES[round_index % len(MODES):] + MODES[:round_index % len(MODES)]
        for mode in order:
            binary = args.baseline if mode in ("eager", "explicit_fusion") else args.arda
            fields, rss = invoke(binary, directory, args.time, array_source(args.elements, mode))
            seconds, planning, workspace, peak, result = map(float, fields)
            expected = (-5 if mode in ("in_ram", "tiled") else 10) * args.elements
            if result != expected:
                raise AssertionError((mode, result, expected))
            if round_index >= args.warmups:
                rows[mode].append(dict(elapsed_seconds=seconds, planning_seconds=planning,
                                       workspace_bytes=int(workspace), peak_budgeted_bytes=int(peak),
                                       rss_kib=rss))
    return {mode: {**{k: median(row[k] for row in samples) for k in samples[0]},
                   "seconds_min": min(row["elapsed_seconds"] for row in samples),
                   "seconds_max": max(row["elapsed_seconds"] for row in samples),
                   "samples": samples} for mode, samples in rows.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arda", type=lambda name: Path(name).resolve(), default=Path("arda"))
    parser.add_argument("--baseline", type=lambda name: Path(name).resolve())
    parser.add_argument("--time", default=shutil.which("time"))
    parser.add_argument("--elements", type=int, default=4_194_304)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--file-mib", type=int, default=0)
    args = parser.parse_args()
    args.arda = args.arda.resolve()
    args.baseline = args.baseline or args.arda
    if not args.time or args.elements < 1 or args.rounds < 1 or args.warmups < 0 or args.file_mib < 0:
        parser.error("GNU time and positive element and round counts are required")
    with tempfile.TemporaryDirectory(prefix="arda-ace-") as directory:
        root = Path(directory)
        result = dict(elements=args.elements, rounds=args.rounds, warmups=args.warmups,
                      arrays=measure_arrays(args, root))
        if args.file_mib:
            result["file"] = measure_file(args, root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
