"""Measure AMDE metadata overhead with verified, interleaved Linux process runs."""
import argparse
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import tempfile
from time import perf_counter


def workloads():
    definitions = "\n".join(
        f"caro f{i}(x):\n    esta y = x*x\n    nan y+{i}"
        for i in range(2000)
    )
    return {
        "startup": ("pedo(1)\n", "1"),
        "compile_definitions": (definitions + "\npedo(f1999(3))\n", "2008"),
        "vm_calls": ("""
caro f(x):
    nan x+1
esta i = 0
esta total = 0
tenna i < 500000:
    total = total + f(i)
    i = i+1
pedo(total)
""", "125000250000"),
        "dense_matmul": ("""
tol numc as nc
esta a = nc.ones(256,256)
esta b = nc.ones(256,256)
esta c = nc.matmul(a,b)
an i in range(8):
    c = nc.matmul(a,b)
pedo(nc.at(c,[255,255]))
""", "256"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path,
                        default=Path(__file__).resolve().parents[1] / "arda")
    parser.add_argument("--baseline", type=Path,
                        help="Optional separately built Arda 0.15.3 executable")
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--cpu", type=int, help="Optional Linux CPU affinity")
    args = parser.parse_args()
    if args.rounds < 3:
        parser.error("at least three rounds required")
    timer = shutil.which("time")
    if not timer:
        parser.error("GNU time must be available on PATH (Ubuntu package: time)")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    modes = {
        "compact": [str(args.current.resolve()), "--mathdoc=compact"],
        "none": [str(args.current.resolve()), "--mathdoc=none"],
    }
    if args.baseline:
        modes["baseline"] = [str(args.baseline.resolve())]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", LC_ALL="C")
    rng = random.Random(42)
    print(platform.platform(), platform.machine(), "Python", platform.python_version())
    if hasattr(os, "sched_getaffinity"):
        print("CPU affinity:", sorted(os.sched_getaffinity(0)))
    for label, command in modes.items():
        version = subprocess.check_output([command[0], "--version"], text=True).strip()
        print(label, version, command[0])
    print("Wall time includes GNU time wrapper, startup, compilation and execution.")
    print("Peak RSS in KiB; one warmup per workload and mode; raw interleaved samples.")
    with tempfile.TemporaryDirectory(prefix="arda-amde-") as directory:
        root = Path(directory)
        rss_file = root / "rss.txt"
        for name, (source, expected) in workloads().items():
            script = root / (name + ".arda")
            script.write_text(source, encoding="utf-8")
            samples = {label: [] for label in modes}

            def run(label):
                command = [timer, "-f", "%M", "-o", str(rss_file),
                           *modes[label], str(script)]
                start = perf_counter()
                result = subprocess.run(command, env=env, capture_output=True,
                                        text=True, check=True, timeout=120)
                elapsed = (perf_counter() - start) * 1000
                if result.stdout.strip() != expected:
                    raise RuntimeError((name, label, expected, result.stdout))
                return elapsed, int(rss_file.read_text().strip())

            for label in modes:
                run(label)
            for _ in range(args.rounds):
                order = list(modes)
                rng.shuffle(order)
                for label in order:
                    samples[label].append(run(label))
            print("\n" + name)
            medians = {}
            for label, values in samples.items():
                times, rss = zip(*values)
                medians[label] = statistics.median(times)
                print(f"  {label:9s} ms median={medians[label]:.3f} "
                      f"range=[{min(times):.3f},{max(times):.3f}] "
                      f"RSS median={statistics.median(rss):.0f} "
                      f"range=[{min(rss)},{max(rss)}]")
                print("    raw (ms, KiB): " +
                      ", ".join(f"({t:.3f},{m})" for t, m in values))
            print(f"  compact vs none wall time: "
                  f"{100 * (medians['compact'] / medians['none'] - 1):+.2f}%")
            if "baseline" in medians:
                print(f"  compact vs baseline wall time: "
                      f"{100 * (medians['compact'] / medians['baseline'] - 1):+.2f}%")


if __name__ == "__main__":
    main()
