#!/usr/bin/env python3
"""Compare portable Arda executables and optionally generate GNU gprof profiles."""

import argparse
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
from textwrap import dedent
import time


CASES = {
    "locals": ("72000006000000", """
        def solve(n):
            let i = 0
            let total = 0
            while i < n:
                i = i + 1
                total = total + i
            return total
        print(solve(12000000))
    """),
    "calls": ("3500000", """
        def add(a,b=1):
            return a+b
        def solve(n):
            let total = 0
            for i in range(n):
                total = add(total)
            return total
        print(solve(3500000))
    """),
    "attributes": ("5000000", """
        class Box:
            def init(self,n):
                self.n = n
            def add(self,x):
                return self.n+x
        def solve(box,n):
            let total = 0
            for i in range(n):
                total = box.add(total)
            return total
        print(solve(Box(2),2500000))
    """),
    "allocations": ("1500000", """
        def solve(n):
            let total = 0
            for i in range(n):
                let a = [i,1]
                let m = {"x":a}
                total = total + m["x"][1]
            return total
        print(solve(1500000))
    """),
    "numeric": ("6144", """
        import numc as n
        let a = n.ones(256,256)
        let total = 0.0
        for i in range(24):
            let b = n.matmul(a,a)
            total = total + b[0][0]
        print(total)
    """),
}

CASES["numeric-long"] = (
    "49152", CASES["numeric"][1].replace("range(24)", "range(192)")
)


def run(binary, program, expected, cwd, extra=()):
    result = subprocess.run([str(binary), *extra, str(program)], cwd=cwd,
                            capture_output=True, text=True, timeout=30)
    if result.returncode or result.stdout.strip() != expected:
        raise RuntimeError(f"{binary} {program.name}: returned {result.returncode}, "
                           f"expected {expected}, got {result.stdout!r}\n{result.stderr}")
    return result


def profile(binary, program, expected, directory):
    directory.mkdir(parents=True, exist_ok=True)
    run(binary, program, expected, directory)
    result = subprocess.run(["gprof", "-b", str(binary), "gmon.out"], cwd=directory,
                            capture_output=True, text=True, timeout=30, check=True)
    (directory / (program.stem + ".gprof")).write_text(result.stdout, encoding="utf-8")


def timed(time_bin, binary, program, expected, directory):
    rss_path = directory / "rss.txt"
    start = time.perf_counter()
    result = subprocess.run([time_bin, "-f", "%M", "-o", str(rss_path), str(binary),
                             str(program)], cwd=directory, capture_output=True, text=True,
                            timeout=30)
    elapsed = 1000 * (time.perf_counter() - start)
    if result.returncode or result.stdout.strip() != expected:
        raise RuntimeError(f"{binary} {program.name}: returned {result.returncode}, "
                           f"expected {expected}, got {result.stdout!r}\n{result.stderr}")
    return elapsed, int(rss_path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--arda", type=Path, default=Path("arda"))
    parser.add_argument("--time", default="/usr/bin/time", help="GNU time binary")
    parser.add_argument("--cpu", type=int, help="restrict runs to one Linux logical CPU")
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--case", action="append", choices=CASES, help="measure only selected cases")
    parser.add_argument("--profile-baseline", type=Path, help="baseline binary built with -pg")
    parser.add_argument("--profile-arda", type=Path, help="candidate binary built with -pg")
    parser.add_argument("--profile-directory", type=Path)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    if (args.profile_baseline or args.profile_arda) and not args.profile_directory:
        parser.error("use --profile-directory with instrumented binaries")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    binaries = (args.baseline.resolve(strict=True), args.arda.resolve(strict=True))
    with tempfile.TemporaryDirectory(prefix="arda-palantir-") as tmp:
        directory = Path(tmp)
        for name in args.case or CASES:
            expected, code = CASES[name]
            program = directory / (name + ".arda")
            program.write_text(dedent(code).lstrip(), encoding="utf-8")
            rows = [[], []]
            for iteration in range(args.runs + 2):
                for slot in ((0, 1) if iteration % 2 == 0 else (1, 0)):
                    values = timed(args.time, binaries[slot], program, expected, directory)
                    if iteration >= 2:
                        rows[slot].append(values)
            for label, row in zip(("before", "after"), rows):
                wall, rss = zip(*row)
                print(f"{name:12} {label:6} median={statistics.median(wall):.3f} ms "
                      f"range=[{min(wall):.3f}, {max(wall):.3f}] ms "
                      f"peak={max(rss)} KiB", flush=True)
            if name == "allocations":
                result = run(binaries[1], program, expected, directory, ("--gc-profile",))
                print(result.stderr.strip(), flush=True)
            for label, binary in (("baseline", args.profile_baseline), ("arda", args.profile_arda)):
                if binary:
                    profile(binary.resolve(strict=True), program, expected,
                            args.profile_directory.resolve() / label)


if __name__ == "__main__":
    main()
