"""Local single-threaded dense GEMM comparisons and CPU microbenchmarks."""
import argparse
import os
from pathlib import Path
import platform
import shutil
from statistics import median
import struct
import subprocess
import tempfile
from time import perf_counter

ROOT = Path(__file__).resolve().parent
SIZES = {128: 30, 512: 5, 1024: 2}
THREAD_ENV = dict(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  BLIS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")


def save_tensor(path, values):
    header = bytearray(128)
    header[:8] = b"ARDATEN1"
    header[8:11] = bytes((8, values.ndim, 1))
    struct.pack_into("<QQ", header, 16, 128, values.size * 8)
    struct.pack_into("<" + "Q" * values.ndim, header, 32, *values.shape)
    with path.open("wb") as target:
        target.write(header)
        values.astype("<f8", copy=False).tofile(target)


def arda_program(rounds, repeat, setup):
    return f"""
import tensor as t
import compute as c
{setup}
let a = t.mmap("a.atn")
let b = t.mmap("b.atn")
let result = t.matmul(a,b)
for round in range({rounds}):
    let elapsed = 0.0
    for i in range({repeat}):
        collect()
        let start = monotonic()
        result = t.matmul(a,b)
        elapsed = elapsed + (monotonic() - start)
    print(elapsed/{repeat})
t.save("result.atn",result)
"""


def arda_times(command, root, perf=False):
    if perf:
        command = ["perf", "stat", "-e", "cycles,instructions,cache-references,cache-misses",
                   "--", *command]
    return [float(value) for value in
            subprocess.check_output(command, cwd=root, text=True).splitlines()]


def numpy_times(np, a, b, rounds, repeat):
    result = a @ b
    samples = []
    for _ in range(rounds):
        start = perf_counter()
        for _ in range(repeat):
            result = a @ b
        samples.append((perf_counter() - start) / repeat)
    return samples, result


def summarize(name, n, samples, error):
    middle = median(samples)
    mad = median(abs(sample - middle) for sample in samples)
    print(f"{name:16s} {n:4d}  median {middle*1e3:9.4f} ms  MAD {mad*1e3:8.4f} ms  "
          f"range [{min(samples)*1e3:.4f}, {max(samples)*1e3:.4f}] ms  "
          f"max error {error:.3g}",
          flush=True)


def fixture(np, n, root):
    rng = np.random.default_rng(181000 + n)
    ai, bi = (rng.integers(-127, 128, (n, n), dtype=np.int64) for _ in range(2))
    a, b = ai.astype(np.float64) / 128, bi.astype(np.float64) / 128
    save_tensor(root / "a.atn", a)
    save_tensor(root / "b.atn", b)
    indices = np.linspace(0, n - 1, 17, dtype=int)
    exact = (ai[indices, :] @ bi[:, indices]).astype(np.float64) / (128 * 128)
    return a, b, indices, exact


def matrix_bench(np, args, root, eigen):
    for n in args.sizes:
        a, b, indices, exact = fixture(np, n, root)
        repeat = SIZES[n]
        numpy_samples, reference = numpy_times(np, a, b, args.rounds, repeat)
        if not np.array_equal(reference[np.ix_(indices, indices)], exact):
            raise ValueError("NumPy differs from the independent exact integer reference")
        runs = {"numpy": (numpy_samples, reference)}
        for name, binary in args.arda:
            program = root / "dense.arda"
            program.write_text(arda_program(args.rounds, repeat, args.setup), encoding="utf-8")
            samples = arda_times([binary, program], root, args.perf)
            result = np.fromfile(root / "result.atn", dtype="<f8", offset=128).reshape(n, n)
            runs[name] = samples, result
        if eigen:
            samples = arda_times([eigen, str(n), str(args.rounds), str(repeat)], root, args.perf)
            result = np.fromfile(root / "result.raw", dtype="<f8").reshape(n, n)
            runs["eigen"] = samples, result
        for name, (samples, result) in runs.items():
            error = np.max(np.abs(result - reference))
            if not np.array_equal(result, reference):
                raise ValueError(f"{name} produced a different Float64 result at {n}: {error}")
            summarize(name, n, samples, error)


MICRO_OPERATIONS = {
    "GEMV": ("t.matmul(a,v)", 12),
    "SpMV": ("sp.matvec(csr,x)", 12),
    "AXPY": ("high.axpy(x,y,0.25)", 12),
    "reduction": ("t.sum(u)", 12),
    "sin": ("t.sin(u)", 8),
    "cos": ("t.cos(u)", 8),
    "exp": ("t.exp(u)", 8),
    "packed_copy": ('t.cast(a,"float64")', 12),
    "strided_copy": ('t.cast(view,"float64")', 12),
    "VM_loop": ("vm_loop()", 15),
}


def micro_bench(args, root):
    program = root / "micro.arda"
    preamble = """
import tensor as t
import numc as nc
import sparse as sp
import high
import compute as c
let a = t.zeros([256,256],"float64")+0.25
let v = t.zeros([256,1],"float64")+0.5
let x = nc.ones(65536)
let y = nc.ones(65536)
let u = t.zeros([65536],"float64")+0.25
let csr = sp.tridiag(65536,-1,4,-1)
let view = t.permute(a,[1,0])
def vm_loop():
    let sum = 0
    for i in range(20000):
        sum = sum+i
    return sum
"""
    for label, (operation, repeat) in MICRO_OPERATIONS.items():
        program.write_text(preamble + args.setup + f"""
let result = {operation}
for round in range({args.rounds}):
    let elapsed = 0.0
    for i in range({repeat}):
        collect()
        let start = monotonic()
        result = {operation}
        elapsed = elapsed + (monotonic()-start)
    print(elapsed/{repeat})
""", encoding="utf-8")
        for name, binary in args.arda:
            samples = arda_times([binary, program], root, args.perf)
            middle = median(samples)
            suffix = ""
            if label.endswith("_copy"):
                suffix = f"  {2*256*256*8/middle/1e9:.3f} GB/s logical read plus write"
            print(f"{name:16s} {label:14s} {middle*1e3:8.4f} ms{suffix}", flush=True)


def read_arda(value):
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("use NAME=PATH for an Arda binary")
    return name, Path(path).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arda", type=read_arda, action="append", required=True)
    parser.add_argument("--eigen-include", type=Path)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--cpu", type=int, default=min(os.sched_getaffinity(0)))
    parser.add_argument("--sizes", type=int, choices=list(SIZES), nargs="+", default=list(SIZES))
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--micro", action="store_true")
    parser.add_argument("--micro-only", action="store_true")
    parser.add_argument("--perf", action="store_true")
    parser.add_argument("--setup", default="", help="Optional Arda source before each workload")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    if args.perf and not shutil.which("perf"):
        parser.error("--perf requires Linux perf")
    os.sched_setaffinity(0, {args.cpu})
    os.environ.update(THREAD_ENV)
    import numpy as np

    print(platform.platform(), "CPU", args.cpu, "NumPy", np.__version__, flush=True)
    np.show_config()
    for name, binary in args.arda:
        print(name, subprocess.check_output([binary, "--version"], text=True).strip(), binary)
    with tempfile.TemporaryDirectory(prefix="arda-cpu-") as tmp:
        root = Path(tmp)
        eigen = None
        if args.eigen_include:
            eigen = root / "eigen"
            command = [args.cxx, "-O3", "-std=c++17", "-march=native", "-DNDEBUG",
                       "-DEIGEN_DONT_PARALLELIZE", "-I", str(args.eigen_include),
                       ROOT / "cpu_eigen.cpp", "-o", eigen]
            subprocess.run(command, check=True)
            print("Eigen", subprocess.check_output([eigen, "--version"], text=True).strip(),
                  subprocess.check_output([args.cxx, "-dumpfullversion"], text=True).strip())
        if not args.micro_only:
            matrix_bench(np, args, root, eigen)
        if args.micro or args.micro_only:
            micro_bench(args, root)


if __name__ == "__main__":
    main()
