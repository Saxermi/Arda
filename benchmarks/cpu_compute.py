"""Reproduce Shadowfax fusion, buffered I/O, and native CPU scaling measurements."""
import argparse
import os
from pathlib import Path
import shutil
from statistics import median
from time import perf_counter
import subprocess
import tempfile

from cpu_baseline import save_tensor


def run(binary, root, source, env, *, rss=False):
    script = root / "bench.arda"
    script.write_text(source, encoding="utf-8")
    command = [str(binary), "--mathdoc=none", str(script)]
    rss_file = root / "rss.txt"
    if rss:
        timer = shutil.which("time")
        if not timer:
            raise RuntimeError("the fusion benchmark requires GNU time on PATH")
        command = [timer, "-f", "%M", "-o", str(rss_file), *command]
    result = subprocess.run(command, cwd=root, env=env, check=True, capture_output=True,
                            text=True, timeout=120)
    return result.stdout.splitlines(), int(rss_file.read_text()) if rss else None


IMPORTS = """
import tensor as t
import compute as c
"""


def fusion(binary, root, env, rounds, np):
    preamble = IMPORTS + """
let a = t.zeros([10000000],"float64")+2.0
let b = t.zeros([10000000],"float64")+3.0
let d = t.zeros([10000000],"float64")+4.0
let e = t.zeros([10000000],"float64")+5.0
let f = t.zeros([10000000],"float64")+6.0
collect()
let start = monotonic()
"""
    expressions = ("a*b+d*e+f", "c.fused_products(a,b,d,e,f)")
    samples = [[], []]
    for index in range(rounds):
        for side in (index % 2, 1 - index % 2):
            result, rss = run(binary, root, preamble + f"""
let x = {expressions[side]}
let elapsed = monotonic()-start
print(elapsed,t.sum(x),c.memory_stats()["peak_budgeted_bytes"])
""", env, rss=True)
            elapsed, result_sum, peak = map(float, result[0].split())
            np.testing.assert_equal(result_sum, 320_000_000)
            samples[side].append((elapsed, rss, int(peak)))
    for name, data in zip(("plain", "fused"), samples):
        seconds, rss, peak = map(median, zip(*data))
        print(f"fusion {name:5s} {seconds*1000:.3f} ms {rss:g} KiB RSS {peak:g} budgeted bytes "
              f"samples={data}", flush=True)


def streaming(binary, root, env, rounds, np, rng):
    data = rng.standard_normal(8_388_608)
    save_tensor(root / "stream.atn", data)
    expected = [len(data), data.mean(), data.var()]
    output, _ = run(binary, root, IMPORTS + f"""
let a = c.stream_stats("stream.atn",65536)
let b = c.stream_stats("stream.atn",65536,{{"prefetch":true}})
for i in range({rounds}):
    for phase in range(2):
        let mode = (i+phase)%2
        let start = monotonic()
        let s = c.stream_stats("stream.atn",65536,{{"prefetch":mode==1}})
        print(mode,monotonic()-start,s["count"],s["mean"],s["variance"])
""", env)
    samples = [[], []]
    for line in output:
        mode, seconds, *statistics = map(float, line.split())
        np.testing.assert_allclose(statistics, expected, rtol=1e-12, atol=1e-12)
        samples[int(mode)].append(seconds)
    for name, data in zip(("serial", "prefetch"), samples):
        print(f"stream {name:8s} {median(data)*1000:.3f} ms samples={data}", flush=True)


def scaling(binary, root, env, rounds, np, rng):
    from scipy import sparse

    for workload, a_shape, b_shape in (
            ("GEMM 128", (128, 128), (128, 128)), ("GEMM 768", (768, 768), (768, 768)),
            ("GEMV 128", (128, 128), (128,)), ("GEMV 4096x2048", (4096, 2048), (2048,))):
        a = rng.standard_normal(a_shape)
        b = rng.standard_normal(b_shape)
        save_tensor(root / "a.atn", a)
        save_tensor(root / "b.atn", b)
        expected = a @ b
        source = IMPORTS + """
let a = t.mmap("a.atn")
let b = t.mmap("b.atn")
let p = c.plan("matmul",a,b,{"threads":REQUEST})
let x = c.execute(p,a,b)
for i in range(ROUNDS):
    collect()
    let start = monotonic()
    x = c.execute(p,a,b)
    print(monotonic()-start,c.last_run()["threads"])
t.save("product.atn",x)
"""
        for request in (1, 2, 4, 8):
            output, _ = run(binary, root, source.replace("REQUEST", str(request))
                            .replace("ROUNDS", str(rounds)), env)
            samples = [float(line.split()[0]) for line in output]
            teams = sorted({int(line.split()[1]) for line in output})
            actual = np.fromfile(root / "product.atn", dtype="<f8", offset=128)
            error = float(np.max(np.abs(actual - expected.ravel())))
            np.testing.assert_allclose(actual, expected.ravel(), rtol=1e-10, atol=1e-11)
            print(f"{workload:15s} requested={request} team={teams} {median(samples)*1000:.3f} ms "
                  f"max_abs={error:.3g} samples={samples}", flush=True)

    length = 262_144
    diag = sparse.diags([-1, 4, -1], [-1, 0, 1], shape=(length, length), format="csr", dtype=np.float64)
    vector = np.ones(length)
    expected = diag @ vector
    scipy_times = []
    for _ in range(rounds):
        start = perf_counter()
        result = diag @ vector
        scipy_times.append(perf_counter() - start)
        np.testing.assert_array_equal(result, expected)
    print(f"SciPy SpMV {length} {median(scipy_times)*1000:.3f} ms samples={scipy_times}", flush=True)
    for request in (1, 2, 4, 8):
        output, _ = run(binary, root, IMPORTS + f"""
import sparse as sp
import numc as nc
c.threads({request})
let a = sp.tridiag({length},-1,4,-1)
let b = nc.ones({length})
let x = sp.matvec(a,b)
for i in range({rounds}):
    collect()
    let start = monotonic()
    x = sp.matvec(a,b)
    print(monotonic()-start)
t.save("sparse.atn",x)
""", env)
        actual = np.fromfile(root / "sparse.atn", dtype="<f8", offset=128)
        np.testing.assert_array_equal(actual, expected)
        samples = [float(value) for value in output]
        print(f"SpMV {length}     requested={request} serial {median(samples)*1000:.3f} ms "
              f"samples={samples}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arda", type=Path, required=True)
    parser.add_argument("--suite", choices=("fusion", "stream", "scaling"), required=True)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--cpus", type=int, nargs="+")
    parser.add_argument("--thread-limit", type=int, default=2)
    args = parser.parse_args()
    if args.rounds < 3 or args.thread_limit < 1:
        parser.error("use at least three rounds and a positive thread limit")
    if args.cpus:
        os.sched_setaffinity(0, set(args.cpus))
    for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[variable] = "1"
    import numpy as np

    env = dict(os.environ, OMP_THREAD_LIMIT=str(args.thread_limit), OMP_PROC_BIND="spread",
               OMP_PLACES="cores", OMP_DYNAMIC="FALSE")
    binary = args.arda.resolve()
    print(subprocess.check_output([str(binary), "--version"], text=True).strip(), flush=True)
    if hasattr(os, "sched_getaffinity"):
        print("CPUs:", sorted(os.sched_getaffinity(0)), "OMP_THREAD_LIMIT:", args.thread_limit,
              "NumPy:", np.__version__, flush=True)
    rng = np.random.default_rng(1806)
    with tempfile.TemporaryDirectory(prefix="arda-compute-") as directory:
        root = Path(directory)
        if args.suite == "fusion":
            fusion(binary, root, env, args.rounds, np)
        elif args.suite == "stream":
            streaming(binary, root, env, args.rounds, np, rng)
        else:
            scaling(binary, root, env, args.rounds, np, rng)


if __name__ == "__main__":
    main()
