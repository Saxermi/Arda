"""Compare Arda Multigrid, Arda CG and SciPy CG on a discrete Poisson problem."""
import argparse
import json
import os
from pathlib import Path
import shutil
from statistics import median
import subprocess
import sys
import tempfile
from time import perf_counter

from cpu_baseline import save_tensor


def manufactured(n, np):
    h = 1 / (n + 1)
    x = np.arange(1, n+1, dtype=np.float64) * h
    a = [0.2, -0.3, 0.7]
    f = [x * (1-x) * (1+c*x) for c in a]
    # The discrete centered second difference is exact for each cubic.
    d = [h*h*(2*(1-c)+6*c*x) for c in a]

    def product(ax, ay, az):
        return ax[:, None, None] * ay[None, :, None] * az[None, None, :]

    u = 64 * product(*f)
    b = 64 * (product(d[0], f[1], f[2]) + product(f[0], d[1], f[2]) +
              product(f[0], f[1], d[2]))
    return u, b


def stencil(vector, n):
    x = vector.reshape((n, n, n))
    y = 6 * x
    y[1:, :, :] -= x[:-1, :, :]
    y[:-1, :, :] -= x[1:, :, :]
    y[:, 1:, :] -= x[:, :-1, :]
    y[:, :-1, :] -= x[:, 1:, :]
    y[:, :, 1:] -= x[:, :, :-1]
    y[:, :, :-1] -= x[:, :, 1:]
    return y.ravel()


def scipy_worker(n, tol):
    import numpy as np
    from scipy.sparse.linalg import LinearOperator, cg
    rhs = np.memmap("rhs.atn", dtype="<f8", mode="r", offset=128, shape=(n**3,))
    operator = LinearOperator((n**3, n**3), matvec=lambda x: stencil(x, n), dtype=np.float64)
    calls = 0

    def iteration(_):
        nonlocal calls
        calls += 1

    start = perf_counter()
    x, info = cg(operator, rhs, rtol=tol, atol=0, maxiter=1000, callback=iteration)
    elapsed = perf_counter() - start
    reported = np.linalg.norm(stencil(x, n) - rhs) / np.linalg.norm(rhs)
    save_tensor(Path("solution.atn"), x)
    print(elapsed, calls, reported, "converged" if info == 0 else "nonconverged")


def arda_script(method, n, tol):
    common = "import tensor as t\n"
    if method == "legacy":
        common += "import high\n"
        setup = 'let b = t.to_numc(t.mmap("rhs_grid.atn"))'
        expression = f"high.poisson(b,{tol},1000)"
    else:
        common += "import solver as s\nimport linop as l\n"
        setup = f'let op = l.poisson([{n},{n},{n}])\nlet b = t.mmap("rhs.atn")'
        if method == "cg":
            expression = f's.solve("cg",op,b,{{"tol":{tol},"maxiter":1000}})'
        else:
            expression = f's.multigrid(op,b,{{"tol":{tol},"maxiter":100}})'
    return common + f"""
{setup}
collect()
let start = monotonic()
let r = {expression}
let elapsed = monotonic()-start
print(elapsed,r["iterations"],r["residual"],r["reason"])
t.save("solution.atn",r["solution"])
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arda", type=Path, default=Path("arda"))
    parser.add_argument("--sizes", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--methods", nargs="+", choices=("multigrid", "cg", "legacy", "scipy"),
                        default=["multigrid", "cg", "scipy"])
    parser.add_argument("--legacy-arda", type=Path, help="Pre-migration Arda binary for legacy CG")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--scipy-worker", nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.scipy_worker:
        scipy_worker(int(args.scipy_worker[0]), float(args.scipy_worker[1]))
        return
    if any(n < 2 or n**3 > 16_000_000 for n in args.sizes) or not 0 < args.tol < 1 or args.rounds < 1:
        parser.error("use a grid that fits Arda's 16 million element limit, tolerance in (0,1), and positive rounds")
    if "legacy" in args.methods and args.legacy_arda is None:
        parser.error("--methods legacy needs --legacy-arda pointing to a pre-migration build")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    import numpy as np

    timer = shutil.which("time")
    if timer is None:
        parser.error("GNU time is needed to measure peak RSS")
    binary = args.arda.resolve()
    legacy = args.legacy_arda.resolve() if args.legacy_arda else None
    script = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="arda-aule-") as tmp:
        root = Path(tmp)
        for n in args.sizes:
            expected, rhs = manufactured(n, np)
            np.testing.assert_allclose(stencil(expected.ravel(), n), rhs.ravel(),
                                       atol=1e-13, rtol=1e-10)
            save_tensor(root / "rhs.atn", rhs.ravel())
            if "legacy" in args.methods:
                save_tensor(root / "rhs_grid.atn", rhs)
            runs = {method: [] for method in args.methods}
            for iteration in range(args.rounds):
                shift = iteration % len(args.methods)
                for method in args.methods[shift:] + args.methods[:shift]:
                    if method == "scipy":
                        command = [sys.executable, str(script), "--scipy-worker", str(n), str(args.tol)]
                    else:
                        (root / "run.arda").write_text(arda_script(method, n, args.tol), encoding="utf-8")
                        command = [str(legacy if method == "legacy" else binary), "--mathdoc=none", "run.arda"]
                    run = subprocess.run([timer, "-f", "%M", "-o", "rss.txt", *command],
                                         cwd=root, check=True, capture_output=True, text=True, timeout=600)
                    elapsed, iterations, reported, reason = run.stdout.split()
                    x = np.memmap(root / "solution.atn", dtype="<f8", mode="r", offset=128, shape=(n**3,))
                    independent = np.linalg.norm(stencil(x, n)-rhs.ravel()) / np.linalg.norm(rhs)
                    error = np.linalg.norm(x-expected.ravel()) / np.linalg.norm(expected)
                    rss = int((root / "rss.txt").read_text().strip())
                    record = dict(n=n, method=method, seconds=float(elapsed), iterations=int(iterations),
                                  reason=reason, reported_residual=float(reported),
                                  reference_residual=float(independent), relative_error=float(error),
                                  peak_rss_kib=rss)
                    runs[method].append(record)
                    print(json.dumps(record, sort_keys=True), flush=True)
                    del x
                    if reason != "converged" or independent > args.tol or error > 1e-6:
                        raise RuntimeError(f"Poisson numerical gate failed for {method} on {n}^3")
                    if n == 128 and method == "multigrid" and rss >= 512 * 1024:
                        raise RuntimeError(f"multigrid exceeded the Poisson RSS target on {n}^3")
            for method, data in runs.items():
                print(json.dumps(dict(n=n, method=method, rounds=args.rounds,
                                      median_seconds=median(r["seconds"] for r in data),
                                      median_rss_kib=median(r["peak_rss_kib"] for r in data)),
                                 sort_keys=True), flush=True)



if __name__ == "__main__":
    main()
