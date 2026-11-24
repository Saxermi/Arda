"""Compare Arda dense utilities and structured solves with single-threaded SciPy."""
import argparse
import math
import os
from pathlib import Path
import statistics
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter


def arda_program(dense_n, small_n, large_n):
    return f"""
tol linalg as la
tol numc as nc
tol sparse as sp
tol time

caro time_call(label,n,repeats,operation,check):
    esta value = operation()
    esta t = time.monotonic()
    an i in range(repeats):
        value = operation()
    esta elapsed = time.monotonic()-t
    pedo(label,n,elapsed/repeats,check(value))

esta n = {dense_n}
esta a = nc.reshape(nc.linspace(-0.001,0.001,n*n),[n,n])
esta identity = nc.zeros(n,n)
an i in range(n):
    nc.set(a,i,i,nc.get(a,i,i)+2)
    nc.set(identity,i,i,1)
time_call("det",n,12,lambda: la.det(a),lambda value: value)
time_call("inverse",n,6,lambda: la.inv(a),lambda value: nc.norm(nc.matmul(a,value)-identity))

esta n = {small_n}
esta x = nc.linspace(0.5,1.5,n)
esta sparse = sp.tridiag(n,-1,4,-1)
esta b = sp.matvec(sparse,x)
esta dense = sp.todense(sparse)
esta off = 0-nc.ones(n-1)
esta diagonal = nc.ones(n)*4
time_call("dense_tridiag",n,4,lambda: nc.solve(dense,b),lambda value: nc.norm(value-x)/nc.norm(x))
time_call("thomas",n,60,lambda: la.solve_tridiag(off,diagonal,off,b),lambda value: nc.norm(value-x)/nc.norm(x))

n = {large_n}
x = nc.linspace(0.5,1.5,n)
b = sp.matvec(sp.tridiag(n,-1,4,-1),x)
off = 0-nc.ones(n-1)
diagonal = nc.ones(n)*4
time_call("thomas",n,12,lambda: la.solve_tridiag(off,diagonal,off,b),lambda value: nc.norm(value-x)/nc.norm(x))

esta bands = nc.zeros(5,n)
b = nc.zeros(n)
an i in range(n):
    nc.set(bands,0,i,0.125)
    nc.set(bands,1,i,-0.75)
    nc.set(bands,2,i,4)
    nc.set(bands,3,i,-0.5)
    nc.set(bands,4,i,0.25)
    esta rhs = 4*x[i]
    qui i+1<n:
        rhs = rhs-0.75*x[i+1]
    qui i+2<n:
        rhs = rhs+0.125*x[i+2]
    qui i>0:
        rhs = rhs-0.5*x[i-1]
    qui i>1:
        rhs = rhs+0.25*x[i-2]
    b[i] = rhs
nc.set(bands,2,0,0)
b[0] = b[0]-4*x[0]
time_call("banded",n,12,lambda: la.solve_banded(2,2,bands,b),lambda value: nc.norm(value-x)/nc.norm(x))
"""


def scipy_workloads(dense_n, small_n, large_n):
    import numpy as np
    from scipy import linalg

    n = dense_n
    a = np.linspace(-0.001, 0.001, n * n).reshape(n, n)
    a[np.diag_indices(n)] += 2
    identity = np.eye(n)
    cases = {
        ("det", n): (12, lambda: linalg.det(a), float),
        ("inverse", n): (6, lambda: linalg.inv(a),
                         lambda result: np.linalg.norm(a @ result - identity)),
    }
    for n in (small_n, large_n):
        x = np.linspace(0.5, 1.5, n)
        rhs = 4 * x
        rhs[:-1] -= x[1:]
        rhs[1:] -= x[:-1]
        bands = np.broadcast_to(np.array([[-1.0], [4.0], [-1.0]]), (3, n)).copy()
        check = lambda result, x=x: np.linalg.norm(result - x) / np.linalg.norm(x)
        cases["thomas", n] = (
            60 if n == small_n else 12,
            lambda bands=bands, rhs=rhs: linalg.solve_banded((1, 1), bands, rhs), check,
        )
        if n == small_n:
            matrix = np.diag(bands[1]) + np.diag(bands[0, 1:], 1) + np.diag(bands[2, :-1], -1)
            cases["dense_tridiag", n] = (
                4, lambda matrix=matrix, rhs=rhs: linalg.solve(matrix, rhs, assume_a="gen"), check,
            )
    n = large_n
    x = np.linspace(0.5, 1.5, n)
    bands = np.broadcast_to(np.array([[0.125], [-0.75], [4.0], [-0.5], [0.25]]), (5, n)).copy()
    bands[2, 0] = 0
    rhs = bands[2] * x
    rhs[:-1] -= 0.75 * x[1:]
    rhs[:-2] += 0.125 * x[2:]
    rhs[1:] -= 0.5 * x[:-1]
    rhs[2:] += 0.25 * x[:-2]
    cases["banded", n] = (
        12, lambda: linalg.solve_banded((2, 2), bands, rhs),
        lambda result: np.linalg.norm(result - x) / np.linalg.norm(x),
    )
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arda", type=Path, default=Path("./arda"))
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    dims = (192, 512, 100_000)
    cases = scipy_workloads(*dims)
    arda_times = {key: [] for key in cases}
    scipy_times = {key: [] for key in cases}
    max_error = {key: 0 for key in cases}
    with TemporaryDirectory() as directory:
        path = Path(directory) / "direct_linalg.arda"
        path.write_text(arda_program(*dims))
        for _ in range(args.rounds):
            lines = subprocess.check_output([str(args.arda.resolve()), str(path)], text=True).splitlines()
            if len(lines) != len(cases):
                raise RuntimeError("Unexpected benchmark output")
            for line in lines:
                label, n, seconds, actual = line.split()
                key = label, int(n)
                reps, operation, check = cases[key]
                result = operation()
                start = perf_counter()
                for _ in range(reps):
                    result = operation()
                scipy_times[key].append((perf_counter() - start) / reps)
                arda_times[key].append(float(seconds))
                expected = float(check(result))
                measured = float(actual)
                if not math.isfinite(measured) or not math.isfinite(expected):
                    raise RuntimeError(f"{key} produced a nonfinite check")
                if label == "det":
                    error = abs(measured / expected - 1)
                else:
                    error = max(measured, expected)
                if not math.isfinite(error) or error > 1e-12:
                    raise RuntimeError(f"{key} failed numerical check: {error}")
                max_error[key] = max(error, max_error[key])

    print("workload size Arda_ms SciPy_ms max_check_error")
    for label, n in arda_times:
        key = label, n
        arda_ms = statistics.median(arda_times[key]) * 1000
        scipy_ms = statistics.median(scipy_times[key]) * 1000
        print(f"{label} {n} {arda_ms:.6g} {scipy_ms:.6g} {max_error[key]:.3g}")


if __name__ == "__main__":
    main()
