"""Measure local Moria assembly, CSR SpMV and PCG with equivalent workloads."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile


def triplet_source(n, workload, builder):
    if workload == "tridiagonal":
        row_setup = "    esta row = i\n"
        records = [(f"row{offset}", value, guard) for offset, value, guard in [
            ("+1", -1, "row+1 < n"), ("", 4, None), ("-1", -1, "row > 0"),
        ]]
    else:
        row_setup = "    esta row = n-1-i\n"
        records = [
            ("(row+2)%n", 5, None), ("row", 3, None), ("(row+1)%n", 2, None),
            ("row", -1, None), ("(row+2)%n", -5, None),
        ]
    code = ["esta t = time.monotonic()"]
    if builder:
        code.append("esta builder = sp.builder(n,n)")
    else:
        code.extend(["esta rows = []", "esta cols = []", "esta values = []"])
    code.extend(["an i in range(n):", row_setup.rstrip()])
    for col, value, guard in records:
        indent = "    "
        if guard:
            code.append(indent + f"qui {guard}:")
            indent *= 2
        if builder:
            code.append(indent + f"sp.append(builder,row,{col},{value})")
        else:
            code.extend(indent + expr for expr in ("push(rows,row)", f"push(cols,{col})", f"push(values,{value})"))
    code.extend(["esta assembly = time.monotonic()-t", "esta t = time.monotonic()"])
    if builder:
        code.extend(["esta coo_time = 0", "esta a = sp.finish(builder)"])
    else:
        code.extend([
            "esta coo = sp.coo(n,n,rows,cols,values)", "esta coo_time = time.monotonic()-t",
            "rows = mun", "cols = mun", "values = mun", "collect()",
            "t = time.monotonic()", "esta a = sp.to_csr(coo)",
        ])
    code.extend(["esta finish = time.monotonic()-t", "pedo(assembly,coo_time,finish)"])
    return "\n".join(code)


def arda_source(n, workload, method):
    code = ["tol sparse as sp", "tol numc as nc", "tol time", f"esta n = {n}"]
    if method in ("direct", "direct-spmv"):
        code.extend(["esta t = time.monotonic()", "esta a = sp.tridiag(n,-1,4,-1)",
                     "pedo(time.monotonic()-t,0,0)"])
    else:
        code.append(triplet_source(n, workload, method == "builder"))
    code.extend([
        "esta x = nc.linspace(0.5,1.5,n)", "esta first = sp.matvec(a,x)",
        "esta t = time.monotonic()", "an i in range(8):", "    first = sp.matvec(a,x)",
        "pedo((time.monotonic()-t)/8,sp.nnz(a),nc.sum(first),first[0],first[n-1])",
    ])
    if method == "direct":
        code.extend([
            "tol solver as s", "esta t = time.monotonic()",
            'esta result = s.solve("cg",a,first,{"tol":1e-9,"maxiter":100,"preconditioner":"jacobi"})',
            "esta elapsed = time.monotonic()-t",
            'esta err = nc.norm(result["solution"]-x)/nc.norm(x)',
            'pedo(elapsed,result["iterations"],result["residual"],err,result["reason"])',
        ])
    return "\n".join(code) + "\n"


SCIPY_SOURCE = """
import sys
from time import perf_counter
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator, cg

n = int(sys.argv[1])
start = perf_counter()
a = diags([-1.0, 4.0, -1.0], [-1, 0, 1], shape=(n, n), format="csr")
print(perf_counter() - start, 0, 0)
x = np.linspace(0.5, 1.5, n)
b = a @ x
start = perf_counter()
for _ in range(8):
    b = a @ x
print((perf_counter() - start) / 8, a.nnz, b.sum(), b[0], b[-1])
iterations = 0

def count(_):
    global iterations
    iterations += 1

start = perf_counter()
m = LinearOperator((n,n), matvec=lambda v: v / 4, dtype=np.float64)
solution, info = cg(a, b, rtol=1e-9, atol=0, maxiter=100, M=m, callback=count)
residual = np.linalg.norm(a @ solution - b) / np.linalg.norm(b)
elapsed = perf_counter() - start
print(elapsed, iterations, residual,
      np.linalg.norm(solution - x) / np.linalg.norm(x), "converged" if info == 0 else "failed")
"""


def measure(command, cpu):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    if cpu is not None:
        command = ["taskset", "-c", str(cpu), *command]
    with tempfile.TemporaryFile(mode="w+") as stdout, tempfile.TemporaryFile(mode="w+") as stderr:
        with subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr) as process:
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
            if process.returncode:
                stderr.seek(0)
                raise RuntimeError(f"{command[0]} exited {process.returncode}: {stderr.read()}")
        stdout.seek(0)
        assembly, product, *solver = [line.split() for line in stdout]
    rss_mib = usage.ru_maxrss / 1024  # Linux reports KiB.
    return tuple(map(float, assembly)), tuple(map(float, product)), solver, rss_mib


def check(result, size, workload):
    _, product, solver, _ = result
    _, nnz, total, first, last = product
    step = 1.0 / (size - 1)
    if workload == "tridiagonal":
        expected = (3 * size - 2, 2 * size + 2, 1.5 - step, 4.5 + step)
    else:
        expected = (2 * size, 4 * size, 2.0 + 2.0 * step, 4.0)
    for actual, ref in zip((nnz, total, first, last), expected, strict=True):
        if not math.isclose(actual, ref, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(f"Invalid {workload} check: {actual} versus {ref}")
    if solver:
        elapsed, it, residual, error, reason = solver[0]
        if reason != "converged" or float(residual) > 1e-9 or float(error) > 2e-9:
            raise RuntimeError(f"Invalid PCG result: {solver}")
        return float(elapsed), int(it), float(residual), float(error)
    return None


def verify_scipy(arda, root, cpu, rows, solver_size):
    """Compare complete results outside the timing and RSS measurements."""
    import numpy as np
    from scipy.io import mmread
    from scipy.sparse import coo_matrix, diags

    prefix = ["taskset", "-c", str(cpu)] if cpu is not None else []
    program = root / "verify.arda"

    def run(code):
        program.write_text(code)
        return subprocess.check_output([*prefix, str(arda), str(program)], text=True)

    indices = np.arange(rows - 1, -1, -1, dtype=np.int32)
    cols = np.column_stack(((indices + 2) % rows, indices,
                            (indices + 1) % rows, indices, (indices + 2) % rows)).ravel()
    values = np.tile(np.array([5, 3, 2, -1, -5], dtype=np.float64), rows)
    expected = coo_matrix((values, (np.repeat(indices, 5), cols)), shape=(rows, rows)).tocsr()
    expected.eliminate_zeros()
    expected.sort_indices()
    expected_coo = expected.tocoo()
    for method in ("lists", "builder"):
        output = root / f"verify-{method}.mtx"
        run("\n".join([
            "tol sparse as sp", "tol time", f"esta n = {rows}",
            triplet_source(rows, "duplicates", method == "builder"),
            f"sp.write_mm({json.dumps(str(output))},a)", "",
        ]))
        # Compare emitted entries in order before any SciPy canonicalization.
        actual = mmread(output)
        if actual.shape != expected.shape:
            raise RuntimeError("Incorrect CSR dimensions")
        np.testing.assert_array_equal(actual.row, expected_coo.row)
        np.testing.assert_array_equal(actual.col, expected_coo.col)
        np.testing.assert_array_equal(actual.data, expected_coo.data)
        print(f"SciPy CSR comparison passed for {method}: {5 * rows:,} triplets, {actual.nnz:,} stored")

    n = solver_size
    output = run("\n".join([
        "tol sparse as sp", "tol numc as nc", f"esta n = {n}",
        "esta a = sp.tridiag(n,-1,4,-1)", "esta x = nc.linspace(0.5,1.5,n)",
        "pedo(nc.tolist(sp.matvec(a,x)))", "",
    ])).strip()
    if not (output.startswith("[") and output.endswith("]")):
        raise RuntimeError("Invalid matvec output")
    actual = np.fromstring(output[1:-1], sep=",")
    a = diags([-1.0, 4.0, -1.0], [-1, 0, 1], shape=(n, n), format="csr")
    expected = a @ np.linspace(0.5, 1.5, n)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
    print(f"SciPy SpMV comparison passed for all {n:,} entries")


def main():
    if sys.platform != "linux":
        raise SystemExit("This benchmark requires Linux peak RSS accounting.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arda", type=Path, default=Path("./arda"))
    parser.add_argument("--size", type=int, default=100_000, help="number of rows in assembly workloads")
    parser.add_argument("--solver-size", type=int, default=1_000_000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--cpu", type=int, help="optionally pin subprocesses to a Linux logical CPU")
    parser.add_argument("--scipy", action="store_true",
                        help="time SciPy PCG and verify full CSR and SpMV results outside timing")
    args = parser.parse_args()
    if min(args.size, args.solver_size) < 3 or args.rounds < 1:
        parser.error("sizes must be at least three and rounds must be positive")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        cases = {}
        for shape in ("tridiagonal", "duplicates"):
            for method in ("lists", "builder"):
                cases[shape, method] = args.size
        cases["tridiagonal", "direct-spmv"] = args.solver_size
        cases["tridiagonal", "direct"] = args.solver_size
        if args.scipy:
            cases["tridiagonal", "scipy"] = args.solver_size
        commands = {}
        for (shape, method), n in cases.items():
            if method == "scipy":
                commands[shape, method] = [sys.executable, "-c", SCIPY_SOURCE, str(n)]
            else:
                file = root / f"{shape}-{method}.arda"
                file.write_text(arda_source(n, shape, method))
                commands[shape, method] = [str(args.arda.resolve()), str(file)]
        samples = {key: [] for key in cases}
        for _ in range(args.rounds):
            for (shape, method), n in cases.items():
                result = measure(commands[shape, method], args.cpu)
                assembly, product, _, rss = result
                solver = check(result, n, shape)
                samples[shape, method].append((*assembly, sum(assembly), product[0], rss, *(solver or ())))
        print("workload method rows assembly_ms coo_ms finish_ms total_ms spmv_ms median_RSS_MiB max_RSS_MiB "
              "[pcg_ms iterations residual relative_error]")
        for (shape, method), results in samples.items():
            medians = [statistics.median(column) for column in zip(*results, strict=True)]
            for i in range(5):
                medians[i] *= 1000
            if method in ("direct", "scipy"):
                medians[6] *= 1000
            medians.insert(6, max(result[5] for result in results))
            print(shape, method, cases[shape, method], *(f"{value:.6g}" for value in medians))
        if args.scipy:
            verify_scipy(args.arda.resolve(), root, args.cpu, args.size, args.solver_size)


if __name__ == "__main__":
    main()
