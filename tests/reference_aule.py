"""Seeded independent SciPy/NumPy checks for Aulë numerical algorithms."""
from pathlib import Path
import subprocess
import tempfile
import unittest

import test_arda

try:
    import numpy as np
    import scipy.linalg as la
    from scipy.sparse import diags
    from scipy.sparse.linalg import gmres, minres
except ImportError:
    np = None

from benchmarks.cpu_baseline import save_tensor


@unittest.skipIf(np is None, "NumPy and SciPy are required")
class SolverReferenceTests(unittest.TestCase):
    def run_problem(self, inputs, code, outputs, *, requires_lapack=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, matrix in inputs.items():
                save_tensor(root / (name + ".atn"), np.asarray(matrix, dtype=np.float64))
            script = root / "check.arda"
            script.write_text("""
import solver as s
import tensor as t
import numc as n
import linop as l
""" + code, encoding="utf-8")
            result = subprocess.run([test_arda.BIN, "--gc-stress", script], cwd=root,
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            if requires_lapack and result.stdout.strip() == "backend":
                self.skipTest("Arnoldi projected eigenproblem requires LAPACK=1")
            self.assertTrue(all(line.startswith("converged") or line.startswith("least_squares")
                                for line in result.stdout.splitlines()), result.stdout)
            return [np.fromfile(root / (name + ".atn"), dtype="<f8", offset=128) for name in outputs]

    def test_cg_and_minres_against_independent_dense_and_scipy_solutions(self):
        rng = np.random.default_rng(1901)
        q, _ = la.qr(rng.standard_normal((19, 19)))
        spd = (q * np.linspace(1, 8, 19)) @ q.T
        indefinite = (q * np.r_[np.linspace(-2, -0.5, 9), np.linspace(1, 5, 10)]) @ q.T
        spd = (spd + spd.T) / 2
        indefinite = (indefinite + indefinite.T) / 2
        rhs = rng.standard_normal(19)
        cg, mr = self.run_problem({"spd": spd, "indef": indefinite, "rhs": rhs}, """
let rhs = t.mmap("rhs.atn")
let cg = s.solve("cg",t.mmap("spd.atn"),rhs,{"tol":1e-12,"preconditioner":"block_jacobi"})
let mr = s.solve("minres",t.mmap("indef.atn"),rhs,{"tol":1e-12})
print(cg["reason"])
print(mr["reason"])
t.save("cg.atn",cg["solution"])
t.save("mr.atn",mr["solution"])
""", ("cg", "mr"))
        np.testing.assert_allclose(cg, la.solve(spd, rhs, assume_a="pos"), rtol=1e-10, atol=1e-11)
        expected, info = minres(indefinite, rhs, rtol=1e-13)
        self.assertEqual(info, 0)
        np.testing.assert_allclose(mr, expected, rtol=1e-10, atol=1e-11)

    def test_restarted_gmres_and_bicgstab_on_a_nonnormal_system(self):
        rng = np.random.default_rng(1902)
        matrix = 4 * np.eye(31) + np.triu(rng.standard_normal((31, 31)) * 0.4, 1)
        matrix += np.diag(np.full(30, -0.6), -1)
        rhs = rng.standard_normal(31)
        actual = self.run_problem({"a": matrix, "b": rhs}, """
let a = t.mmap("a.atn")
let b = t.mmap("b.atn")
let r = s.solve("gmres",a,b,{"restart":3,"tol":1e-12,"maxiter":100})
print(r["reason"])
t.save("gmres.atn",r["solution"])
r = s.solve("bicgstab",a,b,{"tol":1e-12})
print(r["reason"])
t.save("bicgstab.atn",r["solution"])
""", ("gmres", "bicgstab"))
        for x in actual:
            np.testing.assert_allclose(x, la.solve(matrix, rhs), rtol=1e-10, atol=1e-11)

    def test_gmres_preconditioned_scaled_nonnormal_system_at_two_restarts(self):
        rng = np.random.default_rng(1920)
        order = 17
        normalized = 4*np.eye(order) + np.triu(rng.normal(0, 0.25, (order, order)), 1)
        normalized += np.diag(np.full(order-1, -0.4), -1)
        scale = np.logspace(-5, 5, order)
        matrix = scale[:, None]*normalized
        rhs = matrix @ rng.normal(size=order)
        actual = self.run_problem({"a": matrix, "b": rhs}, """
let a = t.mmap("a.atn")
let b = t.mmap("b.atn")
for k in [2,5]:
    let r = s.solve("gmres",a,b,{"restart":k,"tol":1e-13,"preconditioner":"jacobi"})
    print(r["reason"])
    t.save("x"+str(k)+".atn",r["solution"])
""", ("x2", "x5"))
        m = diags(1/np.diag(matrix))
        for restart, x in zip((2, 5), actual):
            reference, info = gmres(matrix, rhs, M=m, restart=restart, rtol=1e-13, maxiter=400)
            self.assertEqual(info, 0)
            self.assertLess(la.norm(matrix@x-rhs)/la.norm(rhs), 1e-13)
            np.testing.assert_allclose(x, reference, rtol=1e-9, atol=1e-10)

    def test_gmres_near_invariant_directions_at_multiple_scales(self):
        for exponent in (-300, 0, 300):
            with self.subTest(exponent=exponent):
                scale = np.ldexp(1., exponent)
                matrix = scale*np.array([[1, 0], [0, 2**-18]])
                rhs = scale*np.array([1, 2**-60])
                recovered, = self.run_problem({"a": matrix, "b": rhs}, """
let r = s.solve("gmres",t.mmap("a.atn"),t.mmap("b.atn"),{"restart":2,"tol":1e-22})
print(r["reason"])
t.save("x.atn",r["solution"])
""", ("x",))
                np.testing.assert_array_equal(recovered, la.solve(matrix, rhs))

    def test_rectangular_rank_deficient_least_squares(self):
        rng = np.random.default_rng(1903)
        matrix = rng.standard_normal((13, 5)) @ rng.standard_normal((5, 9))
        rhs = rng.standard_normal(13)
        actual = self.run_problem({"a": matrix, "b": rhs}, """
let a = t.mmap("a.atn")
let b = t.mmap("b.atn")
for method in ["lsqr","lsmr"]:
    let r = s.solve(method,a,b,{"tol":1e-12})
    print(r["reason"])
    t.save(method+".atn",r["solution"])
""", ("lsqr", "lsmr"))
        expected, *_ = la.lstsq(matrix, rhs, cond=1e-12)
        for x in actual:
            np.testing.assert_allclose(x, expected, rtol=1e-10, atol=1e-11)

    def test_anisotropic_multigrid_against_explicit_kronecker_stencil(self):
        shape = (7, 9, 11)
        rng = np.random.default_rng(1904)
        exact = rng.standard_normal(np.prod(shape))
        operators = [np.diag(np.full(n, 2.0)) - np.diag(np.ones(n-1), 1) -
                     np.diag(np.ones(n-1), -1) for n in shape]
        a, b, c = map(np.eye, shape)
        matrix = (np.kron(np.kron(operators[0], b), c) + np.kron(np.kron(a, operators[1]), c)
                  + 100*np.kron(np.kron(a, b), operators[2]))
        rhs = matrix @ exact
        actual, = self.run_problem({"rhs": rhs}, """
let op = l.poisson([7,9,11],[1,1,100])
let r = s.multigrid(op,t.mmap("rhs.atn"),{"tol":1e-10,"maxiter":100})
print(r["reason"])
t.save("x.atn",r["solution"])
""", ("x",))
        self.assertLess(la.norm(matrix @ actual - rhs) / la.norm(rhs), 1e-10)
        self.assertLess(la.norm(actual - exact) / la.norm(exact), 1e-8)

    def test_lanczos_eigenpairs_and_mixed_refinement(self):
        rng = np.random.default_rng(1905)
        q, _ = la.qr(rng.standard_normal((21, 21)))
        a = (q * np.linspace(1, 20, 21)) @ q.T
        a = (a + a.T) / 2
        seed, rhs = rng.standard_normal((2, 21))
        values, vectors, x = self.run_problem({"a": a, "s": seed, "b": rhs}, """
let a = t.mmap("a.atn")
let report = s.lanczos(a,t.mmap("s.atn"),4,{"basis":21,"tol":1e-12})
print(report["reason"])
t.save("val.atn",report["values"])
t.save("vec.atn",report["vectors"])
let report = s.refine(t.to_numc(a),t.mmap("b.atn"),{"tol":1e-13})
print(report["reason"])
t.save("x.atn",report["solution"])
""", ("val", "vec", "x"))
        vectors = vectors.reshape(21, 4)
        np.testing.assert_allclose(values, la.eigvalsh(a)[-4:][::-1], rtol=1e-12)
        np.testing.assert_allclose(a @ vectors, vectors * values, atol=1e-11)
        np.testing.assert_allclose(vectors.T @ vectors, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(x, la.solve(a, rhs, assume_a="pos"), rtol=1e-11, atol=1e-12)

    def test_arnoldi_matches_a_general_complex_spectrum(self):
        rng = np.random.default_rng(1906)
        matrix = rng.standard_normal((8, 8))
        re, im, vr, vi = self.run_problem({"a": matrix, "seed": rng.standard_normal(8)}, """
let a = t.mmap("a.atn")
try:
    let r = s.arnoldi(a,t.mmap("seed.atn"),8,{"basis":8,"tol":1e-12})
    print(r["reason"])
    t.save("re.atn",r["values_real"])
    t.save("im.atn",r["values_imag"])
    t.save("vr.atn",r["vectors_real"])
    t.save("vi.atn",r["vectors_imag"])
catch err:
    if err["code"] == "backend":
        print("backend")
    else:
        throw err
""", ("re", "im", "vr", "vi"), requires_lapack=True)
        values = re + 1j * im
        vectors = vr.reshape(8, 8) + 1j * vi.reshape(8, 8)
        expected = la.eigvals(matrix)
        for value in values:
            self.assertLess(np.min(np.abs(expected-value)), 1e-11)
        np.testing.assert_allclose(matrix @ vectors, vectors * values, atol=1e-11)
