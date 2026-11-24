"""Independent SciPy checks of factor reconstruction and reusable solves."""
import ast
import unittest

import numpy as np
from scipy import linalg

import test_arda


class LinalgReferenceTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_lu_reconstruction_and_backward_error(self):
        rng = np.random.default_rng(1500)
        q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
        p, _ = np.linalg.qr(rng.normal(size=(12, 12)))
        matrices = [rng.normal(size=(12, 12)), (q * np.geomspace(1, 1e-10, 12)) @ p.T]
        for matrix in matrices:
            b = rng.normal(size=(12, 4))
            output = self.run_code(f"""
            tol numc as nc
            tol linalg as la
            esta a = nc.array({matrix.T.tolist()})
            esta f = la.lu(nc.permute(a,[1,0]))
            esta parts = la.parts(f)
            pedo(parts[0])
            pedo(nc.tolist(parts[1]))
            pedo(nc.tolist(parts[2]))
            pedo(nc.tolist(la.solve(f,nc.array({b.tolist()}))))
            pedo(nc.tolist(nc.solve(nc.permute(a,[1,0]),nc.array({b[:, 0].tolist()}))))
            """, stress=True).splitlines()
            perm, l, u, actual, legacy = map(ast.literal_eval, output)
            self.assertEqual(sorted(perm), list(range(12)))
            np.testing.assert_allclose(np.asarray(l) @ u, matrix[perm], rtol=1e-13, atol=1e-14)
            np.testing.assert_allclose(legacy, np.asarray(actual)[:, 0], rtol=1e-13, atol=1e-14)
            expected = linalg.solve(matrix, b)
            self.assertLess(np.linalg.norm(actual - expected) / np.linalg.norm(expected), 5e-6)
            error = np.linalg.norm(matrix @ actual - b) / (
                np.linalg.norm(matrix) * np.linalg.norm(actual) + np.linalg.norm(b)
            )
            self.assertLess(error, 2e-14)

    def test_cholesky_reconstruction_and_reused_solve(self):
        rng = np.random.default_rng(1501)
        a = rng.normal(size=(10, 10))
        a = a @ a.T + 2 * np.eye(10)
        b = rng.normal(size=(10, 3))
        output = self.run_code(f"""
        tol numc as nc
        tol linalg as la
        esta f = la.cholesky(nc.array({a.tolist()}))
        pedo(nc.tolist(la.parts(f)[0]))
        pedo(nc.tolist(la.solve(f,nc.array({b.tolist()}))))
        """, stress=True).splitlines()
        lower, actual = [np.asarray(ast.literal_eval(line)) for line in output]
        np.testing.assert_allclose(lower, linalg.cholesky(a, lower=True), rtol=2e-13, atol=2e-14)
        np.testing.assert_allclose(lower @ lower.T, a, rtol=2e-13, atol=2e-14)
        np.testing.assert_allclose(actual, linalg.solve(a, b, assume_a="pos"), rtol=2e-13, atol=2e-14)

    def test_householder_qr_tall_wide_and_scaled(self):
        rng = np.random.default_rng(1502)
        for m, n, scale in [(11, 4, 1.0), (4, 11, 1.0), (7, 3, 1e200), (7, 3, 1e-200)]:
            with self.subTest(shape=(m, n), scale=scale):
                mat = rng.normal(size=(m, n))
                b = rng.normal(size=(m, 3))
                source = f"""
                tol numc as nc
                tol linalg as la
                esta factor = la.qr(nc.array({(mat * scale).tolist()}))
                esta parts = la.parts(factor)
                pedo(nc.tolist(parts[0]))
                pedo(nc.tolist(parts[1]))
                """
                if m >= n:
                    source += f"""
                pedo(nc.tolist(la.solve(factor,nc.array({(b * scale).tolist()}))))
                """
                output = self.run_code(source, stress=True).splitlines()
                q, r, *solution = [np.asarray(ast.literal_eval(line)) for line in output]
                k = min(m, n)
                self.assertEqual(q.shape, (m, k))
                self.assertEqual(r.shape, (k, n))
                np.testing.assert_allclose(q.T @ q, np.eye(k), rtol=2e-13, atol=2e-14)
                np.testing.assert_allclose(q @ (r / scale), mat, rtol=2e-13, atol=2e-14)
                if solution:
                    expected = linalg.lstsq(mat, b)[0]
                    np.testing.assert_allclose(solution[0], expected, rtol=2e-13, atol=2e-14)
