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
        mixed_eigenvalues = np.concatenate((-np.linspace(1, 3, 6), np.linspace(1, 3, 6)))
        indefinite = (q * mixed_eigenvalues) @ q.T
        matrices = [
            ("general", rng.normal(size=(12, 12))),
            ("ill-conditioned", (q * np.geomspace(1, 1e-10, 12)) @ p.T),
            ("symmetric indefinite", (indefinite + indefinite.T) / 2),
        ]
        for label, matrix in matrices:
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
            reconstructed = np.asarray(l) @ u
            np.testing.assert_allclose(reconstructed, matrix[perm], rtol=1e-13, atol=1e-14)
            np.testing.assert_allclose(legacy, np.asarray(actual)[:, 0], rtol=1e-13, atol=1e-14)
            expected = linalg.solve(matrix, b)
            solution_difference = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
            self.assertLess(solution_difference, 5e-6 if label == "ill-conditioned" else 1e-13)
            matrix_norm = np.linalg.norm(matrix)
            error = np.linalg.norm(matrix @ actual - b) / (
                matrix_norm * np.linalg.norm(actual) + np.linalg.norm(b)
            )
            self.assertLess(error, 2e-14)
            reconstruction_error = np.linalg.norm(reconstructed - matrix[perm]) / matrix_norm
            print(f"\nLU {label} cond2 {np.linalg.cond(matrix):.3g}, "
                  f"reconstruction {reconstruction_error:.3g}, residual {error:.3g}, "
                  f"relative SciPy solution difference {solution_difference:.3g}")

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

    def test_matrix_utilities_on_strided_ill_conditioned_input(self):
        rng = np.random.default_rng(1503)
        q, _ = np.linalg.qr(rng.normal(size=(7, 7)))
        r, _ = np.linalg.qr(rng.normal(size=(7, 7)))
        a = (q * np.geomspace(1, 1e-8, 7)) @ r
        result = self.run_code(f"""
        tol numc as nc
        tol linalg as la
        esta a = nc.permute(nc.array({a.T.tolist()}),[1,0])
        pedo(la.det(a),la.trace(a))
        pedo(nc.tolist(la.diag(a)))
        pedo(nc.tolist(la.inv(a)))
        """, stress=True).splitlines()
        determinant, trace = map(float, result[0].split())
        np.testing.assert_allclose(determinant, linalg.det(a), rtol=5e-8, atol=0)
        np.testing.assert_allclose(trace, np.trace(a), rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(ast.literal_eval(result[1]), np.diag(a), rtol=1e-13)
        inverse = np.asarray(ast.literal_eval(result[2]))
        identity = np.eye(len(a))
        np.testing.assert_allclose(inverse, linalg.inv(a), rtol=5e-7, atol=0)
        self.assertLess(np.linalg.norm(a @ inverse - identity), 1e-7)
        self.assertLess(np.linalg.norm(inverse @ a - identity), 1e-7)

    def test_scaled_strictly_dominant_thomas(self):
        rng = np.random.default_rng(1504)
        n = 31
        scale = np.geomspace(1e-120, 1e120, n)
        d = 4 * scale * (-1.0) ** np.arange(n)
        lower = scale[1:] * rng.uniform(-1, 1, n - 1)
        upper = scale[:-1] * rng.uniform(-1, 1, n - 1)
        expected = rng.normal(size=(n, 3))
        bands = np.zeros((3, n))
        bands[0, 1:], bands[1, :], bands[2, :-1] = upper, d, lower
        b = d[:, None] * expected
        b[:-1] += upper[:, None] * expected[1:]
        b[1:] += lower[:, None] * expected[:-1]
        output = self.run_code(f"""
        tol numc as nc
        tol linalg as la
        esta lower = nc.array({lower.tolist()})
        esta diagonal = nc.array({d.tolist()})
        esta upper = nc.array({upper.tolist()})
        esta b = nc.permute(nc.array({b.T.tolist()}),[1,0])
        pedo(nc.tolist(la.solve_tridiag(lower,diagonal,upper,b)))
        """, stress=True).strip()
        actual = np.asarray(ast.literal_eval(output))
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
        # Match Thomas's row equilibration for the LAPACK reference. Raw
        # pivoting on this across-row scale range can lose small equations.
        bands[0, 1:] /= d[:-1]
        bands[2, :-1] /= d[1:]
        bands[1, :] = 1.0
        reference = linalg.solve_banded((1, 1), bands, b / d[:, None])
        np.testing.assert_allclose(actual, reference, rtol=1e-13, atol=1e-14)

    def test_pivoted_band_solves_against_scipy(self):
        rng = np.random.default_rng(1505)
        for n, lower, upper in [(1, 0, 0), (7, 0, 2), (8, 3, 0), (9, 1, 1), (15, 3, 2), (20, 2, 5), (11, 10, 10)]:
            with self.subTest(n=n, lower=lower, upper=upper):
                bands = np.zeros((lower + upper + 1, n))
                a = np.zeros((n, n))
                for col in range(n):
                    start, stop = max(col - upper, 0), min(col + lower + 1, n)
                    values = rng.uniform(-3, 3, stop - start)
                    if col == 0 and lower and upper:
                        # Force row pivoting with a zero or very small first pivot.
                        values[0] = 0.0 if n < 10 else 1e-30
                    a[start:stop, col] = values
                    bands[upper + start - col : upper + stop - col, col] = values
                b = rng.normal(size=(n, 3))
                output = self.run_code(f"""
                tol numc as nc
                tol linalg as la
                esta bands = nc.permute(nc.array({bands.T.tolist()}),[1,0])
                esta rhs = nc.array({b.tolist()})
                pedo(nc.tolist(la.solve_banded({lower},{upper},bands,rhs)))
                """, stress=True).strip()
                actual = np.asarray(ast.literal_eval(output))
                expected = linalg.solve_banded((lower, upper), bands, b)
                np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
                residual = np.linalg.norm(a @ actual - b) / (
                    np.linalg.norm(a) * np.linalg.norm(actual) + np.linalg.norm(b)
                )
                self.assertLess(residual, 1e-14)
