"""Independent SciPy checks of factor reconstruction and reusable solves."""
import ast
import unittest

import numpy as np
from scipy import linalg
from scipy.optimize import linear_sum_assignment

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


class LapackReferenceTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def setUp(self):
        if self.run_code("tol calc\npedo(calc.backend())").strip() != "lapack":
            self.skipTest("requires LAPACK=1")

    def test_reduced_svd_and_pseudoinverse(self):
        rng = np.random.default_rng(1510)
        cases = [
            (rng.normal(size=(8, 4)), 1.0),
            (rng.normal(size=(4, 8)), 1.0),
            (rng.normal(size=(1, 6)), 1e-210),
            (rng.normal(size=(6, 1)), 1e210),
            (np.array([[2, 0, 0], [0, 2, 0], [0, 0, 0], [0, 0, 0]]), 1.0),
        ]
        for matrix, scale in cases:
            m, n = matrix.shape
            k = min(m, n)
            with self.subTest(shape=(m, n), scale=scale):
                a = matrix * scale
                output = self.run_code(f"""
                tol numc as nc
                tol linalg as la
                esta a = nc.permute(nc.array({a.T.tolist()}),[1,0])
                esta parts = la.svd(a)
                pedo(nc.tolist(parts[0]))
                pedo(nc.tolist(parts[1]))
                pedo(nc.tolist(parts[2]))
                pedo(nc.tolist(la.pinv(a)))
                pedo(la.rank(a))
                pedo(nc.tolist(a))
                """, stress=True).splitlines()
                u, s, vt, pinv, rank, unchanged = map(ast.literal_eval, output)
                u, s, vt, pinv = map(np.asarray, (u, s, vt, pinv))
                np.testing.assert_allclose(np.asarray(unchanged) / scale, matrix, rtol=1e-14)
                self.assertEqual((u.shape, s.shape, vt.shape), ((m, k), (k,), (k, n)))
                self.assertTrue(np.all(s[:-1] >= s[1:]) and np.all(s >= 0))
                np.testing.assert_allclose(s / scale, linalg.svdvals(matrix), rtol=1e-13, atol=1e-14)
                np.testing.assert_allclose((u * (s / scale)) @ vt, matrix, rtol=1e-13, atol=1e-14)
                np.testing.assert_allclose(u.T @ u, np.eye(k), atol=1e-13)
                np.testing.assert_allclose(vt @ vt.T, np.eye(k), atol=1e-13)

                pinv = pinv * scale
                expected, expected_rank = linalg.pinv(matrix, return_rank=True)
                self.assertEqual(rank, expected_rank)
                np.testing.assert_allclose(pinv, expected, rtol=1e-12, atol=1e-14)
                left, right = matrix @ pinv, pinv @ matrix
                np.testing.assert_allclose(left @ matrix, matrix, rtol=1e-13, atol=1e-14)
                np.testing.assert_allclose(right @ pinv, pinv, rtol=1e-13, atol=1e-14)
                np.testing.assert_allclose(left, left.T, atol=1e-13)
                np.testing.assert_allclose(right, right.T, atol=1e-13)

    def test_rank_cutoff_and_condition_number(self):
        rng = np.random.default_rng(1511)
        u, _ = np.linalg.qr(rng.normal(size=(8, 4)))
        v, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        a = (u * [1, 0.1, 1e-6, 1e-10]) @ v.T
        output = self.run_code(f"""
        tol numc as nc
        tol linalg as la
        esta a = nc.array({a.tolist()})
        pedo(la.cond(a))
        pedo(la.rank(a), la.rank(a,1e-5))
        pedo(nc.tolist(la.pinv(a,1e-5)))
        """, stress=True).splitlines()
        expected, expected_rank = linalg.pinv(a, rtol=1e-5, return_rank=True)
        self.assertEqual(list(map(int, output[1].split())), [4, expected_rank])
        np.testing.assert_allclose(float(output[0]), np.linalg.cond(a), rtol=3e-7)
        np.testing.assert_allclose(ast.literal_eval(output[2]), expected, rtol=1e-12, atol=1e-13)

    def test_symmetric_eigenpairs_on_strided_input(self):
        rng = np.random.default_rng(1512)
        basis, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        a = (basis * [-4, -1, -1, 0.25, 2, 5]) @ basis.T
        a = (a + a.T) / 2
        scale = 1e140
        output = self.run_code(f"""
        tol calc
        tol numc as nc
        tol linalg as la
        esta a = nc.permute(nc.array({(a * scale).tolist()}),[1,0])
        esta p = la.eigh(a)
        pedo(nc.tolist(p[0]))
        pedo(nc.tolist(p[1]))
        pedo(nc.tolist(a))
        """, stress=True).splitlines()
        values, vectors = (np.array(ast.literal_eval(line)) for line in output[:2])
        values /= scale
        expected, reference = linalg.eigh(a)
        np.testing.assert_allclose(values, expected, atol=1e-13)
        np.testing.assert_allclose(vectors[:, 1:3] @ vectors[:, 1:3].T,
                                   reference[:, 1:3] @ reference[:, 1:3].T, atol=1e-13)
        np.testing.assert_allclose(np.asarray(ast.literal_eval(output[2])) / scale, a, atol=1e-14)
        residual = np.linalg.norm(a @ vectors - vectors * values) / np.linalg.norm(a)
        orthogonality = np.linalg.norm(vectors.T @ vectors - np.eye(6))
        self.assertLess(residual, 1e-13)
        self.assertLess(orthogonality, 1e-13)
        print(f"\nEigh relative residual {residual:.3g}, orthogonality {orthogonality:.3g}")

    def test_real_complex_and_defective_general_eigenpairs(self):
        blocks = linalg.block_diag([[0, -2], [2, 0]], [[1, -3], [3, 1]], [[5]])
        basis = np.eye(5) + np.diag([1, -2, 1, 0.5], 1)
        cases = [
            ("mixed", basis @ blocks @ linalg.inv(basis), 1e-140),
            ("real", np.array([[1, 2, 3], [0, 2, 4], [0, 0, 3]]), 1.0),
            ("defective", np.array([[2, 1, 0], [0, 2, 1], [0, 0, 2]]), 1.0),
        ]
        for label, a, scale in cases:
            with self.subTest(case=label):
                output = self.run_code(f"""
                tol numc as nc
                tol linalg as la
                esta a = nc.permute(nc.array({(a.T * scale).tolist()}),[1,0])
                esta parts = la.eig(a)
                an i in range(4):
                    pedo(nc.tolist(parts[i]))
                pedo(nc.tolist(a))
                """, stress=True).splitlines()
                wr, wi, vr, vi, original = (np.array(ast.literal_eval(line)) for line in output)
                values, vectors = wr / scale + 1j * (wi / scale), vr + 1j * vi
                expected = linalg.eigvals(a)
                rows, cols = linear_sum_assignment(abs(values[:, None] - expected))
                np.testing.assert_allclose(values[rows], expected[cols], atol=1e-13)
                np.testing.assert_allclose(np.linalg.norm(vectors, axis=0), 1, atol=1e-13)
                np.testing.assert_allclose(original / scale, a, atol=1e-14)
                for j in np.flatnonzero(wi > 0):
                    self.assertEqual(values[j + 1], values[j].conjugate())
                    np.testing.assert_array_equal(vectors[:, j + 1], vectors[:, j].conjugate())
                self.assertTrue(np.all(vi[:, wi == 0] == 0))
                residuals = np.linalg.norm(a @ vectors - vectors * values, axis=0)
                max_residual = max(residuals) / np.linalg.norm(a)
                self.assertLess(max_residual, 1e-13)
                print(f"\nEig {label} maximum relative residual {max_residual:.3g}")
