"""Independent complex, structured, spectral and optimization references."""
import ast
import cmath
import unittest

import numpy as np
from scipy import linalg, optimize

import test_arda


class StructuredReferenceTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_complex_branches_against_cmath(self):
        numbers = [0.3 + 0.5j, -2 - 0.3j, -3 + 1e-16j, -3 - 1e-16j, 4 - 2j]
        coordinates = [[z.real, z.imag] for z in numbers]
        output = self.run_code(f"""
        tol complex as cx
        an parts in {coordinates}:
            esta z = cx.new(parts[0],parts[1])
            an f in [cx.sqrt,cx.log,cx.exp,cx.sin,cx.cos,cx.tan]:
                esta w = f(z)
                pedo(cx.real(w),cx.imag(w))
            esta w = cx.pow(z,cx.new(0.6,-0.4))
            pedo(cx.real(w),cx.imag(w))
            pedo(cx.abs(z),cx.arg(z))
        """, stress=True).splitlines()
        expected = []
        for z in numbers:
            for operation in (cmath.sqrt, cmath.log, cmath.exp, cmath.sin, cmath.cos, cmath.tan,
                              lambda value: value ** (0.6 - 0.4j)):
                w = operation(z)
                expected.append([w.real, w.imag])
            expected.append([abs(z), cmath.phase(z)])
        np.testing.assert_allclose([list(map(float, line.split())) for line in output], expected,
                                   rtol=6e-14, atol=6e-15)

    def test_rectangular_operators_against_dense_products(self):
        rng = np.random.default_rng(1531)
        u, v = rng.normal(size=(6, 3)), rng.normal(size=(7, 3))
        a, b = rng.normal(size=(4, 2)), rng.normal(size=(3, 5))
        col, row = rng.normal(size=6), rng.normal(size=4)
        row[0] = col[0]
        as_array = lambda x: f'nc.array({x.tolist()})'
        cases = [
            (f'lo.lowrank({as_array(u)},{as_array(v)})', u @ v.T),
            (f'lo.kron({as_array(a)},{as_array(b)})', np.kron(a, b)),
            (f'lo.toeplitz({as_array(col)},{as_array(row)})', linalg.toeplitz(col, row)),
            ('sp.tridiag(7,-1,4,2)', np.diag(np.full(7, 4.0)) + np.diag(np.full(6, -1.0), -1)
             + np.diag(np.full(6, 2.0), 1)),
        ]
        maximum = 0
        for constructor, matrix in cases:
            x = rng.normal(size=(matrix.shape[1], 3))
            y = rng.normal(size=(matrix.shape[0], 3))
            output = self.run_code(f"""
            tol numc as nc
            tol linop as lo
            tol sparse as sp
            esta op = {constructor}
            esta x = nc.permute({as_array(x.T)},[1,0])
            esta y = nc.permute({as_array(y.T)},[1,0])
            pedo(nc.tolist(lo.apply(op,x)))
            pedo(nc.tolist(lo.apply_transpose(op,y)))
            """, stress=True).splitlines()
            for actual, expected in zip(map(ast.literal_eval, output), (matrix @ x, matrix.T @ y)):
                np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
                maximum = max(maximum, np.linalg.norm(actual - expected) / np.linalg.norm(expected))
        print(f'\nOperators maximum relative product error {maximum:.3g}')

    def test_randomized_svd_accuracy_and_no_dense_operator_materialization(self):
        if self.run_code('tol calc\npedo(calc.backend())').strip() != 'lapack':
            self.skipTest('requires LAPACK=1')
        spectrum = np.geomspace(1, 1e-4, 32)
        output = self.run_code(f"""
        tol numc as nc
        tol linalg as la
        tol linop as lo
        esta values = nc.array({spectrum.tolist()})
        esta op = lo.diagonal(values)
        esta parts = la.rsvd(op,5,4,2,90)
        an i in range(3):
            pedo(nc.tolist(parts[i]))
        pedo(parts[3])
        esta large = lo.lowrank(nc.ones(5000,1),nc.ones(6000,1))
        esta large_parts = la.rsvd(large,1,1,1,5)
        pedo(nc.shape(large_parts[0]),nc.shape(large_parts[2]))
        pedo(large_parts[1][0],large_parts[3])
        """, stress=True).splitlines()
        u, s, vt = (np.asarray(ast.literal_eval(line)) for line in output[:3])
        matrix = np.diag(spectrum)
        expected = linalg.svdvals(matrix)
        error = np.linalg.norm((u * s) @ vt - matrix) / np.linalg.norm(matrix)
        estimate = float(output[3])
        np.testing.assert_allclose(s, expected[:5], rtol=2e-4)
        np.testing.assert_allclose(u.T @ u, np.eye(5), atol=1e-13)
        np.testing.assert_allclose(vt @ vt.T, np.eye(5), atol=1e-13)
        self.assertLess(error, 1.01 * np.linalg.norm(expected[5:]) / np.linalg.norm(expected))
        self.assertLess(abs(estimate - error), 0.5 * error)
        self.assertEqual(output[4], '[5000, 1] [1, 6000]')
        singular, large_error = map(float, output[5].split())
        np.testing.assert_allclose(singular, np.sqrt(5000 * 6000), rtol=1e-13)
        self.assertLess(large_error, 1e-12)
        print(f'\nRandomized SVD actual error {error:.3g}, estimate {estimate:.3g}, '
              f'large rank-one estimate {large_error:.3g}')

    def test_objective_only_optimizers_against_scipy(self):
        output = self.run_code("""
        tol numc as nc
        tol opt
        esta center = nc.array([1,-1,2])
        esta weights = nc.array([1,2,3])
        caro f(x):
            esta d = x-center
            nan nc.sum(weights*d*d)/2
        esta start = nc.array([7,-2,3])
        an method in ["coordinate","random"]:
            esta result = opt.zeroth_order(f,start,method,1e-4,1e-7,160,1532,24)
            pedo(nc.tolist(result[0]))
            pedo(result[4],result[5],result[3])
        """, stress=True).splitlines()
        center = np.array([1.0, -1.0, 2.0])
        objective = lambda x: np.sum(np.array([1, 2, 3]) * (x - center) ** 2) / 2
        expected = optimize.minimize(objective, [7, -2, 3], method='BFGS', tol=1e-10)
        for i, method in enumerate(('coordinate', 'random')):
            actual = np.asarray(ast.literal_eval(output[2 * i]))
            objective_value, evaluations, converged = output[2 * i + 1].split()
            self.assertEqual(converged, 'nae')
            np.testing.assert_allclose(actual, expected.x, rtol=0, atol=2e-7)
            self.assertLess(float(objective_value), 2e-14)
            print(f'\nZeroth-order {method} objective {float(objective_value):.3g}, '
                  f'{evaluations} function evaluations')
