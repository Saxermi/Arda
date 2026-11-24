"""Independent reference checks for the 0.15.5 scalar numerical library."""
import ast
import math
import unittest

import numpy as np
from scipy import integrate, interpolate, optimize, special, stats

import test_arda


class NumericsReferenceTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_gamma_and_beta_against_scipy(self):
        points = [-30.125, -10.0000001, -1.9999999, -0.5, 1e-12, 0.3, 0.99, 1.5, 10, 170]
        pairs = [[0.01, 0.25], [0.5, 1e6], [0.8, 1e50], [2, 1e100],
                 [15.99, 16], [16, 16], [16, 500], [1e3, 5e3], [1e50, 1e50]]
        output = self.run_code(f"""
        tol special as s
        an x in {points}:
            pedo(s.gamma(x),s.lgamma(x))
        an ab in {pairs}:
            pedo(s.beta(ab[0],ab[1]),s.lbeta(ab[0],ab[1]))
        """).splitlines()
        values = np.array([list(map(float, line.split())) for line in output])
        np.testing.assert_allclose(values[:len(points), 0], special.gamma(points), rtol=2e-13)
        np.testing.assert_allclose(values[:len(points), 1], special.gammaln(points),
                                   rtol=2e-13, atol=1e-16)
        a, b = np.array(pairs).T
        expected_beta = special.beta(a, b)
        expected_log = special.betaln(a, b)
        # mpmath at 90 decimal digits for the unbalanced case. The SciPy 1.17
        # Float64 reference loses about 9.6e-10 relative accuracy in beta here.
        expected_beta[1] = 0.001772454072462261237774726534
        expected_log[1] = -6.335390211057436964987469022
        np.testing.assert_allclose(values[len(points):, 0], expected_beta, rtol=2e-12)
        np.testing.assert_allclose(values[len(points):, 1], expected_log, rtol=2e-12)

    def test_bessel_and_error_function_against_scipy(self):
        pairs = [[0, 1e-6], [-3, 0.2], [3, 0.2], [2, 5.13562230184],
                 [50, 49.8], [30, 120.5], [100, 105.0], [0, 1e4]]
        points = [-28, -8, -0.125, 0, 0.125, 4, 10, 26, 28]
        output = self.run_code(f"""
        tol special as s
        an pair in {pairs}:
            pedo(s.bessel_j(pair[0],pair[1]),s.bessel_y(pair[0],pair[1]))
        an x in {points}:
            pedo(s.erf(x),s.erfc(x))
        """).splitlines()
        values = np.array([list(map(float, line.split())) for line in output])
        n, x = np.array(pairs).T
        np.testing.assert_allclose(values[:len(pairs), 0], special.jv(n, x), rtol=2e-12, atol=2e-16)
        np.testing.assert_allclose(values[:len(pairs), 1], special.yv(n, x), rtol=2e-12, atol=2e-16)
        np.testing.assert_allclose(values[len(pairs):, 0], special.erf(points), rtol=3e-14)
        np.testing.assert_allclose(values[len(pairs):, 1], special.erfc(points), rtol=3e-13)

    def test_root_brackets_against_scipy(self):
        rng = np.random.default_rng(155)
        # Monotone cubics cover both signs and very different local slopes.
        cases = np.column_stack((rng.uniform(-4, 4, 24), 10 ** rng.uniform(-8, 2, 24)))
        output = self.run_code(f"""
        tol numerics as n
        an pair in {cases.tolist()}:
            esta target = pair[0]
            esta scale = pair[1]
            caro f(x):
                nan x*x*x+scale*x-target
            caro df(x):
                nan 3*x*x+scale
            esta brent = n.brent(f,-2,2)
            esta newton = n.newton(f,df,2,-2)
            pedo(brent,newton,f(brent),f(newton))
        """, stress=True).splitlines()
        roots = [optimize.brentq(lambda x: x**3 + scale*x - target, -2, 2, xtol=1e-14)
                 for target, scale in cases]
        values = np.array([list(map(float, line.split())) for line in output])
        np.testing.assert_allclose(values[:, 0], roots, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(values[:, 1], roots, rtol=2e-12, atol=2e-12)
        self.assertLess(np.max(np.abs(values[:, 2:])), 1e-10)

    def test_interpolants_against_numpy_and_scipy(self):
        x = np.array([-4, -1.5, -1, 0.1, 0.5, 2, 4])
        y = np.sin(x)
        q = np.sort(np.r_[np.linspace(-4, 4, 51), x])
        rows = self.run_code(f"""
        tol numerics as n
        esta linear = n.linear({x.tolist()},{y.tolist()})
        esta cubic = n.spline({x.tolist()},{y.tolist()})
        an q in {q.tolist()}:
            pedo(linear(q),cubic(q))
        """, stress=True).splitlines()
        values = np.array([list(map(float, line.split())) for line in rows])
        np.testing.assert_allclose(values[:, 0], np.interp(q, x, y), rtol=2e-14, atol=2e-15)
        expected = interpolate.CubicSpline(x, y, bc_type="natural")(q)
        np.testing.assert_allclose(values[:, 1], expected, rtol=2e-14, atol=2e-15)

    def test_quadrature_against_scipy(self):
        rows = self.run_code("""
        tol numerics as n
        tol ad
        pedo(n.quad(ad.exp,0,3))
        pedo(n.quad(lambda x: sin(100*x),0,1))
        pedo(n.quad(lambda x: 1/sqrt(x),0,1))
        pedo(n.quad(lambda x: ad.exp(-500*(x-0.3)*(x-0.3)),0,1))
        """, stress=True).splitlines()
        functions = [(math.exp, 3), (lambda x: math.sin(100*x), 1),
                     (lambda x: 1 / math.sqrt(x), 1),
                     (lambda x: math.exp(-500 * (x - 0.3)**2), 1)]
        for row, (function, stop) in zip(rows, functions, strict=True):
            estimate, error, count = ast.literal_eval(row)
            expected, _ = integrate.quad(function, 0, stop, epsabs=1e-12, epsrel=1e-12)
            self.assertLessEqual(abs(estimate - expected), max(2e-12, 2e-9 * abs(expected)))
            self.assertLessEqual(error, max(1e-12, 1e-9 * abs(estimate)))
            self.assertEqual(count % 30, 15)

    def test_distribution_tails_and_dispersion_against_references(self):
        normal_points = [-30, -8, -0.1, 2, 4.5, 12, 32]
        exp_points = [-1, 0, 1e-20, 0.25, 5, 250]
        samples = np.array([1e12, 1e12 + 1.0, 1e12 + 0.25, 1e12 + 3.0, 1e12 - 2.0])
        output = self.run_code(f"""
        tol stats as s
        tol numc as nc
        an x in {normal_points}:
            pedo(s.normal_pdf(x,0.25,0.9),s.normal_cdf(x,0.25,0.9),s.normal_sf(x,0.25,0.9))
        an x in {exp_points}:
            pedo(s.exponential_pdf(x,2),s.exponential_cdf(x,2),s.exponential_sf(x,2))
        esta x = nc.array({samples.tolist()})
        pedo(s.variance(x),s.variance(x,1),s.stddev(x,1))
        """).splitlines()
        values = np.array([list(map(float, line.split())) for line in output])
        normal = stats.norm(loc=0.25, scale=0.9)
        exp = stats.expon(scale=0.5)
        for data, dist, points in ((values[:len(normal_points)], normal, normal_points),
                                   (values[len(normal_points):-1], exp, exp_points)):
            expected = np.column_stack((dist.pdf(points), dist.cdf(points), dist.sf(points)))
            np.testing.assert_allclose(data, expected, rtol=5e-13, atol=0)
        centered = samples - samples[0]
        np.testing.assert_allclose(values[-1], [np.var(centered), np.var(centered, ddof=1),
                                               np.std(centered, ddof=1)], rtol=1e-14)
