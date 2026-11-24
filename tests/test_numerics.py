"""Mathematical library contracts without optional Python dependencies."""
import ast
import math
import unittest

import test_arda


class NumericsTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_special_functions_and_tails(self):
        lines = self.run_code("""
        tol special as s
        pedo(s.gamma(5),s.gamma(-0.5),s.lgamma(-0.5))
        pedo(s.beta(1,1e100),s.beta(2,1e100),s.lbeta(1000,1000))
        pedo(s.bessel_j(1,0),s.bessel_j(0,0),s.bessel_j(-1,2),s.bessel_j(1,2))
        pedo(s.erf(0),s.erfc(10))
        """, stress=True).splitlines()
        gamma, negative, log_abs = map(float, lines[0].split())
        self.assertEqual(gamma, 24)
        self.assertAlmostEqual(negative, -2 * math.sqrt(math.pi), places=13)
        self.assertAlmostEqual(log_abs, math.log(2 * math.sqrt(math.pi)), places=13)
        reciprocal, second, log_beta = map(float, lines[1].split())
        self.assertTrue(math.isclose(reciprocal, 1e-100, rel_tol=1e-14))
        self.assertTrue(math.isclose(second, 1e-200, rel_tol=1e-14))
        self.assertAlmostEqual(log_beta, 2 * math.lgamma(1000) - math.lgamma(2000), places=10)
        zero, one, negative_order, positive_order = map(float, lines[2].split())
        self.assertEqual((zero, one, negative_order), (0, 1, -positive_order))
        erf, erfc = map(float, lines[3].split())
        self.assertEqual(erf, 0)
        self.assertTrue(math.isclose(erfc, math.erfc(10), rel_tol=1e-14))

    def test_special_and_distribution_domains(self):
        cases = {
            "special.gamma(0)": "pole",
            "special.lgamma(-2)": "pole",
            "special.gamma(180)": "Float64 range",
            "special.beta(-1,2)": "positive",
            "special.bessel_j(10001,2)": "Bessel order",
            "special.bessel_y(0,0)": "positive",
            "special.erf(1e300*1e300)": "finite",
            "stats.normal_pdf(0,0,0)": "positive",
            "stats.exponential_cdf(1,0)": "positive",
            "stats.variance(nc.array([1]),1)": "ddof",
            "stats.stddev(nc.array([]))": "ddof",
        }
        imports = "tol special\ntol stats\ntol numc as nc\n"
        for case, error in cases.items():
            with self.subTest(case=case):
                self.run_code(imports + "pedo(" + case + ")", error=error, stress=True)

    def test_distribution_extremes_and_exponential_rng(self):
        output = self.run_code("""
        tol stats
        tol random as r
        tol numc as nc
        pedo(stats.normal_sf(10,0,1),stats.normal_cdf(-10,0,1))
        pedo(stats.exponential_pdf(-1,2),stats.exponential_cdf(-1,2),stats.exponential_sf(-1,2))
        pedo(stats.exponential_cdf(1e-18,2),stats.exponential_sf(50,2))
        esta gen = r.generator(42,54)
        esta twin = r.generator(42,54)
        esta samples = r.exponential(gen,2,3)
        pedo(nc.tolist(samples))
        pedo([r.exponential(twin,2),r.exponential(twin,2),r.exponential(twin,2)])
        esta a = r.generator(42)
        esta b = r.generator(42)
        try:
            r.exponential(a,5e-324,3)
        catch err:
            pedo("overflow")
        pedo(r.uint32(a) == r.uint32(b))
        """, stress=True).splitlines()
        sf, cdf = map(float, output[0].split())
        self.assertEqual(sf, cdf)
        self.assertGreater(sf, 0)
        self.assertEqual(output[1], "0 0 1")
        cdf, sf = map(float, output[2].split())
        self.assertTrue(math.isclose(cdf, 2e-18, rel_tol=1e-14))
        self.assertTrue(math.isclose(sf, math.exp(-100), rel_tol=1e-14))
        self.assertEqual(output[3], output[4])
        self.assertTrue(all(x > 0 for x in ast.literal_eval(output[3])))
        self.assertEqual(output[5:], ["overflow", "nae"])

    def test_dispersion_on_offset_and_scaled_vectors(self):
        lines = self.run_code("""
        tol stats
        tol numc as nc
        esta v = nc.array([1000000000001,1000000000002,1000000000003])
        pedo(stats.variance(v),stats.variance(v,1),stats.stddev(v,1))
        pedo(stats.stddev(nc.array([-1e300,1e300])))
        pedo(stats.stddev(nc.array([-1e-300,1e-300])))
        """, stress=True).splitlines()
        variance, sample, deviation = map(float, lines[0].split())
        self.assertAlmostEqual(variance, 2 / 3, places=14)
        self.assertEqual((sample, deviation), (1, 1))
        self.assertEqual(float(lines[1]), 1e300)
        self.assertEqual(float(lines[2]), 1e-300)

    def test_roots_in_reversed_and_flat_brackets(self):
        values = self.run_code("""
        tol numerics as n
        caro f(x):
            nan x*x-2
        caro flat(x):
            nan 0
        pedo(n.brent(f,2,0),n.newton(f,flat,0,2))
        pedo(n.brent(lambda x: x-2,2,5))
        pedo(n.brent(lambda x: 1e300*(x-0.25),-1,1))
        """, stress=True).split()
        for root in values[:2]:
            self.assertAlmostEqual(float(root), math.sqrt(2), places=11)
        self.assertEqual([float(root) for root in values[2:]], [2, 0.25])

    def test_root_failures_and_callback_errors(self):
        cases = {
            "n.brent(lambda x: x*x+1,-1,1)": "sign-changing bracket",
            "n.brent(lambda x: x*x-2,0,2,1e-15,0,1)": "iteration limit",
            "n.newton(lambda x: x*x-2,lambda x: 1e300*1e300,0,2)": "finite real scalars",
            "n.brent(lambda x: x,-1e308,1e308)": "invalid root bracket",
            "n.newton(callback,lambda x: 1,0,2)": "callback propagated",
        }
        for source, error in cases.items():
            with self.subTest(source=source):
                prefix = 'tol numerics as n\ncaro callback(x):\n    throw "callback propagated"\n'
                self.run_code(prefix + source, error=error, stress=True)

    def test_interpolation_snapshot_and_edges(self):
        self.run_code("""
        tol numerics as n
        tol numc as nc
        esta knots = [0,1,2]
        esta values = [0,1,0]
        esta f = n.linear(knots,values)
        esta g = n.spline(knots,values)
        esta two = n.spline(nc.array([0,4]),nc.array([2,6]))
        knots[1] = 100
        values[1] = 100
        collect()
        pedo(f(0),f(0.5),f(2),g(0.5),g(1),g(1.5),two(2))
        """, "0 0.5 0 0.6875 1 0.6875 4", stress=True)
        self.run_code("tol numerics as n\nn.spline([0,1,1],[0,1,1])",
                      error="strictly increasing", stress=True)
        self.run_code("tol numerics as n\nn.linear([0,1],[0,1])(-0.1)",
                      error="outside the knot interval", stress=True)

    def test_adaptive_quadrature_refines_and_preserves_sign(self):
        lines = self.run_code("""
        tol numerics as n
        esta calls = 0
        caro f(x):
            calls = calls+1
            nan abs(x-0.2)
        pedo(n.quad(f,4,4),calls)
        esta a = n.quad(f,0,1)
        esta b = n.quad(f,1,0)
        pedo(a)
        pedo(b)
        pedo(calls)
        """, stress=True).splitlines()
        self.assertEqual(lines[0], "[0, 0, 0] 0")
        value, estimated, evaluations = ast.literal_eval(lines[1])
        self.assertAlmostEqual(value, 0.34, places=9)
        self.assertLessEqual(estimated, 1e-9 * abs(value))
        self.assertGreater(evaluations, 15)
        self.assertEqual(ast.literal_eval(lines[2]), [-value, estimated, evaluations])
        self.assertEqual(int(lines[3]), 2 * evaluations)

    def test_adaptive_quadrature_failure_and_callback_types(self):
        self.run_code("tol numerics as n\nn.quad(lambda x: abs(x-0.2),0,1,1e-12,1e-14,1)",
                      error="interval limit", stress=True)
        self.run_code("tol numerics as n\nn.quad(lambda x: [x],0,1)",
                      error="finite real scalars", stress=True)
