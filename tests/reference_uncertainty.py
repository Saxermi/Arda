"""Analytic and SciPy references for local uncertainty propagation."""
import ast
import unittest

import numpy as np
from scipy.stats import norm

import test_arda


class UncertaintyReferenceTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_independent_linear_and_monte_carlo_product(self):
        text = self.run_code("""
        import ad
        import tensor as t
        import uncertainty as u
        import random as r
        def force(x):
            return t.at(x,[0])*t.at(x,[1])
        let mean = t.array([10,9.81],"float64")
        let covariance = {"stddev": t.array([0.2,0.05],"float64")}
        let x = ad.variable(mean)
        let result = u.linear(force(x),x,covariance)
        let mc = u.monte_carlo(force,mean,covariance,r.generator(20261126),30000)
        print([result["value"],result["variance"],mc["mean"],mc["variance"]])
        """)
        nominal, linear_variance, sample_mean, sample_variance = ast.literal_eval(text)
        mass, acceleration, sigma_m, sigma_a = 10, 9.81, 0.2, 0.05
        first_order = (acceleration * sigma_m) ** 2 + (mass * sigma_a) ** 2
        exact = first_order + (sigma_m * sigma_a) ** 2
        np.testing.assert_allclose([nominal, linear_variance], [mass * acceleration, first_order],
                                   rtol=2e-14)
        self.assertLess(abs(sample_mean - nominal), 5 * np.sqrt(exact / 30000))
        self.assertLess(abs(sample_variance / exact - 1), 0.05)

    def test_correlated_gaussian_square_at_zero_gradient(self):
        text = self.run_code("""
        import ad
        import tensor as t
        import uncertainty as u
        import random as r
        def square(x):
            let q = t.sum(x)
            return q*q
        let mean = t.zeros([2],"float64")
        let covariance = {"factor": t.array([1,2],"float64",[2,1])}
        let x = ad.variable(mean)
        let result = u.linear(square(x),x,covariance)
        let mc = u.monte_carlo(square,mean,covariance,r.generator(20261126),30000)
        print([result["variance"],mc["mean"],mc["variance"]])
        """)
        linear, sample_mean, sample_variance = ast.literal_eval(text)
        gaussian = norm(loc=0, scale=3)
        expected_mean = gaussian.moment(2)
        expected_variance = gaussian.moment(4) - expected_mean ** 2
        self.assertEqual(linear, 0)
        self.assertLess(abs(sample_mean / expected_mean - 1), 0.05)
        self.assertLess(abs(sample_variance / expected_variance - 1), 0.08)
