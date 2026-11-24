"""NumPy and SciPy comparisons for seeded draws and portable statistics."""
import ast
import math
import unittest

import numpy as np
from scipy import stats

import test_arda


class ProbabilityReferenceTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_uniform_normal_and_integer_distributions(self):
        output = self.run_code("""
        tol numc as nc
        tol random as r
        esta g = r.generator(8912,7)
        pedo(nc.tolist(r.uniform(g,-2,3,12000)))
        pedo(nc.tolist(r.normal(g,4,2,12000)))
        pedo(r.integer(g,-3,4,14000))
        """, stress=True).splitlines()
        uniform, normal, integers = (np.asarray(ast.literal_eval(line)) for line in output)
        self.assertTrue(np.all((uniform >= -2) & (uniform < 3)))
        self.assertTrue(np.all((integers >= -3) & (integers < 4)))
        uniform_distance = stats.kstest(uniform, 'uniform', args=(-2, 5)).statistic
        normal_distance = stats.kstest(normal, 'norm', args=(4, 2)).statistic
        chi_square = stats.chisquare(np.bincount(integers + 3, minlength=7)).statistic
        self.assertLess(uniform_distance, 0.025)
        self.assertLess(normal_distance, 0.025)
        self.assertLess(chi_square, 35)
        print(f"\nRNG uniform KS {uniform_distance:.3g}, normal KS {normal_distance:.3g}, "
              f"integer chi-square {chi_square:.3g}")

    def test_poisson_distributions_both_sides_of_switch_and_large_mean(self):
        means = [0.125, 4, 9.99, 10, 25, 105, 1e8]
        n = 12000
        output = self.run_code(f"""
        tol random as r
        esta g = r.generator(563,9)
        an mean in {means}:
            pedo(r.poisson(g,mean,{n}))
        """, stress=True).splitlines()
        self.assertEqual(len(output), len(means))
        distances = []
        for mean, line in zip(means, output):
            with self.subTest(mean=mean):
                samples = np.sort(np.asarray(ast.literal_eval(line)))
                self.assertTrue(np.issubdtype(samples.dtype, np.integer))
                self.assertTrue(np.all(samples >= 0))
                values, counts = np.unique(samples, return_counts=True)
                totals = np.cumsum(counts) / n
                previous = (np.cumsum(counts) - counts) / n
                distance = max(np.max(abs(totals - stats.poisson.cdf(values, mean))),
                               np.max(abs(previous - stats.poisson.cdf(values - 1, mean))))
                self.assertLess(distance, 0.03)
                self.assertLess(abs(np.mean(samples) - mean) / math.sqrt(mean / n), 6)
                self.assertLess(abs(np.var(samples) / mean - 1), 0.12)
                distances.append(distance)
        print(f"\nRNG Poisson maximum CDF distance {max(distances):.3g} across seven means")

    def test_statistics_against_numpy_and_scipy(self):
        rng = np.random.default_rng(1520)
        x = rng.normal(size=80)
        y = 4 * x + rng.uniform(-0.5, 0.5, size=x.shape) + 1
        edges = [-2, -1, 0, 0.1, 2]
        extended = np.column_stack((x, np.zeros_like(x))).ravel()
        output = self.run_code(f"""
        tol numc as nc
        tol stats as st
        esta x = nc.view(nc.array({extended.tolist()}),0,0,{len(extended)},2)
        esta y = nc.array({y.tolist()})
        pedo(st.median(x),st.quantile(x,0.93),st.covariance(x,y),st.correlation(x,y))
        pedo(st.regression(x,y))
        pedo(nc.tolist(st.histogram(x,nc.array({edges}))))
        pedo(nc.tolist(st.cumsum(x)))
        pedo(st.sum(x),st.argmin(x),st.argmax(x))
        pedo(nc.tolist(st.clip(x,-0.93,1)))
        esta online = st.online()
        an i in range({len(x)}):
            st.update(online,x[i])
        pedo(st.summary(online))
        """, stress=True).splitlines()
        measured = list(map(float, output[0].split()))
        expected = [np.median(x), np.quantile(x, 0.93), np.cov(x, y)[0, 1],
                    stats.pearsonr(x, y).statistic]
        np.testing.assert_allclose(measured, expected, atol=1e-13)
        regression = stats.linregress(x, y)
        np.testing.assert_allclose(ast.literal_eval(output[1]), [regression.slope, regression.intercept],
                                   atol=1e-13)
        np.testing.assert_array_equal(ast.literal_eval(output[2]), np.histogram(x, edges)[0])
        np.testing.assert_allclose(ast.literal_eval(output[3]), np.cumsum(x), atol=1e-13)
        np.testing.assert_allclose(list(map(float, output[4].split())),
                                   [math.fsum(x), np.argmin(x), np.argmax(x)], atol=1e-13)
        np.testing.assert_allclose(ast.literal_eval(output[5]), np.clip(x, -0.93, 1), atol=1e-13)
        np.testing.assert_allclose(ast.literal_eval(output[6]), [len(x), np.mean(x), np.var(x)], atol=1e-13)
        print(f"\nStats maximum absolute scalar difference {max(abs(np.asarray(measured)-expected)):.3g}")

    def test_scaled_paired_statistics_and_shifted_online_variance(self):
        output = self.run_code("""
        tol numc as nc
        tol stats as st
        esta u = nc.array([-1e200,0,1e200])
        esta w = nc.array([-1e-200,0,1e-200])
        pedo(st.covariance(u,w),st.correlation(u,w))
        esta acc = st.online()
        an x in [1,-1,2,-2,0]:
            st.update(acc,1e15+x*0.25)
        pedo(st.summary(acc))
        """, stress=True).splitlines()
        np.testing.assert_allclose(list(map(float, output[0].split())), [1, 1], atol=1e-14)
        np.testing.assert_array_equal(ast.literal_eval(output[1]), [5, 1e15, 0.125])
