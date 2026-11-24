"""Generator state, collection sampling and statistics contracts."""
import ast
import unittest

import test_arda


class ProbabilityTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_pcg_reference_stream_and_independent_state(self):
        self.run_code("""
        tol random as r
        esta first = r.generator(42,54)
        esta second = r.generator(42,54)
        esta other = r.generator(42,55)
        pedo(type(first))
        pedo([r.uint32(first) for i in range(6)])
        pedo(r.uint32(second), r.uint32(other) == 2707161783)
        esta same = r.generator(42,54)
        esta unused = r.uniform(same,0,1,0)
        esta zero = r.poisson(same,0)
        pedo(zero, r.uint32(same))
        """, "generator\n[2707161783, 2068313097, 3122475824, 2211639955, 3215226955, 3421331566]\n"
             "2707161783 law\n0 2707161783", stress=True)

    def test_batch_draws_shuffling_and_exact_integers(self):
        output = self.run_code("""
        tol random as r
        tol numc as nc
        esta first = r.generator(19)
        esta second = r.generator(19)
        pedo(nc.tolist(r.uniform(first,-2,3,5)))
        pedo([r.uniform(second,-2,3) for i in range(5)])
        esta x = nc.view(nc.array([1,99,2,99,3,99,4]),0,0,7,2)
        pedo(nc.tolist(r.shuffle(first,x)))
        pedo(nc.tolist(x))
        pedo(r.choice(first,[17,31],8))
        pedo(r.integer(first,-9223372036854775807-1,9223372036854775807,10))
        pedo(r.shuffle(first,[]))
        """, stress=True).splitlines()
        batch, scalars, shuffled, original, choices, large, empty = map(ast.literal_eval, output)
        self.assertEqual(batch, scalars)
        self.assertTrue(all(-2 <= x < 3 for x in batch))
        self.assertEqual(sorted(shuffled), original)
        self.assertEqual(original, [1, 2, 3, 4])
        self.assertEqual(len(choices), 8)
        self.assertTrue(set(choices) <= {17, 31})
        self.assertTrue(all(isinstance(x, int) and -(2**63) <= x < 2**63 - 1 for x in large))
        self.assertEqual(empty, [])

    def test_random_errors_and_batch_failure_preserves_state(self):
        for expr, message in [
            ("r.generator(1.5)", "seed must be an integer"),
            ("r.generator(1,-1)", "nonnegative integer"),
            ("r.uniform(g,2,1)", "low < high"),
            ("r.normal(g,0,-1,4)", "deviation must be nonnegative"),
            ("r.poisson(g,1e9)", "Poisson mean"),
            ("r.choice(g,[])", "nonempty sequence"),
            ("r.integer(g,1.5,2)", "requires integer bounds"),
            ("r.integer(g,0,2,-1)", "sample count"),
        ]:
            with self.subTest(expression=expr):
                self.run_code("tol random as r\nesta g = r.generator(1)\n" + expr,
                              error=message, stress=True)
        self.run_code("""
        tol random as r
        esta g = r.generator(42,54)
        try:
            r.normal(g,1.7e308,1.7e308,100)
        catch e:
            pedo(e)
        pedo(r.uint32(g))
        """, "normal sample is outside the Float64 range\n2707161783", stress=True)

    def test_statistics_boundaries_and_streaming(self):
        self.run_code("""
        tol numc as nc
        tol stats as st
        esta x = nc.array([4,1,4,2])
        pedo(st.argmin(x),st.argmax(x),st.median(x),st.quantile(x,0.25))
        pedo(nc.tolist(st.histogram(x,nc.array([1,2,4]))))
        esta y = nc.permute(nc.array([[-10,3],[0,10]]),[1,0])
        pedo(nc.tolist(st.clip(y,-2,2)), nc.tolist(y))
        esta cancellation = nc.array([1e100,1,-1e100])
        pedo(st.sum(cancellation), nc.tolist(st.cumsum(cancellation)))
        pedo(st.sum(nc.zeros(0)), nc.shape(st.cumsum(nc.zeros(0))))
        esta moments = st.online()
        pedo(st.summary(moments))
        st.update(moments,999999999999)
        st.update(moments,1000000000001)
        pedo(type(moments), st.summary(moments))
        """, "1 0 3 1.75\n[1, 3]\n[[-2, 0], [2, 2]] [[-10, 0], [3, 10]]\n"
             "1 [1e+100, 1e+100, 1]\n0 [0]\n[0, mun, mun]\nmoments [2, 1000000000000, 1]", stress=True)

    def test_undefined_statistics_and_overflow(self):
        imports = "tol numc as nc\ntol stats as st\n"
        for expr, message in [
            ("st.median(nc.zeros(0))", "nonempty vector"),
            ("st.quantile(nc.ones(2),1.01)", "in [0, 1]"),
            ("st.covariance(nc.ones(2),nc.ones(1))", "equal vector lengths"),
            ("st.correlation(nc.ones(2),nc.ones(2))", "nonconstant vectors"),
            ("st.regression(nc.ones(2),nc.ones(2))", "nonconstant predictor"),
            ("st.histogram(nc.zeros(0),nc.ones(2))", "strictly increasing"),
            ("st.clip(nc.ones(2),3,2)", "low <= high"),
            ("st.argmin(nc.array([1,1e308*1e308]))", "require finite inputs"),
            ("st.cumsum(nc.array([1e308,1e308,-1e308]))", "outside the Float64 range"),
        ]:
            with self.subTest(expression=expr):
                self.run_code(imports + expr, error=message, stress=True)
