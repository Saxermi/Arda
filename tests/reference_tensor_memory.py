"""Independent NumPy checks for bounded tensor operations."""
import ast
import unittest

import numpy as np

import test_arda


class TensorMemoryReferenceTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_streaming_moments_and_partial_tiles(self):
        rng = np.random.default_rng(1603)
        values = 1e12 + rng.normal(0, 4, (7, 5, 11))
        x = values.ravel().tolist()
        output = self.run_code(f"""
        tol tensor as t
        tol compute as c
        t.save("values",t.array({x},"float64",[7,5,11]))
        esta a = c.stream_stats("values",13)
        esta b = c.stream_stats("values",97)
        pedo(a["count"],a["mean"],a["variance"],b["mean"],b["variance"])
        pedo(t.tolist(t.read_tile("values",[3,1,7],[3,2,4])))
        """, stress=True).splitlines()
        count, mean_a, var_a, mean_b, var_b = map(float, output[0].split())
        self.assertEqual(count, values.size)
        centered = values - values.flat[0]
        np.testing.assert_allclose([var_a, var_b], np.var(centered), rtol=1e-13)
        # The CLI prints 15 significant digits, around 0.005 at this scale.
        np.testing.assert_allclose([mean_a, mean_b], np.mean(values), rtol=0, atol=0.006)
        np.testing.assert_allclose(ast.literal_eval(output[1]), values[3:6, 1:3, 7:11],
                                   rtol=0, atol=0.006)
