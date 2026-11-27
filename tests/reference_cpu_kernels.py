"""Independent Float64 checks on rectangular, offset GEMM inputs."""
import ast
import unittest

import numpy as np

import test_arda


class CpuKernelReferenceTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_float64_dispatch_against_extended_precision(self):
        rng = np.random.default_rng(1801)
        a = rng.normal(size=(7, 13))
        b = rng.normal(size=(13, 9))
        left = [0.0, *a.ravel().tolist()]
        right = [0.0, *b.ravel().tolist()]
        result = self.run_code(f"""
        import tensor as t
        import compute as c
        let a = t.view(t.array({left},"float64"),[7,13],[13,1],1)
        let b = t.view(t.array({right},"float64"),[13,9],[9,1],1)
        for kernel in ["auto","reference","sse2","avx2","avx2-fma","neon"]:
            try:
                c.matmul_kernel(kernel)
            catch issue:
                if issue["code"] != "backend":
                    throw issue
                continue
            print(t.tolist(t.matmul(a,b)))
        """, stress=True)
        reference = np.asarray(a.astype(np.longdouble) @ b.astype(np.longdouble),
                               dtype=np.float64)
        self.assertTrue(result.strip())
        for line in result.splitlines():
            np.testing.assert_allclose(ast.literal_eval(line), reference, rtol=1e-13,
                                       atol=2e-14)
