"""Independent NumPy checks for the portable typed tensor paths."""
import ast
import unittest

import numpy as np

import test_arda


class TypedTensorReferenceTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_half_rounding_and_complex_fft(self):
        rng = np.random.default_rng(1601)
        points = np.r_[rng.uniform(-2048, 2048, 60), 2.0**-24, 2.0**-25, 1.00048828125]
        z = rng.normal(size=16) + 1j * rng.normal(size=16)
        pairs = np.column_stack((z.real, z.imag)).tolist()
        rows = self.run_code(f"""
        tol tensor as t
        tol complex as cx
        pedo(t.tolist(t.array({points.tolist()},"float16")))
        esta z = []
        an pair in {pairs}:
            push(z,cx.new(pair[0],pair[1]))
        esta spectrum = t.fft(t.array(z,"complex128"))
        an w in t.tolist(spectrum):
            pedo(cx.real(w),cx.imag(w))
        an w in t.tolist(t.ifft(spectrum)):
            pedo(cx.real(w),cx.imag(w))
        """, stress=True).splitlines()
        np.testing.assert_allclose(ast.literal_eval(rows[0]), points.astype("float16").astype(float),
                                   rtol=6e-15, atol=0)
        coordinates = np.array([list(map(float, row.split())) for row in rows[1:]])
        actual = coordinates[:, 0] + 1j * coordinates[:, 1]
        np.testing.assert_allclose(actual[:16], np.fft.fft(z), rtol=3e-14, atol=2e-14)
        np.testing.assert_allclose(actual[16:], z, rtol=3e-14, atol=2e-14)

    def test_broadcast_contraction_and_sum(self):
        rng = np.random.default_rng(1602)
        a = rng.integers(-50, 50, size=(3, 2, 4))
        b = rng.integers(-50, 50, size=(4, 5))
        rows = self.run_code(f"""
        tol tensor as t
        esta a = t.array({a.ravel().tolist()},"int32",[3,2,4])
        esta b = t.array({b.ravel().tolist()},"int64",[4,5])
        pedo(t.tolist(t.matmul(a,b)))
        pedo(t.tolist(t.einsum("bij,jk->bik",a,b)))
        pedo(t.tolist(t.sum(a,1)))
        pedo(t.tolist(a + t.array([1,2,3,4],"int32")))
        """, stress=True).splitlines()
        targets = (a @ b, np.einsum("bij,jk->bik", a, b), a.sum(axis=1), a + np.arange(1, 5))
        for actual, expected in zip(rows, targets, strict=True):
            np.testing.assert_array_equal(ast.literal_eval(actual), expected)
