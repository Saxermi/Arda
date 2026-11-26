"""Compare Vilya tensor VJPs with analytic NumPy derivatives and finite differences."""
import ast
import unittest

import numpy as np

import test_arda


def literal(x):
    array = np.asarray(x)
    return f't.array({array.ravel().tolist()},"float64",{list(array.shape)})'


class VilyaReferenceTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_broadcast_elementwise_vjp_matches_centered_differences(self):
        rng = np.random.default_rng(20261126)
        x = rng.uniform(0.5, 1.3, (2, 3))
        y = rng.uniform(0.7, 1.1, (3,))
        code = f"""
        import tensor as t
        tol ad
        let x = ad.variable({literal(x)})
        let y = ad.variable({literal(y)})
        let loss = t.sum(t.sin(x*y) + t.cos(y/x) + t.exp(x/5) + t.log(x+y) + t.sqrt(x))
        ad.backward(loss)
        print(t.tolist(ad.grad(x)))
        print(t.tolist(ad.grad(y)))
        """
        dx, dy = (np.asarray(ast.literal_eval(row))
                  for row in self.run_code(code, stress=True).splitlines())

        def loss(x, y):
            return np.sum(np.sin(x*y) + np.cos(y/x) + np.exp(x/5) + np.log(x+y) + np.sqrt(x))

        for actual, param in ((dx, x), (dy, y)):
            finite = np.empty_like(param)
            for index in np.ndindex(param.shape):
                previous = param[index]
                step = 1e-5
                param[index] = previous + step
                above = loss(x, y)
                param[index] = previous - step
                below = loss(x, y)
                param[index] = previous
                finite[index] = (above - below) / (2 * step)
            np.testing.assert_allclose(actual, finite, rtol=2e-9, atol=5e-10)

    def test_batched_matmul_vjp_matches_numpy(self):
        rng = np.random.default_rng(4317)
        a = rng.normal(size=(2, 1, 3, 4))
        b = rng.normal(size=(5, 2, 4))
        # The supplied b is stored as (batch, columns, inner), so the forward
        # operation also exercises permuted strides.
        seed = rng.normal(size=(2, 5, 3, 2))
        code = f"""
        import tensor as t
        import ad
        let a = ad.variable({literal(a)})
        let b = ad.variable({literal(b)})
        let c = t.matmul(a,t.permute(b,[0,2,1]))
        ad.backward(c,{literal(seed)})
        print(t.tolist(ad.value(c)))
        print(t.tolist(ad.grad(a)))
        print(t.tolist(ad.grad(b)))
        """
        c, da, db = (np.asarray(ast.literal_eval(row))
                     for row in self.run_code(code, stress=True).splitlines())
        np.testing.assert_allclose(c, a @ b.swapaxes(-1, -2), rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(da, (seed @ b).sum(axis=1, keepdims=True),
                                   rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(db, (seed.swapaxes(-1, -2) @ a).sum(axis=0),
                                   rtol=1e-13, atol=1e-13)
