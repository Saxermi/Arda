"""Optional CPU JAX checks for Vilya's automatic differentiation."""
import ast
import unittest

import numpy as np

import test_arda

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None


@unittest.skipIf(jax is None, "optional JAX installation unavailable")
class JaxAutodiffTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def setUp(self):
        jax.config.update("jax_enable_x64", True)

    def test_backward_batched_matmul_and_broadcast(self):
        text = self.run_code("""
        import tensor as t
        import ad
        let xs = t.array([0.2,-0.5,0.7,1.2,1.1,-0.1,0.8,0.6],"float64",[2,2,2])
        let ys = t.array([0.1,0.4,-0.3,0.5,0.7,-0.8],"float64",[2,3])
        let x = ad.variable(xs)
        let y = ad.variable(ys)
        let z = t.matmul(x,y)
        let f = t.mean(t.sin(z) + t.exp(z)/10)
        ad.backward(f)
        print(ad.value(f))
        print(t.tolist(ad.grad(x)))
        print(t.tolist(ad.grad(y)))
        """, stress=True)
        loss, dx, dy = map(ast.literal_eval, text.splitlines())
        x = jnp.array([0.2, -0.5, 0.7, 1.2, 1.1, -0.1, 0.8, 0.6]).reshape(2, 2, 2)
        y = jnp.array([0.1, 0.4, -0.3, 0.5, 0.7, -0.8]).reshape(2, 3)

        def objective(a, b):
            product = a @ b
            return jnp.mean(jnp.sin(product) + jnp.exp(product) / 10)

        expected_loss, (ex, ey) = jax.value_and_grad(objective, argnums=(0, 1))(x, y)
        np.testing.assert_allclose(loss, expected_loss, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(dx, ex, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(dy, ey, rtol=1e-13, atol=1e-14)
