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

    def test_vjp_jvp_and_jacobian_match_jax(self):
        output = self.run_code("""
        import ad
        tol tensor as t
        let x = ad.variable(t.array([0.5,1.2,2.0],"float64"))
        let a = t.reshape(t.view(x,[3],[-1],2)*1,[3,1])
        let weights = t.array([1.1,0.7,-0.3,0.5,-0.4,1.3],"float64",[3,2])
        let f = t.sum(t.sin(a + weights)/t.exp(a),0)
        print(t.tolist(ad.vjp(f,x,t.array([-0.6,1.1],"float64"))))
        print(t.tolist(ad.jvp(f,x,t.array([0.1,-0.3,0.7],"float64"))))
        print(t.tolist(ad.jacobian(f,x)))
        """, stress=True)
        vjp, jvp, jac = map(ast.literal_eval, output.splitlines())
        point = jnp.array([0.5, 1.2, 2.0])
        weights = jnp.array([[1.1, 0.7], [-0.3, 0.5], [-0.4, 1.3]])

        def function(x):
            a = x[::-1, None]
            return jnp.sum(jnp.sin(a + weights) / jnp.exp(a), axis=0)

        _, pullback = jax.vjp(function, point)
        _, jvp_expected = jax.jvp(function, (point,), (jnp.array([0.1, -0.3, 0.7]),))
        for actual, expected in ((vjp, pullback(jnp.array([-0.6, 1.1]))[0]),
                                 (jvp, jvp_expected), (jac, jax.jacrev(function)(point))):
            np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-14)

    def test_hessian_matches_jax_on_nonlinear_tensor_graph(self):
        text = self.run_code("""
        import ad
        import tensor as t
        let x = ad.variable(t.array([0.5,1.2,2.0,0.9],"float64",[2,2]))
        let y = t.matmul(x,t.permute(x,[1,0]))
        let p = x*x + 1
        let z = t.sin(p)*t.cos(x/p) + t.exp(p/10) + t.log(p) + t.sqrt(p)
        let w = ad.tanh(x) + ad.pow(x,2.5)
        let loss = t.mean((z+w)*y)
        print(t.tolist(ad.hessian(loss,x)))
        """, stress=True)
        result = ast.literal_eval(text)
        point = jnp.array([[0.5, 1.2], [2.0, 0.9]])

        def loss(x):
            y = x @ x.T
            p = x*x + 1
            z = jnp.sin(p)*jnp.cos(x/p) + jnp.exp(p/10) + jnp.log(p) + jnp.sqrt(p)
            w = jnp.tanh(x) + x ** 2.5
            return jnp.mean((z+w)*y)

        np.testing.assert_allclose(result, jax.hessian(loss)(point), rtol=3e-13, atol=3e-14)

    def test_sparse_csr_uses_the_same_gradient_as_jax_bcoo(self):
        from jax.experimental.sparse import BCOO

        text = self.run_code("""
        import tensor as t
        import sparse as sp
        import ad
        let pattern = sp.csr(3,2,[0,1,1,3],[0,0,1],[1,2,3])
        let values = ad.variable(t.array([2,0,5],"float64"))
        let x = ad.variable(t.array([2,3],"float64"))
        let out = sp.matvec(pattern,values,x)
        let loss = t.sum(out*out)
        ad.backward(loss)
        print(t.tolist(ad.grad(values)))
        print(t.tolist(ad.grad(x)))
        print(t.tolist(ad.hessian(loss,values)))
        """, stress=True)
        values, inputs, hessian = map(ast.literal_eval, text.splitlines())
        indices = jnp.array([[0, 0], [2, 0], [2, 1]])
        point, x = jnp.array([2.0, 0.0, 5.0]), jnp.array([2.0, 3.0])

        def loss(a, x):
            y = BCOO((a, indices), shape=(3, 2)) @ x
            return jnp.sum(y*y)

        ev, ex = jax.grad(loss, argnums=(0, 1))(point, x)
        np.testing.assert_allclose(values, ev, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(inputs, ex, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(hessian, jax.hessian(loss, argnums=0)(point, x),
                                   rtol=1e-13, atol=1e-14)

    def test_checkpoint_composition_matches_cpu_jax(self):
        source = """
        import ad
        import tensor as t
        let x = ad.variable(t.array([0.3,0.8,1.2],"float64"))
        let bias = ad.variable(0.37)
        let state = x
        for i in range(4):
            let step = t.sin(state*state + bias) / (state + 2) + bias * state
            state = ad.checkpoint(step,state,bias)
            collect()
        let loss = t.sum(state)
        print(t.tolist(ad.vjp(loss,x)))
        print(ad.vjp(loss,bias))
        print(t.tolist(ad.hessian(loss,x)))
        print(ad.hessian(loss,bias))
        print(ad.jvp(loss,x,t.array([1,2,-1],"float64")))
        """
        g, b, h, h_bias, jvp = map(ast.literal_eval, self.run_code(source, stress=True).splitlines())
        x, bias = jnp.array([0.3, 0.8, 1.2]), 0.37

        def loss(state, b):
            for _ in range(4):
                state = jnp.sin(state*state + b) / (state + 2) + b * state
            return state.sum()

        dx, db = jax.grad(loss, argnums=(0, 1))(x, bias)
        np.testing.assert_allclose(g, dx, rtol=3e-13, atol=3e-14)
        np.testing.assert_allclose(b, db, rtol=3e-13, atol=3e-14)
        np.testing.assert_allclose(h, jax.hessian(loss, argnums=0)(x, bias),
                                   rtol=3e-13, atol=3e-14)
        np.testing.assert_allclose(h_bias, jax.hessian(loss, argnums=1)(x, bias),
                                   rtol=3e-13, atol=3e-14)
        np.testing.assert_allclose(jvp, dx @ jnp.array([1, 2, -1]), rtol=3e-13, atol=3e-14)

    def test_offload_and_checkpoints_match_cpu_jax(self):
        output = self.run_code("""
        import ad
        import tensor as t
        let plan = ad.offload_plan(".",1024,2)
        let x = ad.variable(t.array([0.1,0.3,0.7],"float64"))
        let state = ad.offload(t.exp(x),plan)
        let activation = ad.offload(ad.checkpoint(t.sin(state*state + 0.2),state),plan)
        let result = t.sum(activation*t.exp(activation))
        collect()
        print(ad.value(result))
        print(t.tolist(ad.vjp(result,x)))
        print(ad.jvp(result,x,t.array([1,2,-1],"float64")))
        print(t.tolist(ad.hessian(result,x)))
        let stats = ad.offload_stats(plan)
        print(stats["files_written"],stats["bytes_read"] > 0,stats["vm_checkpoint_replay_operations"])
        """, stress=True).splitlines()
        value, grad, tangent, hessian = map(ast.literal_eval, output[:4])
        self.assertEqual(output[4], "2 nae 81")
        x = jnp.array([0.1, 0.3, 0.7])

        def loss(x):
            state = jnp.exp(x)
            activation = jnp.sin(state * state + 0.2)
            return (activation * jnp.exp(activation)).sum()

        expected_value, expected_grad = jax.value_and_grad(loss)(x)
        expected_tangent = jax.jvp(loss, (x,), (jnp.array([1.0, 2.0, -1.0]),))[1]
        np.testing.assert_allclose(value, expected_value, rtol=3e-13, atol=3e-14)
        np.testing.assert_allclose(grad, expected_grad, rtol=3e-13, atol=3e-14)
        np.testing.assert_allclose(tangent, expected_tangent, rtol=3e-13, atol=3e-14)
        np.testing.assert_allclose(hessian, jax.hessian(loss)(x), rtol=3e-13, atol=3e-14)

    def test_hessian_vector_product_matches_jax_with_spilled_checkpoints(self):
        output = self.run_code("""
        import ad
        import tensor as t
        let x = ad.variable(t.array([0.1,0.3,0.7],"float64"))
        let plan = ad.offload_plan(".",1024,2)
        let state = ad.offload(t.exp(x),plan)
        let activation = ad.offload(ad.checkpoint(t.sin(state*state+0.2),state),plan)
        let loss = t.sum(activation*t.exp(activation)) + t.sum(activation)*t.sum(state)
        print(t.tolist(ad.hvp(loss,x,t.array([1,2,-1],"float64"))))
        """, stress=True)
        x = jnp.array([0.1, 0.3, 0.7])

        def loss(x):
            state = jnp.exp(x)
            activation = jnp.sin(state * state + 0.2)
            return (activation * jnp.exp(activation)).sum() + activation.sum() * state.sum()

        expected = jax.jvp(jax.grad(loss), (x,), (jnp.array([1.0, 2.0, -1.0]),))[1]
        np.testing.assert_allclose(ast.literal_eval(output), expected, rtol=4e-13, atol=3e-14)
