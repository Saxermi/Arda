"""Independent SciPy and analytic references for Ulmo ODEs and PDEs."""
import ast
import unittest

import test_arda

try:
    import numpy as np
    from scipy.integrate import solve_ivp
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy and SciPy are required")
class UlmoReferenceTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_rk45_irregular_output_against_scipy(self):
        times = np.array([0, 0.013, 0.11, 0.272, 0.52, 0.931, 1.17, 1.4])
        output = self.run_code(f"""
import ode
import numc as n
def rhs(t,y):
    return n.array([y[1],-2*y[0]-0.4*y[1]])
let r = ode.rk45(rhs,0,n.array([1,0.2]),1.4,0.19,1e-10,1e-12,300,
                 {{"saveat":{times.tolist()}}})
for state in r["states"]:
    print(n.tolist(state))
""", stress=True)
        actual = np.asarray([ast.literal_eval(line) for line in output.splitlines()])
        expected = solve_ivp(lambda t, y: [y[1], -2*y[0] - 0.4*y[1]], (0, 1.4), [1, 0.2],
                             rtol=1e-12, atol=1e-14, t_eval=times)
        self.assertTrue(expected.success)
        np.testing.assert_allclose(actual, expected.y.T, rtol=3e-9, atol=5e-10)

    def test_stiff_robertson_radau_against_scipy(self):
        def rhs(t, y):
            a = 0.04*y[0]
            b = 1e4*y[1]*y[2]
            c = 3e7*y[1]**2
            return [-a+b, a-b-c, c]

        expected = solve_ivp(rhs, (0, 0.025), [1, 0, 0], method="Radau",
                             rtol=1e-11, atol=1e-14).y[:, -1]
        output = self.run_code("""
import ode
import numc as n
def rhs(t,y):
    let a = 0.04*y[0]
    let b = 1e4*y[1]*y[2]
    let c = 3e7*y[1]*y[1]
    return n.array([-a+b,a-b-c,c])
def jac(t,y):
    let b = 1e4*y[2]
    let c = 1e4*y[1]
    let d = 6e7*y[1]
    return n.array([[-0.04,b,c],[0.04,-b-d,-c],[0,d,0]])
let r = ode.radau(rhs,0,n.array([1,0,0]),0.025,0.00025,
                  {"jacobian":jac,"atol":1e-13,"rtol":1e-12})
print(n.tolist(r["solution"]))
""")
        actual = np.asarray(ast.literal_eval(output))
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-10)
        self.assertAlmostEqual(actual.sum(), 1, places=11)

    def test_implicit_order_under_step_refinement(self):
        output = self.run_code("""
import ode
import ad
def rhs(t,y):
    return -y
for h in [0.1,0.05,0.025]:
    let e = ode.implicit_euler(rhs,0,1.0,1,h)["solution"]
    let b = ode.bdf2(rhs,0,1.0,1,h)["solution"]
    let r = ode.radau(rhs,0,1.0,1,h)["solution"]
    print([abs(e-ad.exp(-1)),abs(b-ad.exp(-1)),abs(r-ad.exp(-1))])
""")
        error = np.asarray([ast.literal_eval(line) for line in output.splitlines()])
        orders = np.log2(error[:-1] / error[1:])
        np.testing.assert_allclose(orders[-1], [1, 2, 3], atol=0.15)

    def test_poisson_mixed_boundaries_against_kronecker_reference(self):
        rng = np.random.default_rng(202)
        exact = rng.normal(size=35)
        a = 2*np.eye(5)-np.eye(5, k=1)-np.eye(5, k=-1)
        b = 2*np.eye(7)-np.eye(7, k=1)-np.eye(7, k=-1)
        b[0, -1] = b[-1, 0] = -1
        matrix = np.kron(a/0.2**2, np.eye(7)) + np.kron(np.eye(5), b/0.3**2)
        rhs = matrix @ exact
        output = self.run_code(f"""
import pde
import numc as n
import linop as l
let grid = pde.grid([5,7],[0.2,0.3],["dirichlet","periodic"])
let result = pde.poisson(grid,n.array({rhs.tolist()}),{{"tol":1e-12}})
print(n.tolist(result["solution"]))
print(n.tolist(l.apply(grid,n.array({exact.tolist()}))))
""", stress=True).splitlines()
        np.testing.assert_allclose(ast.literal_eval(output[0]), exact, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(ast.literal_eval(output[1]), rhs, rtol=1e-12, atol=1e-11)

    def test_heat_spatial_and_wave_temporal_convergence(self):
        heat_errors = []
        for n in (16, 32, 64):
            h = 1/(n+1)
            mode = np.sin(np.pi*np.arange(1, n+1)*h)
            dt = 0.1*h*h/0.4
            output = self.run_code(f"""
import pde
import numc as n
let r = pde.heat(pde.grid([{n}],{h}),n.array({mode.tolist()}),0.4,0.02,{dt})
print(n.tolist(r["solution"]))
""")
            exact = mode*np.exp(-0.4*np.pi**2*0.02)
            heat_errors.append(np.linalg.norm(np.asarray(ast.literal_eval(output))-exact)/np.linalg.norm(exact))
        order = np.log(heat_errors[-2]/heat_errors[-1])/np.log(65/33)
        self.assertTrue(1.85 < order < 2.15, heat_errors)

        n = 32
        initial = np.cos(2*np.pi*np.arange(n)/n)
        omega = 2*n*np.sin(np.pi/n)
        exact = initial*np.cos(omega*0.2)
        wave_errors = []
        for dt in (0.02, 0.01, 0.005):
            output = self.run_code(f"""
import pde
import numc as n
let g = pde.grid([{n}],{1/n},"periodic")
let r = pde.wave(g,n.array({initial.tolist()}),n.zeros({n}),1,0.2,{dt})
print(n.tolist(r["solution"]))
""")
            wave_errors.append(np.linalg.norm(np.asarray(ast.literal_eval(output))-exact))
        self.assertTrue(3.9 < wave_errors[-2]/wave_errors[-1] < 4.1, wave_errors)
