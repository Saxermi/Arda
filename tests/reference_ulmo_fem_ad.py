"""Independent sparse assembly, manufactured solutions and discrete sensitivities."""
import ast
import math
import unittest

import test_arda

try:
    import numpy as np
    from scipy.integrate import quad
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy and SciPy are required")
class FemAndAdjointReferences(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_triangle_reaction_diffusion_against_independent_quadrature(self):
        xy = np.array([[0, 0], [1, 0], [1.2, 1], [-0.1, 1.2], [0.52, 0.4]])
        cells = [[0, 1, 4], [1, 4, 2], [2, 3, 4], [3, 4, 0]]
        diffusion = [2, 0.2, 1.8, 4]
        reaction = [0.8, 2, 0.1, 3]
        f = np.array([1, -2, 4, 0.2, 0])
        fixed = {0: 0.3, 1: -0.5, 3: 1.3}
        rows, cols, values = [], [], []
        rhs = np.zeros(len(xy))
        # This three-point triangle quadrature integrates all degree-two products.
        basis = np.full((3, 3), 1/6) + np.eye(3)/2
        for ids, k, c in zip(cells, diffusion, reaction):
            x = np.column_stack((np.ones(3), xy[ids]))
            area = abs(np.linalg.det(x))/2
            grad = np.linalg.inv(x)[1:, :]
            local = area * (k * (grad.T @ grad) + c * (basis.T @ basis)/3)
            load = area * (basis.T @ (basis @ f[ids]))/3
            for i, row in enumerate(ids):
                if row in fixed:
                    continue
                rhs[row] += load[i]
                for j, col in enumerate(ids):
                    if col in fixed:
                        rhs[row] -= local[i, j]*fixed[col]
                    else:
                        rows.append(row)
                        cols.append(col)
                        values.append(local[i, j])
        for i, d in fixed.items():
            rows.append(i)
            cols.append(i)
            values.append(1)
            rhs[i] = d
        a = coo_matrix((values, (rows, cols)), shape=(len(xy), len(xy))).tocsr()
        output = self.run_code(f"""
import fem
import numc as n
import solver
import linop
let m = fem.mesh(n.array({xy.tolist()}),{cells})
let s = fem.assemble(m,n.array({f.tolist()}),{list(map(list, fixed.items()))},
                     {{"diffusivity":n.array({diffusion}),"reaction":n.array({reaction})}})
print(n.tolist(s["rhs"]))
for i in range(5):
    let e = n.zeros(5)
    e[i] = 1
    print(n.tolist(linop.apply(s["matrix"],e)))
let fit = solver.solve("cg",s["matrix"],s["rhs"],{{"tol":1e-13}})
print(n.tolist(fit["solution"]))
""", stress=True)
        actual = np.asarray([ast.literal_eval(line) for line in output.splitlines()])
        np.testing.assert_allclose(actual[0], rhs, rtol=3e-14, atol=3e-14)
        np.testing.assert_allclose(actual[1:6].T, a.toarray(), rtol=3e-14, atol=3e-14)
        np.testing.assert_allclose(actual[6], spsolve(a, rhs), rtol=3e-13, atol=3e-13)

    def test_adaptive_line_estimator_and_integrated_convergence(self):
        output = self.run_code("""
import fem
import solver
import numc as n
let m = fem.mesh(n.array([0,0.5,1]),[[0,1],[1,2]])
for level in range(4):
    let points = fem.nodes(m)
    let bc = [[0,0],[2,0]]
    let a = fem.assemble(m,2,bc)
    let r = solver.solve("cg",a["matrix"],a["rhs"],{"tol":1e-13})
    let u = r["solution"]
    let eta = fem.estimate(m,2,bc,u)
    print(n.tolist(points))
    print(n.tolist(u))
    print(fem.elements(m))
    print(n.sum(eta))
    let f = fem.refine(m,u,eta,{"fraction":1})
    m = f["mesh"]
""")
        lines = output.splitlines()
        error, estimates = [], []
        for row in range(0, len(lines), 4):
            x = ast.literal_eval(lines[row])
            u = ast.literal_eval(lines[row+1])
            cells = ast.literal_eval(lines[row+2])
            estimates.append(math.sqrt(float(lines[row+3])))
            norm2 = 0
            for a, b in cells:
                self.assertAlmostEqual(u[a], x[a]*(1-x[a]), places=12)
                slope = (u[b]-u[a])/(x[b]-x[a])
                norm2 += quad(lambda s: (s*(1-s) - u[a] - (s-x[a])*slope)**2,
                              x[a], x[b], epsabs=1e-15)[0]
            error.append(math.sqrt(norm2))
        np.testing.assert_allclose(np.asarray(error[:-1])/error[1:], 4, rtol=2e-11)
        ratio = np.asarray(estimates[:-1])/estimates[1:]
        self.assertTrue(np.all((1.8 < ratio) & (ratio < 2.1)), ratio)

    def test_triangular_poisson_grid_convergence(self):
        output = self.run_code("""
import fem
import solver
import numc as n
let pi = 3.141592653589793
for side in [8,16,32]:
    let width = side+1
    let x = n.zeros(width*width,2)
    let exact = n.zeros(width*width)
    let cells = []
    let bc = []
    for j in range(width):
        for i in range(width):
            let a = j*width+i
            n.put(x,[a,0],i/side)
            n.put(x,[a,1],j/side)
            if i == 0 or j == 0 or i == side or j == side:
                push(bc,[a,0])
            else:
                exact[a] = sin(pi*i/side)*sin(pi*j/side)
            if i < side and j < side:
                push(cells,[a,a+1,a+width+1])
                push(cells,[a,a+width+1,a+width])
    let m = fem.mesh(x,cells)
    let s = fem.assemble(m,2*pi*pi*exact,bc)
    let r = solver.solve("cg",s["matrix"],s["rhs"],{"tol":1e-12})
    print(n.norm(r["solution"]-exact)/n.norm(exact),r["residual"])
""")
        error, residual = np.loadtxt(output.splitlines()).T
        self.assertLess(residual.max(), 1e-12)
        self.assertTrue(np.all((3.65 < error[:-1]/error[1:]) & (error[:-1]/error[1:] < 4.3)),
                        error)

    def test_ode_directions_and_adjoint_against_analytic_and_centered_differences(self):
        output = self.run_code("""
import sim
import numc as n
def f(t,y,p):
    return (t-p)*y
let m = sim.rk4(f)
let y = n.array([2,-0.6,1.3])
let p = n.array([0.2,-0.4,0.9])
let dy = n.array([-0.3,0.6,0.2])
let dp = n.array([0.5,-0.2,0.3])
let seed = n.array([1.5,0.3,-0.8])
let j = sim.jvp(m,0.1,y,p,0.97,0.004,dy,dp)
let a = sim.adjoint(m,0.1,y,p,0.97,0.004,seed,{"checkpoint_stride":11})
let e = 1e-5
let upper = sim.solve(m,0.1,y+e*dy,p+e*dp,0.97,0.004)["solution"]
let lower = sim.solve(m,0.1,y-e*dy,p-e*dp,0.97,0.004)["solution"]
print(n.tolist(j["solution"]))
print(n.tolist(j["tangent"]))
print(n.tolist(a["initial_gradient"]))
print(n.tolist(a["parameter_gradient"]))
print(n.tolist((upper-lower)/(2*e)))
""")
        z, tangent, ygrad, pgrad, differences = map(ast.literal_eval, output.splitlines())
        t, end = 0.1, 0.97
        y = np.array([2, -0.6, 1.3])
        p = np.array([0.2, -0.4, 0.9])
        seed = np.array([1.5, 0.3, -0.8])
        dy, dp = np.array([-0.3, 0.6, 0.2]), np.array([0.5, -0.2, 0.3])
        exact_factor = np.exp((end**2 - t**2)/2 - (end-t)*p)
        np.testing.assert_allclose(z, y*exact_factor, rtol=3e-10)
        np.testing.assert_allclose(tangent, exact_factor*(dy-y*(end-t)*dp), rtol=3e-10)
        np.testing.assert_allclose(ygrad, seed*exact_factor, rtol=3e-10)
        np.testing.assert_allclose(pgrad, -(end-t)*seed*y*exact_factor, rtol=3e-10)
        np.testing.assert_allclose(tangent, differences, rtol=8e-9)
        self.assertAlmostEqual(np.dot(seed, tangent), np.dot(ygrad, dy)+np.dot(pgrad, dp), places=12)

    def test_logistic_parameter_derivative_and_fourth_order_refinement(self):
        output = self.run_code("""
import sim
def rhs(t,y,p):
    return p*y*(1-y)
let model = sim.rk4(rhs)
for h in [0.2,0.1,0.05]:
    let a = sim.adjoint(model,0,0.2,1.4,1,h,1)
    print(a["solution"],a["parameter_gradient"])
""")
        actual = np.loadtxt(output.splitlines())
        terminal = 1/(1+4*math.exp(-1.4))
        expected = [terminal, terminal*(1-terminal)]
        error = np.abs(actual-expected)
        ratios = error[:-1]/error[1:]
        self.assertTrue(np.all((13 < ratios) & (ratios < 19)), ratios)

    def test_heat_discrete_fourier_derivative_and_mass_invariance(self):
        output = self.run_code("""
import sim
import pde
import numc as n
let m = sim.heat(pde.grid([4],0.25,"periodic"))
let y = n.array([1,0,-1,0])
let a = sim.adjoint(m,0,y,0.13,0.43,0.07,y,{"checkpoint_stride":3})
let j = sim.jvp(m,0,y,0.13,0.43,0.07,n.ones(4),1)
let ones = sim.adjoint(m,0,y,0.13,0.43,0.07,n.ones(4))
let e = 1e-5
let plus = sim.solve(m,0,y+e,0.13+e,0.43,0.07)["solution"]
let minus = sim.solve(m,0,y-e,0.13-e,0.43,0.07)["solution"]
print(n.tolist(a["solution"]))
print(n.tolist(a["initial_gradient"]))
print(a["parameter_gradient"])
print(n.tolist(j["tangent"]))
print(ones["parameter_gradient"])
print(n.tolist((plus-minus)/(2*e)))
""", stress=True)
        z, y_grad, p_grad, direction, mass_grad, fd = map(ast.literal_eval, output.splitlines())
        y = np.array([1, 0, -1, 0])
        multipliers = 1-32*0.13*np.array([0.07]*6+[0.01])
        amplitude = multipliers.prod()
        derivative = -32*amplitude*np.sum(np.array([0.07]*6+[0.01])/multipliers)
        np.testing.assert_allclose(z, amplitude*y, atol=2e-15)
        np.testing.assert_allclose(y_grad, amplitude*y, atol=2e-15)
        self.assertAlmostEqual(p_grad, derivative*2, places=13)
        self.assertAlmostEqual(mass_grad, 0, places=14)
        np.testing.assert_allclose(direction, 1+derivative*y, atol=2e-14)
        np.testing.assert_allclose(direction, fd, atol=1e-8)
