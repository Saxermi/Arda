"""Independent numerical references for the alternative Ulmo algorithms."""
import ast
from collections import defaultdict
import math
import unittest

import test_arda

try:
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy and SciPy are required")
class UlmoAlternativesReferences(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_mixed_darcy_against_independent_quadrature_and_sparse_solve(self):
        xy = np.array([[0, 0], [1.1, 0], [1.2, 1.1], [-0.1, 1.2], [0.4, 0.35]])
        cells = [[0, 1, 4], [1, 4, 2], [2, 3, 4], [3, 4, 0]]
        mobility = [0.2, 1.2, 3.0, 0.6]
        f = np.array([0.9, 2.0, -1.0, 0.5, 1.4])
        prescribed = {0: -0.2, 1: 1.4, 3: 0.5}
        groups = defaultdict(list)
        for e, cell in enumerate(cells):
            for i in range(3):
                group = tuple(sorted(cell[(i+j) % 3] for j in (1, 2)))
                groups[group].append((e, i))
        active = {edge: owners for edge, owners in groups.items()
                  if len(owners) == 2 or all(n in prescribed for n in edge)}
        count = len(active)
        indices = np.full((len(cells), 3), -1)
        signs = np.zeros_like(indices)
        rhs = np.zeros(count+len(cells))
        for k, (edge, owners) in enumerate(active.items()):
            for orient, (e, i) in zip((1, -1), owners):
                indices[e, i], signs[e, i] = k, orient
            if len(owners) == 1:
                rhs[k] = -np.mean([prescribed[n] for n in edge])

        rows, cols, values = [], [], []

        def entry(row, column, value):
            rows.append(row)
            cols.append(column)
            values.append(value)

        xg, wg = np.polynomial.legendre.leggauss(3)
        xg, wg = (xg+1)/2, wg/2
        samples = []
        for e, cell in enumerate(cells):
            verts = xy[cell]
            area = abs(np.linalg.det((verts[1:]-verts[0]).T))/2
            mass = np.zeros((3, 3))
            # Product Gauss rule on a Duffy image of the reference square.
            for s, ws in zip(xg, wg):
                for t, wt in zip(xg, wg):
                    point = (1-s)*(1-t)*verts[0] + s*verts[1] + (1-s)*t*verts[2]
                    basis = (point-verts)/(2*area)
                    mass += (2*area*(1-s)*ws*wt/mobility[e])*(basis @ basis.T)
            mean = area*f[cell].mean()
            rhs[count+e] = -mean
            for i, row in enumerate(indices[e]):
                if row < 0:
                    continue
                entry(row, count+e, -signs[e, i])
                entry(count+e, row, -signs[e, i])
                for j, col in enumerate(indices[e]):
                    if col >= 0:
                        entry(row, col, signs[e, i]*signs[e, j]*mass[i, j])
            samples.append(((verts.mean(axis=0)-verts)/(2*area), mean))

        a = coo_matrix((values, (rows, cols)), shape=(len(rhs), len(rhs))).tocsr()
        solution = spsolve(a, rhs)
        exact_flux = solution[indices]*signs
        exact_velocity = [flux @ basis for flux, (basis, _) in zip(exact_flux, samples)]
        output = self.run_code(f"""
import fem
import numc as n
let m = fem.mesh(n.array({xy.tolist()}),{cells})
let r = fem.darcy_mixed(m,n.array({f.tolist()}),{list(map(list,prescribed.items()))},
                       n.array({mobility}),{{"tol":1e-13}})
print(n.tolist(r["pressure"]))
print(n.tolist(r["velocity"]))
print(n.tolist(r["flux"]))
print(r["max_cell_imbalance"])
""", stress=True)
        pressure, velocity, flux, max_error = map(ast.literal_eval, output.splitlines())
        np.testing.assert_allclose(pressure, solution[count:], atol=1e-12)
        np.testing.assert_allclose(velocity, exact_velocity, atol=1e-12)
        np.testing.assert_allclose(flux, exact_flux, atol=1e-12)
        np.testing.assert_allclose(np.sum(flux, axis=1), [mean for _, mean in samples], atol=1e-12)
        self.assertLess(max_error, 1e-12)
        for owners in groups.values():
            if len(owners) == 2:
                (a, i), (b, j) = owners
                self.assertEqual(flux[a][i], -flux[b][j])

    def test_mixed_darcy_first_order_velocity_convergence(self):
        errors = []
        for side in (6, 12, 24):
            width = side+1
            coordinates = [[i/side, j/side] for j in range(width) for i in range(width)]
            triangles = []
            for j in range(side):
                for i in range(side):
                    a = j*width+i
                    triangles.extend(([a, a+1, a+width+1], [a, a+width+1, a+width]))
            boundary = [[j*width, 0] for j in range(width)]
            boundary += [[j*width+side, 1] for j in range(width)]
            output = self.run_code(f"""
import fem
import numc as n
let m = fem.mesh(n.array({coordinates}),{triangles})
let r = fem.darcy_mixed(m,2,{boundary},1,{{"tol":1e-11,"maxiter":4000}})
print(r["max_cell_imbalance"])
print(n.tolist(r["velocity"]))
""")
            balance, velocity = map(ast.literal_eval, output.splitlines())
            self.assertLess(balance, 1e-9)
            centers = np.asarray(coordinates)[triangles].mean(axis=1)
            exact = np.column_stack((2*centers[:, 0]-2, np.zeros(len(centers))))
            errors.append(np.linalg.norm(velocity-exact)/np.linalg.norm(exact))
        np.testing.assert_allclose(np.asarray(errors[:-1])/errors[1:], 2, rtol=0.025)

    def test_nvb_keeps_area_angles_and_conformity_across_generations(self):
        output = self.run_code("""
import fem
import numc as n
let m = fem.mesh(n.array([[0,0],[1,0],[1,1],[0,1]]),[[0,1,2],[0,2,3]])
let f = n.array([1,2,3,4])
for level in range(10):
    let eta = n.zeros(len(fem.elements(m)))
    eta[0] = 1
    let r = fem.refine(m,f,eta,{"method":"nvb"})
    m = r["mesh"]
    f = r["field"]
print(n.tolist(fem.nodes(m)))
print(fem.elements(m))
""", stress=True)
        points, triangles = map(ast.literal_eval, output.splitlines())
        minimum = math.pi
        total_area = 0
        faces = defaultdict(int)
        xy = np.asarray(points)
        for cell in triangles:
            for i in range(3):
                a, b, c = (cell[(i+j) % 3] for j in range(3))
                u, v = xy[b]-xy[a], xy[c]-xy[a]
                det = abs(u[0]*v[1]-u[1]*v[0])
                minimum = min(minimum, math.atan2(det, float(u @ v)))
                faces[tuple(sorted((a, b)))] += 1
                if i == 0:
                    total_area += det/2
        self.assertAlmostEqual(total_area, 1)
        self.assertAlmostEqual(minimum, math.pi/4)
        for (a, b), n in faces.items():
            self.assertIn(n, (1, 2))
            if n == 1:
                self.assertTrue(np.any((xy[a] == xy[b]) & ((xy[a] == 0) | (xy[a] == 1))))
