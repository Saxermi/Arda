"""Independent SciPy comparisons for sparse conversion, algebra and solvers."""
import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg
import test_arda


class SparseReferenceTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_random_conversion(self):
        rng = np.random.default_rng(1400)
        code = ['tol sparse as sp', 'tol numc as nc']
        expected = []
        for rows, cols, n in [(0,3,0),(4,0,0),(1,1,40),(3,11,50),(30,7,250),(100,100,1200)]:
            for repeat in range(4):
                r = rng.integers(0, rows, n).tolist() if n else []
                c = rng.integers(0, cols, n).tolist() if n else []
                x = rng.integers(-9, 10, n).tolist()
                ref = coo_matrix((np.array(x,dtype=float),(np.array(r,dtype=int),np.array(c,dtype=int))), shape=(rows,cols)).tocsr()
                ref.eliminate_zeros()
                code += [f'esta a = sp.coo({rows},{cols},{r},{c},{x})', 'esta b = sp.to_csr(a)', 'pedo(sp.components(b))', 'pedo(sp.components(sp.to_csr(sp.to_coo(b))))']
                expected += [[ref.indptr.tolist(),ref.indices.tolist(),ref.data.tolist()]]*2
        actual = [ast.literal_eval(line) for line in self.run_code('\n'.join(code),stress=True).splitlines()]
        self.assertEqual(actual,expected)

    def test_algebra_against_scipy(self):
        rng = np.random.default_rng(141)
        for rows, cols in [(1, 1), (2, 7), (12, 3), (31, 25)]:
            with self.subTest(shape=(rows, cols)):
                n = rows * cols * 2
                r, c = rng.integers(rows, size=n), rng.integers(cols, size=n)
                x = rng.integers(-4, 5, size=n)
                a = coo_matrix((x.astype(float), (r, c)), shape=(rows, cols)).tocsr()
                a.eliminate_zeros()
                v = rng.normal(size=cols)
                m = rng.normal(size=(cols, 3))
                output = self.run_code(f"""
                tol sparse as sp
                tol numc as nc
                esta b = sp.builder({rows},{cols})
                sp.extend(b,{r.tolist()},{c.tolist()},{x.tolist()})
                esta a = sp.finish(b)
                pedo(sp.components(a))
                pedo(nc.tolist(sp.matvec(a,nc.array({v.tolist()}))))
                pedo(nc.tolist(sp.spmm(a,nc.array({m.tolist()}))))
                pedo(nc.tolist(sp.diagonal(a)))
                pedo(nc.tolist(sp.todense(sp.add(a,sp.scale(a,-0.25)))))
                pedo(nc.tolist(sp.todense(sp.transpose(a))))
                """, stress=True).splitlines()
                self.assertEqual(ast.literal_eval(output[0]), [
                    a.indptr.tolist(), a.indices.tolist(), a.data.tolist(),
                ])
                expected = [a @ v, a @ m, a.diagonal(), 0.75 * a.toarray(), a.T.toarray()]
                for line, ref in zip(output[1:], expected, strict=True):
                    np.testing.assert_allclose(ast.literal_eval(line), ref, rtol=5e-13, atol=5e-14)

    def test_cg_and_pcg_against_scipy(self):
        rng = np.random.default_rng(143)
        n = 123
        edge = rng.uniform(-1, 0, n - 1)
        main = rng.uniform(2.5, 12, n)
        a = diags([edge, main, edge], [-1, 0, 1], format="csr")
        reference = rng.normal(size=n)
        b = a @ reference
        output = self.run_code(f"""
        tol sparse as sp
        tol numc as nc
        esta a = sp.tridiag({n},{edge.tolist()},{main.tolist()},{edge.tolist()})
        esta b = nc.array({b.tolist()})
        tol solver as s
        esta u = s.solve("cg",a,b,{{"tol":1e-11,"maxiter":100}})
        esta v = s.solve("cg",a,b,{{"tol":1e-11,"maxiter":100,"preconditioner":"jacobi"}})
        pedo(u["reason"],v["reason"],v["iterations"] < u["iterations"])
        pedo(u["residual"],v["residual"])
        pedo(nc.tolist(u["solution"]))
        pedo(nc.tolist(v["solution"]))
        """, stress=True).splitlines()
        self.assertEqual(output[0], "converged converged nae")
        residuals = [float(value) for value in output[1].split()]
        for line, reported, m in zip(output[2:], residuals, [None, diags(1 / main)], strict=True):
            actual = ast.literal_eval(line)
            solution, info = cg(a, b, rtol=1e-11, atol=0, maxiter=100, M=m)
            self.assertEqual(info, 0)
            np.testing.assert_allclose(actual, solution, rtol=5e-10, atol=5e-11)
            measured = np.linalg.norm(b - a @ actual) / np.linalg.norm(b)
            self.assertLess(measured, 1.001e-11)
            self.assertAlmostEqual(measured, reported, delta=5e-15)

    def test_matrix_market_interoperability(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "scipy.mtx", root / "arda.mtx"
            matrix = diags([-1.0, 4.0, -1.0], [-1, 0, 1], shape=(7, 7))
            mmwrite(source, matrix, symmetry="symmetric")
            self.run_code(f"""
            tol sparse as sp
            esta a = sp.read_mm({json.dumps(str(source))})
            sp.write_mm({json.dumps(str(output))},a)
            """, stress=True)
            np.testing.assert_array_equal(mmread(output).toarray(), matrix.toarray())

    def test_pcg_scaled_spd(self):
        n = 48
        scale = np.geomspace(1, 1000, n)
        edge = -(scale[:-1] * scale[1:])
        main = 4 * scale**2
        a = diags([edge, main, edge], [-1, 0, 1], format="csr")
        exact = np.random.default_rng(144).normal(size=n) / scale
        b = a @ exact
        output = self.run_code(f"""
        tol sparse as sp
        tol numc as nc
        esta a = sp.tridiag({n},{edge.tolist()},{main.tolist()},{edge.tolist()})
        esta b = nc.array({b.tolist()})
        tol solver as s
        esta r = s.solve("cg",a,b,{{"tol":1e-12,"maxiter":150,"preconditioner":"jacobi"}})
        pedo(r["reason"],r["residual"])
        pedo(nc.tolist(r["solution"]))
        """, stress=True).splitlines()
        reason, reported = output[0].split()
        self.assertEqual(reason, "converged")
        x = np.asarray(ast.literal_eval(output[1]))
        measured = np.linalg.norm(b - a @ x) / np.linalg.norm(b)
        self.assertLess(measured, 1.01e-12)
        self.assertAlmostEqual(measured, float(reported), delta=5e-15)
        np.testing.assert_allclose(x * scale, exact * scale, rtol=1e-8, atol=1e-8)
