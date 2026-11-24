"""Static callable documentation, binding and shape-based cost models."""
import unittest

import test_arda


class MathDocTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_native_tex_levels_and_aliases(self):
        output = self.run_code("""
        tol numc as nc
        tol linalg as la
        tol mathdoc
        esta solve = nc.solve
        pedo(solve.getTex(),mathdoc.getTex(nc.solve,0))
        pedo(solve.getTex(1))
        pedo(solve.getTex(2))
        pedo(solve.getTex(3))
        pedo(solve.getAlgorithm())
        pedo(solve.getAssumptions())
        pedo(solve.getComplexity())
        pedo(la.solve.getTex(0))
        """, stress=True).splitlines()
        self.assertEqual(output[0], "Ax=b Ax=b")
        self.assertEqual(output[1], r"PA=LU,\quad Ly=Pb,\quad Ux=y")
        self.assertIn(r"\mathbb{R}^{n\times n}", output[2])
        self.assertTrue(output[3].startswith(r"\begin{aligned}"))
        self.assertIn(r"\operatorname{arg\,max}", output[3])
        self.assertIn(r"a_{ij}&\gets a_{ij}-\ell_{ik}a_{kj}", output[3])
        self.assertNotIn(r"A^{-1}", output[3])
        self.assertIn("partial row pivoting", output[4])
        self.assertIn("zero pivot", output[5])
        self.assertIn("O(n^3)", output[6])
        self.assertIn(r"\min_X", output[7])

    def test_no_execution_and_bound_query_gc(self):
        self.run_code("""
        tol math as m
        tol mathdoc
        esta calls = 0
        caro effect(x):
            calls = calls+1
            throw "executed target"
        esta ask = effect.getTex
        effect = mun
        collect()
        pedo(ask(3),calls)
        esta cosine = m.cos.getTex
        m = mun
        collect()
        pedo(cosine(0),cosine(3))
        pedo(len.getAlgorithm(),mathdoc.getCost(cos,{}))
        nost Counter:
            caro init(self):
                self.n = 0
            caro step(self):
                self.n = self.n+1
                nan self.n
        esta counter = Counter()
        esta step = counter.step
        pedo(step.getTex(),counter.n,step())
        """, "mun 0\n\\cos(x) mun\nmun mun\nmun 0 1", stress=True)

    def test_selected_backend_and_provider_boundary(self):
        output = self.run_code("""
        tol numc as nc
        tol linalg as la
        tol calc
        pedo(cos.getTex(),nc.cos.getTex())
        pedo(cos.getAlgorithm())
        pedo(cos.getTex(1),cos.getTex(3),cos.getCost({}))
        pedo(nc.backend())
        pedo(nc.matmul.getAlgorithm())
        pedo(nc.matmul.getTex(3))
        pedo(calc.backend())
        pedo(la.rsvd.getAlgorithm())
        pedo(la.rsvd.getTex(3))
        """, stress=True).splitlines()
        self.assertEqual(output[0], r"\cos(x) Y_i=\cos(X_i)")
        self.assertIn("platform C libm cos", output[1])
        self.assertEqual(output[2], 'mun mun mun')
        if output[3] == 'cblas':
            self.assertIn('CBLAS dgemm', output[4])
            self.assertEqual(output[5], r"C=AB\quad\text{via CBLAS dgemm}")
        else:
            self.assertIn('Blocked row-major', output[4])
            self.assertIn(r"\gets", output[5])
        if output[6] == 'lapack':
            self.assertIn('DGESVD', output[7])
            self.assertIn(r"A^TQ=", output[8])
        else:
            self.assertIn('unavailable', output[7])
            self.assertEqual(output[8], 'mun')

    def test_costs_with_shape_only(self):
        self.run_code("""
        tol numc as nc
        tol sparse as sp
        tol mathdoc
        esta solve = nc.solve.getCost({"shape":[2,2]})
        pedo(solve["estimated"],solve["flops"],solve["output_bytes"],solve["workspace_bytes"]>32,solve["backend"])
        esta product = mathdoc.getCost(nc.matmul,{"shape":[2,3,4],"dtype":"float64"})
        pedo(product["flops"],product["output_bytes"],product["workspace_bytes"],product["backend"]==nc.backend())
        esta dot = nc.dot.getCost({"shape":[9]})
        pedo(dot["flops"],dot["output_bytes"])
        esta csr = sp.matvec.getCost({"shape":[1000000,1000000],"nnz":2999998})
        pedo(csr["flops"],csr["output_bytes"],csr["workspace_bytes"])
        pedo(sp.matvec.getCost({"shape":[4,4],"format":"coo"}))
        pedo(nc.solve.getCost({"shape":[0,0]})["flops"])
        """, "nae 9 16 nae portable-c\n48 64 0 nae\n18 8\n5999996 8000000 0\nmun\n0",
                      stress=True)

    def test_invalid_queries(self):
        imports = "tol numc as nc\ntol sparse as sp\ntol mathdoc\n"
        for expression, error in [
            ('cos.getTex(4)', 'level must be an integer from 0 to 3'),
            ('nc.solve.getCost({"shape":[2,3]})', 'requires a square matrix'),
            ('nc.matmul.getCost({"shape":[16000000,2,1]})', 'shape exceeds the tensor'),
            ('nc.dot.getCost({"shape":[2.0]})', 'dimensions must be nonnegative integers'),
            ('nc.dot.getCost({"shape":[2],"dtype":"float32"})', 'require dtype float64'),
            ('sp.matvec.getCost({"shape":[2,3],"nnz":7})', 'nnz within the supplied shape'),
            ('mathdoc.getAlgorithm(3)', 'expects a function or bound method'),
        ]:
            with self.subTest(expression=expression):
                self.run_code(imports + expression, error=error, stress=True)
