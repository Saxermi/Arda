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


    def test_user_arithmetic_levels(self):
        output = self.run_code("""
        tol mathdoc
        caro f(x):
            esta y = x*x
            nan y+2
        pedo(f.getTex())
        pedo(mathdoc.getTex(f,1))
        pedo(f.getTex(2))
        pedo(f.getTex(3))
        pedo(f.getAssumptions())
        pedo(f.getAlgorithm())
        pedo(f.getComplexity(),f.getCost({}),f(3))
        """, stress=True).splitlines()
        self.assertEqual(output[0], r"f\left(x\right)=x^{2}+2")
        self.assertEqual(output[1], r"\begin{aligned}y&=x\cdot x\\f\left(x\right)&=y+2\end{aligned}")
        self.assertIn("Numeric operands assumed", output[2])
        self.assertIn(r"y&\gets ", output[3])
        self.assertIn("not inferred", output[4])
        self.assertIn("Straight-line", output[5])
        self.assertEqual(output[6], "mun mun 11")

    def test_precedence_and_names(self):
        output = self.run_code("""
        caro f(a,b,c):
            nan (a+b)*(a-(b-c))/(a/(b+c))
        caro my_fn(x_1):
            nan -(-x_1)
        caro unicode(α):
            nan α+1
        caro minimum():
            nan -9223372036854775808
        caro decimal(x):
            nan 9.81*x+1.0000000000000002+1e-20
        pedo(f.getTex())
        pedo(my_fn.getTex())
        pedo(unicode.getTex())
        pedo(minimum.getTex())
        pedo(decimal.getTex())
        """, stress=True).splitlines()
        self.assertEqual(output[0], r"f\left(a,b,c\right)=\frac{\left(a+b\right)\cdot \left(a-\left(b-c\right)\right)}{\frac{a}{b+c}}")
        self.assertIn(r"\mathrm{my\_fn}", output[1])
        self.assertEqual(output[1], r"\mathrm{my\_fn}\left(\mathrm{x\_1}\right)=\mathrm{x\_1}")
        self.assertIn("[U+03B1]", output[2])
        self.assertTrue(output[3].endswith("=-9223372036854775808"))
        self.assertTrue(output[4].endswith(r"=9.81\cdot x+1.0000000000000002+1\times 10^{-20}"))

    def test_call_provenance_and_no_execution(self):
        output = self.run_code("""
        tol math as m
        esta calls = 0
        caro effect(x):
            calls = calls+1
            nan x
        caro f(x):
            esta y = effect(x)
            nan y+y+m.cos(x)
        esta q = f.getTex
        f = mun
        collect()
        pedo(q(0))
        pedo(q(2))
        pedo(calls)
        """, stress=True).splitlines()
        self.assertIn("Dynamic call targets unresolved", output[0])
        self.assertIn("Dynamic call targets unresolved", output[1])
        self.assertEqual(output[0].count(r"\mathrm{effect}"), 1)
        self.assertEqual(output[2], "0")

    def test_unsupported_and_closure(self):
        output = self.run_code("""
        caro outer(a):
            caro inner(x):
                nan a*x
            nan inner
        esta inner = outer(3)
        esta q = inner.getTex
        inner = mun
        outer = mun
        collect()
        pedo(q(0))
        caro loop(n):
            esta s = 0
            an i in range(n):
                s = s+i
            nan s
        caro default(x=1):
            nan x
        caro reassign(x):
            x = x+1
            nan x
        pedo(loop.getTex(),default.getTex(),reassign.getTex())
        esta g = lambda x: x+1
        pedo(g.getTex(),g(2))
        """, stress=True).splitlines()
        self.assertIn("External values unresolved", output[0])
        self.assertEqual(output[1], "mun mun mun")
        self.assertEqual(output[2], r"\lambda\left(x\right)=x+1 3")

    def test_disabled_user_metadata_in_imports(self):
        self.run_code("""
        tol helper
        caro f(x):
            nan x+1
        pedo(f.getTex(),helper.g.getTex(),cos.getTex(),f(2))
        """, r"mun mun \cos(x) 3", flags=("--mathdoc=none",),
                      files={"helper.arda": "caro g(x):\n    nan x*x\n"}, stress=True)

    def test_metadata_bounds(self):
        expression = "+".join(["x"] * 90)
        self.run_code("caro f(x):\n    nan " + expression + "\npedo(f.getTex(),f(1))",
                      "mun 90", stress=True)
        # The expansion of these declarations is exponential, but the IR
        # stays small and the renderer falls back to named relations.
        declarations = ["    esta a0 = x+x"]
        for i in range(1, 20):
            declarations.append(f"    esta a{i} = a{i-1}+a{i-1}")
        code = "caro f(x):\n" + "\n".join(declarations) + "\n    nan a19\npedo(f.getTex())"
        output = self.run_code(code, stress=True)
        self.assertIn(r"\mathrm{a19}", output)
        self.assertLess(len(output), 8192)


    def test_limited_simplification_is_display_only(self):
        output = self.run_code("""
        caro f(x):
            nan (x*x+0)*1-0
        caro energy(m,v):
            nan 0.5*m*v*v
        caro identity(x):
            nan (0+x)/1
        caro neg(x):
            nan -(-x)
        pedo(f.getTex(0))
        pedo(f.getTex(1))
        pedo(f.getTex(3))
        pedo(energy.getTex())
        pedo(identity.getTex())
        pedo(neg.getTex())
        pedo(f(3),energy(2,3),identity(5),neg(-7))
        """, stress=True).splitlines()
        self.assertEqual(output[0], r"f\left(x\right)=x^{2}")
        self.assertIn(r"x\cdot x+0", output[1])
        self.assertIn(r"\cdot 1-0", output[2])
        self.assertEqual(output[3], r"\mathrm{energy}\left(m,v\right)=0.5\cdot m\cdot v^{2}")
        self.assertEqual(output[4], r"\mathrm{identity}\left(x\right)=x")
        self.assertEqual(output[5], r"\mathrm{neg}\left(x\right)=x")
        self.assertEqual([float(v) for v in output[6].split()], [9, 9, 5, -7])

    def test_simplification_preserves_domain_and_opaque_calls(self):
        output = self.run_code("""
        esta calls = 0
        esta a = 2
        caro effect(x):
            calls = calls+1
            nan x
        caro singular(x):
            nan x/x
        caro zero(x):
            nan 0*(1/x)
        caro opaque(x):
            nan effect(x)*1+0
        caro external(x):
            nan a*x+0
        pedo(singular.getTex())
        pedo(zero.getTex())
        pedo(opaque.getTex())
        pedo(external.getTex())
        pedo(calls)
        """, stress=True).splitlines()
        self.assertEqual(output[0], r"\mathrm{singular}\left(x\right)=\frac{x}{x}")
        self.assertIn(r"0\cdot \frac{1}{x}", output[1])
        self.assertIn(r"\mathrm{effect}\left(x\right)\cdot 1+0", output[2])
        self.assertIn("Dynamic call targets unresolved", output[2])
        self.assertIn(r"a\cdot x+0", output[3])
        self.assertIn("External values unresolved", output[3])
        self.assertEqual(output[4], "0")
