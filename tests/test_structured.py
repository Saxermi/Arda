"""Complex scalars, operator contracts, and objective-only optimization."""
import ast
import unittest

import test_arda


class StructuredTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_complex_scalar_arithmetic_and_truth(self):
        self.run_code("""
        tol complex as cx
        esta z = cx.new(3,4)
        esta w = cx.new(0,2)
        pedo(type(z), abs(z), cx.real(z), cx.imag(z))
        pedo(z == cx.new(3,4), z != w, cx.new(3,0) == 3, cx.new(9007199254740992,0) != 9007199254740993)
        pedo(cx.real(z*w), cx.imag(z*w), cx.real(z/w), cx.imag(z/w))
        pedo(cx.real(2-z), cx.imag(-z), cx.imag(cx.conj(z)))
        pedo(not cx.new(0,-0.0), not z)
        pedo(cx.imag(cx.sqrt(cx.new(-4,0.0))), cx.imag(cx.sqrt(cx.new(-4,-0.0))))
        esta large = cx.new(1e308,1e308)
        esta same = large/large
        pedo(cx.real(same) == 1 and cx.imag(same) == 0)
        """, "complex 5 3 4\nnae nae nae nae\n-8 6 2 -1.5\n-1 -4 -4\nnae law\n2 -2\nnae",
                      stress=True)

    def test_complex_errors(self):
        for expression, error in [
            ('cx.new(1,0)/cx.new(0,-0.0)', 'complex division by zero'),
            ('cx.new(1,0) < 2', 'unsupported complex operator'),
            ('cx.new(1,"x")', 'expected number'),
        ]:
            with self.subTest(expression=expression):
                self.run_code('tol complex as cx\n' + expression, error=error, stress=True)

    def test_structured_products_and_snapshot_ownership(self):
        self.run_code("""
        tol numc as nc
        tol linop as lo
        esta u = nc.array([[1,2],[3,4],[5,6]])
        esta op = lo.lowrank(u,nc.array([[7,8],[9,10]]))
        nc.put(u,[0,0],100)
        u = mun
        collect()
        esta x = nc.array([1,2,3])
        pedo(type(op),lo.shape(op))
        pedo(nc.tolist(lo.apply(op,nc.array([2,-1]))), nc.tolist(lo.apply_transpose(op,x)))
        esta d = lo.diagonal(nc.array([2,3,4]))
        pedo(nc.tolist(lo.apply(d,x)))
        esta b = lo.banded(1,0,nc.array([[2,3,4],[5,6,0]]))
        pedo(nc.tolist(lo.apply(b,x)), nc.tolist(lo.apply_transpose(b,x)))
        esta t = lo.toeplitz(nc.array([2,3,5]),nc.array([2,-1,4]))
        pedo(nc.tolist(lo.apply(t,x)), nc.tolist(lo.apply_transpose(t,x)))
        esta k = lo.kron(nc.array([[1],[2]]),nc.array([[3,4]]))
        esta block = nc.permute(nc.array([[1,2],[3,4]]),[1,0])
        pedo(nc.tolist(lo.apply(k,block)), nc.tolist(lo.apply_transpose(k,nc.array([1,2]))))
        esta empty = lo.lowrank(nc.zeros(3,0),nc.zeros(2,0))
        pedo(nc.tolist(lo.apply(empty,nc.ones(2))))
        """, "linear_operator [3, 2]\n[17, 39, 61] [378, 478]\n[2, 6, 12]\n"
             "[2, 11, 24] [12, 24, 12]\n[12, 4, 17] [23, 12, 8]\n[[11, 25], [22, 50]] [15, 20]\n[0, 0, 0]",
                      stress=True)

    def test_operator_shape_and_finiteness_errors(self):
        for expression, error in [
            ('lo.lowrank(nc.ones(2,1),nc.ones(2,2))', 'equal column counts'),
            ('lo.toeplitz(nc.array([1,2]),nc.array([2,3]))', 'shared first value'),
            ('lo.banded(1,1,nc.ones(2,4))', 'band storage'),
            ('lo.kron(nc.zeros(4001,0),nc.zeros(4001,0))', 'Kronecker axis'),
            ('lo.apply(lo.diagonal(nc.ones(2)),nc.ones(3))', 'shape mismatch'),
            ('lo.apply(lo.diagonal(nc.array([1e300])),nc.array([1e300]))', 'nonfinite'),
        ]:
            with self.subTest(expression=expression):
                self.run_code('tol numc as nc\ntol linop as lo\n' + expression, error=error, stress=True)

        self.run_code("""
        tol numc as nc
        tol linop as lo
        tol sparse as sp
        esta a = sp.coo(2,2,[0],[1],[1e308])
        esta before = sp.memory()["used"]
        try:
            lo.apply(a,nc.ones(2)*10)
        catch e:
            pedo(e["message"])
        pedo(sp.memory()["used"] == before)
        pedo(nc.tolist(lo.apply(a,nc.ones(2))))
        """, "nonfinite linear operator input or product\nnae\n[1e+308, 0]")

    def test_rsvd_availability_and_rank_contract(self):
        imports = 'tol numc as nc\ntol linalg as la\ntol linop as lo\n'
        if self.run_code('tol calc\npedo(calc.backend())').strip() != 'lapack':
            self.run_code(imports + 'la.rsvd(nc.ones(2,2),1)',
                          error='rebuild with LAPACK=1', stress=True)
        else:
            self.run_code(imports + 'la.rsvd(lo.diagonal(nc.ones(2)),2)', error='rank', stress=True)
            self.run_code(imports + 'la.rsvd(nc.ones(2,2),1,0,-1)', error='power iterations', stress=True)
            self.run_code(imports + 'pedo(la.rsvd(nc.zeros(3,2),1)[3])', '0', stress=True)

    def test_zeroth_order_reports_and_preserves_input(self):
        output = self.run_code("""
        tol numc as nc
        tol opt
        esta calls = 0
        caro f(x):
            calls = calls + 1
            nan (x[0]-2)*(x[0]-2)+(x[1]+1)*(x[1]+1)
        esta start = nc.array([9,-7])
        esta result = opt.zeroth_order(f,start)
        pedo(nc.tolist(result["solution"]))
        pedo(result["iterations"],result["converged"],result["evaluations"] == calls)
        pedo(nc.tolist(start))
        """, stress=True).splitlines()
        npfree = ast.literal_eval(output[0])
        self.assertAlmostEqual(npfree[0], 2, places=6)
        self.assertAlmostEqual(npfree[1], -1, places=6)
        steps, converged, accurate_count = output[1].split()
        self.assertIn(int(steps), range(1, 6))
        self.assertEqual((converged, accurate_count, output[2]), ('nae', 'nae', '[9, -7]'))
        self.run_code("""
        tol numc as nc
        tol opt
        esta center = nc.array([2,-1])
        caro f(x):
            esta delta = x-center
            nan nc.dot(delta,delta)
        esta result = opt.zeroth_order(f,nc.array([5,3]))
        pedo(nc.norm(result["solution"]-center)<1e-6 and result["converged"])
        """, 'nae', stress=True)
        self.run_code('tol numc as nc\ntol opt\nopt.zeroth_order(lambda x: x,nc.ones(2))',
                      error='real scalar', stress=True)
