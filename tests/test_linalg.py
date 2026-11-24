"""Reusable dense factor ownership, compatibility, and API contracts."""
import unittest

import test_arda


class LinalgTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_pivoted_lu_reuse_and_copy_isolation(self):
        self.run_code("""
        tol numc as nc
        tol linalg as la
        esta a = nc.array([[0,2],[1,3]])
        esta f = la.lu(a)
        nc.set(a,0,1,100)
        a = mun
        collect()
        esta parts = la.parts(f)
        pedo(type(f),parts[0],nc.tolist(parts[1]),nc.tolist(parts[2]))
        nc.set(parts[2],0,0,0)
        parts[0][0] = 0
        collect()
        esta rhs = nc.permute(nc.array([[4,7],[2,5]]),[1,0])
        pedo(nc.tolist(la.solve(f,rhs)))
        pedo(nc.tolist(la.solve(f,nc.array([4,7]))))
        pedo(nc.tolist(nc.solve(nc.array([[0,2],[1,3]]),nc.array([4,7]))))
        """, "factor [1, 0] [[1, 0], [0, 1]] [[1, 3], [0, 2]]\n[[1, 2], [2, 1]]\n[1, 2]\n[1, 2]",
                      stress=True)

    def test_cholesky_and_qr_parts(self):
        self.run_code("""
        tol numc as nc
        tol linalg as la
        esta f = la.cholesky(nc.array([[4,2],[2,5]]))
        esta l = la.parts(f)[0]
        pedo(nc.tolist(l),nc.tolist(la.solve(f,nc.array([6,7]))))
        esta tall = nc.array([[1,0],[0,2],[0,0]])
        esta qrf = la.qr(tall)
        esta parts = la.parts(qrf)
        pedo(nc.shape(parts[0]),nc.shape(parts[1]),nc.tolist(la.solve(qrf,nc.array([3,8,7]))))
        esta wide = la.parts(la.qr(nc.array([[0,1,3],[2,1,-1]])))
        pedo(nc.norm(nc.matmul(wide[0],wide[1])-nc.array([[0,1,3],[2,1,-1]]))<1e-13)
        """, "[[2, 0], [1, 2]] [1, 1]\n[3, 2] [2, 2] [3, 4]\nnae", stress=True)

    def test_empty_factors_and_right_hand_sides(self):
        self.run_code("""
        tol numc as nc
        tol linalg as la
        esta f = la.lu(nc.zeros(0,0))
        pedo(nc.shape(la.solve(f,nc.zeros(0))))
        pedo(nc.shape(nc.solve(nc.zeros(0,0),nc.zeros(0))))
        pedo(nc.shape(la.solve(la.cholesky(nc.ones(1,1)),nc.zeros(1,0))))
        esta q = la.qr(nc.zeros(3,0))
        pedo(nc.shape(la.parts(q)[0]),nc.shape(la.solve(q,nc.ones(3,2))))
        """, "[0]\n[0]\n[1, 0]\n[3, 0] [0, 2]", stress=True)

    def test_factor_errors(self):
        for expr, error in [
            ("la.lu(nc.ones(2,3))", "square"),
            ("la.lu(nc.array([[1,2],[2,4]]))", "singular matrix"),
            ("la.cholesky(nc.array([[4,1],[2,5]]))", "symmetric"),
            ("la.cholesky(nc.array([[1,2],[2,1]]))", "positive definite"),
            ("la.qr(nc.ones(2))", "expects a matrix"),
            ("la.qr(nc.array([[1e308*1e308]]))", "finite inputs"),
            ("la.solve(la.lu(nc.ones(1,1)),nc.ones(2))", "matching"),
            ("la.solve(la.lu(nc.ones(1,1)),nc.array([1e308*1e308]))", "finite inputs"),
            ("la.solve(la.qr(nc.ones(2,3)),nc.ones(2))", "as many rows"),
            ("la.solve(la.qr(nc.array([[1,2],[2,4],[3,6]])),nc.ones(3))", "full column rank"),
        ]:
            with self.subTest(expression=expr):
                self.run_code("tol numc as nc\ntol linalg as la\n" + expr, error=error, stress=True)
