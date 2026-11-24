"""Sparse assembly, algebra, Matrix Market, solver and budget regressions."""
import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import test_arda


class MoriaTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_builder_batches_are_atomic(self):
        self.run_code('''
        tol sparse as sp
        tol numc as nc
        esta b = sp.builder(2,3,1)
        sp.append(b,0,1,1e16)
        try:
            sp.extend(b,[0,2],[1,2],[32,1])
        catch e:
            pedo("invalid batch")
        sp.extend(b,nc.array([0,1,0,0,1]),nc.array([1,2,0,1,2]),[-1e16,2,9,1,-2])
        pedo(type(b),sp.components(sp.finish(b)),sp.memory()["used"] < 100)
        try:
            sp.finish(b)
        catch e:
            pedo("finalized")
        ''', 'invalid batch\nsparse_builder [[0, 2, 2], [0, 1], [9, 1]] nae\nfinalized', stress=True)

    def test_builder_growth_budget_and_retry(self):
        self.run_code('''
        tol sparse as sp
        esta b = sp.builder(2,2,2,80)
        sp.extend(b,[1,0],[0,0],[3,1])
        sp.set_budget(79)
        try:
            sp.append(b,1,1,5)
        catch e:
            pedo("growth rejected",sp.memory()["used"])
        sp.set_budget(80)
        sp.append(b,1,1,5)
        try:
            sp.finish(b)
        catch e:
            pedo("finish rejected",sp.memory()["used"])
        sp.set_budget(mun)
        pedo(sp.components(sp.finish(b)))
        ''', 'growth rejected 32\nfinish rejected 48\n[[0, 1, 3], [0, 0, 1], [1, 3, 5]]', stress=True)

    def test_builder_failed_duplicate_sum_keeps_input(self):
        self.run_code('''
        tol sparse as sp
        esta b = sp.builder(1,1)
        sp.extend(b,[0,0],[0,0],[1e308,1e308])
        esta n = sp.memory()["used"]
        an i in range(50):
            try:
                sp.finish(b)
            catch e:
                pass
        pedo(sp.memory()["used"] == n)
        sp.append(b,0,0,-1e308)
        try:
            sp.finish(b)
        catch e:
            pedo("input order retained")
        ''', 'nae\ninput order retained', stress=True)

    def test_direct_diagonals(self):
        self.run_code('''
        tol sparse as sp
        tol numc as nc
        esta a = sp.diags(3,4,[1,-1,0],[nc.array([3,0,2]),[4,5],1])
        pedo(sp.components(a))
        pedo(sp.components(sp.tridiag(3,0,[1,0,3],[2,4])))
        pedo(sp.components(sp.eye(0)),sp.components(sp.finish(sp.builder(3,0))))
        ''', '[[0, 2, 4, 7], [0, 1, 0, 1, 1, 2, 3], [1, 3, 4, 1, 5, 1, 2]]'
             '\n[[0, 2, 3, 4], [0, 1, 2, 2], [1, 2, 4, 3]]'
             '\n[[0], [], []] [[0, 0, 0, 0], [], []]', stress=True)
        for expression, error in [
            ('sp.diags(3,4,[1,1],[5,6])', 'duplicate'),
            ('sp.diags(3,4,[-4],[2])', 'offset'),
            ('sp.tridiag(3,[5],4,0)', 'length'),
            ('sp.extend(sp.builder(3,3),nc.array([0.5]),[1],[5])', 'index'),
        ]:
            with self.subTest(expression=expression):
                self.run_code('tol sparse as sp\ntol numc as nc\n' + expression, error=error, stress=True)

    def test_sparse_algebra_and_strides(self):
        self.run_code('''
        tol sparse as sp
        tol numc as nc
        esta a = sp.coo(2,3,[1,0,1,0],[2,1,0,1],[4,5,3,-3])
        esta b = nc.permute(nc.array([[1,2,3],[4,5,6]]),[1,0])
        pedo(nc.tolist(sp.matvec(a,nc.array([1,2,3]))))
        pedo(nc.tolist(sp.spmm(a,b)))
        pedo(sp.components(sp.transpose(a)))
        pedo(sp.components(sp.add(a,sp.scale(a,-1))))
        pedo(nc.tolist(sp.diagonal(a)),sp.estimate(a)["nnz"],sp.estimate(sp.to_csr(a))["matvec_workspace_bytes"])
        pedo(nc.tolist(sp.matvec(sp.csr(2,0,[0,0,0],[],[]),nc.zeros(0))))
        pedo(nc.shape(sp.spmm(a,nc.zeros(3,0))))
        ''', '[4, 15]\n[[4, 10], [15, 36]]\n[[0, 1, 2, 3], [1, 0, 1], [3, 2, 4]]'
             '\n[[0, 0, 0], [], []]\n[0, 0] 4 0\n[0, 0]\n[2, 0]', stress=True)

    def test_matrix_market_round_trip_and_cleanup(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            source = path / "source.mtx"
            output = path / "output.mtx"
            invalid = path / "invalid.mtx"
            source.write_text("%%MatrixMarket matrix coordinate real general\n"
                              "% duplicates\n2 3 4\n1 2 0.5\n2 1 4\n1 2 -.5\n1 3 2.5e0\n")
            invalid.write_text("%%MatrixMarket matrix coordinate real symmetric\n2 2 2\n2 1 4\n0 2 1\n")
            source_arg, output_arg, invalid_arg = map(lambda p: json.dumps(str(p)), (source, output, invalid))
            self.run_code(f'''
            tol sparse as sp
            esta a = sp.read_mm({source_arg})
            pedo(sp.components(a))
            sp.write_mm({output_arg},sp.tridiag(3,-1,4,-1),true)
            pedo(sp.components(sp.read_mm({output_arg})))
            esta n = sp.memory()["used"]
            an k in range(80):
                try:
                    sp.read_mm({invalid_arg})
                catch e:
                    pass
            pedo(sp.memory()["used"] == n)
            try:
                sp.write_mm({output_arg},a,true)
            catch e:
                pedo("invalid symmetric export")
            ''', '[[0, 1, 2], [2, 0], [2.5, 4]]\n'
                 '[[0, 2, 5, 7], [0, 1, 0, 1, 2, 1, 2], [4, -1, -1, 4, -1, -1, 4]]'
                 '\nnae\ninvalid symmetric export', stress=True)
            self.assertEqual(output.read_text(), "%%MatrixMarket matrix coordinate real symmetric\n"
                             "3 3 5\n1 1 4\n2 1 -1\n2 2 4\n3 2 -1\n3 3 4\n")

    def test_matrix_market_rejects_bad_records(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.mtx"
            banner = "%%MatrixMarket matrix coordinate real general\n"
            for header, body, error in [
                ("%%MatrixMarket matrix array real general\n", "1 1\n5\n", "coordinate real"),
                (banner, "1 1 1\n1 1 7e500\n", "real value"),
                (banner, "1 1 1\n1 1 5 extra\n", "entry"),
                (banner, "1 1 0\n1 1 5\n", "more entries"),
                (banner, "1 1 1\n", "missing"),
                (banner.replace("general", "symmetric"), "2 2 1\n1 2 5\n", "lower triangular"),
            ]:
                with self.subTest(body=body, header=header):
                    path.write_text(header + body)
                    self.run_code(f'tol sparse as sp\nsp.read_mm({json.dumps(str(path))})', error=error, stress=True)

    def test_solver_status_and_true_residual(self):
        output = self.run_code('''
        tol sparse as sp
        tol numc as nc
        esta a = sp.tridiag(5,-1,[4,5,6,7,8],-1)
        esta x = nc.array([1,2,3,4,5])
        esta b = sp.matvec(a,x)
        esta w = sp.workspace(5)
        esta r = sp.pcg(a,b,1e-12,25,w)
        pedo(r["converged"],r["reason"],r["residual"] == sp.verify(a,b,r["solution"]))
        pedo(nc.tolist(r["solution"]))
        esta maxed = sp.cg(a,b,1e-12,1,w)
        pedo(maxed["converged"],maxed["iterations"],maxed["reason"])
        esta zero = sp.cg(a,nc.zeros(5),1e-12,25,w)
        pedo(zero["iterations"],zero["residual"],sp.verify(sp.eye(2),nc.zeros(2),nc.array([3,4])))
        esta breakdown = sp.cg(sp.tridiag(2,2,1,2),nc.array([1,-1]),1e-12,10)
        pedo(breakdown["converged"],breakdown["reason"])
        ''', stress=True).splitlines()
        self.assertEqual(output[0], "nae converged nae")
        for value, expected in zip(ast.literal_eval(output[1]), range(1, 6), strict=True):
            self.assertAlmostEqual(value, expected, places=12)
        self.assertEqual(output[2:], ["law 1 max_iterations", "0 0 5", "law nonpositive_curvature"])

    def test_solver_budget_and_workspace_reuse(self):
        self.run_code('''
        tol sparse as sp
        tol numc as nc
        esta a = sp.tridiag(6,-1,4,-1)
        esta b = nc.ones(6)
        esta live = sp.memory()["used"]
        sp.set_budget(live + 6*8*6 - 1)
        try:
            sp.pcg(a,b,1e-10,20)
        catch e:
            pedo("solver rejected",sp.memory()["used"] == live)
        esta w = sp.workspace(6)
        try:
            sp.pcg(a,b,1e-10,20,w)
        catch e:
            pedo("solution rejected")
        sp.set_budget(live + 6*8*6)
        esta r = sp.pcg(a,b,1e-10,20,w)
        pedo(r["reason"],sp.memory()["used"] == live + 6*8*5)
        w = mun
        collect()
        pedo(sp.memory()["used"] == live)
        ''', 'solver rejected nae\nsolution rejected\nconverged nae\nnae', stress=True)
        for expression, error in [
            ('sp.cg(sp.tridiag(2,1,4,0),nc.ones(2),1e-9,10)', 'symmetric'),
            ('sp.pcg(sp.tridiag(2,-1,0,-1),nc.ones(2),1e-9,10)', 'positive diagonal'),
            ('sp.cg(sp.eye(2),nc.ones(2),1e-9,10,sp.workspace(1))', 'workspace'),
            ('sp.cg(sp.eye(2),nc.ones(3),1e-9,10)', 'shape'),
        ]:
            with self.subTest(expression=expression):
                self.run_code('tol sparse as sp\ntol numc as nc\n' + expression, error=error, stress=True)
