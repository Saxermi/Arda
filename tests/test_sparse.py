"""Moria sparse foundations: exact structure, errors, ownership and GC."""
import unittest
import ctypes
import test_arda


class SparseTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_canonical_roundtrip(self):
        self.run_code('''
        tol sparse as sp
        esta a = sp.coo(3,4,[2,0,0,2,0,1,1,1],[3,1,1,0,3,2,2,0],[7,2,3,4,-1,4,-4,0])
        esta b = sp.to_csr(a)
        pedo(type(a), sp.format(a), sp.format(b), sp.shape(b))
        pedo(sp.nnz(a), sp.nnz(b), sp.bytes(a))
        pedo(sp.components(b))
        pedo(sp.components(sp.to_coo(b)))
        pedo(sp.components(sp.to_csr(sp.to_coo(b))))
        pedo(sp.get(a,0,1),sp.get(b,2,3),sp.get(b,1,2))
        ''', 'sparse coo csr [3, 4]\n8 4 128\n[[0, 2, 2, 4], [1, 3, 0, 3], [5, -1, 4, 7]]\n[[0, 0, 2, 2], [1, 3, 0, 3], [5, -1, 4, 7]]\n[[0, 2, 2, 4], [1, 3, 0, 3], [5, -1, 4, 7]]\n5 7 0', stress=True)

    def test_empty_and_large_shape(self):
        self.run_code('''
        tol sparse as sp
        tol numc as nc
        esta a = sp.coo(0,4,[],[],[])
        esta b = sp.csr(3,0,[0,0,0,0],[],[])
        pedo(sp.components(sp.to_csr(a)))
        pedo(sp.components(sp.to_coo(b)))
        pedo(nc.shape(sp.todense(a)),nc.shape(sp.todense(b)))
        esta huge = sp.coo(4294967295,4294967295,[4294967294],[4294967294],[2])
        pedo(sp.shape(huge),sp.get(huge,4294967294,4294967294),sp.bytes(huge))
        ''', '[[0], [], []]\n[[], [], []]\n[0, 4] [3, 0]\n[4294967295, 4294967295] 2 16', stress=True)

    def test_dense_strided_bridge(self):
        self.run_code('''
        tol sparse as sp
        tol numc as nc
        esta a = nc.array([[0,2,0],[3,0,4]])
        esta t = nc.permute(a,[1,0])
        esta s = sp.from_dense(t)
        pedo(sp.components(s))
        pedo(nc.tolist(sp.todense(s)))
        pedo(sp.components(sp.from_dense(sp.todense(sp.csr(0,3,[0],[],[])))))
        ''', '[[0, 1, 2, 3], [1, 0, 1], [3, 2, 4]]\n[[0, 3], [2, 0], [0, 4]]\n[[0], [], []]', stress=True)

    def test_components_are_independent(self):
        self.run_code('''
        tol sparse as sp
        esta r = [0]
        esta x = [3]
        esta a = sp.coo(2,2,r,[1],x)
        r[0] = 1
        x[0] = 9
        esta b = sp.to_csr(a)
        esta c = sp.components(b)
        c[1][0] = 0
        c[2][0] = 100
        collect()
        pedo(sp.components(b),sp.get(a,0,1))
        ''', '[[0, 1, 1], [1], [3]] 3', stress=True)

    def test_invalid_inputs(self):
        cases = [
            ('sp.coo(-1,2,[],[],[])','dimensions'),
            ('sp.coo(4294967296,2,[],[],[])','dimensions'),
            ('sp.coo(2.0,2,[],[],[])','dimensions'),
            ('sp.coo(2,2,[0],[],[1])','length'),
            ('sp.coo(2,2,[2],[1],[1])','index'),
            ('sp.coo(2,2,[0.0],[1],[1])','index'),
            ('sp.coo(2,2,[0],[-1],[1])','index'),
            ('sp.coo(2,2,[0],[1],["x"])','number'),
            ('sp.coo(2,2,[0],[1],[1e308*1e308])','finite'),
            ('sp.csr(2,2,[0,1],[0],[1])','length'),
            ('sp.csr(2,2,[1,1,1],[0],[1])','begin'),
            ('sp.csr(2,2,[0,1,0],[0],[1])','nondecreasing'),
            ('sp.csr(2,2,[0,0,0],[0],[1])','end'),
            ('sp.csr(2,2,[0,2,2],[0],[1])','offset'),
            ('sp.csr(1,2,[0,2],[1,0],[1,2])','strictly'),
            ('sp.csr(1,2,[0,2],[1,1],[1,2])','strictly'),
            ('sp.csr(1,2,[0,1],[0],[0])','zeros'),
            ('sp.get(sp.coo(1,1,[],[],[]),0,1)','index'),
            ('sp.to_csr(2)','sparse matrix'),
            ('sp.coo(2,2,[],[])','expects'),
            ('sp.todense(sp.coo(1000000,1000000,[],[],[]))','array size limit'),
        ]
        for expression, error in cases:
            with self.subTest(expression=expression):
                self.run_code('tol sparse as sp\n'+expression, error=error, stress=True)

    def test_stable_duplicate_order(self):
        self.run_code('''
        tol sparse as sp
        esta a = sp.coo(1,2,[0,0,0,0],[1,0,1,1],[1e16,2,-1e16,1])
        pedo(sp.components(sp.to_csr(a)),sp.get(a,0,1))
        ''', '[[0, 2], [0, 1], [2, 1]] 1')

    def test_failed_conversion_gc_recovery(self):
        self.run_code('''
        tol sparse as sp
        esta a = sp.coo(1,1,[0,0],[0,0],[1e308,1e308])
        collect()
        esta before = heap_bytes()
        an i in range(100):
            try:
                sp.to_csr(a)
            catch e:
                pass
        collect()
        pedo(heap_bytes() <= before + 4096)
        try:
            sp.get(a,0,0)
        catch e:
            pedo("overflow caught")
        pedo(sp.nnz(a))
        ''', 'nae\noverflow caught\n2', stress=True)

    def test_direct_csr_and_accounting(self):
        size = 4*12 + 4*ctypes.sizeof(ctypes.c_size_t)
        self.run_code('''
        tol sparse as sp
        esta a = sp.csr(3,4,[0,2,2,4],[1,3,0,3],[5,-1,4,7])
        pedo(sp.bytes(a), sp.bytes(sp.to_coo(a)))
        pedo(sp.components(sp.to_csr(a)))
        pedo(sp.get(a,0,0),sp.get(a,0,1),sp.get(a,0,2),sp.get(a,0,3),sp.get(a,2,3))
        esta z = sp.to_csr(sp.coo(2,3,[1,0,1],[2,0,2],[1,0,-1]))
        pedo(sp.components(z))
        ''', f'{size} 64\n[[0, 2, 2, 4], [1, 3, 0, 3], [5, -1, 4, 7]]\n0 5 0 -1 7\n[[0, 0, 0], [], []]', stress=True)

    def test_dense_bridge_errors(self):
        for expr, error in [('sp.from_dense(nc.array([1,2]))', 'rank-two'),
                            ('sp.from_dense(nc.array([[1e308*1e308]]))', 'finite')]:
            with self.subTest(expr=expr):
                self.run_code('tol sparse as sp\ntol numc as nc\n'+expr, error=error, stress=True)

    def test_descriptor(self):
        output = self.run_code('tol sparse as sp\npedo(sp.describe("to_csr"))')
        self.assertIn('v1:', output)
        self.assertIn('prefix sum', output)
