"""Independent SciPy comparison of COO/CSR canonical conversion."""
import ast
import unittest
import numpy as np
from scipy.sparse import coo_matrix
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
