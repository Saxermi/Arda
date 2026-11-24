import unittest
from test_arda import ArdaTests as _ArdaTests

class ScientificTests(unittest.TestCase):
    run_code = _ArdaTests.run_code
    def check(self, code, expected=None, **kw):
        import textwrap
        return self.run_code('tol numc as nc\n'+textwrap.dedent(code).lstrip('\n'), expected, **kw)
    def test_vector(self):
        self.check('''
        esta x = nc.array([1, 2, 3])
        pedo(x + 2, 2 + x, x * x, 10 - x, x / 2, -x)
        pedo(type(x), len(x), nc.shape(x), nc.bytes(x))
        ''','[3, 4, 5] [3, 4, 5] [1, 4, 9] [9, 8, 7] [0.5, 1, 1.5] [-1, -2, -3]\narray 3 [3] 24',stress=True)
    def test_index_iter(self):
        self.check('''
        esta x = nc.linspace(0, 4, 3)
        x[-1] = 8
        an i in x:
            pedo(i)
        ''','0\n2\n8',stress=True)
    def test_matrix(self):
        self.check('''
        esta a = nc.array([[1, 2, 3], [4, 5, 6]])
        esta b = nc.transpose(a)
        pedo(nc.matmul(a, b), nc.shape(b))
        esta row = a[0]
        row[0] = 99
        nc.set(a, -1, -1, 9)
        pedo(nc.get(a, 0, 0), nc.tolist(a))
        an r in a:
            pedo(r)
        ''','[[14, 32], [32, 77]] [3, 2]\n1 [[1, 2, 3], [4, 5, 9]]\n[1, 2, 3]\n[4, 5, 9]',stress=True)
    def test_stats(self):
        self.check('''
        esta x = nc.array([1, 2, 3])
        pedo(nc.sum(x), nc.mean(x), nc.var(x), nc.min(x), nc.max(x))
        pedo(nc.dot(x, x), nc.norm(nc.array([3, 4])))
        pedo(nc.std(nc.array([1, 3])))
        ''','6 2 0.666666666666667 1 3\n14 5\n1')
    def test_solve_pivot(self):
        self.check('''
        esta a = nc.array([[0, 2], [1, 3]])
        esta b = nc.array([4, 7])
        pedo(nc.solve(a, b), a, b)
        ''','[1, 2] [[0, 2], [1, 3]] [4, 7]',stress=True)
    def test_functions(self):
        self.check('pedo(nc.sin(nc.zeros(2)), nc.cos(nc.zeros(2)), nc.exp(nc.zeros(2)), nc.log(nc.ones(2)), nc.sqrt(nc.array([4, 9])))','[0, 0] [1, 1] [1, 1] [0, 0] [2, 3]')
    def test_empty(self):
        self.check('pedo(nc.sum(nc.zeros(0)), nc.dot(nc.zeros(0), nc.zeros(0)), nc.shape(nc.matmul(nc.zeros(2, 0), nc.zeros(0, 3))))','0 0 [2, 3]',stress=True)
    def test_errors(self):
        for code,err in [('nc.array([[1], [2, 3]])','ragged'),('nc.ones(2) + nc.ones(3)','shape mismatch'),('nc.solve(nc.array([[1,2],[2,4]]), nc.ones(2))','singular'),('nc.zeros(10000, 10000)','size limit'),('nc.mean(nc.zeros(0))','empty'),('nc.ones(2)[2]','out of bounds'),('nc.dot(nc.ones(2, 2), nc.ones(2, 2))','expected vector'),('nc.zeros(-1)','invalid array dimension'),('nc.array(["x"])','expected number')]:
            with self.subTest(code=code): self.check(code,error=err,stress=True)
    def test_gc_buffer(self):
        self.check('''
        collect()
        esta before = heap_bytes()
        esta i = 0
        tenna i < 100:
            esta a = nc.ones(10000)
            esta b = a * 3
            i = i + 1
        collect()
        pedo(heap_bytes() - before < 10000)
        ''','nae',stress=True)
    def test_accuracy(self):
        self.check('''
        esta x = nc.array([1000000000001, 1000000000002, 1000000000003])
        pedo(abs(nc.var(x) - 2 / 3) < 0.000000001)
        pedo(nc.norm(nc.array([1e200, 1e200])) > 1e200)
        ''','nae\nnae')
    def test_caught_errors(self):
        self.check('''
        try:
            nc.solve(nc.zeros(2, 2), nc.ones(2))
        catch e:
            pedo(e)
        collect()
        pedo(nc.sum(nc.ones(4)))
        ''','singular matrix\n4',stress=True)
del _ArdaTests
