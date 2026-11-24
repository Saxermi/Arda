import unittest
from test_arda import ArdaTests as _ArdaTests

class PerformanceRegressionTests(unittest.TestCase):
    run_code = _ArdaTests.run_code

    def test_cache_resize_repl_like_scopes(self):
        lines = ['esta x = 7', 'caro getx():', '    nan x', 'pedo(getx())']
        lines += [f'esta name{i} = {i}' for i in range(90)]
        lines += ['x = 9', 'collect()', 'pedo(getx())']
        self.run_code('\n'.join(lines), '7\n9', stress=True)

    def test_cache_scope_and_closure(self):
        self.run_code('''
        esta x = 1
        caro make(x):
            caro inc():
                x = x + 1
                nan x
            nan inc
        esta a = make(10)
        esta b = make(20)
        pedo(a(), b(), a(), b(), x)
        ''', '11 21 12 22 1', stress=True)

    def test_cache_alternating_tables(self):
        self.run_code('''
        caro f(n):
            esta x = n
            qui nae:
                esta x = n + 1
                pedo(x)
            x = x + 2
            nan x
        pedo(f(2), f(4))
        ''','3\n5\n4 6',stress=True)

    def test_fused_dynamic_fallback(self):
        self.run_code('''
        esta x = 1.5
        pedo(x + 2, x - 2, x * 2, "ring" + 2)
        qui x < 2:
            pedo("float branch")
        try:
            qui "bad" < 2:
                pass
        catch e:
            pedo("caught")
        ''','3.5 -0.5 3 ring2\nfloat branch\ncaught',stress=True)

    def test_fused_assignment_expression(self):
        self.run_code('''
        esta x = 0
        caro f(n):
            n = n + 1
            nan (n = n + 1)
        pedo((x = 2), x, f(3))
        x = x + 1
        pedo(x)
        ''','2 2 5\n3',stress=True)

    def test_fused_gc_barriers(self):
        self.run_code('''
        esta x = []
        caro f(a):
            a = ["kept"]
            collect()
            nan a
        x = f(x)
        collect()
        pedo(x[0])
        ''','kept',stress=True)

    def test_fused_bytecode(self):
        out = self.run_code('esta x = 0\ntenna x < 2:\n    x = x + 1\npedo(x)',flags=('--bytecode',))
        for opcode in ('JFALSE_LT_CONST','ADD_INT_CONST','SET_DROP'):
            self.assertIn(opcode,out)

del _ArdaTests
