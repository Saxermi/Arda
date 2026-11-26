import unittest
from test_arda import ArdaTests as _ArdaTests, BIN
import pathlib, subprocess, tempfile

class UpgradeTests(unittest.TestCase):
    run_code = _ArdaTests.run_code
    def test_v08_integer_limits(self):
        self.run_code('pedo(9007199254740993 + 1, 9223372036854775807, -9223372036854775808, type(1), type(1.5))','9007199254740994 9223372036854775807 -9223372036854775808 int float')
    def test_v08_overflow(self):
        for expr in ('9223372036854775807 + 1','-9223372036854775808 - 1','-9223372036854775808 * -1','-(-9223372036854775808)','3037000500 * 3037000500'):
            with self.subTest(expr=expr): self.run_code('pedo('+expr+')',error='integer overflow')
    def test_v08_int_literal_error(self):
        self.run_code('pedo(9223372036854775808)',error='integer literal out of range')
    def test_v08_numeric_mixed(self):
        self.run_code('pedo(9007199254740993 == 9007199254740992.0, 9007199254740993 > 9007199254740992.0, 2 == 2.0, 2 / 4, -9223372036854775808 % -1, -2 * -3, int("9223372036854775807"))','law nae nae 0.5 0 6 9223372036854775807')
    def test_v08_param_slots(self):
        self.run_code('''
        caro f(x):
            x = x + 1
            qui nae:
                esta x = 99
                pedo(x)
            caro g():
                x = x + 1
                nan x
            nan g
        esta g = f(1)
        collect()
        pedo(g(),g())
        ''','99\n3 4',stress=True)
    def test_v08_defaults(self):
        self.run_code('''
        caro f(a, b = a + 1, c = b + 1):
            nan [a,b,c]
        pedo(f(2),f(2,8),f(2,8,0))
        ''','[2, 3, 4] [2, 8, 9] [2, 8, 0]',stress=True)
    def test_v08_default_errors(self):
        self.run_code('caro f(a = 1, b):\n    pass',error='required parameter after default')
        self.run_code('caro f(a = b, b = 2):\n    nan a\npedo(f())',error='parameter not initialized')
    def test_v08_inheritance(self):
        self.run_code('''
        nost A:
            caro init(self, x = 2):
                self.x = x
            caro f(self):
                nan self.x
        nost B(A):
            caro f(self):
                nan A.f(self) + 1
        nost C(B):
            pass
        esta c = C()
        esta bound = c.f
        collect()
        pedo(bound(), isinstance(c,A), isinstance(c,B), isinstance(A(),C))
        ''','3 nae nae law',stress=True)
    def test_v08_bad_inheritance(self):
        self.run_code('nost X(5):\n    pass',error='base must be a class')
    def test_v08_lambda(self):
        self.run_code('''
        caro f(n):
            nan lambda x: x + n
        esta g = f(4)
        collect()
        pedo(g(3), (lambda: 42)())
        ''','7 42',stress=True)
    def test_v08_comp(self):
        self.run_code('''
        esta x = 77
        esta f = [lambda: x an x in range(3)]
        pedo([x * x an x in range(6) qui x % 2 == 0], x, f[0](), f[2]())
        pedo([[y an y in range(x)] an x in range(3)])
        ''','[0, 4, 16] 77 0 2\n[[], [0], [0, 1]]',stress=True)
    def test_v08_slicing(self):
        self.run_code('pedo([1,2,3,4][1:-1], [1,2][:], [1,2][9:], "é🌿x"[:2], "abc"[-2:])','[2, 3] [1, 2] [] é🌿 bc',stress=True)
    def test_v08_utf8(self):
        self.run_code('''
        esta él = "aé🌿"
        pedo(len(él), él[1], él[-1])
        an c in él:
            pedo(c)
        ''','3 é 🌿\na\né\n🌿',stress=True)
    def test_v08_invalid_utf8(self):
        self.run_code('pedo(read_text("bad.txt"))',files={'bad.txt':'\u00e9'},expected='é')
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/'bad.arda';p.write_bytes(b'pedo("\xff")')
            binary=BIN
            r=subprocess.run([str(binary),str(p)],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);self.assertIn('invalid UTF-8',r.stderr)
    def test_v08_modules(self):
        self.run_code('''
        tol math
        tol time
        tol text
        tol collections
        tol io
        pedo(math.sqrt(81),math.floor(1.9),math.ceil(1.1),math.cos(0))
        pedo(text.join("-",text.split("a-é-b","-")),text.startswith("éx","é"),text.endswith("ab","b"))
        pedo(list(collections.range(3)),time.monotonic() > 0)
        io.write_text("test.txt","é")
        pedo(io.read_text("test.txt"))
        ''','9 1 2 1\na-é-b nae nae\n[0, 1, 2] nae\né',stress=True)
    def test_v08_weak_intern(self):
        self.run_code('''
        esta same = "mithril"
        esta i = 0
        tenna i < 300:
            esta s = "x" + str(i)
            i = i + 1
        collect()
        pedo(same == "mith" + "ril", heap() < 100)
        ''','nae nae',stress=True)
    def test_v08_bytecode_slots(self):
        out=self.run_code('caro f(x):\n    x = x + 1\n    nan x\npedo(f(2))',flags=('--bytecode',))
        self.assertIn('GET_LOCAL',out);self.assertIn('SET_LOCAL',out)

del _ArdaTests
