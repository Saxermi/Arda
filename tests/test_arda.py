import os
import pathlib
import subprocess
import tempfile
import textwrap
import unittest

BIN = pathlib.Path(os.environ.get('ARDA_BIN', pathlib.Path(__file__).resolve().parents[1] / 'arda')).resolve()

class ArdaTests(unittest.TestCase):
    def run_code(self, code, expected=None, *, error=None, stress=False, files=None, flags=()):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            for name, source in (files or {}).items():
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(textwrap.dedent(source), encoding='utf-8')
            main = root / 'main.arda'
            main.write_text(textwrap.dedent(code).lstrip('\n'), encoding='utf-8')
            cmd = [str(BIN), *flags, *(['--gc-stress'] if stress else []), str(main)]
            p = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=20)
            if error is not None:
                self.assertNotEqual(p.returncode, 0, p.stdout)
                self.assertIn(error, p.stderr)
            else:
                self.assertEqual(p.returncode, 0, p.stderr)
                if expected is not None:
                    self.assertEqual(p.stdout.strip(), expected)
            return p.stdout

    def test_arithmetic(self):
        self.run_code('pedo(2 + 3 * 4, (2 + 3) * 4, 17 % 5, -2)', '14 20 2 -2')
    def test_comments_blank_lines_eof(self):
        self.run_code('''
        # comment
        esta x = 1
        qui nae:
            # blank comment

            x = x + 1 // comment
        pedo(x)''', '2')
    def test_indentation(self):
        self.run_code('qui nae:\n    pedo(1)\n  pedo(2)', error='inconsistent indentation')
    def test_no_semicolon(self):
        self.run_code('esta x = 1;', error='semicolons are not used')
    def test_tabs(self):
        self.run_code('qui nae:\n\tpedo(1)', error='use spaces')
    def test_unclosed_delimiter(self):
        self.run_code('pedo([1, 2)', error='mismatched delimiter')
    def test_multiline_literal(self):
        self.run_code('''
        esta xs = [
            1,
            2,
            3,
        ]
        pedo(xs[-1])
        ''', '3')
    def test_aliases(self):
        self.run_code('''
        def f(x):
            if x > 1:
                return true
            elif x == 1:
                return false
            else:
                return nil
        print(f(2), f(1), f(0))
        ''', 'nae law mun')
    def test_shortcircuit(self):
        self.run_code('pedo(law and missing, nae or missing, mun or 4, 5 and 6, not law)', 'law nae 4 6 nae')
    def test_recursion(self):
        self.run_code('''
        caro fib(n):
            qui n < 2:
                nan n
            nan fib(n - 1) + fib(n - 2)
        pedo(fib(12))
        ''', '144', stress=True)
    def test_closure(self):
        self.run_code('''
        caro counter(n):
            caro step():
                n = n + 1
                nan n
            nan step
        esta c = counter(9)
        collect()
        pedo(c(), c())
        ''', '10 11', stress=True)
    def test_block_scope(self):
        self.run_code('''
        esta x = 2
        qui nae:
            esta x = 5
            pedo(x)
        pedo(x)
        ''', '5\n2', stress=True)
    def test_while_control(self):
        self.run_code('''
        esta i = 0
        esta sum = 0
        tenna i < 10:
            i = i + 1
            qui i == 2:
                continue
            qui i == 5:
                break
            sum = sum + i
        pedo(sum)
        ''', '8', stress=True)
    def test_for_control(self):
        self.run_code('''
        esta sum = 0
        an i in range(10):
            qui i == 2:
                continue
            qui i == 5:
                break
            sum = sum + i
        pedo(sum)
        ''', '8', stress=True)
    def test_nested_for(self):
        self.run_code('''
        esta sum = 0
        an i in range(3):
            an j in range(3):
                qui j == 1:
                    break
                sum = sum + i
        pedo(sum)
        ''', '3', stress=True)
    def test_for_closures(self):
        self.run_code('''
        esta fs = []
        an i in range(3):
            caro f():
                nan i
            push(fs, f)
        collect()
        pedo(fs[0](), fs[1](), fs[2]())
        ''', '0 1 2', stress=True)
    def test_list(self):
        self.run_code('''
        esta xs = [1, 2]
        push(xs, 3)
        xs[-1] = 9
        pedo(len(xs), pop(xs), xs[0])
        ''', '3 9 1', stress=True)
    def test_index_error(self):
        self.run_code('pedo([1][0.5])', error='index out of bounds')
    def test_map(self):
        self.run_code('''
        esta d = {"one": 1, "two": 2}
        d["three"] = 3
        d["one"] = 10
        esta total = 0
        an key in d:
            total = total + d[key]
        pedo(len(d), total, len(keys(d)))
        ''', '3 15 3', stress=True)
    def test_map_non_string(self):
        self.run_code('pedo({1: 2})', error='expected string')
    def test_string(self):
        self.run_code('''
        esta result = ""
        an ch in "abc":
            result = result + ch
        pedo(result, "abc"[-1], str(42), num("2.5"), sqrt(9))
        ''', 'abc c 42 2.5 3', stress=True)
    def test_native_first_class(self):
        self.run_code('esta f = len\npedo(f([1, 2]))', '2', stress=True)
    def test_class_bound_method(self):
        self.run_code('''
        nost Ring:
            caro init(self, n):
                self.n = n
            caro add(self, x):
                self.n = self.n + x
                nan self.n
        esta r = Ring(3)
        esta f = r.add
        r = mun
        collect()
        pedo(f(2), f(4))
        ''', '5 9', stress=True)
    def test_class_error(self):
        self.run_code('class A:\n    pass\nA(1)', error='no constructor')
    def test_module_cache(self):
        self.run_code('''
        tol forge
        import "./forge.arda" as other
        pedo(forge.double(4), other.metal, forge == other)
        ''', 'loaded\n8 mithril nae', stress=True, files={'forge.arda': '''
        pedo("loaded")
        esta metal = "mithril"
        caro double(x):
            nan x * 2
        '''})
    def test_module_visibility(self):
        self.run_code('tol a\npedo(a.f())\npedo(a._private)', error='private module attribute', files={'a.arda':'esta _private = 42\ncaro f():\n    nan _private'})
    def test_module_builtins_isolation(self):
        self.run_code('tol a\npedo(a.len, len([1, 2]))', '99 2', stress=True, files={'a.arda':'esta len = 99'})
    def test_gc_incremental_live_graph(self):
        self.run_code('''
        esta keep = []
        an i in range(1200):
            push(keep, [i])
        an i in range(1200):
            keep[i] = {"value": [i + 1]}
        collect()
        esta total = 0
        an item in keep:
            total = total + item["value"][0]
        pedo(total)
        ''', '720600', stress=True)
    def test_nested_module(self):
        self.run_code('import "pkg/a.arda" as a\npedo(a.x)', '42', stress=True, files={'pkg/a.arda':'import b\nesta x = b.x','pkg/b.arda':'esta x = 42'})
    def test_module_cycle(self):
        self.run_code('tol a', error='circular import', files={'a.arda':'tol b','b.arda':'tol a'})
    def test_module_failure_caught(self):
        self.run_code('''
        try:
            tol a
        catch e:
            pedo(e)
        try:
            tol a
        catch e:
            pedo(e)
        ''', "broken\nmodule 'a' previously failed", stress=True, files={'a.arda':'throw "broken"'})
    def test_exception_unwind(self):
        self.run_code('''
        caro f():
            throw "mordor"
        try:
            f()
        catch e:
            collect()
            pedo(e)
        pedo(1)
        ''', 'mordor\n1', stress=True)
    def test_runtime_exception(self):
        self.run_code('try:\n    pedo(1 / 0)\ncatch e:\n    pedo(e)', 'division by zero', stress=True)
    def test_try_loop_cleanup(self):
        self.run_code('''
        an i in range(5):
            try:
                qui i == 0:
                    continue
                break
            catch e:
                pedo("unexpected")
        try:
            throw "ok"
        catch e:
            pedo(e)
        ''', 'ok', stress=True)
    def test_try_with_outer_iterator(self):
        self.run_code('''
        an i in range(3):
            try:
                throw str(i)
            catch e:
                collect()
                pedo(e)
        ''', '0\n1\n2', stress=True)
    def test_return_from_try(self):
        self.run_code('''
        caro f():
            try:
                nan 42
            catch e:
                nan 0
        pedo(f())
        throw "outer"
        ''', error='outer', stress=True)
    def test_unhandled_trace(self):
        self.run_code('caro f():\n    pedo(nope)\nf()', error='at f (')
    def test_file_io(self):
        self.run_code('write_text("data.txt", "hello\\n")\npedo(read_text("data.txt"))', 'hello', stress=True)
    def test_gc_cycles(self):
        out = self.run_code('''
        esta i = 0
        esta before = clock_gc()
        tenna i < 30000:
            esta a = []
            push(a, a)
            i = i + 1
        pedo(clock_gc() - before)
        collect()
        collect()
        pedo(heap(), heap_bytes())
        ''')
        rows = out.splitlines()
        self.assertGreater(int(rows[0]), 1)
        live, size = map(int, rows[1].split())
        self.assertLess(live, 100)
        self.assertLess(size, 20000)
    def test_gc_mutations(self):
        self.run_code('''
        esta keep = []
        an i in range(300):
            esta a = {"x": [i]}
            push(keep, a)
            collect()
        pedo(keep[299]["x"][0], len(keep))
        ''', '299 300', stress=True)
    def test_gc_stack_temporaries(self):
        self.run_code('''
        caro f(a, b):
            nan a + b
        caro g():
            collect()
            nan "b"
        pedo(f("a" + "1", g()))
        ''', 'a1b', stress=True)
    def test_compile_error(self):
        self.run_code('break', error='loop control outside loop')
    def test_bytecode(self):
        out = self.run_code('esta x = 2 + 3\npedo(x)', flags=('--bytecode',))
        self.assertIn('ADD', out)
        self.assertIn('DEFINE', out)
        self.assertIn('PRINT', out)
    def test_tokens(self):
        out = self.run_code('qui nae:\n    pedo(1)', flags=('--tokens',))
        self.assertIn('INDENT', out)
        self.assertIn('DEDENT', out)
    def test_repl(self):
        p = subprocess.run([str(BIN), '--gc-stress'], input='caro f(x):\n    nan x + 1\n\npedo(f(2))\n:quit\n', text=True, capture_output=True, timeout=10)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('3\n', p.stdout)

if __name__ == '__main__':
    unittest.main()
