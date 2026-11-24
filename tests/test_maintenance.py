"""Regression tests for the 0.13.1 runtime and frontend fixes."""
import pathlib
import re
import subprocess
import tempfile
import unittest
from test_arda import BIN
import test_arda


class MaintenanceTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def repl(self, code, *, stress=False):
        flags = ['--gc-stats'] + (['--gc-stress'] if stress else [])
        result = subprocess.run([str(BIN), *flags], input=code + ':quit\n',
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('Sanitizer', result.stderr)
        return result

    def roots(self, result):
        return int(re.search(r'program_roots=(\d+)', result.stderr).group(1))

    def test_repl_recovers_from_each_frontend_stage(self):
        for stress in (False, True):
            with self.subTest(stress=stress):
                p = self.repl('esta x = 40\npedo("bad\nesta broken =\n'
                              'break\npedo(x + 2)\n', stress=stress)
                self.assertIn('unterminated string', p.stderr)
                self.assertIn('got NEWLINE', p.stderr)
                self.assertIn('loop control outside loop', p.stderr)
                self.assertRegex(p.stdout, r'arda> 42\n')
                self.assertEqual(self.roots(p), 0)

    def test_repeated_partial_frontend_allocations(self):
        code = ('caro f(a, b=2):\n    esta l = [a,b,3]\n    break\n\n'
                'esta z = [1, 2)\nesta z = [1, ] +\n') * 100
        p = self.repl(code + 'pedo(123)\n', stress=True)
        self.assertEqual(p.stderr.count('loop control outside loop'), 100)
        self.assertEqual(p.stderr.count('mismatched delimiter'), 100)
        self.assertIn('arda> 123\n', p.stdout)
        self.assertEqual(self.roots(p), 0)

    def test_repl_bytecode_does_not_accumulate(self):
        p = self.repl('esta x = 0\n' + 'x = x + 1\n' * 3000 + 'pedo(x)\n')
        self.assertIn('arda> 3000\n', p.stdout)
        self.assertEqual(self.roots(p), 0)

    def test_closure_preserves_code_and_can_release_it(self):
        setup = ('caro make(x):\n    nan lambda y: x + y\n\n'
                 'esta f = make(40)\nmake = mun\n')
        noise = 'esta z = 1 + 2\n' * 100
        for stress in (False, True):
            with self.subTest(stress=stress):
                live = self.repl(setup + noise + 'collect()\npedo(f(2))\n', stress=stress)
                self.assertIn('arda> 42\n', live.stdout)
                self.assertEqual(self.roots(live), 1)
                dead = self.repl(setup + noise + 'f = mun\ncollect()\n', stress=stress)
                self.assertEqual(self.roots(dead), 0)

    def test_imported_syntax_error_can_be_caught(self):
        self.run_code('''
        try:
            tol broken
        catch e:
            pedo("caught")
        tol good
        collect()
        pedo(good.f(40))
        ''', 'caught\n42', stress=True,
            files={'broken.arda': 'caro f(x):\n    break\n',
                   'good.arda': 'caro f(x):\n    nan x + 2\n'})

    def test_cli_errors_exit_nonzero_without_leak_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'bad.arda'
            for text, flags, expected in [('pedo("oops', ['--tokens'], 'unterminated string'),
                                          ('esta x =', ['--bytecode'], 'got NEWLINE'),
                                          ('break', [], 'loop control outside loop')]:
                path.write_text(text)
                p = subprocess.run([str(BIN), *flags, str(path)], capture_output=True, text=True)
                self.assertEqual(p.returncode, 1, p.stderr)
                self.assertIn(expected, p.stderr)
                self.assertNotIn('Sanitizer', p.stderr)
            p = subprocess.run([str(BIN), str(path) + '.missing'], capture_output=True, text=True)
            self.assertEqual(p.returncode, 1)
            self.assertIn('cannot read', p.stderr)
            self.assertNotIn('Sanitizer', p.stderr)
