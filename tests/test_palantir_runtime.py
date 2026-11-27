"""Language and resource contracts for the Palantír runtime."""
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import test_arda


class PalantirTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_collections_and_bounded_ranges(self):
        self.run_code("""
            let a = (1, "x", (3,))
            let b = {3, 2, 3.0, 1}
            print(a, a[-1], a[1:], len(a), a == (1.0, "x", (3,)))
            print(b, len(b), set([1, 1, 2]), set(), (), 2 in b, 4 in b)
            print([x+1 for x in b], "x" in {"x": 1}, 3.0 in range(9, -1, -2))
            print({1,(2,),{3,4}} == {{4,3},(2.0,),1.0}, len(set(range(2000))))
            try:
                a[0] = 2
            catch error:
                print(a[0])
        """, "(1, x, (3,)) (3,) (x, (3,)) 3 nae\n"
             "{3, 2, 1} 3 {1, 2} set() () nae law\n[4, 3, 2] nae nae\nnae 2000\n1",
                      stress=True)

    def test_iterators_resume_independently_and_hold_tensor_source(self):
        self.run_code("""
            import tensor as t
            let a = iter(range(3))
            let b = iter(range(4, 0, -2))
            print(next(a), next(b), next(iter(a)))
            for x in a:
                print(x)
            print(next(a, "done"), next(b), next(b, nil))
            let values = iter([nil])
            print(next(values, 9), next(values, 9))
            try:
                next(values)
            catch error:
                print(error["code"])
            t.save("data.atn", t.array([1,2,3,4], "int64"))
            let source = t.blocks("data.atn",2)
            let blocks = iter(source)
            print(t.tolist(next(blocks)))
            for block in blocks:
                print(t.tolist(block))
            print(next(blocks, "end"))
            close_iterator(blocks)
            print(t.tolist(next(iter(source))))
            t.close_blocks(source)
        """, "0 4 1\n2\ndone 2 mun\nmun 9\nstop_iteration\n[1, 2]\n[3, 4]\nend\n[1, 2]",
                      stress=True)

    def test_generator_retains_frames_handlers_and_closures(self):
        self.run_code("""
            let calls = 0
            def source(n):
                calls = calls + 1
                let start = 10
                def inc(x):
                    start = start + x
                    return start
                for i in range(n):
                    try:
                        yield inc(i)
                        if i == 1:
                            throw {"code":"ok", "message":"test"}
                    catch error:
                        yield error["code"]
                yield nil
            let g = source(3)
            print(calls, next(g), calls)
            let cursor = iter(g)
            print(next(cursor), next(g))
            print(type(g), g == cursor)
            print([x for x in cursor])
            print(next(g, "done"))
            def passthrough(it):
                for x in it:
                    yield x
            print([x for x in passthrough(source(2))])
        """, "0 10 1\n11 ok\niterator nae\n[13, mun]\ndone\n[10, 11, ok, mun]", stress=True)

        repl = (
            "def make():\n    yield 42\n\n"
            "let cursor = make()\nmake = nil\ncollect()\n"
            "print(next(cursor))\nclose_iterator(cursor)\ncursor = nil\n:quit\n"
        )
        run = subprocess.run([test_arda.BIN, "--gc-stress", "--gc-stats"], input=repl,
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("42", run.stdout)
        self.assertIn("program_roots=0", run.stderr)

    def test_generator_error_exhausts_and_reentry_is_rejected(self):
        self.run_code("""
            let g = nil
            def source():
                yield 1
                next(g)
            g = source()
            print(next(g))
            try:
                next(g)
            catch error:
                print(error["code"])
            print(next(g, "done"))
            let suspended = source()
            print(next(suspended))
            close_iterator(suspended)
            print(next(suspended, "closed"))
            def broken(x):
                throw {"code":"from_generator","message":x}
                yield 0
            let b = broken("first advance")
            try:
                for value in b:
                    print(value)
            catch error:
                print(error["code"],error["message"])
            print(next(b, "done"))
        """, "1\nstate\ndone\n1\nclosed\nfrom_generator first advance\ndone", stress=True)

    def test_named_arguments_signatures_and_resolution_order(self):
        self.run_code("""
            let order = []
            def record(x):
                push(order, x)
                return x
            def f(a: Int64, b: Int64=record(8), c: Real=7) -> Real:
                return a*100+b*10+c
            print(f(c: record(3), a: record(1)), order)
            let signature_info = signature(f)
            print(signature_info["parameters"][0]["annotation"], signature_info["return"])
            let a = 0
            print(f(a=2), a)
            class Box:
                def init(self, x: Int64=0):
                    self.x = x
                def add(self, y=1):
                    return self.x+y
            let box = Box(x: 40)
            def replace_method():
                box.add = lambda y: y * 2
                return 2
            print(box.add(y: replace_method()), box.add(y: 3), signature(Box.add)["parameters"][0]["name"])
            try:
                f(1, a: 2)
            catch error:
                print(error["code"])
            try:
                f(c: 2)
            catch error:
                print(error["code"])
            try:
                f(1, invalid: 2)
            catch error:
                print(error["code"])
        """, "183 [3, 1, 8]\nInt64 Real\n287 2\n42 6 self\narity\narity\narity", stress=True)

    def test_formatted_string_expressions_and_escaping(self):
        self.run_code(r'''
            let x = 3
            let str = 1
            print(f"{{x}} = {x + 1}: { {"x": 5}["x"] }")
            print(f'{(x,)[0]} {f"{x*2}"}', f"", f"\t{{}}")
        ''', "{x} = 4: 5\n3 6  \t{}", stress=True)
        self.run_code('print(f"{}")', error="empty interpolation field")
        self.run_code('print(f"}")', error="unmatched brace")

    def test_context_exits_on_all_control_transfers(self):
        self.run_code("""
            let events = []
            class Lease:
                def init(self, name, suppress=false):
                    self.name = name
                    self.suppress = suppress
                def __enter__(self):
                    push(events, self.name)
                    return self.name
                def __exit__(self, error):
                    push(events, "end " + self.name)
                    if error != nil:
                        push(events, error["code"])
                    return self.suppress
            def leave():
                with Lease("outer") as name:
                    with Lease(name + " inner"):
                        return name
            print(leave())
            for i in range(3):
                with Lease(str(i)):
                    if i == 0:
                        continue
                    break
            with Lease("suppress", true):
                throw {"code":"ignored", "message":"test"}
            try:
                with Lease("propagate"):
                    throw {"code":"seen", "message":"test"}
            catch error:
                print(error["code"])
            print(events)
            class Fails(Lease):
                def __exit__(self, error):
                    throw {"code":"exit_failed", "message":"cleanup error"}
            try:
                with Lease("outside"):
                    with Fails("inside"):
                        pass
            catch error:
                print(error["code"])
            print(events[-2:])
        """, "outer\nseen\n[outer, outer inner, end outer inner, end outer, "
             "0, end 0, 1, end 1, suppress, end suppress, ignored, propagate, end propagate, seen]"
             "\nexit_failed\n[end outside, exit_failed]",
                      stress=True)
        self.run_code("""
            def invalid():
                with nil:
                    yield 1
        """, error="yield inside with")

    def test_stack_locals_shadow_and_release_on_exception(self):
        self.run_code("""
            let x = 7
            def scoped():
                let a = x
                let x = 10
                try:
                    let x = 12
                    throw {"code":"local", "message":"test"}
                catch error:
                    let y = x
                    print(a,x,y)
                for x in range(2):
                    let a = x
                    print(a)
                return x
            def closures():
                let values = []
                for i in range(3):
                    push(values, lambda: i)
                return values
            print(scoped())
            let f = closures()
            print(f[0](),f[1](),f[2]())
        """, "7 10 10\n0\n1\n10\n0 1 2", stress=True)

    def test_cli_limits_and_profiling(self):
        self.run_code("""
            try:
                while true:
                    pass
            catch error:
                print("caught")
        """, error="instruction limit exceeded", flags=("--max-instructions=40",))
        self.run_code("print((1,2,3,4))", error="operand stack limit",
                      flags=("--max-stack=2",))
        self.run_code("""
            def recurse():
                return recurse()
            try:
                recurse()
            catch error:
                print(error["code"])
        """, "stack_limit", flags=("--max-frames=20",))
        self.run_code("let data = [i for i in range(10000)]", error="managed VM memory limit",
                      flags=("--max-vm-bytes=100000",))
        self.run_code("print(1)", "1", flags=("--max-instructions=30", "--gc-profile"))

    def test_host_file_capability_preserves_standard_imports_and_numerics(self):
        self.run_code("""
            import numc as n
            import tensor as t
            import experiment as e
            import mathdoc as m
            import "std:compute" as c
            print(n.sum(n.ones(2)))
            let attempts = [lambda: read_text("x"), lambda: write_text("x","x"),
                lambda: t.save("x.atn",t.zeros([2],"float64")),
                lambda: c.stream_stats("x.atn",2), lambda: e.save("x.amf",{}),
                lambda: m.export(n.sum,"x.md","markdown")]
            for action in attempts:
                try:
                    action()
                catch error:
                    print(error["code"])
            try:
                import "local.arda" as local
            catch error:
                print(error["code"])
        """, "2\n" + "\n".join(["host_io"] * 7), flags=("--no-file-io",),
                      files={"local.arda": "print(42)\n"}, stress=True)

    def test_interruptible_process_checks_at_bytecode_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            program = path / "spin.arda"
            program.write_text('write_text("ready","1")\nwhile true:\n    pass\n', encoding="utf-8")
            with subprocess.Popen([test_arda.BIN, "--interruptible", program], cwd=tmp,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as child:
                try:
                    end = time.monotonic() + 5
                    while not (path / "ready").exists() and child.poll() is None:
                        self.assertLess(time.monotonic(), end)
                        time.sleep(0.01)
                    child.send_signal(signal.SIGINT)
                    _, err = child.communicate(timeout=5)
                    self.assertNotEqual(child.returncode, 0)
                    self.assertIn("execution interrupted", err)
                finally:
                    if child.poll() is None:
                        child.kill()
