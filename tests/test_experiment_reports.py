"""Persistent experiments and bounded AMDE exports."""
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from textwrap import dedent
import unittest

import test_arda


class ExperimentReportTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def run_script(self, root, source):
        script = root / "check.arda"
        script.write_text(dedent(source).lstrip(), encoding="utf-8")
        result = subprocess.run([test_arda.BIN, "--gc-stress", script], cwd=root,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def test_fingerprint_matches_hashlib_including_padding_and_block_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = []
            for size in [0, 1, 55, 56, 63, 64, 65, 8193, 16403]:
                data = bytes(i % 251 for i in range(size))
                (root / str(size)).write_bytes(data)
                expected.append(hashlib.sha256(data).hexdigest())
            actual = self.run_script(root, """
                import experiment as e
                for size in [0,1,55,56,63,64,65,8193,16403]:
                    print(e.fingerprint(str(size),size))
                try:
                    e.fingerprint("65",64)
                catch error:
                    print(error["code"])
            """)
            self.assertEqual(actual.splitlines(), expected + ["disk_budget"])

    def test_manifest_resumes_a_heat_run_in_a_new_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.run_script(root, """
                import pde as p
                import numc as n
                import experiment as e
                let g = p.grid([5],1,"periodic")
                let y = n.array([1,3,5,2,6])
                let a = p.heat(g,y,0.1,0.5,0.125,{"checkpoint":true})
                p.heat_save("heat.ckpt",a["checkpoint"],4096)
                e.save("run.amf",{"file":"heat.ckpt","hash":e.fingerprint("heat.ckpt",4096),
                    "seed":9007199254740993,"options":{"factor":1.25},"label":"Wärme",
                    "flags":[true,false,nil]},1024)
            """)
            for filename, magic in [("run.amf", b"ARDAEXP1"), ("heat.ckpt", b"ARDACKP2")]:
                data = (root / filename).read_bytes()
                self.assertEqual(data[:8], magic)
                self.assertEqual(data[-32:], hashlib.sha256(data[:-32]).digest())
                damaged = bytearray(data)
                damaged[-39] ^= 1
                (root / (filename + ".corrupt")).write_bytes(damaged)
                (root / (filename + ".truncated")).write_bytes(data[:-1])
            # The previous heat layout had neither a digest nor the new magic.
            heat = (root / "heat.ckpt").read_bytes()
            (root / "legacy.ckpt").write_bytes(b"ARDACKP1" + heat[8:-32])
            result = self.run_script(root, """
                import pde as p
                import numc as n
                import experiment as e
                import compute as c
                let manifest = e.load("run.amf",1024)
                print(manifest["seed"],manifest["options"]["factor"],manifest["label"],manifest["flags"])
                print(manifest["hash"]==e.fingerprint(manifest["file"],4096))
                let b = p.heat_resume(manifest["file"],0.5)
                let legacy = p.heat_resume("legacy.ckpt",0.5)
                let g = p.grid([5],1,"periodic")
                let full = p.heat(g,n.array([1,3,5,2,6]),0.1,1,0.125)
                print(b["time"],n.norm(b["solution"]-full["solution"])==0,
                      n.norm(b["solution"]-legacy["solution"])==0)
                let live = c.memory_stats()["tensor_heap_bytes"]
                for path in ["run.amf.corrupt","run.amf.truncated","heat.ckpt.corrupt","heat.ckpt.truncated"]:
                    try:
                        if startswith(path,"run"):
                            e.load(path)
                        else:
                            p.heat_resume(path,0.25)
                    catch error:
                        print(error["code"],c.memory_stats()["tensor_heap_bytes"]<=live)
            """)
            self.assertEqual(result.splitlines(), [
                "9007199254740993 1.25 Wärme [nae, law, mun]", "nae", "1 nae nae",
                "manifest_format nae", "manifest_format nae", "checkpoint_format nae",
                "checkpoint_format nae",
            ])

    def test_manifest_quota_and_unsupported_metadata_preserve_the_previous_file(self):
        self.run_code("""
import experiment as e
import tensor as t
import compute as c
let n = e.save("r.amf",{"n":-9007199254740993,"x":[]})
try:
    e.save("r.amf",{"n":4,"x":[]},n-1)
catch error:
    print(error["code"])
try:
    e.save("r.amf",{"table":t.zeros([2],"float64")})
catch error:
    print(error["code"])
let loop = []
push(loop,loop)
try:
    e.save("r.amf",{"cycle":loop})
catch error:
    print(error["code"])
collect()
collect()
let live = c.memory_stats()["tensor_heap_bytes"]
c.memory_budget(live+n-1)
try:
    e.load("r.amf")
catch error:
    print(error["code"])
c.memory_budget(live+n)
print(e.load("r.amf",n)["n"],c.memory_stats()["tensor_heap_bytes"]==live)
""", "disk_budget\nmetadata\nmetadata\nmemory_budget\n-9007199254740993 nae",
                      stress=True)

    def test_manifest_rejects_invalid_types_even_with_a_valid_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, payload in {
                "unknown": b"\x99",
                "badshape": b"\x06" + struct.pack("<Q", 2**63),
                "duplicate": b"\x07" + struct.pack("<Q", 2) +
                             (struct.pack("<Q", 1) + b"x\x00")*2,
                "trailing": b"\x07" + struct.pack("<Q", 0) + b"\x00",
                "invalidutf8": b"\x07" + struct.pack("<Q", 1) + struct.pack("<Q", 1) + b"\xff\x00",
            }.items():
                body = b"ARDAEXP1" + struct.pack("<Q", len(payload)) + payload
                (root / name).write_bytes(body + hashlib.sha256(body).digest())
            actual = self.run_script(root, """
                import experiment as e
                for path in ["unknown","badshape","duplicate","trailing","invalidutf8"]:
                    try:
                        e.load(path)
                    catch error:
                        print(error["code"])
            """)
            self.assertEqual(actual.splitlines(), ["manifest_format"]*5)

    def test_amde_formats_keep_static_and_observed_data_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = self.run_script(root, r'''
                import mathdoc as m
                import numc as n
                import solver as s
                let a = n.array([[2]])
                let result = s.verify(a,n.array([4]),n.array([2]))
                let options = {"title":"Heat & <test>","units":{"x":"K"},
                               "verification":result,"trace":[{"step":1,"note":"</pre>"}]}
                for format in ["markdown","json","html","latex"]:
                    print(m.export(n.solve,format,format,options))
                def marker(x):
                    write_text("must-not-exist","side effect")
                    return x*x
                m.export(marker,"marker.json","json")
                try:
                    m.export(n.solve,"json","json",{"max_bytes":8})
                catch error:
                    print(error["code"])
                try:
                    m.export(n.solve,"json","json",{"trace":[n.ones(3)]})
                catch error:
                    print(error["code"])
            ''')
            lengths = [str((root / fmt).stat().st_size)
                       for fmt in ["markdown", "json", "html", "latex"]]
            self.assertEqual(output.splitlines(), lengths + ["disk_budget", "metadata"])
            self.assertFalse((root / "must-not-exist").exists())
            data = json.loads((root / "json").read_text())
            self.assertEqual(data["version"], 1)
            self.assertEqual(data["title"], "Heat & <test>")
            self.assertEqual(data["static"]["formula_tex"], "Ax=b")
            self.assertIn("pivoting", data["static"]["algorithm"])
            self.assertEqual(data["units_declared"], {"x": "K"})
            self.assertTrue(data["observed"]["verification"]["verified"])
            self.assertEqual(data["observed"]["trace"], [{"step": 1, "note": "</pre>"}])
            self.assertIn("## Verification (observed)", (root / "markdown").read_text())
            html = (root / "html").read_text()
            self.assertIn("Heat &amp; &lt;test&gt;", html)
            self.assertIn(r"\u003c/pre\u003e", html)
            self.assertIn("Ax=b", (root / "latex").read_text())
            self.assertEqual(list(root.glob("*.tmp.*")), [])
            latex = shutil.which("pdflatex")
            if latex:
                (root / "report.tex").write_bytes((root / "latex").read_bytes())
                built = subprocess.run([latex, "-no-shell-escape", "-halt-on-error",
                                        "-interaction=nonstopmode", "report.tex"],
                                       cwd=root, capture_output=True, text=True, timeout=20)
                self.assertEqual(built.returncode, 0, built.stdout)
