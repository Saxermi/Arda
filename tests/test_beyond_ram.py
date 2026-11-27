"""Portable contracts for bounded file processing and visualization."""
import struct
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import test_arda


class BeyondRamTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_map_filter_and_reduce_release_consumed_blocks(self):
        self.run_code("""
import tensor as t
import compute as c
t.save("source.atn",t.array([1,2,3,4,5,6],"int64",[3,2]))
collect()
def times_two(block):
    return block*2
def keep(value):
    return value > 3
def add(total,block):
    return total + t.sum(block)
c.memory_budget(32)
print(c.map_file("source.atn","mapped.atn",times_two,nil,2,176))
print(c.filter_file("mapped.atn","subset.atn",keep,2,176))
print(c.reduce_file("subset.atn",0,add,2))
c.memory_budget(nil)
print(t.file_info("mapped.atn")["shape"],t.tolist(t.read_block("mapped.atn",0,6)))
print(t.file_info("subset.atn")["shape"],t.tolist(t.read_block("subset.atn",0,5)))
""", "6\n5\n40\n[3, 2] [2, 4, 6, 8, 10, 12]\n[5] [4, 6, 8, 10, 12]", stress=True)

    def test_transformation_rollback_and_empty_selection(self):
        self.run_code("""
import tensor as t
import compute as c
t.save("source.atn",t.array([1,2,3,4],"int64"))
t.save("kept.atn",t.array([33],"int64"))
def bad(block):
    if t.at(block,[0]) == 3:
        throw {"code":"test","message":"callback rollback"}
    return block
def none(value):
    return false
try:
    c.map_file("source.atn","kept.atn",bad,nil,2)
catch e:
    print(e["code"])
print(t.tolist(t.mmap("kept.atn")))
print(c.filter_file("source.atn","empty.atn",none,3),t.file_info("empty.atn")["shape"])
""", "test\n[33]\n0 [0]", stress=True)

    def test_matmul_uses_bounded_workspace_and_handles_empty_inner_axis(self):
        self.run_code("""
import tensor as t
import compute as c
t.save("a.atn",t.array([1,2,3,4,5,6],"float64",[2,3]))
t.save("b.atn",t.array([1,2,3,4,5,6],"float64",[3,2]))
t.save("zero_a.atn",t.zeros([2,0],"float64"))
t.save("zero_b.atn",t.zeros([0,3],"float64"))
collect()
c.memory_budget(23)
try:
    c.file_matmul("a.atn","b.atn","out.atn",1)
catch e:
    print(e["code"])
c.memory_budget(24)
c.file_matmul("a.atn","b.atn","out.atn",1,160)
c.file_matmul("zero_a.atn","zero_b.atn","empty_product.atn",2)
c.memory_budget(nil)
print(t.tolist(t.mmap("out.atn")))
print(t.tolist(t.mmap("empty_product.atn")))
""", "memory_budget\n[[22, 28], [49, 64]]\n[[0, 0, 0], [0, 0, 0]]",
                      stress=True)

    def test_spmv_validates_csr_before_replacing_result(self):
        self.run_code("""
import tensor as t
import compute as c
t.save("ptr.atn",t.array([0,2,2,4],"int64"))
t.save("cols.atn",t.array([0,2,0,1],"int64"))
t.save("bad.atn",t.array([0,2,1,1],"int64"))
t.save("val.atn",t.array([2,1,-1,5],"float64"))
t.save("x.atn",t.array([3,4,2],"float64"))
collect()
c.memory_budget(24)
c.file_spmv("ptr.atn","cols.atn","val.atn","x.atn","out.atn",1,152)
try:
    c.file_spmv("ptr.atn","bad.atn","val.atn","x.atn","out.atn",1)
catch e:
    print(e["code"])
c.memory_budget(nil)
print(t.tolist(t.mmap("out.atn")))
""", "domain\n[8, 0, 17]", stress=True)

    def invoke(self, root, source):
        script = root / "case.arda"
        script.write_text(source, encoding="utf-8")
        result = subprocess.run([str(test_arda.BIN), "--gc-stress", str(script)], cwd=root,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_svg_geometry_and_downsampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.invoke(root, """
import plot as p
import tensor as t
import compute as c
t.save("series.atn",t.array([0,0,1,2,2,0,3,1,4,2],"float64",[5,2]))
t.save("grid.atn",t.array([0,1,0,1],"float64",[2,2]))
t.save("field.atn",t.array([1,0,0,0,0,1,-1,0],"float64",[2,2,2]))
collect()
c.memory_budget(32)
p.svg("line","series.atn","line.svg",{"stride":2})
p.svg("scatter","series.atn","scatter.svg",{"stride":2})
p.svg("histogram","grid.atn","histogram.svg",{"bins":2})
p.svg("heatmap","grid.atn","heatmap.svg")
p.svg("contour","grid.atn","contour.svg",{"level":0.5})
p.svg("vector","field.atn","vector.svg")
""")
            ns = {"svg": "http://www.w3.org/2000/svg"}

            def svg(name):
                return ET.parse(root / f"{name}.svg").getroot()

            line = svg("line").findall("svg:path", ns)[1].attrib["d"]
            self.assertEqual(line.split(), ["M24.0000", "376.0000", "L320.0000", "376.0000",
                                            "L616.0000", "24.0000"])
            self.assertEqual(len(svg("scatter").findall("svg:circle", ns)), 3)
            histogram = svg("histogram").findall("svg:rect[@data-count]", ns)
            self.assertEqual([cell.attrib["data-count"] for cell in histogram], ["2", "2"])
            self.assertEqual(len(svg("heatmap").findall("svg:rect", ns)), 5)
            contour = svg("contour").find("svg:path", ns).attrib["d"]
            self.assertEqual(contour, "M320.0000 0.0000L320.0000 400.0000")
            self.assertEqual(len(svg("vector").findall("svg:line", ns)), 3)

    def test_vtk_layout_endian_and_export_quota(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kept.svg").write_text("original", encoding="utf-8")
            status = self.invoke(root, """
import plot as p
import tensor as t
import compute as c
t.save("volume.atn",t.array([1.25,-2,3,4,5,6,7,8],"float64",[2,2,2]))
t.save("series.atn",t.array([0,1,2],"float64"))
collect()
c.memory_budget(8)
p.vtk("volume.atn","volume.vtk",512)
try:
    p.svg("line","series.atn","kept.svg",{"disk_budget":170})
catch e:
    print(e["code"])
""")
            self.assertEqual(status, "disk_budget\n")
            vtk = (root / "volume.vtk").read_bytes()
            header, payload = vtk.split(b"LOOKUP_TABLE default\n")
            self.assertIn(b"BINARY\nDATASET STRUCTURED_POINTS\nDIMENSIONS 2 2 2\n", header)
            self.assertIn(b"POINT_DATA 8\nSCALARS values double 1\n", header)
            self.assertEqual(struct.unpack(">8d", payload[:-1]), (1.25, -2, 3, 4, 5, 6, 7, 8))
            self.assertEqual((root / "kept.svg").read_text(), "original")
            self.assertFalse(list(root.glob("*.tmp.*")))
