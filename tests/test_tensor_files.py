"""Streaming text import and staged tile output contracts."""
import subprocess
import tempfile
import unittest
from pathlib import Path

import test_arda


class TensorFileTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_import_streams_csv_and_tsv_under_the_tensor_budget(self):
        self.run_code("""
import compute as c
import tensor as t
c.memory_budget(0)
t.import_delimited("pairs.csv","pairs.atn",[2,2],{"header":true,"disk_budget":160})
t.import_delimited("values.tsv","values.atn",[1,3],{"delimiter":"\t"})
print(c.memory_stats()["tensor_heap_bytes"])
c.memory_budget(nil)
print(t.tolist(t.read_tile("pairs.atn",[0,0],[2,2])))
print(t.tolist(t.read_tile("values.atn",[0,0],[1,3])))
""", "0\n[[1.25, -2], [300, 4.5]]\n[[10, -20, 30]]", stress=True,
                      files={"pairs.csv": '"first, ""key""",second\r\n1.25,"-2"\r\n3e2,4.5',
                             "values.tsv": "10\t-20\t30\n"})

    def test_delimited_shape_and_disk_failures_preserve_destination(self):
        self.run_code("""
import tensor as t
t.save("kept.atn",t.array([9],"float64"))
for path in ["wrong.csv","extra.csv","empty.csv","large_field.csv"]:
    try:
        t.import_delimited(path,"kept.atn",[1,2])
    catch e:
        print(e["code"])
try:
    t.import_delimited("good.csv","kept.atn",[1,2],{"disk_budget":143})
catch e:
    print(e["code"])
try:
    t.import_delimited("empty.csv","kept.atn",[0,1],{"header":true})
catch e:
    print(e["code"])
print(t.tolist(t.mmap("kept.atn")))
""", "file_format\nfile_format\nfile_format\nfile_format\ndisk_budget\nfile_format\n[9]", stress=True,
                      files={"wrong.csv": "1,2,3\n", "extra.csv": "1,2\n3,4\n",
                             "empty.csv": "", "good.csv": "1,2\n",
                             "large_field.csv": "0" * 257 + ",2\n"})

    def test_delimited_values_cross_the_copy_buffer(self):
        self.run_code("""
import compute as c
import tensor as t
c.memory_budget(0)
t.import_delimited("long.csv","long.atn",[5000,2])
c.memory_budget(nil)
print(t.tolist(t.read_tile("long.atn",[4095,0],[2,2])))
""", "[[4095, -4095], [4096, -4096]]", stress=True,
                      files={"long.csv": "".join(f"{i},{-i}\n" for i in range(5000))})

    def test_staged_tiles_publish_once_and_preserve_existing_mappings(self):
        self.run_code("""
import tensor as t
import compute as c
let original = t.array([1,2,3,4],"int64",[2,2])
t.save("out.atn",original)
let old = t.mmap("out.atn")
let writer = t.open_output("out.atn",[2,2],"int64",160)
print(type(writer),t.tolist(t.read_tile(writer,[0,0],[2,2])))
let reversed = t.view(original,[2,2],[2,-1],1)
let live = c.memory_stats()["tensor_heap_bytes"]
c.memory_budget(live+8)
t.write_tile(writer,[0,0],reversed)
c.memory_budget(nil)
t.write_tile(writer,[1,1],t.array([9],"int64",[1,1]))
print(t.tolist(t.read_tile(writer,[0,0],[2,2])))
print(t.tolist(t.mmap("out.atn")))
t.commit_output(writer)
print(t.tolist(t.mmap("out.atn")),t.tolist(old))
try:
    t.read_tile(writer,[0,0],[1,1])
catch e:
    print(e["code"])
""", "tensor_output [[0, 0], [0, 0]]\n[[2, 1], [4, 9]]\n"
             "[[1, 2], [3, 4]]\n[[2, 1], [4, 9]] [[1, 2], [3, 4]]\nio",
                      stress=True)

    def test_output_abandonment_quota_and_sparse_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "existing_directory").mkdir()
            script = root / "case.arda"
            script.write_text("""
import tensor as t
import compute as c
let shape = [2000000,9]
try:
    t.open_output("over.atn",shape,"float64",144000127)
catch e:
    print(e["code"])
def abandon():
    let output = t.open_output("unused.atn",[1],"float64")
    t.write_tile(output,[0],t.array([9],"float64"))
abandon()
collect()
let canceled = t.open_output("canceled.atn",[1],"float64")
t.abort_output(canceled)
t.abort_output(canceled)
let cannot_publish = t.open_output("existing_directory",[1],"float64")
try:
    t.commit_output(cannot_publish)
catch e:
    print(e["code"])
let writer = t.open_output("big.atn",shape,"float64",144000128)
let tile = t.array([3.5],"float64",[1,1])
let live = c.memory_stats()["tensor_heap_bytes"]
c.memory_budget(live)
t.write_tile(writer,[1999999,8],tile)
t.commit_output(writer)
c.memory_budget(nil)
print(t.tolist(t.read_tile("big.atn",[1999999,7],[1,2])))
""", encoding="utf-8")
            result = subprocess.run([str(test_arda.BIN), "--gc-stress", str(script)], cwd=root,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "disk_budget\nio\n[[0, 3.5]]\n")
            self.assertTrue((root / "existing_directory").is_dir())
            self.assertEqual({p.name for p in root.iterdir()},
                             {"case.arda", "big.atn", "existing_directory"})
            self.assertEqual((root / "big.atn").stat().st_size, 144000128)
