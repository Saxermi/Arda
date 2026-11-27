"""NumPy format and payload references for streaming NPY import."""
from pathlib import Path
import subprocess
import tempfile
import unittest

import test_arda

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipUnless(np is not None, "NumPy is required for file references")
class TensorFileReferences(unittest.TestCase):
    def invoke(self, root, source):
        path = root / "import.arda"
        path.write_text(source, encoding="utf-8")
        run = subprocess.run([str(test_arda.BIN), "--gc-stress", str(path)], cwd=root,
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        return run.stdout

    def test_npy_versions_and_binary_values_match_numpy(self):
        cases = [
            ((1, 0), np.array([[-2, 0], [3, 8]], dtype="<i4")),
            ((2, 0), np.array([-(2**62), 2**62 + 1], dtype="<i8")),
            ((3, 0), np.array([[1.5 + 2j, -3.5j]], dtype="<c16")),
            ((1, 0), np.array([[True, False], [False, True]], dtype="bool")),
            ((1, 0), np.linspace(-1, 1, 50000, dtype="<f4").reshape(100, 500)),
            ((1, 0), np.empty((0, 3), dtype="<f8")),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, (version, expected) in enumerate(cases):
                with (root / f"src{index}.npy").open("wb") as source:
                    np.lib.format.write_array(source, expected, version=version, allow_pickle=False)
            output = self.invoke(root, """
import tensor as t
import compute as c
c.memory_budget(0)
for index in range(6):
    t.import_npy("src"+str(index)+".npy","out"+str(index)+".atn")
print(c.memory_stats()["tensor_heap_bytes"])
""")
            self.assertEqual(output, "0\n")
            for index, (_, expected) in enumerate(cases):
                with self.subTest(index=index, shape=expected.shape, dtype=expected.dtype):
                    data = np.fromfile(root / f"out{index}.atn", dtype=expected.dtype, offset=128)
                    np.testing.assert_array_equal(data.reshape(expected.shape), expected)

    def test_npy_unsupported_layout_and_budget_keep_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "fortran.npy", np.asfortranarray(np.ones((2, 4))))
            np.save(root / "big_endian.npy", np.array([1], dtype=">f8"))
            np.save(root / "valid.npy", np.array([1, 2], dtype="<f8"))
            (root / "short.npy").write_bytes((root / "valid.npy").read_bytes()[:-1])
            result = self.invoke(root, """
import tensor as t
t.save("kept.atn",t.array([17],"float64"))
for file in ["fortran.npy","big_endian.npy","short.npy"]:
    try:
        t.import_npy(file,"kept.atn")
    catch e:
        print(e["code"])
try:
    t.import_npy("valid.npy","kept.atn",143)
catch e:
    print(e["code"])
print(t.tolist(t.mmap("kept.atn")))
""")
            self.assertEqual(result, "file_format\nfile_format\nfile_format\ndisk_budget\n[17]\n")
            self.assertFalse(list(root.glob("*.tmp.*")))
