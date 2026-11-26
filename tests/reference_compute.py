"""Independent seeded NumPy checks for tiled and fused Float64 compute."""
import sys
from pathlib import Path
import subprocess
import tempfile
import unittest

import test_arda

try:
    import numpy as np
except ImportError:
    np = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from cpu_baseline import save_tensor


@unittest.skipIf(np is None, "NumPy is required")
class ComputeReferenceTests(unittest.TestCase):
    def test_ragged_strided_gemm_signed_gemv_and_fusion(self):
        rng = np.random.default_rng(1806)
        left = rng.standard_normal((47, 35))
        right = rng.standard_normal((29, 35))
        vector = rng.standard_normal(35)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_tensor(root / "left.atn", left)
            save_tensor(root / "right.atn", right)
            save_tensor(root / "vector.atn", vector)
            script = root / "check.arda"
            script.write_text("""
import tensor as t
import compute as c
let a = t.mmap("left.atn")
let b = t.permute(t.mmap("right.atn"),[1,0])
let p = c.plan("matmul",a,b,{"kernel":"tiled","block":16})
t.save("matrix.atn",c.execute(p,a,b))
let v = t.mmap("vector.atn")
let reversed = t.view(v,[35],[-1],34)
t.save("gemv.atn",c.tiled_matmul(a,reversed,7))
t.save("fused.atn",c.fused_products(v,reversed,reversed,reversed,v))
print(c.map_reduce("sum_squares",reversed))
""")
            result = subprocess.run([test_arda.BIN, "--gc-stress", script], cwd=root,
                                    capture_output=True, text=True, timeout=15, check=True)

            def read(name):
                return np.fromfile(root / name, dtype="<f8", offset=128)

            np.testing.assert_allclose(read("matrix.atn").reshape(47, 29), left @ right.T,
                                       rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(read("gemv.atn"), left @ vector[::-1], rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(read("fused.atn"),
                                          (vector * vector[::-1] + vector[::-1] ** 2) + vector)
            np.testing.assert_allclose(float(result.stdout), np.sum(vector**2), rtol=1e-13)
