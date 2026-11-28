"""Independent references for lazy Float64 graphs and structured ACE plans."""
import ast
import math
from pathlib import Path
import subprocess
import tempfile
import unittest

from reference_compute import np, save_tensor
import test_arda

try:
    from scipy.sparse import csr_matrix
except ImportError:
    csr_matrix = None


@unittest.skipIf(np is None, "NumPy is required")
class AceV2ReferenceTests(unittest.TestCase):
    def test_seeded_fusion_tiling_negative_strides_and_reduction(self):
        rng = np.random.default_rng(224)
        x, y, z = rng.normal(size=(3, 17, 31))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in zip("xyz", (x, y, z)):
                save_tensor(root / f"{name}.atn", value)
            script = root / "graph.arda"
            script.write_text("""
import lazy as l
import compute as c
import tensor as t
let x = l.file("x.atn")
let y = l.input(t.view(t.mmap("y.atn"),[17,31],[31,-1],30))
let z = l.input(t.mmap("z.atn"))
let g = l.add(l.mul(x,y),z)
for kernel in ["fused","tiled","in_ram"]:
    t.save(kernel+".atn",c.execute(c.plan("graph",g,{"kernel":kernel,"block_elements":37})))
let product = l.mul(l.sin(x),l.exp(l.neg(x)))
let sum = l.sum(product)
print(l.materialize(sum,{"block_elements":37}),
      l.materialize(sum,{"kernel":"tiled","block_elements":13}))
let shared = l.add(x,y)
let general = l.div(l.sub(l.mul(shared,shared),z),l.exp(l.neg(z)))
t.save("general.atn",l.materialize(general,{"block_elements":29}))
""", encoding="utf-8")
            result = subprocess.run([test_arda.BIN, "--gc-stress", script], cwd=root,
                                    capture_output=True, text=True, timeout=15, check=True)
            expected = x * y[:, ::-1] + z
            for kernel in ("fused", "tiled", "in_ram"):
                with self.subTest(kernel=kernel):
                    actual = np.fromfile(root / f"{kernel}.atn", dtype="<f8", offset=128)
                    np.testing.assert_array_equal(actual.reshape(x.shape), expected)
            shared = x + y[:, ::-1]
            actual = np.fromfile(root / "general.atn", dtype="<f8", offset=128)
            np.testing.assert_allclose(actual.reshape(x.shape), (shared * shared - z) / np.exp(-z),
                                       rtol=7e-16, atol=1e-15)
            reduction = math.fsum(math.sin(a) * math.exp(-a) for a in x.flat)
            fused, tiled = map(float, result.stdout.split())
            self.assertEqual(fused, tiled)
            np.testing.assert_allclose(fused, reduction, rtol=2e-14, atol=1e-13)

    @unittest.skipIf(csr_matrix is None, "SciPy is required")
    def test_csr_with_empty_rows_and_cancellation(self):
        data = [2.0, -1.0, 3.0, 2.0, -2.0, 4.0]
        offsets = [0, 2, 2, 5, 6, 6]
        columns = [1, 3, 0, 2, 4, 1]
        vector = [10.0, -3.0, 4.0, 8.0, 19.0]
        matrix = csr_matrix((data, columns, offsets), shape=(5, 5))
        source = f"""
        import sparse as s
        import tensor as t
        import compute as c
        let a = s.csr(5,5,{offsets},{columns},{data})
        let x = t.array({vector},"float64")
        let p = c.plan("apply",a,x)
        print(t.tolist(c.execute(p)))
        """
        result = test_arda.ArdaTests.run_code(self, source, stress=True)
        np.testing.assert_array_equal(ast.literal_eval(result), matrix @ vector)
