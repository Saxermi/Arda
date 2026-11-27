"""Reference comparisons with NumPy and SciPy on independent file fixtures."""
from pathlib import Path
import subprocess
import tempfile
import unittest

import test_arda

try:
    import numpy as np
    import scipy.sparse as sp
except ImportError:
    np = None


@unittest.skipUnless(np is not None, "NumPy and SciPy are required")
class BeyondRamReferences(unittest.TestCase):
    def invoke(self, directory, code):
        entry = directory / "reference.arda"
        entry.write_text("import tensor as t\nimport compute as c\n" + code, encoding="utf-8")
        result = subprocess.run([str(test_arda.BIN), "--gc-stress", str(entry)], cwd=directory,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def fixture(self, directory, name, array):
        np.save(directory / (name + ".npy"), array)
        return f't.import_npy("{name}.npy","{name}.atn")\n'

    def read(self, directory, name, shape):
        return np.fromfile(directory / (name + ".atn"), dtype="<f8", offset=128).reshape(shape)

    def test_out_of_core_products_match_numpy_and_scipy(self):
        rng = np.random.default_rng(21)
        a, b = rng.normal(size=(31, 17)), rng.normal(size=(17, 24))
        matrix = sp.random(50, 30, density=0.12, random_state=21, format="csr")
        x = rng.normal(size=30)
        inputs = {"a": a, "b": b, "ptr": matrix.indptr.astype("<i8"),
                  "cols": matrix.indices.astype("<i8"), "values": matrix.data, "x": x}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup = "".join(self.fixture(root, name, array) for name, array in inputs.items())
            self.invoke(root, setup + """
c.memory_budget(1176)
c.file_matmul("a.atn","b.atn","product.atn",7)
c.file_spmv("ptr.atn","cols.atn","values.atn","x.atn","sparse_product.atn",5)
""")
            np.testing.assert_allclose(self.read(root, "product", (31, 24)), a @ b, atol=2e-13)
            np.testing.assert_allclose(self.read(root, "sparse_product", (50,)), matrix @ x,
                                       atol=2e-14)

    def test_tall_qr_matches_independent_gram_matrix(self):
        rng = np.random.default_rng(212)
        a = rng.normal(size=(93, 7))
        a[:, 6] = a[:, 0] + a[:, 1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup = self.fixture(root, "tall", a)
            self.invoke(root, setup + """
c.memory_budget(1512)
let r = c.file_qr_r("tall.atn",13)
t.save("r.atn",r)
""")
            r = self.read(root, "r", (7, 7))
            atol = 3e-12
            np.testing.assert_allclose(r, np.triu(r), atol=0)
            np.testing.assert_allclose(r.T @ r, a.T @ a, atol=atol)

    def test_right_svd_matches_numpy(self):
        rng = np.random.default_rng(212)
        a = rng.normal(size=(93, 7))
        a[:, 6] = a[:, 0] + a[:, 1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup = self.fixture(root, "tall", a)
            status = self.invoke(root, setup + """
try:
    let result = c.file_svd_right("tall.atn",13)
    t.save("singular.atn",result[0])
    t.save("right.atn",result[1])
catch e:
    print(e["message"])
""")
            if "LAPACK unavailable" in status:
                self.skipTest("file right SVD requires LAPACK=1")
            self.assertEqual(status, "")
            d = self.read(root, "singular", (7,))
            vt = self.read(root, "right", (7, 7))
            np.testing.assert_allclose(d, np.linalg.svd(a, compute_uv=False), atol=3e-13)
            np.testing.assert_allclose(vt @ vt.T, np.eye(7), atol=3e-13)
            np.testing.assert_allclose((vt.T * (d ** 2)) @ vt, a.T @ a, atol=3e-12)

    def test_randomized_right_svd_approximates_a_low_rank_row_space(self):
        rng = np.random.default_rng(213)
        left, _ = np.linalg.qr(rng.normal(size=(83, 3)))
        right, _ = np.linalg.qr(rng.normal(size=(41, 3)))
        a = (left * [10.0, 3.0, 1.0]) @ right.T
        a += rng.normal(scale=1e-8, size=a.shape)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup = self.fixture(root, "lowrank", a)
            status = self.invoke(root, setup + """
try:
    let result = c.file_rsvd_right("lowrank.atn",3,4,36,13)
    t.save("singular.atn",result[0])
    t.save("right.atn",result[1])
    result = c.file_rsvd_right("lowrank.atn",3,4,36,13)
    t.save("again.atn",result[1])
catch e:
    print(e["message"])
""")
            if "LAPACK unavailable" in status:
                self.skipTest("file randomized SVD requires LAPACK=1")
            self.assertEqual(status, "")
            d = self.read(root, "singular", (3,))
            vt = self.read(root, "right", (3, 41))
            np.testing.assert_array_equal(vt, self.read(root, "again", (3, 41)))
            np.testing.assert_allclose(d, np.linalg.svd(a, compute_uv=False)[:3], atol=5e-11)
            np.testing.assert_allclose(vt @ vt.T, np.eye(3), atol=2e-13)
            residual = np.linalg.norm(a - (a @ vt.T) @ vt) / np.linalg.norm(a)
            self.assertLess(residual, 2e-7)
