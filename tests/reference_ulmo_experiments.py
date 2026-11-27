"""Compare explicit slab and ghost products with an independent NumPy stencil."""
import subprocess
import tempfile
import unittest
from pathlib import Path

import test_arda
from benchmarks.cpu_baseline import save_tensor

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipUnless(np is not None, "NumPy is required for grid decomposition references")
class SlabReferenceTests(unittest.TestCase):
    def test_slabs_match_independent_stencil_across_axes_and_tile_boundaries(self):
        rng = np.random.default_rng(205)
        fixtures = (
            ((3, 5, 4), (1, 2, 4), ("neumann", "periodic", "dirichlet"), 1, 3),
            ((3, 5, 4), (1, 2, 4), ("periodic", "dirichlet", "neumann"), 2, 4),
            ((3, 4101), (2, 1), ("periodic", "neumann"), 0, 2),
            ((2, 7), (2, 1), ("dirichlet", "neumann"), 1, 3),
        )
        for shape, spacing, boundaries, axis, parts in fixtures:
            with self.subTest(shape=shape, axis=axis, parts=parts):
                u = rng.integers(-16, 17, size=shape).astype(float)
                expected = np.zeros(shape)
                for dim, (h, bc) in enumerate(zip(spacing, boundaries)):
                    before = np.roll(u, 1, axis=dim)
                    after = np.roll(u, -1, axis=dim)
                    if bc != "periodic":
                        lower = [slice(None)] * u.ndim
                        upper = lower.copy()
                        lower[dim], upper[dim] = 0, -1
                        lower, upper = tuple(lower), tuple(upper)
                        before[lower] = 0 if bc == "dirichlet" else u[lower]
                        after[upper] = 0 if bc == "dirichlet" else u[upper]
                    expected += (2 * u - before - after) / h**2
                slabs = np.array_split(u, parts, axis=axis)
                targets = np.array_split(expected, parts, axis=axis)
                with tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    for j, slab in enumerate(slabs):
                        save_tensor(root / f"slab{j}.atn", slab.ravel(order="C"))
                    source = f'''import pde as p
import tensor as t
let grid = p.grid({list(shape)},{list(spacing)},{list(boundaries)})
let states = [t.mmap("slab"+str(j)+".atn") for j in range({parts})]
for j in range({parts}):
    let part = p.partition(grid,{axis},{parts},j)
    let lower = nil
    let upper = nil
    if part["lower"] != nil:
        let k = part["lower"]
        lower = p.subdomain_face(grid,{axis},{parts},k,states[k],"upper")
    if part["upper"] != nil:
        let k = part["upper"]
        upper = p.subdomain_face(grid,{axis},{parts},k,states[k],"lower")
    let result = p.apply_subdomain(grid,{axis},{parts},j,states[j],lower,upper)
    t.save("result"+str(j)+".atn",result)
'''
                    (root / "slabs.arda").write_text(source, encoding="utf-8")
                    run = subprocess.run([str(test_arda.BIN), "--gc-stress", "slabs.arda"], cwd=root,
                                         capture_output=True, text=True, timeout=20)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    for j, target in enumerate(targets):
                        got = np.fromfile(root / f"result{j}.atn", dtype="<f8", offset=128)
                        np.testing.assert_array_equal(got, target.ravel(order="C"))
