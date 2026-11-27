"""Workspace quotas, streaming statistics and tensor tiles."""
from pathlib import Path
import resource
import signal
import struct
import subprocess
import tempfile
import unittest

import test_arda
from test_tensor_storage import tensor_header


class TensorMemoryTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_memory_stats_count_shared_storage_once(self):
        self.run_code("""
        tol compute as c
        tol tensor as t
        tol numc as nc
        esta x = nc.array([0,1,2,3])
        esta view = t.view(x,[2],[2],1)
        esta m = c.memory_stats(view)
        pedo(m["logical_bytes"],m["backing_bytes"],m["tensor_heap_bytes"],m["storage"])
        t.save("values",x)
        esta mapped = t.mmap("values")
        esta mapped_view = t.view(mapped,[2],[-1],2)
        mapped = mun
        x = mun
        collect()
        m = c.memory_stats(mapped_view)
        pedo(m["logical_bytes"],m["backing_bytes"],m["tensor_heap_bytes"],m["mapped_bytes"],m["storage"])
        view = mun
        collect()
        pedo(c.memory_budget(0),c.memory_stats()["tensor_heap_bytes"],t.at(mapped_view,[0]))
        mapped_view = mun
        collect()
        pedo(c.memory_stats()["mapped_bytes"],c.memory_budget(mun))
        """, "16 32 32 heap\n16 32 32 160 mapped\n0 0 2\n0 mun", stress=True)

    def test_budget_preflight_and_workspace_cache(self):
        self.run_code("""
        tol compute as c
        tol tensor as t
        tol numc as nc
        t.save("values",t.array([1,2,3,4,5],"float64"))
        collect()
        c.memory_budget(15)
        esta plan = c.plan("stream_stats","values",2)
        pedo(plan["elements"],plan["block_elements"],plan["workspace_bytes"],plan["output_bytes"],plan["fits"])
        try:
            c.stream_stats("values",2)
        catch error:
            pedo(error["message"])
        esta m = c.memory_stats()
        pedo(m["tensor_heap_bytes"],m["workspace_cached_bytes"],m["workspace_reserved_bytes"])
        c.memory_budget(16)
        plan = c.plan("stream_stats","values",2)
        esta stats = c.stream_stats("values",2)
        pedo(plan["fits"],stats["count"],stats["mean"],stats["variance"])
        stats = c.stream_stats("values",2)
        m = c.memory_stats()
        pedo(m["workspace_cached_bytes"],m["workspace_reserved_bytes"],stats["variance"])
        esta dense = nc.ones(2)
        m = c.memory_stats()
        pedo(m["tensor_heap_bytes"],m["workspace_cached_bytes"])
        try:
            c.memory_budget(15)
        catch error:
            pedo(error["message"])
        try:
            t.zeros([1],"uint8")
        catch error:
            pedo(error["message"])
        pedo(c.memory_budget())
        dense = mun
        collect()
        c.stream_stats("values",2)
        pedo(c.trim_pool(),c.memory_stats()["workspace_cached_bytes"])
        """, "5 2 16 0 law\nunable to reserve streaming workspace within the memory budget\n"
             "0 0 0\nnae 5 3 2\n16 0 2\n16 0\n"
             "tensor memory budget is below live buffers and reservations\n"
             "tensor memory budget exceeded\n16\n16 0", stress=True)

    def test_streaming_empty_nonfinite_and_int64_precision(self):
        self.run_code("""
        tol compute as c
        tol tensor as t
        t.save("empty",t.zeros([0],"float32"))
        c.memory_budget(0)
        esta x = c.stream_stats("empty",2)
        pedo(x["count"],x["mean"],x["variance"],c.plan("stream_stats","empty",2)["fits"])
        c.memory_budget(mun)
        t.save("bad",t.zeros([2],"float64") / 0.0)
        try:
            c.stream_stats("bad",1)
        catch error:
            pedo(error["message"])
        pedo(c.memory_stats()["workspace_reserved_bytes"])
        t.save("large",t.array([9223372036854775804,9223372036854775805,9223372036854775806,9223372036854775807],"int64"))
        x = c.stream_stats("large",3)
        pedo(x["count"],x["variance"])
        """, "0 mun mun nae\nstream_stats requires finite values and representable moments\n"
             "0\n4 1.25", stress=True)

    def test_block_iteration_preserves_old_file_snapshot_and_returns_copies(self):
        self.run_code("""
        tol tensor as t
        t.save("values",t.array(list(range(5)),"int32"))
        esta source = t.blocks("values",2)
        esta blocks = []
        an block in source:
            push(blocks,block)
        t.put(blocks[0],[0],-1)
        t.save("values",t.array([9],"int32"))
        collect()
        pedo(t.tolist(blocks[0]),t.tolist(blocks[1]),t.tolist(blocks[2]))
        an block in source:
            pedo(t.tolist(block))
        t.close_blocks(source)
        t.close_blocks(source)
        try:
            an block in source:
                pedo(block)
        catch error:
            pedo(error["message"])
        """, "[-1, 1] [2, 3] [4]\n[0, 1]\n[2, 3]\n[4]\ntensor blocks are closed", stress=True)

    def test_tile_boundaries_quota_and_atomic_replacement(self):
        self.run_code("""
        tol compute as c
        tol tensor as t
        t.save("values",t.array(list(range(24)),"int32",[2,3,4]))
        esta patch = t.permute(t.array([70,71,80,81],"int32",[1,2,2]),[0,2,1])
        esta old = t.mmap("values")
        pedo(t.tolist(t.read_tile("values",[1,1,1],[1,2,2])))
        collect()
        c.memory_budget(16)
        try:
            t.write_tile("values",[1,1,1],patch,224)
        catch error:
            pedo(error["message"])
        c.memory_budget(mun)
        try:
            t.write_tile("values",[1,1,1],patch,223)
        catch error:
            pedo(error["message"])
        pedo(t.tolist(t.read_tile("values",[1,1,1],[1,2,2])))
        t.write_tile("values",[1,1,1],patch,224)
        pedo(t.tolist(t.read_tile("values",[1,0,0],[1,3,4])))
        pedo(t.at(old,[1,1,1]),c.memory_stats()["workspace_reserved_bytes"])
        pedo(t.shape(t.read_tile("values",[2,0,0],[0,3,4])))
        """, "[[[17, 18], [21, 22]]]\n"
             "unable to reserve tensor tile workspace within the memory budget\n"
             "tensor file exceeds disk budget or budget is invalid\n[[[17, 18], [21, 22]]]\n"
             "[[[12, 13, 14, 15], [16, 70, 80, 19], [20, 71, 81, 23]]]\n17 0\n[0, 3, 4]",
                      stress=True)

    def test_oversized_output_rejected_and_small_tile_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "file").open("wb") as handle:
                handle.write(tensor_header([4001, 4000], dtype=3, width=4))
                handle.seek(128 + (4001 * 4000 - 2) * 4)
                handle.write(struct.pack("<ii", -27, 91))
            script = root / "tile.arda"
            script.write_text("""
tol tensor as t
tol compute as c
try:
    t.read_tile("file",[0,0],[4001,4000])
catch error:
    pedo(error["message"])
pedo(c.memory_stats()["tensor_heap_bytes"])
pedo(t.tolist(t.read_tile("file",[4000,3998],[1,2])))
""")
            result = subprocess.run([test_arda.BIN, "--gc-stress", script], cwd=root,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "tensor tile exceeds the allocation limit or memory budget\n"
                                            "0\n[[-27, 91]]\n")

    def test_partial_write_releases_workspace_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = tensor_header([2]) + struct.pack("<qq", 1, 2)
            (root / "file").write_bytes(original)
            script = root / "fail.arda"
            script.write_text("""
tol tensor as t
tol compute as c
try:
    t.write_tile("file",[0],t.array([9],"int64"))
catch error:
    pedo(startswith(error["message"],"cannot publish tensor tile"))
pedo(c.memory_stats()["workspace_reserved_bytes"])
pedo(t.tolist(t.read_tile("file",[0],[2])))
""")

            def cap_file_size():
                signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
                resource.setrlimit(resource.RLIMIT_FSIZE, (132, 132))

            result = subprocess.run([test_arda.BIN, "--gc-stress", script], cwd=root,
                                    preexec_fn=cap_file_size, capture_output=True,
                                    text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "nae\n0\n[1, 2]\n")
            self.assertEqual((root / "file").read_bytes(), original)
            self.assertEqual({path.name for path in root.iterdir()}, {"file", "fail.arda"})
