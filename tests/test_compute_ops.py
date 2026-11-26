"""Explicit Float64 kernels, arithmetic order and bounded file prefetch."""
import unittest

import test_arda


class ComputeOpsTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_tiled_products_strides_tails_and_budget(self):
        self.run_code("""
        tol tensor as t
        tol compute as c
        esta a = t.array([1,2,3,4,5,6],"float64",[2,3])
        esta b = t.permute(t.array([1,3,5,2,4,6],"float64",[2,3]),[1,0])
        esta v = t.view(t.array([1,2,3],"float64"),[3],[-1],2)
        esta live = c.memory_stats()["tensor_heap_bytes"]
        c.memory_budget(live+63)
        try:
            c.tiled_matmul(a,b,2)
        catch e:
            pedo(e["code"])
        c.memory_budget(live+64)
        pedo(t.tolist(c.tiled_matmul(a,b,2)))
        pedo(c.memory_stats()["workspace_reserved_bytes"],c.memory_stats()["workspace_cached_bytes"])
        c.memory_budget(mun)
        pedo(t.tolist(c.tiled_matmul(a,v,2)))
        pedo(t.tolist(c.tiled_matmul(t.zeros([2,0],"float64"),t.zeros([0,2],"float64"))))
        """, "memory_budget\n[[22, 28], [49, 64]]\n0 32\n[10, 28]\n[[0, 0], [0, 0]]", stress=True)

    def test_fusion_rounds_each_operation_and_handles_views(self):
        self.run_code("""
        tol tensor as t
        tol compute as c
        esta e = 1.0/134217728.0
        esta a = t.array([1.0+e],"float64")
        esta b = t.array([1.0-e],"float64")
        esta d = t.array([-1],"float64")
        pedo(t.tolist(c.mul_add(a,b,d)))
        esta x = t.array([1,2,3],"float64")
        esta y = t.view(x,[3],[-1],2)
        pedo(t.tolist(c.axpy(x,y,2)))
        pedo(t.tolist(c.fused_products(x,y,y,y,x)))
        pedo(c.map_reduce("sum_squares",y),c.map_reduce("sum",y))
        pedo(abs(c.map_reduce("sum_sin",y)-t.sum(t.sin(y))) < 1e-15)
        """, "[0]\n[5, 6, 7]\n[13, 10, 7]\n14 6\nnae", stress=True)

    def test_prefetch_matches_serial_and_releases_failed_reads(self):
        self.run_code("""
        tol tensor as t
        tol compute as c
        t.save("data.atn",t.array([1,2,3,4,5,6,7],"float64"))
        esta o = {"prefetch":nae}
        esta p = c.plan("stream_stats","data.atn",3,o)
        esta single = c.plan("stream_stats","data.atn",7,o)
        pedo(p["prefetch"],p["workspace_bytes"],single["prefetch"],single["workspace_bytes"])
        esta a = c.stream_stats("data.atn",3)
        esta b = c.stream_stats("data.atn",3,o)
        pedo(a["count"] == b["count"],a["mean"] == b["mean"],a["variance"] == b["variance"])
        c.memory_budget(47)
        try:
            c.stream_stats("data.atn",3,o)
        catch e:
            pedo(e["code"])
        c.memory_budget(48)
        pedo(c.stream_stats("data.atn",3,o)["count"])
        pedo(c.memory_stats()["workspace_reserved_bytes"])
        """, "nae 48 law 56\nnae nae nae\nmemory_budget\n7\n0", stress=True)
