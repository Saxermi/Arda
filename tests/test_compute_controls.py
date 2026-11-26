"""Backend comparisons, affinity restoration and first-touch allocation."""
import os
import unittest

import test_arda


class ComputeControlsTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_compare_budget_warmup_and_nonfinite_results(self):
        self.run_code("""
        import tensor as t
        import compute as c
        let a = t.array([1,2,3,4],"float64",[2,2])
        let b = t.permute(a,[1,0])
        let cfg = {"first":{"block":1}, "second":{"kernel":"tiled","block":2},
                   "warmup":0, "repeats":2, "atol":0, "rtol":0}
        let stats = c.compare(a,b,cfg)
        print(stats["equivalent"],stats["max_abs_error"],stats["warmup"],stats["repeats"])
        print(stats["first"]["block"],stats["second"]["block"],stats["first"]["median_seconds"]>=0)
        print(c.last_run(),c.memory_stats()["workspace_reserved_bytes"])
        let live = c.memory_stats()["tensor_heap_bytes"]
        c.memory_budget(live+95)
        try:
            c.compare(a,b,cfg)
        catch err:
            print(err["code"])
        c.memory_budget(live+96)
        print(c.compare(a,b,cfg)["equivalent"])
        c.memory_budget(mun)
        let x = t.array([1e308],"float64",[1,1])
        let overflow = c.compare(x,x)
        print(overflow["equivalent"],overflow["max_abs_error"])
        let cpu = c.cpu()
        let verified = true
        if cpu["auto_matmul"] == "cblas":
            let external = c.compare(a,a,{"first":{"kernel":"cblas"},"second":{"kernel":"tiled"}})
            verified = external["equivalent"] and external["first"]["threads"] == mun
        else:
            if cpu["avx2"] and cpu["fma"]:
                let e = 1.0/134217728.0
                let left = t.array([-1,1.0+e],"float64",[1,2])
                let right = t.array([1,1.0-e],"float64",[2,1])
                let mismatch = c.compare(left,right,{"first":{"kernel":"avx2-fma"},
                    "second":{"kernel":"tiled"},"atol":0,"rtol":0})
                verified = not mismatch["equivalent"] and mismatch["max_abs_error"] == e*e
        print(verified)
        """, "nae 0 0 2\n1 2 nae\nmun 0\nmemory_budget\nnae\nlaw mun\nnae", stress=True)

    @unittest.skipUnless(hasattr(os, "sched_getaffinity"), "Linux affinity required")
    def test_topology_and_calling_thread_affinity(self):
        self.run_code("""
        import tensor as t
        import compute as c
        let original = c.affinity()
        let topo = c.topology()
        print(topo["source"]=="linux" or topo["source"]=="hwloc")
        print(len(topo["cpus"]) == len(original))
        let cpu = original[0]
        let pinned = c.affinity([cpu])
        print(len(pinned),pinned[0]==cpu,len(c.topology()["cpus"]))
        try:
            c.affinity([-1])
        catch err:
            print(err["code"])
        print(c.affinity()[0]==cpu)
        let restored = c.affinity(mun)
        print(len(restored)==len(original),restored[0]==cpu)
        """, "nae\nnae\n1 nae 1\ndomain\nnae\nnae nae", stress=True)

    def test_first_touch_and_thread_reports(self):
        self.run_code("""
        import tensor as t
        import compute as c
        let parallel = c.cpu()["native_threads_available"]
        if parallel:
            c.threads(8)
        let a = c.first_touch([1024,1025])
        let b = t.zeros([1025],"float64")+2.0
        let p = c.plan("matmul",a,b)
        let cap = c.explain(p)["estimated_threads"]
        let result = c.execute(p,a,b)
        let threads = c.last_run()["threads"]
        print(t.shape(a),t.sum(a),t.sum(result),threads>=1,threads<=cap)
        print(t.shape(c.first_touch([0,2])))
        c.memory_budget(c.memory_stats()["tensor_heap_bytes"])
        try:
            c.first_touch([1])
        catch err:
            print(err["code"])
        """, "[1024, 1025] 0 0 nae nae\n[0, 2]\nmemory_budget", stress=True)
