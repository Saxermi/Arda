"""ACE plan contracts, tuning cache and execution under GC stress."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import test_arda


def cacheable():
    online = Path("/sys/devices/system/node/online")
    return hasattr(os, "sched_getaffinity") and online.is_file() and online.read_text().strip().isdigit()


class ComputePlanTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_explain_calibration_execution_and_compatible_reuse(self):
        self.run_code("""
        import tensor as t
        import compute as c
        let a = t.array([1,2,3,4],"float64",[2,2])
        let b = t.permute(a,[1,0])
        let p = c.plan("matmul",a,b,{"block":1, "calibration":{
            "flops_per_second":32, "ram_bytes_per_second":96, "overhead_seconds":0.25}})
        let info = c.explain(p)
        print(type(p),info["kernel"],info["block"],info["output_bytes"],info["workspace_bytes"])
        print(info["flops"], info["logical_ram_bytes"],info["lower_bound_seconds"],info["estimated_seconds"])
        print(c.last_run(),info["measured_seconds"])
        collect()
        print(t.tolist(c.execute(p,a,b)))
        let run = c.last_run()
        print(run["kernel"],run["block"],run["threads"],run["workspace_bytes"])
        t.put(a,[0,0],2)
        print(t.tolist(c.execute(p,a,b)))
        try:
            c.execute(p,a,a)
        catch err:
            print(err["code"])
        """, "compute_plan tiled 1 32 8\n16 96 1 1.25\nmun mun\n[[5, 11], [11, 25]]\n"
             "tiled 1 1 8\n[[8, 14], [14, 25]]\nshape", stress=True)

    def test_tuning_cache_budget_and_unvalidated_options(self):
        self.run_code("""
        import tensor as t
        import compute as c
        let a = t.array([1,2,3,4,5,6],"float64",[2,3])
        let b = t.array([1,2,3],"float64")
        let p = c.plan("matmul",a,b,{"tune":true})
        let q = c.plan("matmul",a,b,{"tune":true})
        let e = c.explain(p)
        let f = c.explain(q)
        print(e["cache_hit"],f["cache_hit"],not f["cache_hit"] or e["block"]==f["block"],e["measured_seconds"]>=0)
        let live = c.memory_stats()["tensor_heap_bytes"]
        c.memory_budget(live+15)
        try:
            c.execute(p,a,b)
        catch err:
            print(err["code"])
        c.memory_budget(live+16)
        print(t.tolist(c.execute(q,a,b)))
        try:
            c.plan("matmul",a,b,{"tune":true,"block":4})
        catch err:
            print(err["code"])
        try:
            c.plan("matmul",a,b,{"unknown":1})
        catch err:
            print(err["code"])
        """, f"law {'nae' if cacheable() else 'law'} nae nae\nmemory_budget\n[14, 32]\ndomain\ndomain",
                      stress=True)

    def test_mapped_file_tier_and_empty_reference(self):
        self.run_code("""
        import tensor as t
        import compute as c
        let a = t.array([1,2,3,4],"float64",[2,2])
        t.save("data.atn",a)
        let b = t.mmap("data.atn")
        let p = c.plan("matmul",a,b,{"kernel":"tiled","calibration":{
            "flops_per_second":128,"ram_bytes_per_second":96,"disk_bytes_per_second":8}})
        let e = c.explain(p)
        print(e["logical_file_bytes"],e["lower_bound_seconds"])
        try:
            c.execute(p,a,a)
        catch err:
            print(err["code"])
        print(t.tolist(c.execute(p,a,b)))
        let x = t.zeros([2,0],"float64")
        let y = t.zeros([0,3],"float64")
        print(t.tolist(c.execute(c.plan("matmul",x,y),x,y)),c.last_run()["kernel"])
        """, "32 4\nshape\n[[7, 10], [15, 22]]\n[[0, 0, 0], [0, 0, 0]] none", stress=True)

    def test_candidate_budget_classes_and_comparison(self):
        self.run_code("""
        import tensor as t
        import compute as c
        let a = t.zeros([64,64],"float64")
        let b = t.permute(a,[1,0])
        let config = {"tune":true}
        let full = c.explain(c.plan("matmul",a,b,config))
        let live = c.memory_stats()["tensor_heap_bytes"]
        let output = 64*64*8
        let small = 32*32*8
        c.memory_budget(live+output+small)
        let p = c.plan("matmul",a,b,config)
        let narrow = c.explain(p)
        print(narrow["block"],narrow["cache_hit"],narrow["measured_threads"],
              t.shape(c.execute(p,a,b)))
        collect()
        let repeat = c.explain(c.plan("matmul",a,b,config))
        print(repeat["cache_hit"]==cacheable(),c.memory_stats()["workspace_reserved_bytes"])
        c.memory_budget(live+output+small-1)
        try:
            c.plan("matmul",a,b,config)
        catch e:
            print(e["code"],c.memory_stats()["tensor_heap_bytes"]==live)
        c.memory_budget(live+2*output+small)
        let cmp = c.compare(a,b,{"first":config,"second":config,"repeats":1})
        print(cmp["equivalent"],cmp["first"]["block"],cmp["second"]["block"])
        """.replace("cacheable()", "true" if cacheable() else "false"),
                      "32 law 1 [64, 64]\nnae 0\nmemory_budget nae\nnae 32 32", stress=True)

    @unittest.skipUnless(cacheable(), "single-node Linux affinity is required for timing reuse")
    def test_affinity_changes_scope_cache_to_the_exact_mask(self):
        if len(os.sched_getaffinity(0)) < 2:
            self.skipTest("test needs two allowed CPUs")
        with patch.dict(os.environ, {"OMP_PROC_BIND": "FALSE"}):
            self.run_code("""
            import compute as c
            import tensor as t
            let a = t.zeros([16,16],"float64")
            let cpus = c.affinity()
            def tune():
                return c.explain(c.plan("matmul",a,a,{"tune":true}))["cache_hit"]
            print(tune(),tune())
            c.affinity([cpus[0]])
            print(tune(),tune())
            c.affinity([cpus[1]])
            print(tune(),tune())
            c.affinity([cpus[1]])
            print(tune())
            c.affinity(nil)
            print(tune())
            """, "law nae\nlaw nae\nlaw nae\nnae\nnae", stress=True)

    @unittest.skipUnless(cacheable(), "single-node Linux affinity is required for timing reuse")
    def test_effective_thread_limit_and_independent_backend_control(self):
        with patch.dict(os.environ, {"OMP_THREAD_LIMIT": "1", "OMP_DYNAMIC": "FALSE",
                                     "OMP_PROC_BIND": "FALSE"}):
            self.run_code("""
            import compute as c
            import tensor as t
            let a = t.zeros([216,216],"float64")
            let one = c.explain(c.plan("matmul",a,a,{"tune":true,"threads":1}))
            try:
                c.matmul_kernel("reference")
            catch e:
                c.matmul_kernel("cblas")
            let requested = 1
            if c.cpu()["native_threads_available"]:
                c.threads(4)
                requested = 4
            let p = c.plan("matmul",a,a,{"tune":true})
            let e = c.explain(p)
            c.execute(p,a,a)
            print(one["cache_hit"],e["cache_hit"],e["requested_threads"]==requested,
                  e["estimated_threads"],e["measured_threads"],c.last_run()["threads"])
            """, "law nae nae 1 1 1", stress=True)
