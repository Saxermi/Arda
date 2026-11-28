"""Lazy graph, adaptive execution and static analysis contracts."""
import unittest

import test_arda


class AceV2Tests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_float64_order_and_determinism(self):
        self.run_code("""
        import lazy as l
        import tensor as t
        import compute as c
        let eps = 0.000000007450580596923828125
        let x = t.array([1+eps,0.0,-2.0,4.0,7.0],"float64")
        let y = t.array([1-eps,2.0,3.0,4.0,5.0],"float64")
        let z = t.array([-1.0,-0.0,5.0,7.0,9.0],"float64")
        let g = l.add(l.mul(l.input(x),l.input(y)),l.input(z))
        let reference = t.tolist(x*y+z)
        for method in ["fused","tiled","in_ram"]:
            let p = c.plan("graph",g,{"kernel":method,"block_elements":2})
            let value = t.tolist(c.execute(p))
            let equal = true
            for i in range(len(value)):
                equal = equal and value[i] == reference[i]
            print(equal,value[0] == 0,c.explain(p)["deterministic"])
        let a = l.input(x)
        let s = l.sum(l.mul(l.sin(a),l.exp(l.neg(a))))
        print(l.materialize(s) == c.execute(c.plan("graph",s,{"kernel":"tiled","block_elements":1})))
        """, "nae nae nae\nnae nae nae\nnae nae nae\nnae", stress=True)

    def test_shared_dag_and_noncontiguous_alias_versions(self):
        self.run_code("""
        import lazy as l
        import tensor as t
        import compute as c
        let a = t.array([1,2,3,4,5,6],"float64",[2,3])
        let view = t.permute(a,[1,0])
        let x = l.input(view)
        let u = l.add(x,x)
        let g = l.mul(u,u)
        let p = c.plan("graph",g,{"kernel":"tiled","block_elements":2})
        a = nil
        collect()
        print(c.explain(p)["nodes"],t.tolist(c.execute(p)))
        t.put(view,[0,1],99)
        try:
            c.execute(p)
        catch e:
            print(e["code"],c.memory_stats()["workspace_reserved_bytes"])
        print(t.tolist(l.materialize(l.input(t.zeros([0,2],"float64")))))
        """, "3 [[4, 64], [16, 100], [36, 144]]\nmutation 0\n[]", stress=True)

    def test_budget_calibration_and_revalidation(self):
        self.run_code("""
        import lazy as l
        import tensor as t
        import compute as c
        let x = l.input(t.zeros([16],"float64"))
        let g = l.mul(l.add(x,x),l.sub(x,x))
        let opts = {"block_elements":16,"calibration":{
            "flops_per_second":64,"ram_bytes_per_second":1024,"overhead_seconds":0.125}}
        let live = c.memory_stats()["tensor_heap_bytes"]
        c.memory_budget(live+128+32)
        let p = c.plan("graph",g,opts)
        let e = c.explain(p)
        print(e["kernel"],e["block_elements"],e["workspace_bytes"],e["output_bytes"])
        print(e["estimated_seconds"] > 0,e["candidates"][2]["fits"])
        c.memory_budget(live+128+31)
        try:
            c.execute(p)
        catch e:
            print(e["code"])
        c.memory_budget(live+128+32)
        let run = c.run(p,2)
        let r = c.report(run)
        print(r["blocks"],r["trace"][1]["start"],r["trace_omitted"],r["estimate_error_seconds"] != nil)
        print(c.memory_stats()["workspace_reserved_bytes"])
        """, "tiled 2 32 128\nnae law\nmemory_budget\n8 2 6 nae\n0", stress=True)

    def test_file_stream_inode_and_mapped_storage(self):
        self.run_code("""
        import lazy as l
        import tensor as t
        import compute as c
        t.save("source.atn",t.array([1,2,3,4,5],"float64"))
        let a = l.file("source.atn")
        let g = l.sum(l.add(a,a))
        let p = c.plan("graph",g,{"block_elements":2})
        let mapped = c.plan("graph",l.sum(l.input(t.mmap("source.atn"))))
        t.save("source.atn",t.array([99],"float64"))
        collect()
        let run = c.report(c.run(p,3))
        print(c.explain(p)["storage"],run["logical_file_bytes"],run["value"])
        print(run["trace"][2]["start"],run["trace"][2]["elements"],run["trace_omitted"])
        print(c.explain(mapped)["storage"],c.execute(mapped))
        """, "file_blocks 40 30\n4 1 0\nmapped 15", stress=True)

    def test_graph_plans_retain_buffers_and_gc_releases_them(self):
        self.run_code("""
        import lazy as l
        import tensor as t
        import compute as c
        def plan():
            let x = l.input(t.zeros([1024],"float64")+1.0)
            return c.plan("graph",l.sum(l.mul(x,x)),{"block_elements":16})
        let baseline = c.memory_stats()["tensor_heap_bytes"]
        let p = plan()
        collect()
        collect()
        print(c.memory_stats()["tensor_heap_bytes"]-baseline,c.execute(p))
        p = nil
        collect()
        collect()
        c.trim_pool()
        let stats = c.memory_stats()
        print(stats["tensor_heap_bytes"] == baseline,stats["workspace_reserved_bytes"],
              stats["workspace_cached_bytes"])
        """, "8192 1024\nnae 0 0", stress=True)

    def test_file_mutation_corrupt_header_and_cleanup(self):
        self.run_code("""
        import lazy as l
        import tensor as t
        import compute as c
        t.save("source.atn",t.array([1,2,3,4],"float64"))
        let p = c.plan("graph",l.sum(l.file("source.atn")))
        write_text("source.atn","")
        try:
            c.execute(p)
        catch e:
            print(e["code"],c.memory_stats()["workspace_reserved_bytes"])
        try:
            l.file("source.atn")
        catch e:
            print(e["code"])
        """, "mutation 0\nfile_format", stress=True)
        self.run_code("""
        import lazy as l
        import compute as c
        print(c.capabilities()["ssd"],c.capabilities()["device"],c.capabilities()["network"])
        try:
            l.file("missing.atn")
        catch e:
            print(e["code"])
        """, "law law law\nhost_io", flags=("--no-file-io",), stress=True)

    def test_invalid_graph_and_trace_limits(self):
        self.run_code("""
        import lazy as l
        import tensor as t
        import compute as c
        let x = l.input(t.zeros([2],"float64"))
        let y = l.input(t.zeros([3],"float64"))
        let attempts = [lambda: l.input(t.zeros([2],"int32")), lambda: l.add(x,y),
                        lambda: l.sin(l.sum(x)),
                        lambda: c.plan("graph",l.neg(x),{"kernel":"fused"}),
                        lambda: c.plan("graph",x,{"block_elements":0}),
                        lambda: c.run(c.plan("graph",x),257)]
        for action in attempts:
            try:
                action()
            catch e:
                print(e["code"])
        let g = x
        try:
            for i in range(80):
                g = l.neg(g)
        catch e:
            print(e["code"])
        """, "dtype\nshape\nshape\nbackend\ndomain\ndomain\nlimit", stress=True)

    def test_structured_diagonal_csr_and_output_failure(self):
        self.run_code("""
        import numc as n
        import sparse as s
        import linop as l
        import compute as c
        import mathdoc as m
        import tensor as t
        let x = n.array([1,2,3])
        let d = l.diagonal(n.array([2,3,4]))
        let r = s.csr(2,3,[0,1,3],[2,0,1],[4,5,6])
        let p = c.plan("apply",r,x)
        let run = c.run(p,1)
        collect()
        print(t.tolist(c.execute(c.plan("apply",d,x))),t.tolist(c.report(run)["value"]))
        print(m.getPlan(p)["output_bytes"],m.getCost(p)["flops"],c.report(run)["trace_omitted"])
        print(run.getAlgorithm() == m.getAlgorithm(run))
        n.put(x,[1],8)
        try:
            c.execute(p)
        catch e:
            print(e["code"])
        let q = c.plan("apply",l.diagonal(n.array([1e300])),n.array([1e300]))
        collect()
        let live = c.memory_stats()["tensor_heap_bytes"]
        try:
            c.execute(q)
        catch e:
            print(e["code"],c.memory_stats()["tensor_heap_bytes"] == live)
        """, "[2, 6, 12] [12, 17]\n16 6 0\nnae\nmutation\ndomain nae", stress=True)

    def test_nested_ranges_branches_and_overflow(self):
        self.run_code("""
        import mathdoc as m
        def total(n,k):
            let result = 0
            for i in range(n):
                for j in range(k,0,-2):
                    result = result + i*j
            return result
        let p = total.getPlan({"parameters":{"n":[2,4],"k":5}})
        print(p["loop_depth"],p["cost"]["min_scalar_operations"],p["cost"]["max_scalar_operations"])
        print(total(3,5),total.getCost() == nil)
        print(total.getCost({"parameters":{"n":9223372036854775807,"k":5}}) == nil)
        let empty = total.getCost({"parameters":{"n":-1,"k":5}})
        print(empty["min_scalar_operations"],empty["max_scalar_operations"])
        def conditional(x):
            let y = x+1
            if x > 0:
                y = y*x
            return y
        let c = m.getCost(conditional)
        print(c["min_scalar_operations"],c["max_scalar_operations"])
        """, "2 12 24\n27 nae\nnae\n0 0\n2 3", stress=True)

    def test_recursion_assumption_purity_and_shadowing(self):
        self.run_code("""
        import mathdoc as m
        def depth(n):
            if n <= 0:
                return 0
            return 1+depth(n-1)
        let d = m.getCost(depth,{"recursion_depth":5})
        print(d["min_scalar_operations"],d["max_scalar_operations"],d["max_invocations"],depth(4))
        print(m.getCost(depth) == nil,m.getCost(depth,{"recursion_depth":513}) == nil)
        def notify(n):
            write_text("side-effect.txt","unexpected")
            return n+1
        print(m.getPlan(notify) == nil,m.getCost(notify) == nil)
        try:
            read_text("side-effect.txt")
        catch e:
            print("no side effect")
        let saved = depth
        depth = lambda n: n
        print(m.getPlan(saved) == nil)
        def f(n):
            for i in range(n):
                n = n-1
            return n
        print(m.getPlan(f) == nil)
        """, "0 15 5 4\nnae nae\nnae nae\nno side effect\nnae\nnae", stress=True)

    def test_contracts_and_disable_static_metadata(self):
        self.run_code("""
        import contract as c
        print(c.require(true),c.ensure(true),c.assume(true))
        for check in [c.require,c.ensure,c.assume]:
            try:
                check(false,"constraint")
            catch e:
                print(e["code"],e["message"])
        try:
            c.require(1)
        catch e:
            print(e["code"])
        """, "nae nae nae\nprecondition constraint\npostcondition constraint\nassumption constraint\ntype",
                      stress=True)
        self.run_code("""
        import mathdoc as m
        def f(n):
            return n+1
        print(f(4),m.getPlan(f),m.getCost(f))
        """, "5 mun mun", flags=("--mathdoc=none",), stress=True)
