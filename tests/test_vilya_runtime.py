"""Vilya runtime behavior and transactional tensor gradients."""
import json
import pathlib
import tempfile
import unittest

import test_arda


class VilyaRuntimeTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_ranges_are_lazy_exact_and_explicitly_materialized(self):
        self.run_code("""
        let wide = range(-9223372036854775808, 9223372036854775807, 9223372036854775807)
        print(type(wide), len(wide), wide[0], wide[-1])
        let huge = range(9223372036854775807)
        let before = heap_bytes()
        for i in huge:
            print(i, len(huge))
            break
        print(heap_bytes()-before<4096, list(range(5,0,-2)), list(range(3,0)))
        try:
            list(huge)
        catch issue:
            print(issue["code"])
        try:
            range(-9223372036854775808,9223372036854775807)
        catch issue:
            print(issue["code"])
        """, "range 3 -9223372036854775808 9223372036854775806\n"
             "0 9223372036854775807\nnae [5, 3, 1] []\nrange\nrange", stress=True)

    def test_imports_distinguish_standard_modules_and_source_files(self):
        self.run_code("""
        import "std:tensor" as std_tensor
        tol tensor
        import "./tensor.arda" as local
        print(tensor == std_tensor, local.origin)
        try:
            import "std:other" as invalid
        catch issue:
            print(issue["code"])
        """, "nae local\nimport", files={
            "tensor.arda": 'let origin = "local"',
            "other.arda": 'let origin = "other"',
        }, stress=True)

    def test_structured_diagnostics_and_real_float_policy(self):
        self.run_code("""
        import tensor as t
        import ad
        let error = {"code":"shape", "message":"example", "expected":[2,3]}
        try:
            throw error
        catch issue:
            collect()
            print(issue == error, issue["code"], issue["expected"])
        try:
            throw "text"
        catch issue:
            print(issue["code"], type(issue["file"]), issue["line"] > 0)
        print(1.0/0.0, t.tolist(t.array([1.0],"float64")/0.0))
        let invalid = sqrt(-1)
        print(invalid != invalid)
        try:
            if t.zeros([0],"float64"):
                print("unexpected")
        catch issue:
            print(issue["code"])
        try:
            ad.variable(1.0/0.0)
        catch issue:
            print(issue["code"])
        """, "nae shape [2, 3]\ntype string nae\ninf [inf]\nnae\nambiguous_truth\nnonfinite",
             stress=True)

    def test_shared_parents_scalar_tensors_and_repeat_backward(self):
        self.run_code("""
        tol tensor as t
        import ad
        let x = ad.variable(t.array([1,2,3,4],"float64",[2,2]))
        let s = ad.variable(2)
        let twice = x*s
        let y = t.sum(twice*twice + twice)
        print(ad.grad(x))
        ad.backward(y)
        print(ad.value(y), t.tolist(ad.grad(x)), ad.grad(s))
        ad.backward(y,2)
        print(t.tolist(ad.grad(x)), ad.grad(s))
        ad.backward(y)
        print(t.tolist(ad.grad(x)), ad.grad(s))
        """, "mun\n140 [[10, 18], [26, 34]] 130\n[[20, 36], [52, 68]] 260\n"
             "[[10, 18], [26, 34]] 130", stress=True)

    def test_view_gradient_scatter_permutation_and_index(self):
        self.run_code("""
        import tensor as t
        tol ad
        let x = ad.variable(t.array([1,2,3,4,5,6],"float64"))
        let repeats = t.view(x,[3],[0],1)
        let reverse = t.view(x,[6],[-1],5)
        let layout = t.reshape(reverse * 1,[2,3])
        let permuted = t.permute(layout,[1,0])
        let sampled = t.slice(permuted,0,0,3,2)
        let loss = t.at(sampled,[1,1]) + t.sum(repeats)
        ad.backward(loss)
        print(ad.value(loss),t.tolist(ad.grad(x)))
        """, "7 [1, 3, 0, 0, 0, 0]", stress=True)

    def test_alias_mutation_rejects_backward_and_invalid_writes_preserve_version(self):
        self.run_code("""
        tol tensor as t
        tol numc as nc
        import ad
        let raw = nc.array([1,2])
        let x = ad.variable(t.readonly(raw))
        let loss = t.sum(x*x)
        ad.backward(loss)
        try:
            t.put(raw,[0],"invalid")
        catch issue:
            ad.backward(loss)
            print(t.tolist(ad.grad(x)))
        raw[1] = 8
        try:
            ad.backward(loss)
        catch issue:
            print(issue["code"],t.tolist(ad.grad(x)))
        let linear = t.sum(x + x)
        nc.put(raw,[1],9)
        ad.backward(linear)
        print(t.tolist(ad.grad(x)))
        """, "[2, 4]\nmutation [2, 4]\n[2, 2]", stress=True)

    def test_backprop_budget_error_and_overflow_keep_previous_gradients(self):
        self.run_code("""
        tol tensor as t
        tol compute as c
        import ad
        let x = ad.variable(t.array([0,0],"float64"))
        let f = t.sum(x*2)
        ad.backward(f)
        let overflow = t.sum((x*1e200)*1e200)
        try:
            ad.backward(overflow)
        catch issue:
            print(issue["code"],t.tolist(ad.grad(x)))
        collect()
        let baseline = c.memory_stats()["tensor_heap_bytes"]
        c.memory_budget(baseline)
        try:
            ad.backward(f)
        catch issue:
            print(issue["code"],t.tolist(ad.grad(x)))
        print(c.memory_stats()["tensor_heap_bytes"] == baseline)
        """, "nonfinite [2, 2]\nmemory_budget [2, 2]\nnae", stress=True)

    def test_backward_validates_seed_storage_and_dtypes(self):
        self.run_code("""
        import tensor as t
        import ad
        let x = ad.variable(t.array([1,2],"float64"))
        try:
            ad.backward(x*x,t.zeros([2,1],"float64"))
        catch issue:
            print(issue["code"])
        ad.backward(x*x,t.array([4,5],"float64"))
        print(t.tolist(ad.grad(x)))
        try:
            ad.variable(t.zeros([2],"int64"))
        catch issue:
            print(issue["code"])
        t.save("fixture.atn",t.zeros([2],"float64"))
        try:
            ad.variable(t.mmap("fixture.atn"))
        catch issue:
            print(issue["code"])
        """, "shape\n[8, 20]\ndtype\ndtype", stress=True)

    def test_tracking_controls_are_scoped_and_gradients_can_be_cleared(self):
        self.run_code("""
        import ad
        tol tensor as t
        let x = ad.variable(t.array([1,2],"float64"))
        let raw = ad.detach(x)
        print(raw == ad.value(x))
        let foo = ad.no_grad
        def nested(y):
            try:
                foo(t.at,y,[9])
            catch issue:
                print(type(t.sum(y*y)))
            return t.sum(ad.sin(y) + y*y)
        let r = foo(foo,nested,x)
        print(type(r),type(foo(ad.variable,17)))
        try:
            foo(t.at,x,[8])
        catch issue:
            print(type(t.sum(x*x)))
        let f = t.sum(x*x)
        ad.backward(f)
        let snapshot = ad.grad(x)
        ad.zero_grad(x)
        print(ad.grad(x), t.tolist(snapshot))
        ad.backward(f)
        print(t.tolist(ad.grad(x)))
        """, "nae\nfloat\nfloat int\nad\nmun [2, 4]\n[2, 4]", stress=True)

    def test_queries_stop_at_selected_node_and_leave_published_gradients_alone(self):
        self.run_code("""
        import ad
        tol tensor as t
        let x = ad.variable(3)
        let u = x*x
        let f = u*u+x
        ad.backward(f)
        print(ad.vjp(f,u),ad.jvp(f,u,2),ad.jacobian(f,u),ad.hessian(f,u))
        let independent = ad.variable(t.array([7,8],"float64"))
        print(t.tolist(ad.vjp(f,independent)),ad.jvp(f,independent,t.array([1,2],"float64")))
        print(t.tolist(ad.hessian(f,independent)), ad.grad(x), ad.grad(u))
        """, "18 36 18 2\n[0, 0] 0\n[[0, 0], [0, 0]] 109 18", stress=True)

    def test_derivative_shapes_empty_and_budget_checks(self):
        self.run_code("""
        import ad
        tol tensor as t
        tol compute as c
        let x = ad.variable(t.array([1,2,3,4],"float64",[2,2]))
        let v = x*x
        let f = t.sum(v)
        print(t.shape(ad.jacobian(v,x)),t.shape(ad.hessian(f,x)))
        let empty = ad.variable(t.zeros([0,2],"float64"))
        print(t.shape(ad.jacobian(empty,empty)),t.shape(ad.hessian(t.sum(empty),empty)))
        let highrank = ad.variable(t.zeros([1,1,1,1,1],"float64"))
        try:
            ad.jacobian(highrank,highrank)
        catch issue:
            print(issue["code"])
        try:
            ad.hessian(v,x)
        catch issue:
            print(issue["code"])
        try:
            ad.vjp(v,x)
        catch issue:
            print(issue["code"])
        try:
            ad.jvp(v,x,t.zeros([4],"float64"))
        catch issue:
            print(issue["code"])
        collect()
        let before = c.memory_stats()["tensor_heap_bytes"]
        c.memory_budget(before+128)
        try:
            ad.hessian(f,x)
        catch issue:
            print(issue["code"],c.memory_stats()["tensor_heap_bytes"]==before)
        """, "[2, 2, 2, 2] [2, 2, 2, 2]\n[0, 2, 0, 2] [0, 2, 0, 2]\n"
             "shape\nshape\nshape\nshape\nmemory_budget nae", stress=True)

    def test_sparse_matvec_derivatives_keep_csr_order_and_validate_alias_versions(self):
        self.run_code("""
        import ad
        tol sparse as sp
        import tensor as t
        let pattern = sp.csr(3,2,[0,1,1,3],[0,0,1],[1,2,3])
        let backing = t.array([2,0,5],"float64")
        let values = ad.variable(backing)
        let x = ad.variable(t.array([2,3],"float64"))
        let y = sp.matvec(pattern,values,x)
        let f = t.sum(y)
        ad.backward(f)
        print(t.tolist(ad.value(y)),t.tolist(ad.grad(values)),t.tolist(ad.grad(x)))
        print(t.tolist(ad.jvp(y,values,t.array([2,3,4],"float64"))))
        print(t.tolist(ad.vjp(t.sum(sp.matvec(pattern,x)),x)))
        print(t.tolist(ad.no_grad(sp.matvec,pattern,x)))
        t.put(backing,[0],9)
        # The values gradient uses x. The input gradient needs the changed values.
        print(t.tolist(ad.vjp(f,values)))
        try:
            ad.vjp(f,x)
        catch issue:
            print(issue["code"],t.tolist(ad.grad(x)))
        """, "[4, 0, 15] [2, 2, 3] [2, 5]\n[4, 0, 18]\n[3, 3]\n[2, 0, 13]\n"
             "[2, 2, 3]\nmutation [2, 5]", stress=True)

    def test_retained_backward_reuses_buffers_and_consumes_only_on_success(self):
        self.run_code("""
        import ad
        import tensor as t
        import compute as c
        let x = ad.variable(t.zeros([1000],"float64"))
        let y = x + 1
        for i in range(11):
            y = y + 1
        let loss = t.sum(y)
        collect()
        let base = c.memory_stats()["tensor_heap_bytes"]
        c.memory_budget(base + 16000)
        try:
            ad.backward(loss)
        catch issue:
            print(issue["code"])
        ad.backward(loss,mun,{"retain":[x]})
        print(t.at(ad.grad(x),[0]), ad.grad(y))
        collect()
        let snapshot = ad.grad(x)
        let before = c.memory_stats()["tensor_heap_bytes"]
        c.memory_budget(before + 8000)
        try:
            ad.backward(loss,mun,{"retain":[x],"consume":true})
        catch issue:
            print(issue["code"],c.memory_stats()["tensor_heap_bytes"] == before)
        c.memory_budget(mun)
        y = mun
        ad.backward(loss,mun,{"retain":[x],"consume":true})
        print(t.at(ad.grad(x),[0]),t.at(snapshot,[999]),ad.value(loss))
        print(c.memory_stats()["tensor_heap_bytes"] < before)
        try:
            ad.backward(loss)
        catch issue:
            print(issue["message"])
        """, "memory_budget\n1 mun\nmemory_budget nae\n1 1 12000\nnae\n"
             "backward tape was already consumed", stress=True)

    def test_batch_means_use_their_sample_counts_and_publish_atomically(self):
        self.run_code("""
        tol ad
        import tensor as t
        let w = ad.variable(t.array([1,2],"float64"))
        let d = ad.variable(2)
        let l = t.sum(w*4)+d*7
        ad.backward(l)
        try:
            ad.backward(l,mun,{"retain":[w,d],"samples":2})
        catch issue:
            print(issue["message"])
        ad.zero_grad(w)
        ad.zero_grad(d)
        ad.backward(l,mun,{"retain":[w,d],"samples":2})
        let original = ad.grad(w)
        ad.backward(t.sum(w*8)+d,mun,{"retain":[w,d],"samples":6})
        print(t.tolist(ad.grad(w)),ad.grad(d),t.tolist(original))
        try:
            ad.backward(l,mun,{"retain":[w,d],"samples":9223372036854775807})
        catch issue:
            print(issue["code"],t.tolist(ad.grad(w)),ad.grad(d))
        ad.zero_grad(w)
        ad.backward(t.sum(w*5),mun,{"retain":[w],"samples":10})
        print(t.tolist(ad.grad(w)))
        """, "clear the unweighted gradient before accumulating batches\n[7, 7] 2.5 [4, 4]\n"
             "overflow [7, 7] 2.5\n[5, 5]", stress=True)

    def test_checkpoint_replays_only_math_and_keeps_rng_draws(self):
        self.run_code("""
        import ad
        import tensor as t
        import random as r
        let rng = r.generator(7)
        let reference = r.generator(7)
        let ignored = r.normal(reference,0,1,3)
        let calls = 0
        def forward(x):
            calls = calls + 1
            let draw = r.normal(rng,0,1,3)
            return x*x + x*draw
        let x = ad.variable(t.array([1,2,3],"float64"))
        let ordinary = forward(x)
        let packed = ad.checkpoint(ordinary,x)
        let old = ad.vjp(t.sum(ordinary),x)
        ordinary = mun
        collect()
        let got = ad.vjp(t.sum(packed),x)
        print(t.sum((old-got)*(old-got)) < 1e-26,calls,r.uint32(rng)==r.uint32(reference))
        """, "nae 1 nae", stress=True)

    def test_checkpoint_validates_mutations_boundaries_and_operation_set(self):
        self.run_code("""
        import ad
        tol tensor as t
        let raw = t.array([1,2],"float64")
        let x = ad.variable(raw)
        let c = t.array([2,3],"float64")
        let y = x*c
        let packed = ad.checkpoint(y,x)
        ad.backward(t.sum(packed))
        t.put(c,[0],4)
        try:
            ad.backward(t.sum(packed))
        catch issue:
            print(issue["code"],t.tolist(ad.grad(x)))
        try:
            ad.checkpoint(y,x)
        catch issue:
            print(issue["code"])
        try:
            ad.checkpoint(t.sum(x),x)
        catch issue:
            print(issue["code"])
        let z = ad.variable(4)
        try:
            ad.checkpoint(x*z,x)
        catch issue:
            print(issue["code"])
        """, "mutation [2, 3]\nmutation\nshape\nautodiff", stress=True)

    def test_empty_checkpoint_fits_zero_tensor_budget(self):
        self.run_code("""
        import ad
        import compute as c
        import tensor as t
        c.memory_budget(0)
        let x = ad.variable(t.zeros([0,2],"float64"))
        let packed = ad.checkpoint(x*x+x,x)
        let loss = t.sum(packed)
        ad.backward(loss,mun,{"retain":[x]})
        print(t.shape(ad.grad(x)),type(ad.no_grad(ad.checkpoint,packed,x)))
        print(c.memory_stats()["tensor_heap_bytes"])
        """, "[0, 2] tensor\n0", stress=True)

    def test_offloaded_snapshot_budget_contents_and_gc(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_code("""
            tol ad
            import "std:tensor" as t
            import compute as c
            let plan = ad.offload_plan(SPILL_DIRECTORY,144,1)
            let x = ad.variable(t.array([0,0],"float64"))
            let y = t.exp(x)
            let z = ad.offload(y,plan)
            let snapshot = ad.value(z)
            print(type(plan),t.tolist(snapshot),c.memory_stats()["mapped_bytes"])
            try:
                ad.offload(t.exp(x),plan)
            catch issue:
                print(issue["code"],ad.offload_stats(plan)["files_written"])
            try:
                t.put(snapshot,[0],7)
            catch issue:
                print(issue["code"])
            t.put(ad.value(y),[0],7)
            try:
                ad.offload(y,plan)
            catch issue:
                print(issue["code"])
            print(t.tolist(ad.vjp(t.sum(z),x)))
            let stats = ad.offload_stats(plan)
            print(stats["bytes_written"],stats["bytes_read"],stats["read_chunks"],stats["peak_disk_bytes"])
            z = mun
            collect()
            print(ad.offload_stats(plan)["live_files"])
            snapshot = mun
            collect()
            print(ad.offload_stats(plan)["live_files"],ad.offload_stats(plan)["live_disk_bytes"])
            """.replace("SPILL_DIRECTORY", json.dumps(directory)),
                "offload_plan [1, 1] 144\ndisk_budget 1\nreadonly\nmutation\n[1, 1]\n"
                "144 16 2 144\n1\n0 0", stress=True)
            self.assertEqual(list(pathlib.Path(directory).iterdir()), [])

    def test_offload_failures_preserve_gradients_and_release_read_workspace(self):
        self.run_code("""
        import ad
        import tensor as t
        import compute as c
        let x = ad.variable(t.array([700,700],"float64"))
        let plan = ad.offload_plan(".",144,1)
        let z = ad.offload(t.exp(x),plan)
        ad.backward(t.sum(x*2),mun,{"retain":[x]})
        let seed = t.array([1e10,1e10],"float64")
        collect()
        let base = c.memory_stats()["tensor_heap_bytes"]
        c.memory_budget(base+32)
        try:
            ad.backward(z,seed,{"retain":[x],"consume":true})
        catch issue:
            print(issue["code"],c.memory_stats()["workspace_reserved_bytes"],t.tolist(ad.grad(x)))
        c.memory_budget(mun)
        try:
            ad.backward(z,seed,{"retain":[x],"consume":true})
        catch issue:
            print(issue["code"],c.memory_stats()["workspace_reserved_bytes"],t.tolist(ad.grad(x)))
        ad.backward(z,t.array([1,1],"float64"),{"retain":[x],"consume":true})
        print(t.at(ad.grad(x),[0]) > 1e300)
        """, "memory_budget 0 [2, 2]\nnonfinite 0 [2, 2]\nnae", stress=True)

    def test_offload_checks_inputs_and_packs_logical_tensor_order(self):
        self.run_code("""
        import ad
        import tensor as t
        import compute as c
        try:
            ad.offload_plan("./missing",10)
        catch issue:
            print(issue["code"])
        try:
            ad.offload_plan(".",-1)
        catch issue:
            print(issue["code"])
        try:
            ad.offload_plan(".",160,0)
        catch issue:
            print(issue["code"])
        let plan = ad.offload_plan(".",160,1)
        let x = ad.variable(t.array([1,2,3,4],"float64",[2,2]))
        try:
            ad.offload(x,plan)
        catch issue:
            print(issue["code"])
        try:
            ad.offload(t.sum(x),plan)
        catch issue:
            print(issue["code"])
        try:
            ad.offload(t.view(x,[4],[-1],3),plan)
        catch issue:
            print(issue["code"])
        let y = ad.offload(t.permute(x,[1,0]),plan)
        print(t.tolist(ad.value(y)),t.tolist(ad.vjp(t.sum(y*y),x)))
        print(t.tolist(ad.vjp(t.sum(t.matmul(y,x)),x)))
        collect()
        let empty = ad.offload_plan(".",0)
        let e = ad.variable(t.zeros([0,2],"float64"))
        let empty_result = ad.offload(e+1,empty)
        print(t.shape(ad.value(empty_result)),ad.offload_stats(empty)["live_files"])
        """, "io\ndisk_budget\nshape\nautodiff\ndtype\nautodiff\n[[1, 3], [2, 4]] [[2, 4], [6, 8]]\n[[6, 6], [14, 14]]\n[0, 2] 0",
            stress=True)
