"""Vilya runtime behavior and transactional tensor gradients."""
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
