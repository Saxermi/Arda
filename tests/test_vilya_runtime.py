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
