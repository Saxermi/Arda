"""Runtime CPU dispatch, packed matrix tails and optional vector math."""
import unittest

import test_arda


class CpuKernelTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_matmul_dispatch_and_unaligned_rectangular_tails(self):
        self.run_code("""
        import tensor as t
        import compute as c
        let flags = c.cpu()
        let initial = flags["matmul_kernel"]
        let left = t.view(t.zeros([33*65+1],"float64")+0.25,[33,65],[65,1],1)
        let right = t.view(t.zeros([65*67+1],"float64")-0.5,[65,67],[67,1],1)
        let automatic = t.matmul(left,right)
        let empty = t.matmul(t.zeros([2,0],"float64"),t.zeros([0,2],"float64"))
        print(t.shape(automatic),t.at(automatic,[32,66]),t.tolist(empty))
        let ok = true
        for name in ["reference","sse2","avx2","avx2-fma","neon"]:
            try:
                c.matmul_kernel(name)
                let diff = t.matmul(left,right)-automatic
                ok = ok and t.sum(diff*diff) == 0
            catch issue:
                ok = ok and issue["code"] == "backend"
        c.matmul_kernel("auto")
        print(ok,c.matmul_kernel()==initial,c.vector_math()==false)
        """, "[33, 67] -8.125 [[0, 0], [0, 0]]\nnae nae nae", stress=True)

    def test_fused_multiply_add_requires_explicit_selection(self):
        self.run_code("""
        import tensor as t
        import compute as c
        let cpu = c.cpu()
        if cpu["auto_matmul"] == "avx2" and cpu["fma"]:
            let e = 1/134217728
            let a = t.zeros([5,2],"float64") + t.array([-1,1+e],"float64")
            let b = t.zeros([2,9],"float64") + t.reshape(t.array([1,1-e],"float64"),[2,1])
            let ordinary = t.matmul(a,b)
            c.matmul_kernel("avx2-fma")
            let fused = t.matmul(a,b)
            print(t.at(ordinary,[0,0]) == 0,t.at(fused,[0,0]) == -e*e,t.at(fused,[4,8]) == -e*e)
        else:
            print(true,true,true)
        """, "nae nae nae", stress=True)

    def test_vector_math_is_opt_in_and_preserves_tracked_path(self):
        self.run_code("""
        import tensor as t
        import compute as c
        import ad
        let x = t.zeros([17],"float64")+0.37
        let reference = t.sin(x)+t.cos(x)+t.exp(x)
        if c.cpu()["vector_math_available"]:
            c.vector_math(true)
            let diff = t.sin(x)+t.cos(x)+t.exp(x)-reference
            let tracked = ad.variable(x)
            let ad_result = ad.value(t.sin(tracked)+t.cos(tracked)+t.exp(tracked))
            print(t.sum(diff*diff) < 1e-27,t.sum(ad_result-reference) == 0,c.vector_math(false)==false)
        else:
            try:
                c.vector_math(true)
            catch issue:
                print(issue["code"]=="backend",c.vector_math()==false,true)
        """, "nae nae nae", stress=True)
