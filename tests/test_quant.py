"""Quantized block layout, budgets, snapshots and explicit precision."""
import unittest

import test_arda


class QuantTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_block_layout_rounding_row_tail_and_snapshot(self):
        self.run_code("""
        import tensor as t
        import quant as q
        let w = t.zeros([2,33],"float64")
        t.put(w,[0,0],7)
        t.put(w,[0,1],-7)
        t.put(w,[0,2],0.5)
        t.put(w,[0,3],-0.5)
        t.put(w,[0,32],70)
        let packed = q.quantize(w,"q4")
        let q8 = q.quantize(t.zeros([2,33],"float32"),"q8")
        let meta = q.info(packed)
        print(type(packed),meta["shape"],meta["encoding"],meta["block_size"],meta["blocks"])
        print(meta["bytes"],meta["code_bytes"],meta["scale_bytes"],q.info(q8)["bytes"])
        t.put(w,[0,0],0)
        collect()
        let d = q.dequantize(packed)
        print(t.at(d,[0,0]),t.at(d,[0,2]),t.at(d,[0,3]),t.at(d,[0,32]),t.at(d,[1,32]))
        let output = q.gemv(packed,t.zeros([33],"float64")+1)
        print(t.tolist(output))
        print(t.tolist(q.gemm(packed,t.zeros([33,3],"float64")+2)))
        """, "quantized_matrix [2, 33] q4 32 4\n80 64 16 144\n7 1 -1 70 0\n[70, 0]\n[[140, 140, 140], [0, 0, 0]]",
            stress=True)

    def test_invalid_input_releases_partial_encoding_and_budget_is_checked(self):
        self.run_code("""
        import tensor as t
        import quant as q
        import compute as c
        import ad
        let w = t.zeros([4,65],"float64")
        let overflow = t.array([1e200],"float64",[1,1])
        let nonfinite = t.zeros([1,33],"float64")
        t.put(nonfinite,[0,32],1/0)
        let tracked = ad.variable(w)
        collect()
        let base = c.memory_stats()["tensor_heap_bytes"]
        try:
            q.quantize(nonfinite,"q4")
        catch issue:
            print(issue["code"],c.memory_stats()["tensor_heap_bytes"]==base)
        try:
            q.quantize(overflow,"q8")
        catch issue:
            print(issue["code"],c.memory_stats()["tensor_heap_bytes"]==base)
        try:
            q.quantize(tracked,"q4")
        catch issue:
            print(issue["code"])
        try:
            q.quantize(w,"q2")
        catch issue:
            print(issue["code"])
        c.memory_budget(base+239)
        try:
            q.quantize(w,"q4")
        catch issue:
            print(issue["code"],c.memory_stats()["tensor_heap_bytes"]==base)
        c.memory_budget(base+240)
        let packed = q.quantize(w,"q4")
        print(q.info(packed)["bytes"],c.memory_stats()["tensor_heap_bytes"]==base+240)
        """, "nonfinite nae\noverflow nae\nautodiff\ndomain\nmemory_budget nae\n240 nae",
            stress=True)

    def test_multiplication_fits_output_only_budget_and_empty_axes(self):
        self.run_code("""
        import tensor as t
        import quant as q
        import compute as c
        let packed = q.quantize(t.zeros([64,65],"float64")+7,"q4")
        let x = t.zeros([65],"float64")+1
        collect()
        collect()
        let base = c.memory_stats()["tensor_heap_bytes"]
        c.memory_budget(base+512)
        let output = q.gemv(packed,x)
        print(t.shape(output),t.at(output,[63]),c.memory_stats()["tensor_heap_bytes"]==base+512)
        try:
            q.dequantize(packed)
        catch issue:
            print(issue["code"])
        c.memory_budget(mun)
        let empty = q.quantize(t.zeros([2,0],"float64"),"q8")
        print(q.info(empty)["bytes"],t.tolist(q.gemv(empty,t.zeros([0],"float64"))))
        print(t.shape(q.gemm(empty,t.zeros([0,4],"float64"))))
        try:
            q.gemv(packed,t.zeros([64],"float64"))
        catch issue:
            print(issue["code"])
        """, "[64] 455 nae\nmemory_budget\n0 [0, 0]\n[2, 4]\nshape", stress=True)

    def test_strided_inputs_and_explicit_low_precision(self):
        self.run_code("""
        import quant as q
        import tensor as t
        let a = t.view(t.array([7,2,-7],"float64"),[1,3],[0,-1],2)
        let x = t.view(t.array([1,2,3],"float64"),[3],[-1],2)
        let w = q.quantize(a,"q4")
        print(t.tolist(q.dequantize(w)),t.tolist(q.gemv(w,x)))
        let b = t.permute(t.array([1,2,3,4,5,6],"float64",[2,3]),[1,0])
        print(t.tolist(q.gemm(w,b)))
        try:
            q.gemm(w,b,"simd")
        catch issue:
            print(issue["code"])
        for dtype in ["float16","bfloat16"]:
            let left = t.zeros([5,3],dtype)+0.25
            left = t.cast(left,dtype)
            let right = t.cast(t.zeros([3,11],"float64")+2,dtype)
            let ref = q.lowp_matmul(left,right,"reference")
            let fast = q.lowp_matmul(left,right)
            let diff = t.cast(fast,"float64") - t.cast(ref,"float64")
            print(t.dtype(fast),t.at(fast,[4,10]),t.sum(diff*diff)==0)
        """, "[[-7, 2, 7]] [-10]\n[[18, 24]]\nbackend\nfloat32 1.5 nae\nfloat32 1.5 nae",
            stress=True)

    def test_small_scales_and_low_precision_subnormal_values(self):
        self.run_code("""
        import quant as q
        import tensor as t
        let w = t.array([1e-42,1e-300],"float64",[2,1])
        let d = q.dequantize(q.quantize(w,"q4"))
        print(abs(t.at(d,[0,0])-1e-42)<1e-43,t.at(d,[1,0])==0)
        let standards = [5.960464477539063e-8,9.183549615799121e-41]
        let types = ["float16","bfloat16"]
        for i in range(2):
            let tiny = standards[i]
            let dtype = types[i]
            let a = t.array([tiny],dtype,[1,1])
            let b = t.cast(t.zeros([1,19],"float64")+1,dtype)
            let output = q.lowp_matmul(a,b)
            print(t.at(output,[0,0])==tiny,t.at(output,[0,18])==tiny)
        """, "nae nae\nnae nae\nnae nae", stress=True)
