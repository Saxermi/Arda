"""Typed tensor storage, layout, and arithmetic contracts."""
import unittest

import test_arda


class TypedTensorTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_casts_storage_and_promotion(self):
        self.run_code("""
        tol tensor as t
        tol numc as nc
        esta ints = t.array([9223372036854775807,-9223372036854775808],"int64")
        esta half = t.array([1.00048828125,1.00146484375,0.000000059604644775390625],"float16")
        esta brain = t.array([1.00390625,1.01171875],"bfloat16")
        esta wide = t.array([1,2],"int32") + t.array([0.5,0.125],"float32")
        pedo(t.tolist(ints),t.bytes(ints))
        pedo(t.tolist(half),t.bytes(half),t.dtype(half + half))
        pedo(t.tolist(brain),t.bytes(brain))
        pedo(t.dtype(wide),nc.tolist(t.to_numc(wide)))
        pedo(t.tolist(t.array([true,false],"bool")),t.dtype(t.array([9],"uint8")/3))
        """, "[9223372036854775807, -9223372036854775808] 16\n"
             "[1, 1.001953125, 5.96046447753906e-08] 6 float32\n"
             "[1, 1.015625] 4\nfloat64 [1.5, 2.125]\n[nae, law] float64", stress=True)
        for expr, error in [
            ('t.array([256],"uint8")', "overflow"),
            ('t.array([1.5],"int32")', "integral"),
            ('t.array([65520],"float16")', "overflow"),
            ('t.array([9223372036854775807],"int64") + 1', "integer overflow"),
            ('t.array([cx.new(1,1)],"float64")', "imaginary"),
            ('t.array([2],"bool")', "overflow"),
        ]:
            with self.subTest(expr=expr):
                self.run_code("tol tensor as t\ntol complex as cx\npedo(" + expr + ")",
                              error=error, stress=True)

    def test_signed_views_and_mutability(self):
        self.run_code("""
        tol tensor as t
        tol numc as nc
        esta source = nc.array([2,4,6,8])
        esta reverse = t.view(source,[4],[-1],3)
        esta broadcast = t.view(source,[2,3],[0,0],1)
        source[1] = 5
        source = mun
        collect()
        pedo(t.tolist(reverse),t.tolist(broadcast))
        pedo(t.tolist(reverse + t.array([0.5],"float32")))
        esta copy = t.cast(reverse,"int32")
        t.put(copy,[0],99)
        pedo(t.at(copy,[0]),t.at(reverse,[0]))
        """, "[8, 6, 5, 2] [[5, 5, 5], [5, 5, 5]]\n[8.5, 6.5, 5.5, 2.5]\n99 8", stress=True)
        for expr, error in [
            ("t.view(x,[3],[-1],1)", "outside"),
            ("t.view(x,[2],[9223372036854775807],0)", "outside"),
            ("t.put(t.view(x,[3],[0],1),[0],3)", "read-only"),
        ]:
            self.run_code('tol tensor as t\nesta x=t.array([1,2,3],"int32")\n' + expr,
                          error=error, stress=True)

    def test_batch_contractions_and_indexing(self):
        self.run_code("""
        tol tensor as t
        esta a = t.array([1,2,3,4,5,6,7,8],"int32",[2,2,2])
        esta b = t.array([1,0,2,1],"int32",[2,2])
        esta product = t.matmul(a,b)
        pedo(t.dtype(product),t.tolist(product))
        pedo(t.tolist(t.einsum("bij,jk->bik",a,b)))
        pedo(t.tolist(t.sum(a,1)))
        pedo(t.tolist(t.tensordot(a,b,2,0)))
        pedo(t.tolist(t.gather(a,[7,0,3])))
        pedo(t.tolist(t.scatter(a,[0,0,7],[9,8,0])))
        pedo(t.tolist(t.mask(b,t.array([true,false,false,true],"bool",[2,2]))))
        pedo(t.einsum("i,i->",t.array([1,2],"int32"),t.array([3,4],"int32")))
        pedo(t.tolist(t.matmul(t.zeros([2,0],"float32"),t.zeros([0,2],"float32"))))
        """, "int64 [[[5, 2], [11, 4]], [[17, 6], [23, 8]]]\n"
             "[[[5, 2], [11, 4]], [[17, 6], [23, 8]]]\n"
             "[[4, 6], [12, 14]]\n"
             "[[[5, 2], [11, 4]], [[17, 6], [23, 8]]]\n[8, 1, 4]\n"
             "[[[8, 2], [3, 4]], [[5, 6], [7, 0]]]\n[1, 1]\n11\n[[0, 0], [0, 0]]", stress=True)
