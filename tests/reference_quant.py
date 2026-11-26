"""NumPy and high precision references for block quantization and matrix products."""
import ast
import unittest

import numpy as np

import test_arda


def reconstruct(weights, limit):
    """Independent symmetric per-row scale and ties-away-from-zero rounding."""
    result = np.empty_like(weights)
    for begin in range(0, weights.shape[1], 32):
        block = weights[:, begin:begin + 32]
        ideal = np.max(np.abs(block.astype(np.longdouble)), axis=1, keepdims=True) / limit
        scale = ideal.astype(np.float32)
        scale = np.where(scale.astype(np.longdouble) < ideal,
                         np.nextafter(scale, np.float32(np.inf)), scale).astype(np.float64)
        relative = np.divide(block, scale, out=np.zeros_like(block), where=scale > 0)
        quantized = np.copysign(np.floor(np.abs(relative) + 0.5), relative)
        result[:, begin:begin + 32] = np.clip(quantized, -limit, limit) * scale
    return result


class QuantReferenceTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_all_complete_blocks_and_ragged_tails_for_both_encodings(self):
        rng = np.random.default_rng(1803)
        w = rng.normal(size=(5, 67))
        w[-1, :32] = 0
        w[0, 64:] *= 1e-3
        x = rng.normal(size=(67, 19))
        vector = rng.normal(size=67)
        lines = iter(self.run_code(f"""
        import tensor as t
        import quant as q
        let w = t.array({w.ravel().tolist()},"float64",[5,67])
        let x = t.array({x.ravel().tolist()},"float64",[67,19])
        let v = t.array({vector.tolist()},"float64")
        for format in ["q4","q8"]:
            let packed = q.quantize(w,format)
            print(t.tolist(q.dequantize(packed)))
            let reference = q.gemm(packed,x,"reference")
            let auto = q.gemm(packed,x)
            print(t.tolist(reference))
            print(t.tolist(auto))
            let ref_v = q.gemv(packed,v,"reference")
            let auto_v = q.gemv(packed,v)
            print(t.tolist(ref_v))
            print(t.tolist(auto_v))
            if q.backends()["packed"] != "reference":
                let r = q.gemm(packed,x,"simd")-reference
                let s = q.gemv(packed,v,"simd")-ref_v
                print(t.sum(r*r)==0,t.sum(s*s)==0)
            else:
                print(true,true)
        """, stress=True).splitlines())
        for limit in (7, 127):
            rebuilt = reconstruct(w, limit)
            np.testing.assert_allclose(ast.literal_eval(next(lines)), rebuilt, rtol=1e-14)
            expected = rebuilt.astype(np.longdouble) @ x.astype(np.longdouble)
            for _ in range(2):
                np.testing.assert_allclose(ast.literal_eval(next(lines)), expected, rtol=4e-14,
                                           atol=2e-13)
            expected_vector = rebuilt.astype(np.longdouble) @ vector.astype(np.longdouble)
            for _ in range(2):
                np.testing.assert_allclose(ast.literal_eval(next(lines)), expected_vector,
                                           rtol=4e-14, atol=2e-13)
            self.assertEqual(next(lines), "nae nae")

    def test_fp16_and_bf16_widening_with_a_float32_reference(self):
        rng = np.random.default_rng(813)
        a = rng.integers(-24, 25, size=(7, 13)) / 16
        b = rng.integers(-24, 25, size=(13, 11)) / 16
        result = self.run_code(f"""
        import tensor as t
        import quant as q
        for dtype in ["float16","bfloat16"]:
            let a = t.array({a.ravel().tolist()},dtype,[7,13])
            let b = t.array({b.ravel().tolist()},dtype,[13,11])
            print(t.tolist(q.lowp_matmul(a,b,"reference")))
            print(t.tolist(q.lowp_matmul(a,b)))
            if q.backends()[dtype] != "reference":
                print(t.tolist(q.lowp_matmul(a,b,"simd")))
            else:
                print(t.tolist(q.lowp_matmul(a,b,"reference")))
        """, stress=True)
        expected = a.astype(np.float32) @ b.astype(np.float32)
        for line in result.splitlines():
            np.testing.assert_array_equal(ast.literal_eval(line), expected)
