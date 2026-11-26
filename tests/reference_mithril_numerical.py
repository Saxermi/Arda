"""Independent numerical tests for Arda low-rank formats and scientific units."""
import ast
import math
import os
import unittest

import numpy as np

import test_arda


def tensor(values):
    return f"nc.array({np.asarray(values, dtype=np.float64).tolist()!r})"


def tensors(values):
    return "[" + ",".join(map(tensor, values)) + "]"


def cp_dense(weights, factors):
    return np.einsum("r,ir,jr,kr->ijk", weights, *factors, optimize=True)


def tucker_dense(core, factors):
    return np.einsum("abc,ia,jb,kc->ijk", core, *factors, optimize=True)


def tt_dense(cores):
    return np.einsum("aib,bjc,ckd->ijk", *cores, optimize=True)


def parsed(line):
    return np.asarray(ast.literal_eval(line))


def metric(label, value):
    if os.environ.get("ARDA_NUMERICAL_TEST_METRICS"):
        print(f"numerical metric {label} = {value:.9g}", flush=True)


class MithrilNumericalReferenceTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    @staticmethod
    def close(label, actual, expected, rtol=8e-13, atol=1e-13):
        a, b = np.asarray(actual), np.asarray(expected)
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=label)
        metric(label, np.max(np.abs(a - b), initial=0.0))

    def compare_format(self, name, x_expr, y_expr, a, b, rng):
        probes = [(0, 0, 0), tuple(n - 1 for n in a.shape), (1, 2, 0)]
        script = [
            "tol numc as nc",
            "tol lowrank as lr",
            f"esta x = {x_expr}",
            f"esta y = {y_expr}",
            "collect()",
            "pedo(lr.shape(x))",
            "pedo(nc.tolist(lr.materialize(x)))",
            "pedo(nc.tolist(lr.materialize(y)))",
            "pedo(lr.norm(x),lr.norm(y),lr.dot(x,y),lr.dot(y,x))",
            'pedo(' + ','.join(f"lr.at(x,{list(index)!r})" for index in probes) + ')',
        ]
        vectors = [rng.normal(size=n) for n in a.shape]
        for axis, vector in enumerate(vectors):
            script.append(
                f"pedo(nc.tolist(lr.materialize(lr.contract(x,{axis},{tensor(vector)}))))"
            )
        lines = self.run_code("\n".join(script), stress=True).splitlines()
        self.assertEqual(len(lines), len(script) - 5)
        self.assertEqual(tuple(ast.literal_eval(lines[0])), a.shape)
        self.close(f"{name} left reconstruction max abs", parsed(lines[1]), a)
        self.close(f"{name} right reconstruction max abs", parsed(lines[2]), b)
        reference_dot = np.vdot(a, b)
        self.close(f"{name} norm and dot max abs", list(map(float, lines[3].split())),
                   [np.linalg.norm(a), np.linalg.norm(b), reference_dot, reference_dot])
        self.close(f"{name} at max abs", list(map(float, lines[4].split())),
                   [a[i] for i in probes])
        for axis, (line, vector) in enumerate(zip(lines[5:], vectors, strict=True)):
            expected = np.tensordot(a, vector, axes=([axis], [0]))
            self.close(f"{name} contract axis {axis} max abs", parsed(line), expected)

    def test_cp_signed_weights_and_unequal_latent_ranks(self):
        rng = np.random.default_rng(164)
        sizes = (3, 4, 2)
        factors = [rng.normal(size=(n, 3)) for n in sizes]
        other = [rng.normal(size=(n, 2)) for n in sizes]
        weights = np.array([1.3, -0.7, 2.1])
        other_weights = np.array([-1.5, 0.5])
        self.compare_format("CP", f"lr.cp({tensors(factors)},{tensor(weights)})",
                            f"lr.cp({tensors(other)},{tensor(other_weights)})",
                            cp_dense(weights, factors), cp_dense(other_weights, other), rng)

    def test_norm_and_dot_resolve_signed_cancellation(self):
        delta = 2.0 ** -36
        first = np.array([[1.0, 1.0 + delta], [1.0, 1.0]])
        lines = self.run_code(f"""
        tol numc as nc
        tol lowrank as lr
        esta first = {tensor(first)}
        esta cp_factors = [first,nc.ones(2,2)]
        esta tucker_factors = [first,nc.ones(2,1)]
        esta core = nc.array([[1],[-1]])
        esta left = nc.reshape(first,[1,2,2])
        esta right = nc.reshape(nc.array([[1,1],[-1,-1]]),[2,2,1])
        esta cp = [lr.cp(cp_factors,[1,-1]),lr.cp(cp_factors,[-2,2])]
        esta tucker = [lr.tucker(core,tucker_factors),lr.tucker(-2*core,tucker_factors)]
        esta tt = [lr.tt([left,right]),lr.tt([left,-2*right])]
        an pair in [cp,tucker,tt]:
            esta x = pair[0]
            esta y = pair[1]
            pedo(lr.at(x,[0,0]),lr.norm(x),lr.dot(x,x),lr.dot(x,y),lr.dot(y,x))
        """, stress=True).splitlines()
        for name, row in zip(("CP", "Tucker", "TT"), lines, strict=True):
            square = 2 * delta * delta
            self.close(f"{name} cancellation max abs", list(map(float, row.split())),
                       [-delta, math.sqrt(2) * delta, square, -2 * square, -2 * square],
                       rtol=3e-14, atol=0)

    def test_large_exactly_orthogonal_tensors(self):
        n = 200_000
        lines = self.run_code(f"""
        tol numc as nc
        tol lowrank as lr
        esta long = nc.ones({n},1)
        esta u = nc.ones(3,1)
        esta v = nc.array([[2],[-2],[0]])
        esta cp1 = lr.cp([u,long])
        esta cp2 = lr.cp([nc.array([[2,0],[0,-2],[0,0]]),nc.ones({n},2)])
        esta tk1 = lr.tucker(nc.ones(1,1),[u,long])
        esta tk2 = lr.tucker(nc.ones(1,1),[v,long])
        esta right = nc.reshape(long,[1,{n},1])
        esta tt1 = lr.tt([nc.reshape(u,[1,3,1]),right])
        esta tt2 = lr.tt([nc.reshape(v,[1,3,1]),right])
        an pair in [[cp1,cp2],[tk1,tk2],[tt1,tt2]]:
            esta x = pair[0]
            esta y = pair[1]
            pedo(lr.norm(x),lr.norm(y),lr.dot(x,y),lr.dot(y,x))
        """, stress=True).splitlines()
        for name, row in zip(("CP", "Tucker", "TT"), lines, strict=True):
            norm_x, norm_y, xy, yx = map(float, row.split())
            self.close(f"{name} orthogonal tensor norm max abs", [norm_x, norm_y],
                       [math.sqrt(3 * n), math.sqrt(8 * n)])
            self.assertEqual((xy, yx), (0.0, 0.0), name)

    def test_tucker_nonsquare_factors_and_all_mode_contractions(self):
        rng = np.random.default_rng(165)
        sizes = (3, 4, 2)
        ranks, other_ranks = (2, 3, 2), (3, 2, 1)
        core = rng.normal(size=ranks)
        other_core = rng.normal(size=other_ranks)
        factors = [rng.normal(size=(n, r)) for n, r in zip(sizes, ranks, strict=True)]
        other = [rng.normal(size=(n, r)) for n, r in zip(sizes, other_ranks, strict=True)]
        self.compare_format("Tucker", f"lr.tucker({tensor(core)},{tensors(factors)})",
                            f"lr.tucker({tensor(other_core)},{tensors(other)})",
                            tucker_dense(core, factors), tucker_dense(other_core, other), rng)

    def test_tt_unequal_bond_ranks_and_all_mode_contractions(self):
        rng = np.random.default_rng(166)
        left = [rng.normal(size=s) for s in [(1, 3, 2), (2, 4, 3), (3, 2, 1)]]
        right = [rng.normal(size=s) for s in [(1, 3, 3), (3, 4, 2), (2, 2, 1)]]
        self.compare_format("TT", f"lr.tt({tensors(left)})", f"lr.tt({tensors(right)})",
                            tt_dense(left), tt_dense(right), rng)

    def test_large_logical_shapes_with_small_real_budget(self):
        dims = (64, 68, 70, 67)
        n = math.prod(dims)
        self.assertGreater(n, 16_000_000)
        factors = "[" + ",".join(f"nc.ones({size},1)" for size in dims) + "]"
        cores = "[" + ",".join(f"nc.reshape(nc.ones({size}),[1,{size},1])"
                                for size in dims) + "]"
        lines = self.run_code(f"""
        tol numc as nc
        tol lowrank as lr
        tol compute as c
        c.memory_budget(250000)
        esta cp = lr.cp({factors},nc.array([-2]))
        esta tk = lr.tucker(nc.array([[[[2]]]]),{factors})
        esta tt = lr.tt({cores})
        an x in [cp,tk,tt]:
            pedo(lr.shape(x))
            esta y = lr.contract(x,2,nc.ones({dims[2]}))
            pedo(lr.at(x,[7,11,13,19]),lr.norm(x),lr.dot(x,x),lr.at(y,[5,9,12]))
            try:
                lr.materialize(x)
                pedo("unexpected materialization")
            catch err:
                pedo("materialization blocked")
        pedo(c.memory_stats()["peak_budgeted_bytes"])
        """, stress=True).splitlines()
        for i, sign in enumerate((-2, 2, 1)):
            row = i * 3
            self.assertEqual(ast.literal_eval(lines[row]), list(dims))
            self.close("large compressed reductions max abs",
                       list(map(float, lines[row + 1].split())),
                       [sign, abs(sign) * math.sqrt(n), sign * sign * n, sign * dims[2]])
            self.assertEqual(lines[row + 2], "materialization blocked")
        peak = int(lines[-1])
        self.assertLessEqual(peak, 250_000)
        metric("large test charged peak bytes", peak)

    def test_logical_element_product_beyond_int32(self):
        lines = self.run_code("""
        tol numc as nc
        tol lowrank as lr
        tol compute as c
        c.memory_budget(5000000)
        esta column = nc.ones(50000,1)
        esta x = lr.cp([column,column])
        pedo(lr.at(x,[49999,49999]),lr.norm(x),lr.dot(x,x))
        pedo(c.memory_stats()["peak_budgeted_bytes"])
        """, stress=True).splitlines()
        self.assertEqual(list(map(float, lines[0].split())), [1, 50_000, 2_500_000_000])
        self.assertLessEqual(int(lines[1]), 5_000_000)
        metric("2.5 billion logical element peak budgeted bytes", int(lines[1]))

    def require_lapack(self):
        if self.run_code('tol lowrank as lr\npedo(lr.backend())').strip() != "lapack":
            self.skipTest("low-rank SVD requires LAPACK")

    def test_tt_svd_controlled_truncation_on_known_singular_spectrum(self):
        self.require_lapack()
        dense = np.zeros((4, 4, 3))
        for i, weight in enumerate((4.0, 3.0, 1.0)):
            dense[i, i, i] = weight
        lines = self.run_code(f"""
        tol numc as nc
        tol lowrank as lr
        esta x = {tensor(dense)}
        esta full = lr.tt_svd(x,0.0)
        esta cut = lr.tt_svd(x,0.25,2)
        pedo(nc.tolist(lr.materialize(full[0])))
        pedo(full[1])
        pedo(nc.tolist(lr.materialize(cut[0])))
        pedo(cut[1])
        """, stress=True).splitlines()
        self.close("TT-SVD exact reconstruction max abs", parsed(lines[0]), dense)
        self.assertLessEqual(float(lines[1]), 1e-14)
        low = dense.copy()
        low[2, 2, 2] = 0.0
        self.close("TT-SVD rank-two optimum max abs", parsed(lines[2]), low)
        actual_relative_error = np.linalg.norm(parsed(lines[2]) - dense) / np.linalg.norm(dense)
        expected_relative_tail = 1 / math.sqrt(26)
        np.testing.assert_allclose(float(lines[3]), expected_relative_tail, rtol=5e-13)
        self.assertLessEqual(actual_relative_error, float(lines[3]) + 2e-13)
        metric("TT-SVD rank-two relative residual", actual_relative_error)
        metric("TT-SVD rank-two reported tail", float(lines[3]))

    def test_round_preserves_error_bound_under_bond_rescaling(self):
        self.require_lapack()
        rng = np.random.default_rng(1663)
        cores = [rng.normal(size=s) for s in [(1, 4, 3), (3, 5, 4), (4, 3, 1)]]
        # These inverse gauges agree exactly in the represented dense tensor.
        left_scale = np.array([1e-3, 3.0, 1e3])
        right_scale = np.array([1e2, 8e-3, 7.0, 2e-2])
        gauged = [cores[0] * left_scale,
                  cores[1] * right_scale / left_scale[:, None, None],
                  cores[2] / right_scale[:, None, None]]
        dense = tt_dense(cores)
        np.testing.assert_allclose(tt_dense(gauged), dense, rtol=4e-14, atol=7e-14)
        lines = self.run_code(f"""
        tol numc as nc
        tol lowrank as lr
        esta x = lr.tt({tensors(gauged)})
        esta base = lr.round(lr.tt({tensors(cores)}),0.0)
        esta keep = lr.round(x,0.0)
        esta reduced = lr.round(x,0.8,1)
        pedo(nc.tolist(lr.materialize(base[0])))
        pedo(nc.tolist(lr.materialize(keep[0])))
        pedo(keep[1])
        pedo(nc.tolist(lr.materialize(reduced[0])))
        pedo(reduced[1])
        """, stress=True).splitlines()
        self.close("TT rounding full-rank error", parsed(lines[0]), dense)
        # Undoing the 10^6 gauge creates more roundoff than the standard case.
        # The discarded singular-value estimator excludes arithmetic roundoff.
        without_truncation = np.linalg.norm(parsed(lines[1]) - dense) / np.linalg.norm(dense)
        self.assertLessEqual(without_truncation, 1e-11)
        self.assertLessEqual(float(lines[2]), 1e-13)
        metric("TT round scaled-gauge zero-truncation relative roundoff", without_truncation)
        reduced = parsed(lines[3])
        for split in range(1, dense.ndim):
            self.assertEqual(np.linalg.matrix_rank(
                reduced.reshape(math.prod(dense.shape[:split]), -1), tol=1e-10), 1)
        actual_relative_error = np.linalg.norm(reduced - dense) / np.linalg.norm(dense)
        reported = float(lines[4])
        self.assertGreater(actual_relative_error, 0.01)
        self.assertLessEqual(actual_relative_error, reported + 5e-12)
        metric("TT round gauged rank-one relative residual", actual_relative_error)
        metric("TT round gauged rank-one reported tail", reported)

    def test_tt_svd_random_truncations_match_independent_residuals(self):
        self.require_lapack()
        rng = np.random.default_rng(1644)
        cases = [rng.normal(size=shape) for shape in [(2, 3, 4), (3, 2, 3), (4, 4, 2),
                                                       (3, 5, 3), (2, 4, 5), (4, 6, 2)]]
        script = ["tol numc as nc", "tol lowrank as lr"]
        for i, data in enumerate(cases):
            script += [
                f"esta x{i} = lr.tt_svd({tensor(data)},0.1,2)",
                f"pedo(x{i}[1])",
                f"pedo(nc.tolist(lr.materialize(x{i}[0])))",
            ]
        lines = self.run_code("\n".join(script), stress=True).splitlines()
        self.assertEqual(len(lines), 2 * len(cases))
        maximum_disagreement = 0
        for i, data in enumerate(cases):
            reported = float(lines[2 * i])
            reduced = parsed(lines[2 * i + 1])
            residual = np.linalg.norm(reduced - data) / np.linalg.norm(data)
            self.assertLessEqual(residual, reported + 3e-13)
            maximum_disagreement = max(maximum_disagreement, abs(residual - reported))
            for axis in range(1, data.ndim):
                split = reduced.reshape(math.prod(data.shape[:axis]), -1)
                self.assertLessEqual(np.linalg.matrix_rank(split, tol=1e-10), 2)
        metric("six random TT-SVD residual-to-report max abs", maximum_disagreement)

    def test_zero_tensor_tt_svd_and_round_have_finite_zero_error(self):
        self.require_lapack()
        zeros = np.zeros((3, 4, 2))
        lines = self.run_code(f"""
        tol numc as nc
        tol lowrank as lr
        esta x = lr.tt_svd({tensor(zeros)},1e-6,1)
        esta y = lr.round(x[0],1e-6,1)
        pedo(x[1],y[1],lr.norm(x[0]),lr.norm(y[0]))
        pedo(nc.tolist(lr.materialize(y[0])))
        """, stress=True).splitlines()
        self.assertEqual(list(map(float, lines[0].split())), [0, 0, 0, 0])
        np.testing.assert_array_equal(parsed(lines[1]), zeros)

    def test_si_affine_temperatures_and_vectorized_matmul(self):
        lines = self.run_code("""
        tol numc as nc
        tol units as u
        esta length = u.quantity(2,"m") + u.quantity(150,"cm")
        esta velocity = length / u.quantity(2,"s")
        esta force = u.quantity(1,"N")
        pedo(u.value(length,"cm"),u.value(velocity,"m/s"),u.value(force,"kg*m/s^2"))
        pedo(u.dimension(velocity))
        esta c = u.quantity(100,"degC")
        esta f = u.convert(c,"degF")
        esta cold = u.quantity(32,"degF")
        pedo(u.value(c,"K"),u.value(f),u.value(cold,"degC"),u.value(c-cold,"deltaF"))
        pedo(u.value(cold+u.quantity(9,"deltaF"),"degC"))
        esta x = u.quantity(nc.array([[1,2],[3,4]]),"cm")
        esta y = u.quantity(nc.array([[2,1],[1,0]]),"m")
        pedo(nc.tolist(u.value(u.matmul(x,y),"m^2")))
        pedo(u.value(u.sum(x),"m"))
        """, stress=True).splitlines()
        self.close("SI unit conversion max abs", list(map(float, lines[0].split())),
                   [350, 1.75, 1])
        self.assertEqual(ast.literal_eval(lines[1]), [1, 0, -1, 0, 0, 0, 0])
        self.close("absolute and delta temperature conversion max abs",
                   list(map(float, lines[2].split())), [373.15, 212, 0, 180])
        self.close("mixed absolute and delta temperature max abs", float(lines[3]), 5)
        self.close("unit-aware matmul max abs", parsed(lines[4]),
                   np.array([[1, 2], [3, 4]]) @ np.array([[2, 1], [1, 0]]) / 100)
        self.close("unit-aware sum max abs", float(lines[5]), 0.1)

    def test_vectorized_units_against_random_numpy_oracles(self):
        rng = np.random.default_rng(1665)
        cm = rng.uniform(3, 50, (3, 4))
        metres = rng.uniform(-4, 5, (1, 4))
        seconds = rng.uniform(1, 15, (3, 1))
        celsius = np.array([[-273.15, -100, -40, 0], [0.5, 21, 37, 100], [150, 190, 320, 1000]])
        lines = self.run_code(f"""
        tol numc as nc
        tol units as u
        esta a = u.quantity({tensor(cm)},"cm")
        esta b = u.quantity({tensor(metres)},"m")
        esta t = u.quantity({tensor(seconds)},"s")
        esta thermal = u.quantity({tensor(celsius)},"degC")
        pedo(nc.tolist(u.value(a+b,"m")))
        pedo(nc.tolist(u.value(a*b,"m^2")))
        pedo(nc.tolist(u.value(a/t,"m/s")))
        pedo(nc.tolist(u.value(thermal,"degF")))
        """, stress=True).splitlines()
        expected = [cm / 100 + metres, cm / 100 * metres, cm / 100 / seconds,
                    celsius * 9 / 5 + 32]
        for i, (line, ref) in enumerate(zip(lines, expected, strict=True)):
            self.close(f"vectorized unit operation {i + 1} max abs", parsed(line), ref)

    def test_unit_matmul_and_axis_sums_of_strided_arrays(self):
        rng = np.random.default_rng(1645)
        speed = rng.normal(size=(7, 3))
        time = rng.uniform(0.1, 20, size=(5, 7))
        lines = self.run_code(f"""
        tol numc as nc
        tol units as u
        esta x = u.quantity(nc.permute({tensor(speed)},[1,0]),"km/h")
        esta y = u.quantity(nc.permute({tensor(time)},[1,0]),"min")
        esta travel = u.matmul(x,y)
        collect()
        pedo(u.dimension(travel))
        pedo(nc.tolist(u.value(travel,"m")))
        pedo(nc.tolist(u.value(u.sum(travel,0),"m")))
        pedo(nc.tolist(u.value(u.sum(travel,-1),"m")))
        pedo(u.value(u.sum(travel),"m"))
        """, stress=True).splitlines()
        expected = (speed.T * (1000 / 3600)) @ (time.T * 60)
        self.assertEqual(ast.literal_eval(lines[0]), [1, 0, 0, 0, 0, 0, 0])
        self.close("strided quantity matmul max abs", parsed(lines[1]), expected)
        self.close("quantity sum axis 0 max abs", parsed(lines[2]), expected.sum(axis=0))
        self.close("quantity sum axis -1 max abs", parsed(lines[3]), expected.sum(axis=-1))
        self.close("quantity sum all axes max abs", float(lines[4]), expected.sum())


if __name__ == "__main__":
    unittest.main()
