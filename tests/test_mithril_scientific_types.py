"""Portable runtime contracts for the Mithril low-rank and scientific APIs."""
import unittest

import test_arda


class ScientificTypesTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_contract_uses_metadata_and_preserves_identity(self):
        self.run_code("""
        tol compute as c
        tol lowrank as lr
        tol numc as nc
        tol tensor as t
        esta input = t.array([1,2,3,4],"int32",[2,2])
        t.save("data.atn",input)
        esta mapped = t.mmap("data.atn")
        esta factor = nc.ones(2000,1)
        esta compact = lr.cp([factor,factor,factor,factor])
        collect()
        esta baseline = c.memory_stats()["tensor_heap_bytes"]
        c.memory_budget(baseline)
        pedo(t.contract(mapped,[2,mun],"int32")==mapped)
        pedo(t.contract(compact,[mun,2000,2000,2000])==compact)
        pedo(c.memory_stats()["tensor_heap_bytes"]==baseline)
        """, "nae\nnae\nnae", stress=True)
        code = 'tol tensor as t\nesta x=t.zeros([2,2],"int32")\n'
        for expr, error in (
            ('t.contract(x,[2])', "rank mismatch"),
            ('t.contract(x,[2,2],"float64")', "dtype mismatch"),
            ('t.contract(x,[2,4294967298])', "dimension mismatch"),
            ('t.contract(x,[2,-1])', "nonnegative integers"),
        ):
            with self.subTest(expr=expr):
                self.run_code(code + expr, error=error, stress=True)

    def test_tensor_spec_checks_64_bit_size_without_payload(self):
        self.run_code("""
        tol compute as c
        tol tensor as t
        c.memory_budget(0)
        caro show(shape, dtype):
            esta s = t.spec(shape,dtype)
            pedo(s["elements"],s["bytes"],s["materializable"])
        show([3000000000,2],"float64")
        show([9223372036854775807],"uint8")
        show([9223372036854775807],"float64")
        show([9223372036854775807,2],"float64")
        show([9223372036854775807,2,0],"float64")
        show([0,12],"float64")
        pedo(c.memory_stats()["tensor_heap_bytes"])
        """, "6000000000 48000000000 law\n9223372036854775807 9223372036854775807 law\n"
             "9223372036854775807 mun law\nmun mun law\n0 0 law\n0 0 nae\n0", stress=True)

    def test_quantity_snapshots_and_axis_sum_budget(self):
        self.run_code("""
        tol compute as c
        tol numc as nc
        tol units as u
        esta data = nc.array([[1,2,3],[4,5,6]])
        esta q = u.quantity(data,"m")
        nc.put(data,[0,0],99)
        esta copy = u.value(q)
        nc.put(copy,[0,0],99)
        data = mun
        copy = mun
        collect()
        esta baseline = c.memory_stats()["tensor_heap_bytes"]
        c.memory_budget(baseline + 24)
        esta result = u.sum(u.convert(q,"cm"),0)
        collect()
        pedo(c.memory_stats()["tensor_heap_bytes"] - baseline)
        c.memory_budget(mun)
        pedo(nc.tolist(u.value(result)),u.unit(result))
        """, "24\n[500, 700, 900] cm", stress=True)

    def test_quantities_reject_dimension_and_temperature_kind_mismatch(self):
        for expr, error in (
            ('u.quantity(5,"m")+u.quantity(3,"s")', "dimensions do not match"),
            ('u.quantity(5,"K")+u.quantity(3,"degC")', "absolute temperatures cannot be added"),
            ('u.quantity(5,"K")*2', "absolute temperature cannot be multiplied"),
            ('u.convert(u.quantity(5,"degF"),"deltaK")', "temperature kind"),
            ('u.quantity(5,"degC/s")', "offset temperature"),
            ('u.quantity(-274,"degC")', "zero kelvin"),
        ):
            with self.subTest(expr=expr):
                self.run_code("tol units as u\n" + expr, error=error, stress=True)

    def test_mathir_subscripts_preserve_operations_without_evaluating(self):
        source = """
        esta calls = 0
        caro step(i):
            calls = calls + 1
            nan i + 1
        caro f(A,i):
            nan A[step(i)][1] + 0
        caro loop(bounds):
            esta result = 0
            an i in range(bounds[0]):
                result = result + i
            nan result
        pedo(f.getTex(0))
        pedo(f.getTex(2))
        pedo(loop.getTex())
        pedo(calls)
        pedo(f([[1,2],[3,4]],0),calls)
        """
        lines = self.run_code(source, stress=True).splitlines()
        self.assertIn(r"_{\mathrm{step}\left(i\right)}", lines[0])
        self.assertIn("_{1}+0", lines[0])
        self.assertIn("bounds unknown", lines[1])
        self.assertEqual(lines[2:], ["mun", "0", "4 1"])
        self.run_code(source, "mun\nmun\nmun\n0\n4 1", flags=("--mathdoc=none",), stress=True)

    def test_failed_lowrank_materialization_releases_reserved_workspace(self):
        self.run_code("""
        tol compute as c
        tol lowrank as lr
        tol numc as nc
        esta core = nc.reshape(nc.array([1e308]),[1,1,1])
        esta train = lr.tt([core,core])
        an i in range(8):
            try:
                lr.materialize(train)
                pedo("unexpected")
            catch error:
                pedo("rejected")
            qui c.memory_stats()["workspace_reserved_bytes"] != 0:
                throw {"code": "user_error", "message": "retained workspace"}
            collect()
        pedo(lr.shape(train))
        """, "rejected\n" * 8 + "[1, 1]", stress=True)
