import unittest
from test_arda import ArdaTests as _Base, BIN

class TensorTests(unittest.TestCase):
    run_code = _Base.run_code
    def check(self, code, expected=None, **kw):
        import textwrap
        return self.run_code('tol numc as nc\n'+textwrap.dedent(code).lstrip('\n'),expected,**kw)
    def test_rank_and_broadcast(self):
        self.check('''
        esta a = nc.reshape(nc.linspace(1,12,12), [2,2,3])
        pedo(nc.shape(a), a + nc.array([10,20,30]))
        pedo(nc.ones(2,1,3) + nc.ones(1,4,1))
        ''','[2, 2, 3] [[[11, 22, 33], [14, 25, 36]], [[17, 28, 39], [20, 31, 42]]]\n[[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]], [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]]',stress=True)
    def test_reshape_gc_alias(self):
        self.check('''
        caro create():
            esta x = nc.array([1,2,3,4])
            nan nc.reshape(x,[2,2])
        esta v = create()
        collect()
        nc.put(v,[1,0],42)
        pedo(v, nc.sum(v))
        ''','[[1, 2], [42, 4]] 49',stress=True)
    def test_readonly_views_share_storage_and_copy_is_writable(self):
        self.run_code("""
        tol numc as nc
        esta x = nc.array([1,2,3,4])
        esta r = nc.readonly(x)
        esta m = nc.reshape(r,[2,2])
        x[2] = 8
        x = mun
        r = mun
        collect()
        pedo(nc.tolist(m))
        esta c = nc.copy(m)
        nc.set(c,1,0,9)
        pedo(nc.get(c,1,0),nc.get(m,1,0))
        """, "[[1, 2], [8, 4]]\n9 8", stress=True)
        for code in (
            "esta r = nc.readonly(nc.ones(2))\nr[0] = 9",
            "esta r = nc.permute(nc.readonly(nc.ones(2,2)),[1,0])\nnc.set(r,0,0,9)",
            "esta r = nc.view(nc.readonly(nc.ones(2)),0,0,2,1)\nnc.put(r,[0],9)",
        ):
            self.run_code("tol numc as nc\n" + code, error="read-only", stress=True)

    def test_view_permute(self):
        self.check('''
        esta a = nc.array([[1,2,3],[4,5,6]])
        esta v = nc.permute(a,[1,0])
        esta w = nc.view(v,0,0,3,2)
        nc.put(w,[1,1],9)
        pedo(a, w, nc.contiguous(w), nc.copy(w))
        pedo(w + nc.array([10,20]), nc.sum(w))
        pedo(nc.matmul(w,nc.transpose(w)))
        ''','[[1, 2, 3], [4, 5, 9]] [[1, 4], [3, 9]] law [[1, 4], [3, 9]]\n[[11, 24], [13, 29]] 17\n[[17, 39], [39, 90]]',stress=True)
    def test_strided_vector(self):
        self.check('''
        esta a = nc.array([0,1,2,3,4,5])
        esta b = nc.view(a,0,0,6,2)
        b[1] = 10
        pedo(a, b, nc.dot(b,b), 2*b, nc.sqrt(b*b))
        ''','[0, 1, 10, 3, 4, 5] [0, 10, 4] 116 [0, 20, 8] [0, 10, 4]',stress=True)
    def test_zero_and_max_rank(self):
        self.check('''
        pedo(nc.shape(nc.zeros(0,3)+nc.ones(1,3)))
        esta x = nc.ones(1,1,1,1,1,1,1,2)
        pedo(nc.shape(x), nc.sum(x), nc.at(x,[0,0,0,0,0,0,0,-1]))
        ''','[0, 3]\n[1, 1, 1, 1, 1, 1, 1, 2] 2 1',stress=True)
    def test_nd_nested_copy_index(self):
        self.check('''
        esta a = nc.array([[[1,2]],[[3,4]]])
        esta b = a[1]
        nc.put(b,[0,0],90)
        pedo(a,b,nc.tolist(a))
        ''','[[[1, 2]], [[3, 4]]] [[90, 4]] [[[1, 2]], [[3, 4]]]',stress=True)
    def test_invalid_shapes(self):
        for s,err in [('nc.ones(2,3)+nc.ones(2,2)','shape mismatch'),('nc.reshape(nc.ones(4),[3,2])','reshape size mismatch'),('nc.permute(nc.ones(2,2),[0,0])','duplicate'),('nc.view(nc.ones(2),0,0,2,0)','invalid view'),('nc.at(nc.ones(2),[2])','out of bounds'),('nc.array([[[1]],[[2,3]]])','ragged'),('nc.reshape(nc.permute(nc.ones(2,3),[1,0]),[6])','contiguous')]:
            with self.subTest(s=s): self.check(s,error=err,stress=True)
    def test_view_accounting(self):
        self.check('''
        esta a = nc.ones(100000)
        collect()
        esta before = heap_bytes()
        esta v = nc.reshape(a,[100,1000])
        collect()
        pedo(heap_bytes()-before < 4000)
        ''','nae',stress=True)
    def test_broadcast_strided_outer(self):
        self.check('''
        esta x = nc.reshape(nc.linspace(1,24,24),[2,3,4])
        esta p = nc.permute(x,[2,0,1])
        esta y = p + nc.array([[10],[20]])
        pedo(nc.at(y,[3,1,2]), nc.sum(y))
        ''','44 660',stress=True)

class CalculusTests(unittest.TestCase):
    run_code = _Base.run_code
    def check(self,code,expected=None,**kw):
        import textwrap
        return self.run_code('tol numc as nc\ntol calc\ntol ode\n'+textwrap.dedent(code).lstrip('\n'),expected,**kw)
    def test_quadrature(self):
        self.check('''
        esta x = nc.linspace(0,1,101)
        pedo(abs(calc.simpson(x*x,0.01)-1/3)<1e-14)
        pedo(abs(calc.trapz(x*x,0.01)-1/3)<0.00002)
        caro f(x):
            nan x*x*x
        pedo(abs(ode.integrate(f,0,1,10)-0.25)<1e-14)
        ''','nae\nnae\nnae',stress=True)
    def test_fft_roundtrip_and_known_spectrum(self):
        self.check('''
        esta x = nc.array([1,0,-1,0])
        esta y = calc.fft(x)
        pedo(abs(nc.get(y,1,0)-2)<1e-12, abs(nc.get(y,3,0)-2)<1e-12)
        esta z = calc.ifft(y)
        pedo(nc.norm(nc.view(z,1,0,1,1) - nc.reshape(x,[4,1]))<1e-12)
        pedo(nc.norm(nc.view(z,1,1,2,1))<1e-12)
        ''','nae nae\nnae\nnae',stress=True)
    def test_fft_complex(self):
        self.check('''
        esta x = nc.array([[1,2],[3,4],[5,6],[7,8]])
        pedo(nc.norm(calc.ifft(calc.fft(x))-x)<1e-12)
        ''','nae',stress=True)
    def test_rk4_scalar_and_linear(self):
        self.check('''
        caro f(t,y):
            nan -y
        esta y = ode.rk4(f,0,1,0.01,100)
        esta z = calc.rk4_linear(nc.array([[-1]]),nc.array([1]),0.01,100)
        pedo(abs(y[-1]-0.3678794411714423)<1e-9)
        pedo(abs(nc.get(z,100,0)-y[-1])<1e-13)
        ''','nae\nnae',stress=True)
    def test_rk4_vector_oscillator(self):
        self.check('''
        caro f(t,y):
            nan nc.array([y[1],-y[0]])
        esta y = ode.rk4(f,0,nc.array([1,0]),0.01,100)
        pedo(nc.norm(y[-1]-nc.array([0.54030230586814,-0.8414709848079]))<1e-8)
        ''','nae',stress=True)
    def test_fft_reference_dft(self):
        import cmath, math, ast
        x=[0.125,-2,4,3.5,0,1,-1,2]
        expected=[sum(a*cmath.exp(-2j*math.pi*k*j/len(x)) for j,a in enumerate(x)) for k in range(len(x))]
        out=ast.literal_eval(self.check('pedo(calc.fft(nc.array('+repr(x)+')))').strip())
        for row,z in zip(out,expected): self.assertLess(abs(complex(*row)-z),1e-11)
    def test_rk4_order(self):
        self.check('''
        caro f(t,y):
            nan y
        esta coarse=ode.rk4(f,0,1,0.1,10)
        esta fine=ode.rk4(f,0,1,0.05,20)
        esta e=2.718281828459045
        esta ratio=abs(coarse[-1]-e)/abs(fine[-1]-e)
        pedo(ratio>14 and ratio<17)
        ''','nae',stress=True)
    def test_calc_errors(self):
        for s,err in [('calc.fft(nc.ones(3))','power of two'),('calc.ifft(nc.ones(4))','FFT expects'),('calc.simpson(nc.ones(4),1)','sample count'),('calc.rk4_linear(nc.ones(2,2),nc.ones(3),1,1)','shape mismatch'),('ode.integrate(lambda x: x,0,1,3)','even interval')]:
            with self.subTest(s=s): self.check(s,error=err,stress=True)
    def test_callback_exception(self):
        self.check('''
        caro f(x):
            throw "bad callback"
        try:
            ode.integrate(f,0,1)
        catch e:
            pedo(e)
        pedo(ode.integrate(lambda x: x,0,1))
        ''','bad callback\n0.5',stress=True)
    def test_lapack(self):
        self.check('''
        qui calc.backend() == "lapack":
            esta a = nc.array([[2,1],[1,2]])
            esta b = nc.array([3,3])
            pedo(nc.norm(calc.lapack_solve(a,b)-nc.ones(2))<1e-12)
            esta e = calc.eigh(a)
            pedo(nc.norm(e[0]-nc.array([1,3]))<1e-12)
            esta v = e[1]
            pedo(nc.norm(nc.matmul(a,v)-v*e[0])<1e-12)
        ai:
            try:
                calc.eigh(nc.ones(2,2))
            catch e:
                pedo("nae")
            pedo("nae", "nae")
        ''','nae\nnae nae' if self.run_code('tol calc\npedo(calc.backend())').strip()!='lapack' else 'nae\nnae\nnae',stress=True)

class ADTests(unittest.TestCase):
    run_code = _Base.run_code
    def check(self,code,expected=None,**kw):
        import textwrap
        return self.run_code('tol ad\ntol numc as nc\n'+textwrap.dedent(code).lstrip('\n'),expected,**kw)
    def test_shared_graph(self):
        self.check('''
        esta x = ad.variable(3)
        esta y = ad.variable(2)
        esta p = x*y
        esta q = p*p + p + x
        ad.backward(q)
        pedo(ad.value(q),ad.grad(x),ad.grad(y),ad.grad(p))
        ad.backward(q)
        pedo(ad.grad(x))
        ''','45 27 39 13\n27',stress=True)
    def test_chain_rule(self):
        self.check('''
        esta x = ad.variable(2)
        esta f = ad.log(ad.exp(x)) + ad.pow(x,3) + 1/x - ad.sqrt(x*x)
        ad.backward(f)
        pedo(abs(ad.grad(x)-11.75)<1e-12)
        esta z = ad.sin(x)*ad.cos(x)+ad.tanh(x)
        ad.backward(z)
        pedo(abs(ad.grad(x)-(-0.6536436208636119+0.0706508248531645))<1e-12)
        ''','nae\nnae',stress=True)
    def test_finite_difference_reference(self):
        import math
        def f(x,y): return math.sin(x*y)+x*x/(y+1)
        h=1e-5; x=1.3; y=0.7
        expected=[(f(x+h,y)-f(x-h,y))/(2*h),(f(x,y+h)-f(x,y-h))/(2*h)]
        out=self.check('''
        esta x=ad.variable(1.3)
        esta y=ad.variable(0.7)
        esta z=ad.sin(x*y)+x*x/(y+1)
        ad.backward(z)
        pedo(ad.grad(x),ad.grad(y))
        ''',stress=True)
        for actual,want in zip(map(float,out.split()),expected): self.assertLess(abs(actual-want),1e-8)
    def test_deep_graph(self):
        self.check('''
        esta x = ad.variable(1)
        esta y = x
        esta i = 0
        tenna i < 10000:
            y = y + 1
            i = i + 1
        ad.backward(y)
        pedo(ad.value(y),ad.grad(x))
        ''','10001 1')
    def test_optimizer(self):
        self.check('''
        esta p = nc.array([3,-2])
        esta g = nc.array([2,-4])
        pedo(ad.sgd(p,g,0.5))
        esta r = ad.adam(p,g,nc.zeros(2),nc.zeros(2),1,0.1)
        pedo(nc.norm(r[0]-nc.array([2.9,-1.9]))<1e-8)
        pedo(nc.norm(r[1]-nc.array([0.2,-0.4]))<1e-12)
        ''','[2, 0]\nnae\nnae',stress=True)
    def test_ad_gc(self):
        self.check('''
        caro f():
            esta x = ad.variable(3)
            nan x*x
        esta y = f()
        collect()
        ad.backward(y)
        pedo(ad.value(y))
        esta i = 0
        collect()
        esta before = heap_bytes()
        tenna i < 500:
            esta x = ad.variable(i)
            esta z = x*x + 2*x
            ad.backward(z)
            i = i + 1
        collect()
        pedo(heap_bytes()-before<10000)
        ''','9\nnae',stress=True)
    def test_ad_errors(self):
        for s,e in [('ad.log(ad.variable(0))','log domain'),('ad.variable(1)/0','division by zero'),('ad.sqrt(ad.variable(0))','sqrt derivative'),('ad.backward(1)','expected autodiff node'),('ad.adam(nc.ones(1),nc.ones(1),nc.zeros(1),nc.zeros(1),0,0.1)','start at 1')]:
            with self.subTest(s=s): self.check(s,error=e,stress=True)
    def test_learning_example(self):
        from pathlib import Path
        import subprocess
        root=Path(__file__).resolve().parents[1]
        p=subprocess.run([str(BIN),str(root/'examples/learning.arda')],capture_output=True,text=True,timeout=20)
        self.assertEqual(p.returncode,0,p.stderr)
        vals=[float(l.split(':')[1]) for l in p.stdout.strip().splitlines()]
        self.assertLess(abs(vals[0]-2),.005)
        self.assertLess(abs(vals[1]-1),.005)
        self.assertLess(vals[2],1e-5)

del _Base
