"""Independent NumPy/SciPy comparisons. Run with make test-reference."""
import ast
import json
import subprocess
import tempfile
import pathlib
import unittest
import numpy as np
from scipy import linalg, fft, integrate, sparse, optimize
from scipy.sparse.linalg import spsolve
from test_arda import BIN


def literal(x):
    return json.dumps(np.asarray(x).tolist(), allow_nan=False)


def execute(code):
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / 'ref.arda'
        # Convert tensors to lists before printing to avoid abbreviated arrays.
        code = code.replace('pedo(', 'emit(')
        path.write_text('tol numc as nc\ntol calc as ca\ntol high\ntol ode\ntol opt\ncaro emit(x):\n    qui type(x) == "array":\n        pedo(nc.tolist(x))\n    ai:\n        pedo(x)\n' + code + '\n')
        p = subprocess.run([str(BIN), '--gc-stress', str(path)], text=True,
                           capture_output=True, timeout=30)
        if p.returncode:
            raise AssertionError(p.stderr)
        return [ast.literal_eval(line.replace('nae', 'True').replace('law', 'False'))
                for line in p.stdout.splitlines()]


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1301)

    def compare(self, name, actual, expected, *, rtol=1e-10, atol=1e-12):
        a, b = np.asarray(actual), np.asarray(expected)
        self.assertEqual(a.shape, b.shape)
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=name)
        return float(np.max(np.abs(a-b))) if a.size else 0.0

    def test_solve_seeded_and_pivot(self):
        errors = []
        for n in (2, 4, 8, 16, 32):
            for _ in range(4):
                a = self.rng.normal(size=(n,n)) + n*np.eye(n)
                b = self.rng.normal(size=n)
                x, = execute(f'pedo(nc.solve(nc.array({literal(a)}),nc.array({literal(b)})))\n')
                errors.append(self.compare('solve', x, linalg.solve(a,b)))
                self.assertLess(np.linalg.norm(a@x-b)/np.linalg.norm(b), 1e-12)
        x, = execute('pedo(nc.solve(nc.array([[0,2],[1,3]]),nc.array([4,7])))\n')
        self.compare('pivot', x, [1,2])
        print(f'\nsolve: 21 systems; max abs error {max(errors):.3g}')

    def test_matmul_views_and_dot(self):
        errors = []
        for m,k,n in [(1,3,2),(3,7,5),(16,17,15),(65,67,69),(4,0,3)]:
            if k == 0:
                a,b = np.zeros((m,k)),np.zeros((k,n))
                x, = execute(f'pedo(nc.matmul(nc.zeros({m},0),nc.zeros(0,{n})))\n')
            else:
                a,b = self.rng.normal(size=(m,k)),self.rng.normal(size=(k,n))
                x, = execute(f'esta a=nc.permute(nc.array({literal(a.T)}),[1,0])\n'
                             f'pedo(nc.matmul(a,nc.array({literal(b)})))\n')
            errors.append(self.compare('matmul', x, a@b))
        a,b = self.rng.normal(size=1001),self.rng.normal(size=1001)
        x, = execute(f'pedo(nc.dot(nc.array({literal(a)}),nc.array({literal(b)})))\n')
        e = self.compare('dot', x, a@b)
        print(f'\nmatmul: 5 shapes; max abs error {max(errors):.3g}; dot {e:.3g}')

    def test_elementwise_reductions_and_broadcast(self):
        a=self.rng.normal(size=(3,4,5)); b=self.rng.normal(size=3)
        # Transposed non-contiguous tensor: shape (5,4,3).
        p=a.transpose(2,1,0)
        results=execute(f'esta a=nc.permute(nc.array({literal(a)}),[2,1,0])\n'
                        f'pedo(nc.cos(a))\npedo(nc.min(a))\npedo(nc.max(a))\n'
                        'pedo(nc.mean(a))\npedo(nc.var(a))\npedo(nc.norm(a))\n'
                        f'pedo(a+nc.array({literal(b)}))\ntol tensor as t\npedo(t.tolist(t.sum(a,1)))\n')
        expected=[np.cos(p),p.min(),p.max(),p.mean(),p.var(),np.linalg.norm(p),p+b,p.sum(axis=1)]
        errors=[self.compare(name,x,y) for name,x,y in zip(('cos','min','max','mean','var','norm','broadcast','sum_axis'),results,expected)]
        print(f'\n8 elementwise/reduction results; max abs error {max(errors):.3g}')

    def test_fft_and_inverse(self):
        errors=[]
        for n in (1,8,32,128,1024):
            a=self.rng.normal(size=n)
            x,y=execute(f'esta a=nc.array({literal(a)})\nesta f=ca.fft(a)\npedo(f)\npedo(ca.ifft(f))\n')
            z=fft.fft(a)
            errors.append(self.compare('fft',x,np.column_stack((z.real,z.imag)),atol=1e-10))
            errors.append(self.compare('ifft',y,np.column_stack((a,np.zeros(n))),atol=1e-10))
        print(f'\nFFT/IFFT: 10 comparisons; max abs error {max(errors):.3g}')

    def test_cg_and_lstsq(self):
        a=self.rng.normal(size=(12,12)); a=a.T@a+2*np.eye(12); b=self.rng.normal(size=12)
        r=execute(f'esta r=high.cg(nc.array({literal(a)}),nc.array({literal(b)}),1e-11,100)\npedo(r["solution"])\npedo(r["converged"])\n')
        self.assertTrue(r[1]); cg=self.compare('CG',r[0],linalg.solve(a,b),atol=1e-10)
        a=self.rng.normal(size=(30,8)); b=self.rng.normal(size=30)
        r=execute(f'esta r=high.lstsq(nc.array({literal(a)}),nc.array({literal(b)}),1e-11,100)\npedo(r["solution"])\npedo(r["converged"])\n')
        self.assertTrue(r[1]); ls=self.compare('CGLS',r[0],linalg.lstsq(a,b)[0],atol=1e-10)
        print(f'\nCG max abs error {cg:.3g}; CGLS {ls:.3g}')

    def test_poisson_independent_sparse_operator(self):
        errors=[]
        for shape in [(5,), (3,4), (2,3,4), (2,2,3,4), (2,2,2,2,2,2,2,2)]:
            n=int(np.prod(shape)); matrix=sparse.csr_matrix((n,n))
            for axis,m in enumerate(shape):
                t=sparse.diags([2*np.ones(m),-np.ones(m-1),-np.ones(m-1)],[0,-1,1],format='csr')
                left=int(np.prod(shape[:axis])); right=int(np.prod(shape[axis+1:]))
                matrix+=sparse.kron(sparse.eye(left),sparse.kron(t,sparse.eye(right)),format='csr')
            b=self.rng.normal(size=shape)
            x,stencil,ok=execute(f'esta b=nc.array({literal(b)})\nesta r=high.poisson(b,1e-11,500)\npedo(r["solution"])\npedo(high.laplace(b))\npedo(r["converged"])\n')
            self.assertTrue(ok)
            expected=spsolve(matrix,b.ravel()).reshape(shape)
            errors.append(self.compare('poisson',x,expected,atol=1e-10))
            self.compare('laplace',stencil,(matrix@b.ravel()).reshape(shape))
        print(f'\nPoisson: 5 independent sparse systems, up to 8D; max abs error {max(errors):.3g}')

    def test_rk45_and_quadrature(self):
        rates=np.array([0.5,1.,2.,4.])
        x,=execute(f'esta rate=nc.array({literal(rates)})\ncaro f(t,y):\n    nan 0-rate*y\n'
                   'esta r=ode.rk45(f,0,nc.ones(4),2,0.5,1e-9,1e-11)\npedo(r["solution"])\n')
        expected=integrate.solve_ivp(lambda t,y:-rates*y,(0,2),np.ones(4),rtol=1e-12,atol=1e-14).y[:,-1]
        err=self.compare('rk45',x,expected,rtol=1e-7,atol=1e-10)
        y=np.sin(np.linspace(0,np.pi,101)); h=np.pi/100
        a,b=execute(f'esta x=nc.array({literal(y)})\npedo(ca.trapz(x,{h}))\npedo(ca.simpson(x,{h}))\n')
        self.compare('trapz',a,integrate.trapezoid(y,dx=h)); self.compare('simpson',b,integrate.simpson(y,dx=h))
        print(f'\nRK45 max abs error {err:.3g}; trapezoid {a:.15g}; Simpson {b:.15g}')

    def test_lbfgs(self):
        x,ok=execute('caro f(x):\n    esta a=x[0]\n    esta b=x[1]\n    esta d=b-a*a\n'
                     '    nan {"value": (1-a)*(1-a)+100*d*d, "gradient": nc.array([2*(a-1)-400*a*d,200*d])}\n'
                     'esta r=opt.lbfgs(f,nc.array([-1.2,1]),1e-7,1000,10)\npedo(r["solution"])\npedo(r["converged"])\n')
        self.assertTrue(ok)
        ref=optimize.minimize(optimize.rosen,[-1.2,1],jac=optimize.rosen_der,method='BFGS',options={'gtol':1e-9})
        e=self.compare('lbfgs',x,ref.x,atol=1e-6,rtol=1e-6)
        print(f'\nL-BFGS Rosenbrock max abs error {e:.3g}')
