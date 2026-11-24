import unittest
import test_arda

class HighDimTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code
    def check(self, code, expected=None, **kw):
        import textwrap
        return self.run_code('tol numc as nc\ntol high\ntol ode\ntol opt\n'+textwrap.dedent(code).lstrip('\n'), expected, **kw)
    def test_dense_cg(self):
        self.check('''
        esta A = nc.array([[4,1],[1,3]])
        esta r = high.cg(A,nc.array([1,2]),0.000000000001,20)
        pedo(nc.norm(r[0]-nc.array([1/11,7/11]))<1e-12,r[1],r[3])
        ''','nae 2 nae',stress=True)
    def test_cg_invalid(self):
        for a,err in [('[[1,2],[0,1]]','symmetric'),('[[1,2],[2,1]]','nonpositive curvature'),('[[0,0],[0,1]]','positive diagonal')]:
            with self.subTest(a=a):
                self.check(f'high.cg(nc.array({a}),nc.array([1,-1]),1e-8,10)',error=err,stress=True)
    def test_cg_zero_and_limit(self):
        self.check('''
        esta A=nc.array([[4,1],[1,3]])
        esta z=high.cg(A,nc.zeros(2),1e-9,5)
        esta r=high.cg(A,nc.array([1,2]),1e-12,1)
        pedo(z[1],z[2],z[3],r[1],r[3])
        ''','0 0 nae 1 law',stress=True)
    def test_stencil(self):
        self.check('''
        pedo(high.laplace(nc.ones(3)))
        pedo(high.laplace(nc.ones(2,3)))
        pedo(high.laplace(nc.ones(1,1,1)))
        ''','[1, 0, 1]\n[[2, 1, 2], [2, 1, 2]]\n[[[6]]]',stress=True)
    def test_poisson_nd(self):
        self.check('''
        esta exact=nc.reshape(nc.linspace(-1,1,120),[2,3,4,5])
        esta b=high.laplace(exact)
        esta r=high.poisson(b,1e-11,200)
        pedo(r[3],nc.shape(r[0]),nc.norm(r[0]-exact)<1e-9,nc.norm(high.laplace(r[0])-b)/nc.norm(b)<1e-11)
        esta x=nc.ones(1,2,1,2,1,2,1,2)
        esta r=high.poisson(high.laplace(x),1e-11,100)
        pedo(r[3],nc.norm(r[0]-x)<1e-10)
        ''','nae [2, 3, 4, 5] nae nae\nnae nae',stress=True)
    def test_cgls(self):
        self.check('''
        esta A=nc.array([[1,0],[1,1],[1,2]])
        esta r=high.lstsq(A,nc.array([1,4,4]),1e-11,30)
        pedo(r[3],nc.norm(r[0]-nc.array([1.5,1.5]))<1e-9)
        esta u=high.lstsq(nc.array([[1,1]]),nc.array([2]),1e-10,20)
        pedo(u[3],nc.norm(u[0]-nc.ones(2))<1e-10)
        ''','nae nae\nnae nae',stress=True)
    def test_fused_and_axis(self):
        self.check('''
        esta x=nc.reshape(nc.linspace(1,12,12),[2,2,3])
        esta y=nc.permute(x,[2,0,1])
        pedo(high.sum_axis(y,1),high.inner(y,y),high.sum_axis(nc.zeros(2,0,3),1))
        pedo(high.axpy(nc.ones(2),nc.array([3,4]),2),high.sum_axis(nc.ones(5),0))
        ''','[[8, 14], [10, 16], [12, 18]] 650 [[0, 0, 0], [0, 0, 0]]\n[5, 6] 5',stress=True)
    def test_rk45_scalar_and_reverse(self):
        self.check('''
        caro f(t,y):
            nan 0-y
        esta r=ode.rk45(f,0,1,1,1,1e-9,1e-11)
        esta z=ode.rk45(f,1,r[0],0,0.5,1e-9,1e-11)
        pedo(abs(r[0]-0.367879441171442)<1e-9,r[2]>0,abs(z[0]-1)<1e-8)
        ''','nae nae nae',stress=True)
    def test_rk45_tensor(self):
        self.check('''
        esta rate=nc.reshape(nc.linspace(1,3,24),[2,3,4])
        caro f(t,y):
            nan 0-rate*y
        esta r=ode.rk45(f,0,nc.ones(2,3,4),1,0.7,1e-8,1e-10)
        pedo(nc.shape(r[0]),nc.norm(r[0]-nc.exp(0-rate))<1e-7)
        ''','[2, 3, 4] nae',stress=True)
    def test_rk45_limit(self):
        self.check('''
        caro f(t,y):
            nan y
        ode.rk45(f,0,1,10,1,1e-10,1e-12,1)
        ''',error='step limit',stress=True)
    def test_lbfgs_rosenbrock(self):
        self.check('''
        caro f(x):
            esta a=x[0]
            esta b=x[1]
            esta d=b-a*a
            nan [(1-a)*(1-a)+100*d*d,nc.array([2*(a-1)-400*a*d,200*d])]
        esta r=opt.lbfgs(f,nc.array([-1.2,1]),1e-6,1000,10)
        pedo(r[3],nc.norm(r[0]-nc.ones(2))<1e-5)
        ''','nae nae',stress=True)
    def test_lbfgs_tensor(self):
        self.check('''
        esta goal=nc.ones(2,3,4)
        caro f(x):
            esta d=x-goal
            nan [high.inner(d,d)/2,d]
        esta r=opt.lbfgs(f,nc.zeros(2,3,4))
        pedo(r[3],r[1],nc.norm(r[0]-goal))
        ''','nae 1 0',stress=True)
    def test_odd_matmul(self):
        self.check('''
        esta a=nc.ones(3,35)
        esta b=nc.ones(35,37)
        esta c=nc.matmul(a,b)
        pedo(nc.shape(c),nc.min(c),nc.max(c))
        ''','[3, 37] 35 35',stress=True)
