import unittest
import test_arda

class HighDimTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code
    def check(self, code, expected=None, **kw):
        import textwrap
        return self.run_code('tol numc as nc\ntol high\ntol linop as l\ntol solver as s\ntol tensor as t\ntol ode\ntol opt\n'+textwrap.dedent(code).lstrip('\n'), expected, **kw)
    def test_dense_cg(self):
        self.check('''
        esta A = nc.array([[4,1],[1,3]])
        esta r = s.solve("cg",A,nc.array([1,2]),{"tol":1e-12,"maxiter":20,"preconditioner":"jacobi"})
        pedo(nc.norm(r["solution"]-nc.array([1/11,7/11]))<1e-12,r["iterations"],r["converged"])
        ''','nae 2 nae',stress=True)
    def test_cg_invalid(self):
        for a,err in [('[[1,2],[0,1]]','symmetric'),('[[0,0],[0,1]]','positive diagonal')]:
            with self.subTest(a=a):
                self.check(f's.solve("cg",nc.array({a}),nc.array([1,-1]))',error=err,stress=True)
        self.check('''
        esta r = s.solve("cg",nc.array([[1,2],[2,1]]),nc.array([1,-1]))
        pedo(r["reason"],r["breakdown"])
        ''','nonpositive_curvature nae',stress=True)
    def test_cg_zero_and_limit(self):
        self.check('''
        esta A=nc.array([[4,1],[1,3]])
        esta z=s.solve("cg",A,nc.zeros(2),{"tol":1e-9,"maxiter":5})
        esta r=s.solve("cg",A,nc.array([1,2]),{"tol":1e-12,"maxiter":1})
        pedo(z["iterations"],z["residual"],z["converged"],r["iterations"],r["converged"])
        ''','0 0 nae 1 law',stress=True)
    def test_stencil(self):
        self.check('''
        pedo(l.apply(l.poisson([3]),nc.ones(3)))
        pedo(nc.reshape(l.apply(l.poisson([2,3]),nc.ones(6)),[2,3]))
        pedo(nc.reshape(l.apply(l.poisson([1,1,1]),nc.ones(1)),[1,1,1]))
        ''','[1, 0, 1]\n[[2, 1, 2], [2, 1, 2]]\n[[[6]]]',stress=True)
    def test_poisson_nd(self):
        self.check('''
        esta shape=[2,3,4,5]
        esta exact=nc.linspace(-1,1,120)
        esta op=l.poisson(shape,[1,2,3,4])
        esta b=l.apply(op,exact)
        esta r=s.solve("cg",op,b,{"tol":1e-11,"maxiter":200,"preconditioner":"block_jacobi"})
        pedo(r["converged"],nc.shape(nc.reshape(r["solution"],shape)),nc.norm(r["solution"]-exact)<1e-9,s.verify(op,b,r["solution"])["residual"]<1e-11)
        esta op8=l.poisson([1,2,1,2,1,2,2,1])
        esta x=nc.ones(16)
        esta r=s.solve("cg",op8,l.apply(op8,x),{"tol":1e-11,"maxiter":100,"preconditioner":"block_jacobi"})
        pedo(r["converged"],nc.norm(r["solution"]-x)<1e-10)
        try:
            s.multigrid(op8,x)
        catch e:
            pedo(e["code"])
        ''','nae [2, 3, 4, 5] nae nae\nnae nae\nshape',stress=True)
    def test_least_squares(self):
        self.check('''
        esta A=nc.array([[1,0],[1,1],[1,2]])
        esta r=s.solve("lsqr",A,nc.array([1,4,4]),{"tol":1e-11,"maxiter":30})
        pedo(r["converged"],nc.norm(r["solution"]-nc.array([1.5,1.5]))<1e-9)
        esta u=s.solve("lsqr",nc.array([[1,1]]),nc.array([2]),{"tol":1e-10,"maxiter":20})
        pedo(u["converged"],nc.norm(u["solution"]-nc.ones(2))<1e-10)
        ''','nae nae\nnae nae',stress=True)
    def test_fused_and_axis(self):
        self.check('''
        esta x=nc.reshape(nc.linspace(1,12,12),[2,2,3])
        esta y=nc.permute(x,[2,0,1])
        pedo(t.tolist(t.sum(y,1)),high.inner(y,y),t.tolist(t.sum(nc.zeros(2,0,3),1)))
        pedo(high.axpy(nc.ones(2),nc.array([3,4]),2),t.sum(nc.ones(5),0))
        ''','[[8, 14], [10, 16], [12, 18]] 650 [[0, 0, 0], [0, 0, 0]]\n[5, 6] 5',stress=True)
    def test_rk45_scalar_and_reverse(self):
        self.check('''
        caro f(t,y):
            nan 0-y
        esta r=ode.rk45(f,0,1,1,1,1e-9,1e-11)
        esta z=ode.rk45(f,1,r["solution"],0,0.5,1e-9,1e-11)
        pedo(abs(r["solution"]-0.367879441171442)<1e-9,r["rejected_steps"]>0,abs(z["solution"]-1)<1e-8)
        ''','nae nae nae',stress=True)
    def test_rk45_tensor(self):
        self.check('''
        esta rate=nc.reshape(nc.linspace(1,3,24),[2,3,4])
        caro f(t,y):
            nan 0-rate*y
        esta r=ode.rk45(f,0,nc.ones(2,3,4),1,0.7,1e-8,1e-10)
        pedo(nc.shape(r["solution"]),nc.norm(r["solution"]-nc.exp(0-rate))<1e-7)
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
            nan {"value": (1-a)*(1-a)+100*d*d, "gradient": nc.array([2*(a-1)-400*a*d,200*d])}
        esta r=opt.lbfgs(f,nc.array([-1.2,1]),1e-6,1000,10)
        pedo(r["converged"],nc.norm(r["solution"]-nc.ones(2))<1e-5)
        ''','nae nae',stress=True)
    def test_lbfgs_tensor(self):
        self.check('''
        esta goal=nc.ones(2,3,4)
        caro f(x):
            esta d=x-goal
            nan {"value": high.inner(d,d)/2, "gradient": d}
        esta r=opt.lbfgs(f,nc.zeros(2,3,4))
        pedo(r["converged"],r["iterations"],nc.norm(r["solution"]-goal))
        ''','nae 1 0',stress=True)
    def test_odd_matmul(self):
        self.check('''
        esta a=nc.ones(3,35)
        esta b=nc.ones(35,37)
        esta c=nc.matmul(a,b)
        pedo(nc.shape(c),nc.min(c),nc.max(c))
        ''','[3, 37] 35 35',stress=True)
