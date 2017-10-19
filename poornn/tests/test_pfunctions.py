from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
import pdb,time

from ..pfunctions import *
from ..checks import check_numdiff
from ..utils import typed_randn

def test_prelu():
    for var_mask in [True,False]:
        func=PReLU(input_shape=(-1,2), itype='float64',leak=0.1, var_mask=var_mask)
        print('Test numdiff for \n%s.'%func)
        assert_(all(check_numdiff(func)))

def test_mul():
    for var_mask in [True,False]:
        func=PMul(input_shape=(-1,2), itype='float64',c=0.5, var_mask=var_mask)
        print('Test numdiff for \n%s.'%func)
        assert_(all(check_numdiff(func)))

def test_poly():
    kernels = ['polynomial','chebyshev', 'legendre','laguerre','hermite','hermiteE']
    for factorial_rescale in [True,False]:
        for kernel in kernels:
            for var_mask in [[True]*3,[False]*3]:
                func=Poly(input_shape=(-1,2), itype='complex128',params=[3.,2,2+1j], kernel=kernel, factorial_rescale = factorial_rescale)
                print('Test numdiff for \n%s.'%func)
                assert_(all(check_numdiff(func)))

def test_mobiusgeogaussian():
    for var_mask in [[True]*3,[False]*3]:
        func1=Mobius(input_shape=(-1,2), itype='complex128',params=[1,2j,1e10], var_mask=var_mask[:3])
        func2 = Georgiou1992(input_shape=(-1,2), itype='complex128',params=[1,2.], var_mask=var_mask[:2])
        func3 = Gaussian(input_shape=(-1,2), itype='complex128',params=[1j,2.], var_mask=var_mask[:2])
        for func in [func1, func2, func3]:
            print('Test numdiff for \n%s.'%func)
            assert_(all(check_numdiff(func, eta_w=1e-2 if func is func3 else None)))

def test_all():
    random.seed(2)
    test_poly()
    test_mul()
    test_prelu()
    test_mobiusgeogaussian()

if __name__=='__main__':
    test_all()
