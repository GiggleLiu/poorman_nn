from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
import pdb,time

from ..layers import *
from ..checks import check_numdiff
from ..utils import typed_randn

def test_poly():
    kernels = ['polynomial','chebyshev', 'legendre','laguerre','hermite','hermiteE']
    for factorial_rescale in [True,False]:
        for kernel in kernels:
            func=Poly(input_shape=(-1,2), itype='complex128',params=[3.,2,2+1j], kernel=kernel, factorial_rescale = factorial_rescale)
            print('Test numdiff for %s.'%func)
            assert_(all(check_numdiff(func)))

def test_all():
    random.seed(3)
    test_poly()

if __name__=='__main__':
    test_all()
