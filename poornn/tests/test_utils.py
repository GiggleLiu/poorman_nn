'''
Tests for MPS and MPO
'''
from numpy import *
from numpy.linalg import norm,svd
from copy import deepcopy
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from scipy import sparse as sps
import pdb,time

from ..utils import *

def test_fsign():
    def npsign(x):
        return x/maximum(1e-15,abs(x))
    for dtype in ['float32','float64','complex128','complex64']:
        for order in ['C','F']:
            print('Testing fsign dtype = %s, order = %s'%(dtype, order))
            x=typed_randn(dtype,(300,300))
            x=asarray(x,order=order)
            t0=time.time()
            s1 = npsign(x)
            t1=time.time()
            s2 = fsign(x)
            t2=time.time()
            print('Elapse np=%s, f=%s'%(t1-t0,t2-t1))
            assert_allclose(s1,s2)

def run_all():
    test_fsign()

if __name__=='__main__':
    run_all()
