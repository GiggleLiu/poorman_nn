'''
Tests for MPS and MPO
'''
from numpy import *
from numpy.testing import dec, assert_, assert_raises,\
    assert_almost_equal, assert_allclose
from scipy import sparse as sps
import pdb
import time

from ..checks import check_numdiff
from ..utils import typed_randn
from ..linears import Linear
from ..nets import ANN
from .. import functions

random.seed(2)


def test_hash():
    num_batch = 2
    dim_in = 30
    dtype = 'complex128'

    input_shape = (-1, dim_in)
    x = asfortranarray(typed_randn(dtype, [num_batch, dim_in]))
    weight = asfortranarray(typed_randn(dtype, [dim_in, dim_in]))
    bias = typed_randn(dtype, [dim_in])
    ll = Linear(input_shape=input_shape, itype=dtype, weight=weight, bias=bias)
    ann = ANN(layers=[ll])

    d = {}
    d[ann] = 2

    assert_(d[ann]==2)
    N = 10000
    t0 = time.time()
    for i in range(N):
        ann.__hash__()
    t1 = time.time()
    for i in range(N):
        ann.uuid
    t2 = time.time()
    print('%d run, hash time %.4f, uuid time %.4f'%(N, t1-t0, t2-t1))

if __name__ == '__main__':
    test_hash()
