'''
Tests for MPS and MPO
'''
from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from scipy import sparse as sps
import pdb,time

from ..checks import check_numdiff
from ..utils import typed_randn
from ..linears import Linear
from ..nets import ParallelNN
from .. import functions

random.seed(2)

def test_pa():
    num_batch = 2
    dim_in = 30
    dtype = 'complex128'

    input_shape = (-1, dim_in)
    output_shape = (-1, -1, dim_in)
    pnet = ParallelNN(input_shape, output_shape, dtype=dtype, otype=dtype, axis=1)
    x=asfortranarray(typed_randn(dtype, [num_batch,dim_in]))
    weight=asfortranarray(typed_randn(dtype, [dim_in, dim_in]))
    bias=typed_randn(dtype, [dim_in])
    pnet.add_layer(Linear, weight=weight, bias=bias)
    pnet.add_layer(functions.Power, order=2)
    pnet.add_layer(functions.Reshape)
    y = pnet.forward(x)
    ll = pnet.layers[0]
    assert_allclose(y[:,0,:], x.dot(ll.weight.T)+ll.bias)
    assert_allclose(y[:,1,:], x**2)
    assert_allclose(y[:,2,:], x)
    print("Testing numdiff for %s"%pnet)
    assert_(all(check_numdiff(pnet, num_check=100)))

if __name__ == '__main__':
    test_pa()
