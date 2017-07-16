from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
import sys,pdb,time

from linears import *

def test_tensor1D():
    W=arange(2**2).reshape([2]*2)
    layer=L_Tensor(zeros([2,2]),einsum_tokens=['ij','jk','ik'])
    x=array([[0,1],[1j,2]])
    layer.set_variables(W.ravel())
    assert_allclose(layer.get_variables(),W.ravel())
    y=layer.forward(x)
    dy=array([[1,2j],[-2,0]])
    dvar,dx=layer.backward(x,y,dy)
    assert_allclose(y,[[2,3],[4,1j+6]])
    assert_allclose(dvar,[[-2j,0],[-3,2j]])
    assert_allclose(dx,[[2j,2+6j],[0,-4]])

    layer.set_variables(0)
    y=layer.forward(x)
    assert_allclose(y,zeros([2,2]))

def test_conv1D():
    W=arange(4).reshape([2,1,2])+0j
    layer=L_Conv(zeros([2,1,2],dtype='complex128'),strides=[1],num_strides=[3],einsum_tokens=['ij','jkf','ikf'])
    x=array([[0,1,2],[1j,2,-1j]])
    layer.set_variables(W.ravel())
    assert_allclose(layer.get_variables(),W.ravel())
    y=layer.forward(x)
    dy=array([[[1,2j],[-2,0],[0,0]],[[1,1],[1j,1j],[0,0]]])
    dvar,dx=layer.backward(x,y,dy)
    pdb.set_trace()
    assert_allclose(y,[[2,3],[4,1j+6]])
    assert_allclose(dvar,[[-2j,0],[-3,2j]])
    assert_allclose(dx,[[2j,2+6j],[0,-4]])

    layer.set_variables(0)
    y=layer.forward(x)
    assert_allclose(y,zeros([2,2]))

def test_all():
    test_tensor1D()
    test_conv1D()

test_all()
