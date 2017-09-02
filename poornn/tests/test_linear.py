'''
Tests for MPS and MPO
'''
from numpy import *
from numpy.linalg import norm,svd
from copy import deepcopy
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from scipy import sparse as sps
import pdb,time
import torch.nn as nn
from torch import autograd
import torch

from ..linears import Linear, Apdot, SPLinear
from ..checks import check_numdiff
from ..utils import typed_randn

def test_linear():
    random.seed(2)
    num_batch=10
    nfi=1024
    nfo=512
    ts=random.randn(num_batch,nfi)
    ts=autograd.Variable(torch.Tensor(ts),requires_grad=True)
    #2 features, kernel size 3x3
    cv=nn.Linear(nfi,nfo)
    weight=cv.weight.data.numpy()
    sv=Linear((num_batch, nfi), 'float32', float32(weight),float32(cv.bias.data.numpy()))
    print("Testing forward for %s"%sv)
    xin_np=asfortranarray(ts.data.numpy())
    ntest=5
    t0=time.time()
    for i in range(ntest):
        y1=cv(ts)
    t1=time.time()
    for i in range(ntest):
        y2=sv.forward(xin_np)
    t2=time.time()
    print( "Elapse old = %s, new = %s"%(t1-t0,t2-t1))
    res1=y1.data.numpy()
    res2=y2
    assert_allclose(res1,res2,atol=1e-4)

    print( "Testing backward")
    dy=torch.randn(*y1.size())
    dy_np=asfortranarray(dy.numpy())
    dweight=zeros_like(sv.weight)
    dbias=zeros_like(sv.bias)
    dx=zeros_like(xin_np)

    t0=time.time()
    y1.backward(dy)
    t1=time.time()
    for i in range(ntest):
        dwb, dx=sv.backward([xin_np, y2], dy_np)
    t2=time.time()
    print( "Elapse old = %s, new = %s"%(t1-t0,(t2-t1)/ntest))

    #reshape back
    dx0=ts.grad.data.numpy()

    dweight, dbias = dwb[:sv.weight.size], dwb[sv.weight.size:]
    wfactor=dweight.mean()
    bfactor=dbias.mean()
    dweight=reshape(dweight,cv.weight.size(), order='F')/wfactor
    assert_allclose(dx0,dx,atol=2e-3)
    assert_allclose(cv.weight.grad.data.numpy()/wfactor,dweight,atol=2e-3)
    assert_allclose(cv.bias.grad.data.numpy()/bfactor,dbias/bfactor,atol=3e-3)
    assert_(all(check_numdiff(sv, xin_np)))

def test_linear_complex():
    num_batch=1
    dim_in=30
    dim_out=400
    xin_np=asfortranarray(typed_randn('complex128',[num_batch,dim_in]))
    weight=asfortranarray(typed_randn('complex128',[dim_out, dim_in]))
    bias=typed_randn('complex128',[dim_out])
    sv=Linear((num_batch, dim_in), 'complex128',weight, bias)
    print( "Testing numdiff for %s"%sv)
    assert_(all(check_numdiff(sv, num_check=100)))

    print('Testing var_mask')
    sv.var_mask=(False,False)
    assert_allclose(sv.get_variables(),zeros(0))
    sv=Linear((num_batch, dim_in), 'complex128',weight, bias, var_mask=(False,False))
    assert_allclose(sv.get_variables(),zeros(0))

def test_splinear():
    num_batch=2
    dim_in=30
    dim_out=400
    dtype='complex128'
    x=asfortranarray(typed_randn(dtype,[num_batch,dim_in]))
    weight=asfortranarray(typed_randn(dtype,[dim_out, dim_in]))
    bias=typed_randn(dtype,[dim_out])
    sv=Linear((num_batch, dim_in), dtype,weight, bias)
    sv2=SPLinear((num_batch, dim_in), dtype, sps.csc_matrix(weight), bias)

    print( "Testing forward for %s"%sv2)
    ntest=5
    t0=time.time()
    for i in range(ntest):
        y1=sv.forward(x)
    t1=time.time()
    for i in range(ntest):
        y2=sv2.forward(x)
    t2=time.time()
    print( "Elapse old = %s, new = %s"%(t1-t0,t2-t1))
    assert_allclose(y1,y2,atol=1e-4)

    print( "Testing backward")
    dy=typed_randn(dtype, y1.shape)
    t0=time.time()
    dwb0, dx0 = sv.backward([x, y1],dy)
    t1=time.time()
    dwb1, dx1=sv2.backward([x, y1], dy)
    mat0 = dwb0[:dim_in*dim_out].reshape([dim_out, dim_in], order='F')
    mat1 = sps.csr_matrix((dwb1[:-dim_out], sv2.weight.indices, sv2.weight.indptr)).toarray()
    t2=time.time()
    print( "Elapse old = %s, new = %s"%(t1-t0,t2-t1))
    assert_(all(check_numdiff(sv, num_check=100)))
    assert_(all(check_numdiff(sv2, num_check=100)))
    assert_allclose(dx0, dx1,atol=1e-4)
    assert_allclose(mat1, mat0,atol=1e-4)

    print( "Testing numdiff for %s"%sv)
    assert_(all(check_numdiff(sv2, num_check=100)))

def test_linear1():
    random.seed(2)
    nfi=1024
    nfo=512
    ts=random.randn(1,nfi)
    ts=autograd.Variable(torch.Tensor(ts),requires_grad=True)
    #2 features, kernel size 3x3
    cv=nn.Linear(nfi,nfo)
    weight=cv.weight.data.numpy()
    sv=Linear(input_shape=(nfi,), dtype='float32', weight=float32(weight),bias=float32(cv.bias.data.numpy()))
    print( "Testing forward for %s"%sv)
    xin_np=asfortranarray(ts.data.numpy())
    ntest=5
    t0=time.time()
    for i in range(ntest):
        y1=cv(ts)
    t1=time.time()
    for i in range(ntest):
        y2=sv.forward(xin_np[0])
    t2=time.time()
    print( "Elapse old = %s, new = %s"%(t1-t0,t2-t1))
    res1=y1.data.numpy()
    res2=y2
    assert_allclose(res1[0],res2,atol=1e-4)

    print( "Testing backward")
    dy=torch.randn(*y1.size())
    dy_np=asfortranarray(dy.numpy())
    dweight=zeros_like(sv.weight)
    dbias=zeros_like(sv.bias)
    dx=zeros_like(xin_np)

    t0=time.time()
    y1.backward(dy)
    t1=time.time()
    for i in range(ntest):
        dwb, dx=sv.backward([xin_np[0], y2], dy_np)
    t2=time.time()
    print( "Elapse old = %s, new = %s"%(t1-t0,(t2-t1)/ntest))

    #reshape back
    dx0=ts.grad.data.numpy()

    dweight, dbias = dwb[:sv.weight.size], dwb[sv.weight.size:]
    wfactor=dweight.mean()
    bfactor=dbias.mean()
    dweight=reshape(dweight,cv.weight.size(), order='F')/wfactor
    assert_allclose(dx0[0],dx,atol=2e-3)
    assert_allclose(cv.weight.grad.data.numpy()/wfactor,dweight,atol=2e-3)
    assert_allclose(cv.bias.grad.data.numpy()/bfactor,dbias/bfactor,atol=3e-3)
    assert_(all(check_numdiff(sv, xin_np[0])))

def test_apdot_complex():
    num_batch=3
    dim_in=10
    dim_out=4
    xin_np=asfortranarray(typed_randn('complex128',[num_batch,dim_in]))
    weight=asfortranarray(typed_randn('complex128',[dim_out, dim_in]))
    bias=typed_randn('complex128',[dim_out])
    sv=Apdot((num_batch, dim_in), 'complex128',weight, bias)
    print( "Testing numdiff for %s"%sv)
    assert_(all(check_numdiff(sv, num_check=100)))


def test_all():
    random.seed(2)
    torch.manual_seed(2)

    test_splinear()
    test_apdot_complex()
    test_linear_complex()
    test_linear()
    test_linear1()

if __name__=='__main__':
    test_all()
