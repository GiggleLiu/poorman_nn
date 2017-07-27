'''
Tests for MPS and MPO
'''
from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
import sys,pdb,time
sys.path.insert(0,'../')
from spconv import SPConv
from core import check_numdiff
from utils import typed_random
import torch
import torch.nn.functional as F
from torch.nn import Conv1d,Conv2d
from torch import autograd

random.seed(2)
torch.manual_seed(2)

def test_conv2d():
    ts=arange(16).reshape([1,4,4])
    #[ 0, 1, 2, 3]
    #[ 4, 5, 6, 7]
    #[ 8, 9,10,11]
    #[12,13,14,15]
    ts=autograd.Variable(1.0*torch.Tensor([ts,ts+1]))
    #2 features, kernel size 3x3
    cv=Conv2d(1,2,kernel_size=(3,3),stride=(1,1),padding=(0,0))
    cv.weight.data[...]=torch.Tensor([[[[1.,1,1],[1,0,1],[1,1,1.]]],[[[0,1,0],[1,0,1],[0,1,0]]]])
    cv.bias.data[...]=torch.Tensor([0,1])
    res=cv(ts)
    assert_allclose(res.data.numpy(),[[[[40,48],[72,80]],[[21,25],[37,41]]],[[[48,56],[80,88]],[[25,29],[41,45]]]])

    #new
    sv=SPConv((2,1,4,4), fltr=cv.weight.data.numpy(), bias=cv.bias.data.numpy(), dtype='complex128',strides=(1,1),boundary='O')
    x=asfortranarray(ts.data.numpy(), dtype=complex128)
    res2=sv.forward(x)
    assert_allclose(res2,[[[[40,48],[72,80]],[[21,25],[37,41]]],[[[48,56],[80,88]],[[25,29],[41,45]]]])

def test_conv2d_per():
    num_batch=1
    dim_x=30
    dim_y=40
    K1=3
    K2=4
    nfin=10
    nfout=16
    ts=random.random([num_batch,nfin,dim_x,dim_y])
    ts=autograd.Variable(torch.Tensor(ts),requires_grad=True)
    #2 features, kernel size 3x3
    cv=Conv2d(nfin,nfout,kernel_size=(K1,K2),stride=(1,1),padding=(0,0))
    fltr=cv.weight.data.numpy()
    sv=SPConv((-1,nfin,dim_x,dim_y), float32(fltr),float32(cv.bias.data.numpy()),strides=(1,1),boundary='O', w_contiguous=True, output_shape=(-1,nfout,dim_x-K1+1, dim_y-K2+1))
    sv2=SPConv((nfin,dim_x,dim_y), float32(fltr),float32(cv.bias.data.numpy()),strides=(1,1),boundary='O', w_contiguous=True, output_shape=(nfout,dim_x-K1+1, dim_y-K2+1))
    print "Testing forward for %s, 2D"%sv
    xin_np=ts.data.numpy()
    xin_np1=xin_np[0]
    ntest=5
    t0=time.time()
    for i in xrange(ntest):
        y1=cv(ts)
    t1=time.time()
    for i in xrange(ntest):
        y2=sv.forward(xin_np)
    t2=time.time()
    for i in xrange(ntest):
        y3=sv2.forward(xin_np1)
    t3=time.time()
    print "Elapse old = %s, new = %s, new_1 = %s"%(t1-t0,t2-t1,t3-t2)
    res1=y1.data.numpy()
    res2=y2
    res3=y3[newaxis]
    assert_allclose(res1,res2,atol=1e-4)
    assert_allclose(res1,res3,atol=1e-4)

    print "Testing backward"
    dy=torch.randn(*y1.size())
    dy_np=asfortranarray(dy.numpy())
    dy_np1=dy_np[0]

    t0=time.time()
    y1.backward(dy)
    t1=time.time()
    for i in xrange(ntest):
        dwb,dx=sv.backward(xin_np, y2, dy_np, mask=(1,1))
    t2=time.time()
    for i in xrange(ntest):
        dwb1,dx1=sv2.backward(xin_np1, y3, dy_np1, mask=(1,1))
    t3=time.time()
    print "Elapse old = %s, new = %s, new-1 = %s"%(t1-t0,(t2-t1)/ntest,(t3-t2)/ntest)

    #reshape back
    dx0=ts.grad.data.numpy()
    dx1=dx1[newaxis]

    dweight, dbias = dwb[:sv.fltr.size], dwb[sv.fltr.size:]
    dweight1, dbias1 = dwb1[:sv2.fltr.size], dwb1[sv.fltr.size:]
    wfactor=dweight.mean()
    bfactor=dbias.mean()
    dweight=reshape(dweight,cv.weight.size(),order='F')/wfactor
    dweight1=reshape(dweight1,cv.weight.size(),order='F')/wfactor
    assert_allclose(cv.bias.grad.data.numpy()/bfactor,dbias/bfactor,atol=2e-3)
    assert_allclose(cv.bias.grad.data.numpy()/bfactor,dbias1/bfactor,atol=2e-3)

    assert_allclose(dx0,dx,atol=2e-3)
    assert_allclose(dx0,dx1,atol=2e-3)

    assert_allclose(cv.weight.grad.data.numpy()/wfactor,dweight1,atol=2e-3)
    assert_allclose(cv.weight.grad.data.numpy()/wfactor,dweight,atol=2e-3)
    assert_(check_numdiff(sv, xin_np))

def test_conv1d_per():
    num_batch=1
    dim_x=30
    K1=3
    nfin=10
    nfout=16
    ts=random.random([num_batch,nfin,dim_x])
    ts=autograd.Variable(torch.Tensor(ts),requires_grad=True)
    #2 features, kernel size 3x3
    cv=Conv2d(nfin,nfout,kernel_size=(K1,),stride=(1,),padding=(0,))
    fltr=cv.weight.data.numpy()
    sv=SPConv((-1,nfin,dim_x), float32(fltr),float32(cv.bias.data.numpy()),strides=(1,),boundary='O', w_contiguous=True, output_shape=(-1,nfout,dim_x-K1+1))
    sv2=SPConv((nfin,dim_x), float32(fltr),float32(cv.bias.data.numpy()),strides=(1,),boundary='O', w_contiguous=True, output_shape=(nfout,dim_x-K1+1))
    print "Testing forward for %s, 1D"%sv
    xin_np=asfortranarray(ts.data.numpy())
    xin_np1=xin_np[0]
    ntest=5
    t0=time.time()
    for i in xrange(ntest):
        y1=cv(ts)
    t1=time.time()
    for i in xrange(ntest):
        y2=sv.forward(xin_np)
    t2=time.time()
    for i in xrange(ntest):
        y3=sv2.forward(xin_np1)
    t3=time.time()
    print "Elapse old = %s, new = %s, new_1 = %s"%(t1-t0,t2-t1,t3-t2)
    res1=y1.data.numpy()
    res2=y2
    res3=y3[newaxis]
    assert_allclose(res1,res2,atol=1e-4)
    assert_allclose(res1,res3,atol=1e-4)

    print "Testing backward"
    dy=torch.randn(*y1.size())
    dy_np=asfortranarray(dy.numpy())
    dy_np1=dy_np[0]

    t0=time.time()
    y1.backward(dy)
    t1=time.time()
    for i in xrange(ntest):
        dwb,dx=sv.backward(xin_np, y2, dy_np, mask=(1,1))
    t2=time.time()
    for i in xrange(ntest):
        dwb1,dx1=sv2.backward(xin_np1, y3, dy_np1, mask=(1,1))
    t3=time.time()
    print "Elapse old = %s, new = %s, new-1 = %s"%(t1-t0,(t2-t1)/ntest,(t3-t2)/ntest)

    #reshape back
    dx0=ts.grad.data.numpy()
    dx1=dx1[newaxis]

    dweight, dbias = dwb[:sv.fltr.size], dwb[sv.fltr.size:]
    dweight1, dbias1 = dwb1[:sv2.fltr.size], dwb1[sv.fltr.size:]
    wfactor=dweight.mean()
    bfactor=dbias.mean()
    dweight=reshape(dweight,cv.weight.size(), order='F')/wfactor
    dweight1=reshape(dweight1,cv.weight.size(), order='F')/wfactor
    assert_allclose(cv.bias.grad.data.numpy()/bfactor,dbias/bfactor,atol=2e-3)
    assert_allclose(cv.bias.grad.data.numpy()/bfactor,dbias1/bfactor,atol=2e-3)

    assert_allclose(dx0,dx,atol=2e-3)
    assert_allclose(dx0,dx1,atol=2e-3)

    assert_allclose(cv.weight.grad.data.numpy()/wfactor,dweight1,atol=2e-3)
    assert_allclose(cv.weight.grad.data.numpy()/wfactor,dweight,atol=2e-3)
    assert_(check_numdiff(sv, xin_np))

def test_conv2d_complex():
    num_batch=1
    dim_x=30
    dim_y=40
    K1=3
    K2=4
    nfin=10
    nfout=16
    xin_np=asfortranarray(typed_random(complex128,[num_batch,nfin,dim_x,dim_y]))
    fltr=asfortranarray(typed_random(complex128,[nfout,nfin,K1,K2]))
    bias=typed_random(complex128,nfout)
    sv=SPConv((-1,nfin,dim_x,dim_y), fltr, bias, strides=(1,1), boundary='O', w_contiguous=True, output_shape=(-1,nfout,dim_x-K1+1, dim_y-K2+1),dtype='complex128')
    sv2=SPConv((nfin,dim_x,dim_y), fltr, bias, strides=(1,1), boundary='O', w_contiguous=True, output_shape=(nfout,dim_x-K1+1, dim_y-K2+1),dtype='complex128')
    print "Testing forward for %s"%sv
    xin_np1=xin_np[0]
    ntest=5
    t1=time.time()
    for i in xrange(ntest):
        y2=sv.forward(xin_np)
    t2=time.time()
    for i in xrange(ntest):
        y3=sv2.forward(xin_np1)
    t3=time.time()
    print "Elapse new = %s, new_1 = %s"%(t2-t1,t3-t2)
    res2=y2
    res3=y3[newaxis]
    assert_allclose(res2,res2,atol=1e-4)

    print "Testing backward"
    dy_np=typed_random(complex128, y2.shape)
    dy_np1=dy_np[0]

    t1=time.time()
    for i in xrange(ntest):
        dwb,dx=sv.backward(xin_np, y2, dy_np, mask=(1,1))
    t2=time.time()
    for i in xrange(ntest):
        dwb1,dx1=sv2.backward(xin_np1, y3, dy_np1, mask=(1,1))
    t3=time.time()
    print "Elapse new = %s, new-1 = %s"%((t2-t1)/ntest,(t3-t2)/ntest)

    #reshape back
    dx1=dx1[newaxis]

    dweight, dbias = dwb[:sv.fltr.size], dwb[sv.fltr.size:]
    dweight1, dbias1 = dwb1[:sv2.fltr.size], dwb1[sv.fltr.size:]
    dweight1=reshape(dweight1,dweight.shape,order='F')

    assert_allclose(dbias,dbias1,atol=1e-3)
    assert_allclose(dx,dx1,atol=1e-3)
    assert_allclose(dweight1,dweight,atol=1e-3)

    assert_(check_numdiff(sv, xin_np))

test_conv2d_complex()
test_conv2d()
test_conv2d_per()
test_conv1d_per()
